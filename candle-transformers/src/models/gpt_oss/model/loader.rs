use candle::{DType, Device, Result};
use candle_nn::VarBuilder;

use super::{GptOssAttentionWeights, GptOssLayerWeights, GptOssModel};
use crate::models::gpt_oss::config::GptOssConfig;
use crate::models::gpt_oss::experts::{ExpertMlp, GptOssExperts};
use crate::models::gpt_oss::quantization::{
    load_all_experts_linear_mxfp4_grouped, load_expert_linear_mxfp4_grouped,
    load_linear_maybe_mxfp4,
};
use crate::models::gpt_oss::rotary::{GptOssRopeConfig, GptOssRotaryEmbedding};
use crate::models::with_tracing::{linear, Embedding, RmsNorm};

const DEFAULT_RMS_EPS: f64 = 1e-5;
const ENV_KEEP_RMS_FP32: &str = "CANDLE_RMS_FP32";

impl GptOssModel {
    pub fn load(vb: VarBuilder, cfg: &GptOssConfig) -> Result<Self> {
        let vb_bf16 = vb.to_dtype(DType::BF16);
        let dev = vb.device().clone();
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;
        let keep_rms_in_fp32 = match std::env::var(ENV_KEEP_RMS_FP32).ok().as_deref() {
            Some("0") | Some("false") | Some("FALSE") => false,
            _ => true,
        };

        let embed = Embedding::new(cfg.vocab_size, hidden, vb_bf16.pp("model.embed_tokens"))?;
        let norm = {
            let eps = cfg.rms_norm_eps.unwrap_or(DEFAULT_RMS_EPS);
            RmsNorm::new_with_mode(hidden, eps, vb_bf16.pp("model.norm"), keep_rms_in_fp32)?
        };

        let rope_cfg = {
            let rs = cfg.rope_scaling.clone().unwrap_or_default();
            let factor = rs.factor.unwrap_or(32.0);
            let beta_fast = rs.beta_fast.unwrap_or(32.0);
            let beta_slow = rs.beta_slow.unwrap_or(1.0);
            let original_max = rs.original_max_position_embeddings.unwrap_or(4096);
            let theta = cfg.rope_theta.unwrap_or(150000.0);
            GptOssRopeConfig::new(
                cfg.head_dim(),
                cfg.max_position_embeddings,
                theta,
                factor,
                beta_fast,
                beta_slow,
                original_max,
            )
        };
        let rope = GptOssRotaryEmbedding::new_yarn(DType::BF16, &dev, &rope_cfg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let l_vb = vb_bf16.pp(&format!("model.layers.{i}"));

            let attn_vb = l_vb.pp("self_attn");
            let q_proj = if attn_vb.contains_tensor("q_proj.bias") {
                linear(
                    hidden,
                    cfg.num_attention_heads * head_dim,
                    attn_vb.pp("q_proj"),
                )?
            } else {
                crate::models::with_tracing::linear_no_bias(
                    hidden,
                    cfg.num_attention_heads * head_dim,
                    attn_vb.pp("q_proj"),
                )?
            };
            let k_proj = if attn_vb.contains_tensor("k_proj.bias") {
                linear(
                    hidden,
                    cfg.num_key_value_heads * head_dim,
                    attn_vb.pp("k_proj"),
                )?
            } else {
                crate::models::with_tracing::linear_no_bias(
                    hidden,
                    cfg.num_key_value_heads * head_dim,
                    attn_vb.pp("k_proj"),
                )?
            };
            let v_proj = if attn_vb.contains_tensor("v_proj.bias") {
                linear(
                    hidden,
                    cfg.num_key_value_heads * head_dim,
                    attn_vb.pp("v_proj"),
                )?
            } else {
                crate::models::with_tracing::linear_no_bias(
                    hidden,
                    cfg.num_key_value_heads * head_dim,
                    attn_vb.pp("v_proj"),
                )?
            };
            let o_proj = if attn_vb.contains_tensor("o_proj.bias") {
                linear(
                    cfg.num_attention_heads * head_dim,
                    hidden,
                    attn_vb.pp("o_proj"),
                )?
            } else {
                crate::models::with_tracing::linear_no_bias(
                    cfg.num_attention_heads * head_dim,
                    hidden,
                    attn_vb.pp("o_proj"),
                )?
            };
            let sinks = if attn_vb.contains_tensor("sinks") {
                attn_vb.get(cfg.num_attention_heads, "sinks")?
            } else {
                candle::Tensor::zeros(cfg.num_attention_heads, DType::BF16, &Device::Cpu)?
                    .to_device(&dev)?
            };
            let attn = GptOssAttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                sinks,
            };

            // HF/GPT-OSS uses `mlp.router` with a bias for the MoE router.
            let router =
                candle_nn::linear(hidden, cfg.num_local_experts, l_vb.pp("mlp.router"))?;
            let experts = {
                let mlp_vb = l_vb.pp("mlp");
                let inter = cfg.intermediate_size;
                let limit = cfg.swiglu_limit.unwrap_or(7.0);
                let alpha = 1.702f32;

                // Try batched loading (much faster: one dequant kernel instead of E)
                let gate_up_all = load_all_experts_linear_mxfp4_grouped(
                    hidden,
                    2 * inter,
                    true,
                    mlp_vb.clone(),
                    "experts.gate_up_proj",
                    cfg.num_local_experts,
                );
                let down_all = load_all_experts_linear_mxfp4_grouped(
                    inter,
                    hidden,
                    true,
                    mlp_vb.clone(),
                    "experts.down_proj",
                    cfg.num_local_experts,
                );

                let all: Vec<ExpertMlp> =
                    if let (Ok(gate_ups), Ok(downs)) = (gate_up_all, down_all) {
                        // Batched load succeeded - zip into ExpertMlp structs
                        gate_ups
                            .into_iter()
                            .zip(downs.into_iter())
                            .map(|(gate_up, down)| ExpertMlp::new(gate_up, down, limit, alpha))
                            .collect()
                    } else {
                        // Batched load failed - fall back to per-expert loading
                        let mut all: Vec<ExpertMlp> = Vec::with_capacity(cfg.num_local_experts);
                        for e in 0..cfg.num_local_experts {
                            let gate_up = match load_expert_linear_mxfp4_grouped(
                                hidden,
                                2 * inter,
                                true,
                                mlp_vb.clone(),
                                "experts.gate_up_proj",
                                e,
                                cfg.num_local_experts,
                            ) {
                                Ok(l) => l,
                                Err(_) => load_linear_maybe_mxfp4(
                                    hidden,
                                    2 * inter,
                                    true,
                                    mlp_vb.pp(&format!("experts.{e}")),
                                    "gate_up_proj",
                                )?,
                            };
                            let down = match load_expert_linear_mxfp4_grouped(
                                inter,
                                hidden,
                                true,
                                mlp_vb.clone(),
                                "experts.down_proj",
                                e,
                                cfg.num_local_experts,
                            ) {
                                Ok(l) => l,
                                Err(_) => load_linear_maybe_mxfp4(
                                    inter,
                                    hidden,
                                    true,
                                    mlp_vb.pp(&format!("experts.{e}")),
                                    "down_proj",
                                )?,
                            };
                            all.push(ExpertMlp::new(gate_up, down, limit, alpha));
                        }
                        all
                    };
                GptOssExperts::new(router.clone(), all, Some(cfg.num_experts_per_tok))
            };
            let eps = cfg.rms_norm_eps.unwrap_or(DEFAULT_RMS_EPS);
            let input_layernorm = RmsNorm::new_with_mode(
                hidden,
                eps,
                vb_bf16.pp(&format!("model.layers.{i}.input_layernorm")),
                keep_rms_in_fp32,
            )?;
            let post_attention_layernorm = RmsNorm::new_with_mode(
                hidden,
                eps,
                vb_bf16.pp(&format!("model.layers.{i}.post_attention_layernorm")),
                keep_rms_in_fp32,
            )?;

            layers.push(GptOssLayerWeights {
                attn,
                input_layernorm,
                post_attention_layernorm,
                router,
                experts,
            });
        }

        let lm_head = candle_nn::linear_no_bias(hidden, cfg.vocab_size, vb_bf16.pp("lm_head"))?;
        let mut kv_caches = Vec::with_capacity(cfg.num_hidden_layers);
        let layer_types = cfg.effective_layer_types();
        for (i, _ly) in layer_types.iter().enumerate() {
            let attn_mode = crate::models::gpt_oss::select_attn_mode_for_layer(
                &crate::models::gpt_oss::GptOssConfigMinimal {
                    num_hidden_layers: cfg.num_hidden_layers,
                    layer_types: layer_types.clone(),
                    max_position_embeddings: cfg.max_position_embeddings,
                    sliding_window: cfg.sliding_window,
                },
                i,
            );
            let window = match attn_mode {
                crate::models::gpt_oss::AttnMode::Full => cfg.max_position_embeddings,
                crate::models::gpt_oss::AttnMode::Sliding { left, .. } => left,
            };
            kv_caches.push(candle_nn::kv_cache::RotatingKvCache::new(2, window));
        }

        Ok(Self {
            cfg: cfg.clone(),
            embed,
            norm,
            layers,
            lm_head,
            rope,
            kv_caches,
        })
    }
}
