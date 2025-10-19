use super::config::GptOssConfig;
use super::experts::{ExpertMlp, GptOssExperts};
use super::load_linear_maybe_mxfp4;
use crate::models::with_tracing::{linear_no_bias, Embedding, RmsNorm};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

// Constants
const DEFAULT_RMS_EPS: f64 = 1e-5;

#[derive(Debug, Clone)]
pub struct GptOssAttentionWeights {
    pub q_proj: crate::models::with_tracing::Linear,
    pub k_proj: crate::models::with_tracing::Linear,
    pub v_proj: crate::models::with_tracing::Linear,
    pub o_proj: crate::models::with_tracing::Linear,
    pub sinks: Tensor, // (num_heads)
}

#[derive(Debug, Clone)]
pub struct GptOssLayerWeights {
    pub attn: GptOssAttentionWeights,
    pub router: candle_nn::Linear, // hidden -> num_local_experts
    pub experts: GptOssExperts,
}

#[derive(Debug, Clone)]
pub struct GptOssModel {
    pub cfg: GptOssConfig,
    pub embed: Embedding,
    pub norm: RmsNorm,
    pub layers: Vec<GptOssLayerWeights>,
    pub lm_head: candle_nn::Linear,
}

impl GptOssModel {
    pub fn load(vb: VarBuilder, cfg: &GptOssConfig) -> Result<Self> {
        let vb_bf16 = vb.to_dtype(DType::BF16);
        let dev = vb.device().clone();
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;

        // Embedding and pre/post norms
        let embed = Embedding::new(cfg.vocab_size, hidden, vb_bf16.pp("model.embed_tokens"))?;
        let norm = {
            let eps = cfg.rms_norm_eps.unwrap_or(DEFAULT_RMS_EPS);
            RmsNorm::new(hidden, eps, vb_bf16.pp("model.norm"))?
        };

        // Per-layer weights
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let l_vb = vb_bf16.pp(&format!("model.layers.{i}"));

            // Attention projections
            let attn_vb = l_vb.pp("self_attn");
            let q_proj = linear_no_bias(hidden, cfg.num_attention_heads * head_dim, attn_vb.pp("q_proj"))?;
            let k_proj = linear_no_bias(hidden, cfg.num_key_value_heads * head_dim, attn_vb.pp("k_proj"))?;
            let v_proj = linear_no_bias(hidden, cfg.num_key_value_heads * head_dim, attn_vb.pp("v_proj"))?;
            let o_proj = linear_no_bias(cfg.num_attention_heads * head_dim, hidden, attn_vb.pp("o_proj"))?;
            let sinks = if attn_vb.contains_tensor("sinks") {
                attn_vb.get(cfg.num_attention_heads, "sinks")?
            } else {
                Tensor::zeros(cfg.num_attention_heads, DType::BF16, &Device::Cpu)?
                    .to_device(&dev)?
            };
            let attn = GptOssAttentionWeights { q_proj, k_proj, v_proj, o_proj, sinks };

            // MoE router and experts
            let mlp_vb = l_vb.pp("mlp");
            let router = candle_nn::linear_no_bias(hidden, cfg.num_local_experts, mlp_vb.pp("router"))?;

            // Experts: fused gate_up and down. Use MXFP4 dequant path if available.
            let experts = {
                let mut all = Vec::with_capacity(cfg.num_local_experts);
                for e in 0..cfg.num_local_experts {
                    let ebase = mlp_vb.pp(&format!("experts.{e}"));
                    // Fused gate_up has output 2*intermediate_size
                    let inter = cfg.intermediate_size;
                    let gate_up = load_linear_maybe_mxfp4(
                        hidden,
                        2 * inter,
                        false,
                        ebase.clone(),
                        "gate_up_proj",
                    )?;
                    let down = load_linear_maybe_mxfp4(
                        inter,
                        hidden,
                        false,
                        ebase.clone(),
                        "down_proj",
                    )?;
                    all.push(ExpertMlp::new(gate_up, down, candle_nn::Activation::Silu));
                }
                GptOssExperts::new(router.clone(), all, Some(cfg.num_experts_per_tok))
            };

            layers.push(GptOssLayerWeights { attn, router, experts });
        }

        // Untied LM head
        let lm_head = candle_nn::linear_no_bias(hidden, cfg.vocab_size, vb_bf16.pp("lm_head"))?;

        Ok(Self { cfg: cfg.clone(), embed, norm, layers, lm_head })
    }

    /// Minimal forward used for shape/dtype validation: embed tokens and project to logits via lm_head.
    /// This does not execute transformer layers; it validates assembly wiring and weight dtypes.
    pub fn forward_logits_minimal(&self, input_ids: &Tensor) -> Result<Tensor> {
        // input_ids: (b, t)
        let xs = self.embed.forward(input_ids)?; // (b,t,hidden)

        // Use F32 matmul on CPU for stability/numeric support, then cast back to BF16 to match weights dtype.
        let (b, t, h) = xs.dims3()?;
        let xs2 = xs.reshape(((), h))?.to_dtype(DType::F32)?; // (bt, h)
        let w = self.lm_head.weight().to_dtype(DType::F32)?; // (vocab, h)
        let logits = xs2.matmul(&w.t()?)?; // (bt, vocab)
        let logits = logits.reshape((b, t, self.cfg.vocab_size))?.to_dtype(DType::BF16)?;
        Ok(logits)
    }
}
