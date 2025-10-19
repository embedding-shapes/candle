use super::config::GptOssConfig;
use super::experts::{ExpertMlp, GptOssExperts};
use super::{load_expert_linear_mxfp4_grouped, load_linear_maybe_mxfp4, select_attn_mode_for_layer, AttnMode};
use crate::models::with_tracing::{linear, Embedding, RmsNorm};
use crate::models::gpt_oss::rotary::{GptOssRopeConfig, GptOssRotaryEmbedding};
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
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
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
    // Rotary embedding (YARN) shared across layers.
    pub rope: GptOssRotaryEmbedding,
    // Per-layer rotating KV cache (stores KV in (b, h_kv, t, d) layout).
    pub kv_caches: Vec<candle_nn::kv_cache::RotatingKvCache>,
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
        
        // Rotary (YARN) configuration
        let rope_cfg = {
            let rs = cfg.rope_scaling.clone().unwrap_or_default();
            let factor = rs.factor.unwrap_or(32.0);
            let beta_fast = rs.beta_fast.unwrap_or(32.0);
            let beta_slow = rs.beta_slow.unwrap_or(1.0);
            let original_max = rs.original_max_position_embeddings.unwrap_or(4096);
            let theta = cfg.rope_theta.unwrap_or(150000.0);
            GptOssRopeConfig::new(cfg.head_dim(), cfg.max_position_embeddings, theta, factor, beta_fast, beta_slow, original_max)
        };
        let rope = GptOssRotaryEmbedding::new_yarn(DType::BF16, &dev, &rope_cfg)?;

        // Per-layer weights
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let l_vb = vb_bf16.pp(&format!("model.layers.{i}"));

            // Attention projections (attention_bias=true in config; be robust to missing bias in tests)
            let attn_vb = l_vb.pp("self_attn");
            let q_proj = if attn_vb.contains_tensor("q_proj.bias") {
                linear(hidden, cfg.num_attention_heads * head_dim, attn_vb.pp("q_proj"))?
            } else {
                crate::models::with_tracing::linear_no_bias(hidden, cfg.num_attention_heads * head_dim, attn_vb.pp("q_proj"))?
            };
            let k_proj = if attn_vb.contains_tensor("k_proj.bias") {
                linear(hidden, cfg.num_key_value_heads * head_dim, attn_vb.pp("k_proj"))?
            } else {
                crate::models::with_tracing::linear_no_bias(hidden, cfg.num_key_value_heads * head_dim, attn_vb.pp("k_proj"))?
            };
            let v_proj = if attn_vb.contains_tensor("v_proj.bias") {
                linear(hidden, cfg.num_key_value_heads * head_dim, attn_vb.pp("v_proj"))?
            } else {
                crate::models::with_tracing::linear_no_bias(hidden, cfg.num_key_value_heads * head_dim, attn_vb.pp("v_proj"))?
            };
            let o_proj = if attn_vb.contains_tensor("o_proj.bias") {
                linear(cfg.num_attention_heads * head_dim, hidden, attn_vb.pp("o_proj"))?
            } else {
                crate::models::with_tracing::linear_no_bias(cfg.num_attention_heads * head_dim, hidden, attn_vb.pp("o_proj"))?
            };
            let sinks = if attn_vb.contains_tensor("sinks") {
                attn_vb.get(cfg.num_attention_heads, "sinks")?
            } else {
                Tensor::zeros(cfg.num_attention_heads, DType::BF16, &Device::Cpu)?
                    .to_device(&dev)?
            };
            let attn = GptOssAttentionWeights { q_proj, k_proj, v_proj, o_proj, sinks };

            // MoE router and experts
            let mlp_vb = l_vb.pp("mlp");
            let router = if mlp_vb.contains_tensor("router.bias") {
                candle_nn::linear(hidden, cfg.num_local_experts, mlp_vb.pp("router"))?
            } else {
                candle_nn::linear_no_bias(hidden, cfg.num_local_experts, mlp_vb.pp("router"))?
            };

            // Experts: fused gate_up and down. Use MXFP4 dequant path if available.
            let experts = {
                let mut all = Vec::with_capacity(cfg.num_local_experts);
                for e in 0..cfg.num_local_experts {
                    // Fused gate_up has output 2*intermediate_size
                    let inter = cfg.intermediate_size;
                    // Prefer grouped MXFP4 if present, otherwise try per-expert path.
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
                    // Use GPT-OSS clamped SwiGLU with residual tweak
                    let limit = cfg.swiglu_limit.unwrap_or(7.0);
                    let alpha = 1.702f32;
                    all.push(ExpertMlp::new(gate_up, down, limit, alpha));
                }
                GptOssExperts::new(router.clone(), all, Some(cfg.num_experts_per_tok))
            };
            // Per-layer norms
            let eps = cfg.rms_norm_eps.unwrap_or(DEFAULT_RMS_EPS);
            let input_layernorm = RmsNorm::new(hidden, eps, vb_bf16.pp(&format!("model.layers.{i}.input_layernorm")))?;
            let post_attention_layernorm = RmsNorm::new(hidden, eps, vb_bf16.pp(&format!("model.layers.{i}.post_attention_layernorm")))?;

            layers.push(GptOssLayerWeights { attn, input_layernorm, post_attention_layernorm, router, experts });
        }

        // Untied LM head
        let lm_head = candle_nn::linear_no_bias(hidden, cfg.vocab_size, vb_bf16.pp("lm_head"))?;
        // Per-layer KV caches initialized with appropriate window sizes
        let mut kv_caches = Vec::with_capacity(cfg.num_hidden_layers);
        let layer_types = cfg.effective_layer_types();
        for (i, _ly) in layer_types.iter().enumerate() {
            let attn_mode = select_attn_mode_for_layer(
                &super::GptOssConfigMinimal {
                    num_hidden_layers: cfg.num_hidden_layers,
                    layer_types: layer_types.clone(),
                    max_position_embeddings: cfg.max_position_embeddings,
                    sliding_window: cfg.sliding_window,
                },
                i,
            );
            let window = match attn_mode {
                AttnMode::Full => cfg.max_position_embeddings,
                AttnMode::Sliding { left, .. } => left,
            };
            // Cache along the sequence dimension (index 2) for K/V tensors shaped (b, h_kv, t, d)
            kv_caches.push(candle_nn::kv_cache::RotatingKvCache::new(2, window));
        }

        Ok(Self { cfg: cfg.clone(), embed, norm, layers, lm_head, rope, kv_caches })
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

    /// Full forward for causal decode with KV cache, GQA, YARN RoPE, sinks + flash-attn.
    /// Returns logits for all provided tokens (b, t, vocab).
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        // Embed tokens
        let mut xs = self.embed.forward(input_ids)?; // (b,t,hidden)
        let (b, t, hidden) = xs.dims3()?;
        let head_dim = self.cfg.head_dim();
        let n_q = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        // Apply YARN attention scaling in the score path. We follow the variant where
        // sin/cos tables are unscaled and only the attention softmax scale is multiplied
        // by mscale^2. Do not also scale sin/cos to avoid double-application.
        let yarn_mscale = {
            let factor = self
                .cfg
                .rope_scaling
                .as_ref()
                .and_then(|r| r.factor)
                .unwrap_or(1.0);
            crate::models::gpt_oss::rotary::yarn_get_mscale(factor)
        };
        let softmax_scale = (1.0f32 / (head_dim as f32).sqrt()) * (yarn_mscale * yarn_mscale);
        

        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Pre-attention norm
            let x_norm = layer.input_layernorm.forward(&xs)?; // (b,t,h)

            // Projections
            let q = x_norm.apply(&layer.attn.q_proj)?; // (b,t,n_q*hd)
            let k = x_norm.apply(&layer.attn.k_proj)?; // (b,t,n_kv*hd)
            let v = x_norm.apply(&layer.attn.v_proj)?; // (b,t,n_kv*hd)

            // Reshape to heads
            let q = q.reshape((b, t, n_q, head_dim))?;
            let k = k.reshape((b, t, n_kv, head_dim))?;
            let v = v.reshape((b, t, n_kv, head_dim))?;

            // Apply RoPE (expects (b,h,t,d)) with offset
            let q_bhtd = q.transpose(1, 2)?; // (b,n_q,t,d)
            let k_bhtd = k.transpose(1, 2)?; // (b,n_kv,t,d)
            let (q_bhtd, k_bhtd) = self.rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, seqlen_offset)?;
            let q = q_bhtd.transpose(1, 2)?; // (b,t,n_q,d)
            let k_step = k_bhtd; // (b,n_kv,t,d) for cache
            let v_step = v.transpose(1, 2)?; // (b,n_kv,t,d)

            // Append to KV cache (trimmed to layer’s window if sliding)
            let (k_all, v_all) = self.kv_caches[i].append(&k_step.contiguous()?, &v_step.contiguous()?)?; // (b,n_kv,tk,d)

            // Prepare K,V for attention: repeat for GQA and transpose to (b,tk,n_q,d)
            let n_rep = n_q / n_kv;
            let k_rep = crate::utils::repeat_kv(k_all.clone(), n_rep)?; // (b,n_q,tk,d)
            let v_rep = crate::utils::repeat_kv(v_all.clone(), n_rep)?; // (b,n_q,tk,d)
            let k_btkhd = k_rep.transpose(1, 2)?; // (b,tk,n_q,d)
            let v_btkhd = v_rep.transpose(1, 2)?; // (b,tk,n_q,d)

            // Choose attention mode based on layer_types
            let attn_mode = super::select_attn_mode_for_layer(
                &super::GptOssConfigMinimal {
                    num_hidden_layers: self.cfg.num_hidden_layers,
                    layer_types: self.cfg.effective_layer_types(),
                    max_position_embeddings: self.cfg.max_position_embeddings,
                    sliding_window: self.cfg.sliding_window,
                },
                i,
            );

            // Allow disabling sinks renormalization for ablation via env var.
            let sinks = if matches!(std::env::var("CANDLE_DISABLE_SINKS").ok().as_deref(), Some("1") | Some("true") | Some("TRUE")) {
                None
            } else {
                Some(&layer.attn.sinks)
            };
            #[cfg(feature = "flash-attn")]
            let y = {
                let use_fa = !matches!(std::env::var("CANDLE_DISABLE_FLASH").ok().as_deref(), Some("1") | Some("true") | Some("TRUE"));
                if use_fa {
                    match attn_mode {
                        AttnMode::Full => super::flash_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                        AttnMode::Sliding { left, right } => super::flash_attn_windowed_with_sinks(
                            &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                        )?,
                    }
                } else {
                    match attn_mode {
                        AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                        AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(
                            &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                        )?,
                    }
                }
            };
            #[cfg(not(feature = "flash-attn"))]
            let y = {
                match attn_mode {
                    AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                    AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(
                        &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                    )?,
                }
            }; // (b,t,n_q,d)

            // Merge heads and project out
            let y = y.reshape((b, t, n_q * head_dim))?;
            let y = y.apply(&layer.attn.o_proj)?; // (b,t,h)
            xs = (xs + y)?;

            // Post-attention norm + MoE MLP
            let x_norm2 = layer.post_attention_layernorm.forward(&xs)?; // (b,t,h)
            let mlp_out = layer.experts.forward(&x_norm2)?; // (b,t,h)
            xs = (xs + mlp_out)?;
        }

        // Final norm and lm head
        let xs = self.norm.forward(&xs)?; // (b,t,h)
        let xs2 = xs.reshape(((), hidden))?;
        let logits = xs2.apply(&self.lm_head)?.reshape((b, t, self.cfg.vocab_size))?;
        Ok(logits)
    }
}
