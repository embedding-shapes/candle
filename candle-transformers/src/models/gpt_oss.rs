//! GPT-OSS specific helpers.
//!
//! This module implements:
//! - MXFP4 expert weight load-time detection and dequantization path.
//! - Attention utilities (eager/FA) with sinks renormalization, including windowed variants.
//! - Minimal config parsing for `layer_types` and `sliding_window`, plus helpers to
//!   select per-layer attention mode (full vs sliding/windowed).

use candle::{DType, Result, Tensor, D};
use candle_nn::Linear;

// Centralized GPT-OSS implementation: inline submodules.
// Keep public API paths stable (config::, experts::, rotary::, model::).

pub mod config {
    use super::GptOssLayerType;

    #[derive(Debug, Clone, serde::Deserialize, Default)]
    pub struct RopeScalingConfig {
        #[serde(default, alias = "type")]
        pub r#type: Option<String>,
        #[serde(default)]
        pub rope_type: Option<String>,
        #[serde(default)]
        pub factor: Option<f32>,
        #[serde(default)]
        pub beta_fast: Option<f32>,
        #[serde(default)]
        pub beta_slow: Option<f32>,
        #[serde(default)]
        pub original_max_position_embeddings: Option<usize>,
    }

    #[derive(Debug, Clone, serde::Deserialize)]
    pub struct GptOssConfig {
        pub vocab_size: usize,
        pub hidden_size: usize,
        pub num_hidden_layers: usize,
        pub num_attention_heads: usize,
        pub num_key_value_heads: usize,
        #[serde(default)]
        pub head_dim: Option<usize>,

        // MoE
        pub num_local_experts: usize,
        pub num_experts_per_tok: usize,
        pub intermediate_size: usize,

        // Positional
        pub max_position_embeddings: usize,
        #[serde(default)]
        pub rope_theta: Option<f32>,
        #[serde(default)]
        pub rope_scaling: Option<RopeScalingConfig>,

        // Attention layout
        #[serde(default)]
        pub layer_types: Vec<GptOssLayerType>,
        #[serde(default)]
        pub sliding_window: Option<usize>,

        // Numerics / norms
        #[serde(default)]
        pub rms_norm_eps: Option<f64>,

        // Feed-forward variant tuning
        #[serde(default)]
        pub swiglu_limit: Option<f32>,
    }

    impl GptOssConfig {
        pub fn head_dim(&self) -> usize {
            self.head_dim
                .unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
        }

        pub fn effective_layer_types(&self) -> Vec<GptOssLayerType> {
            if self.layer_types.is_empty() {
                vec![GptOssLayerType::FullAttention; self.num_hidden_layers]
            } else {
                self.layer_types.clone()
            }
        }
    }
}

pub mod experts {
    use candle::{DType, Module, Result, Tensor, D};
    use candle_nn::{ops, Linear};

    // Configurable constants
    const DEFAULT_TOP_K: usize = 4;

    #[derive(Debug, Clone)]
    pub struct ExpertMlp {
        pub gate_up: Linear,
        pub down: Linear,
        pub limit: f32,
        pub alpha: f32,
    }

    impl ExpertMlp {
        pub fn new(gate_up: Linear, down: Linear, limit: f32, alpha: f32) -> Self {
            Self { gate_up, down, limit, alpha }
        }
    }

    impl Module for ExpertMlp {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let gu = xs.apply(&self.gate_up)?;
            let gu = gu.to_dtype(DType::F32)?;
            let (_n, two_inter) = gu.dims2()?;
            let inter = two_inter / 2;
            let mut gate = gu.narrow(D::Minus1, 0, inter)?;
            let mut up = gu.narrow(D::Minus1, inter, inter)?;

            let limit_t = Tensor::new(self.limit, xs.device())?.to_dtype(DType::F32)?;
            let gate_lim = limit_t.broadcast_as(gate.shape().dims())?;
            gate = gate.minimum(&gate_lim)?;
            up = up.clamp(-self.limit, self.limit)?;

            let alpha_t = Tensor::new(self.alpha, xs.device())?.to_dtype(DType::F32)?;
            let gate_alpha = gate.broadcast_mul(&alpha_t)?;
            let sig = ops::sigmoid(&gate_alpha)?;
            let glu = gate.broadcast_mul(&sig)?;

            let one_t = Tensor::new(1.0f32, xs.device())?.to_dtype(DType::F32)?;
            let up_plus = up.broadcast_add(&one_t)?;
            let fused = up_plus.broadcast_mul(&glu)?;
            let fused = fused.to_dtype(xs.dtype())?;
            fused.apply(&self.down)
        }
    }

    #[derive(Debug, Clone)]
    pub struct GptOssExperts {
        pub router: Linear,
        pub experts: Vec<ExpertMlp>,
        pub num_experts_per_tok: usize,
    }

    impl GptOssExperts {
        pub fn new(router: Linear, experts: Vec<ExpertMlp>, num_experts_per_tok: Option<usize>) -> Self {
            let k_env = std::env::var("CANDLE_MOE_TOPK").ok().and_then(|s| s.parse::<usize>().ok());
            let k = k_env.or(num_experts_per_tok).unwrap_or(DEFAULT_TOP_K);
            Self { router, experts, num_experts_per_tok: k }
        }

        pub fn router_topk_softmax(&self, logits: &Tensor) -> Result<(Tensor, Tensor)> {
            let k = self.num_experts_per_tok;
            let topk_idx = logits
                .arg_sort_last_dim(false)?
                .narrow(D::Minus1, 0, k)?
                .contiguous()?;
            let selected = logits.gather(&topk_idx, D::Minus1)?;
            let probs = candle_nn::ops::softmax_last_dim(&selected.to_dtype(DType::F32)?)?;
            Ok((topk_idx, probs))
        }
    }

    impl Module for GptOssExperts {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let (b, t, h) = xs.dims3()?;
            let xs2 = xs.reshape(((), h))?;
            let logits = xs2.apply(&self.router)?;

            let (topk_idx, probs) = self.router_topk_softmax(&logits)?;

            let probs = probs.to_dtype(DType::F32)?;
            let probs_host = probs.to_vec2::<f32>()?;
            let idx_host = topk_idx.to_vec2::<u32>()?;

            let n_experts = self.experts.len();
            let mut token_ids: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
            let mut token_wts: Vec<Vec<f32>> = vec![Vec::new(); n_experts];
            for (row, (row_probs, row_experts)) in probs_host.iter().zip(idx_host.iter()).enumerate() {
                for (&p, &e) in row_probs.iter().zip(row_experts.iter()) {
                    token_ids[e as usize].push(row as u32);
                    token_wts[e as usize].push(p);
                }
            }

            let mut ys = xs2.zeros_like()?;
            for (e_idx, expert) in self.experts.iter().enumerate() {
                let ids = &token_ids[e_idx];
                if ids.is_empty() {
                    continue;
                }
                let ids_t = Tensor::new(ids.as_slice(), xs2.device())?;
                let wts_t = Tensor::new(token_wts[e_idx].as_slice(), xs2.device())?
                    .reshape(((), 1))?
                    .to_dtype(xs2.dtype())?;
                let x_sel = xs2.index_select(&ids_t, 0)?;
                let y_sel = expert.forward(&x_sel)?;
                let y_sel = y_sel.broadcast_mul(&wts_t)?;
                ys = ys.index_add(&ids_t, &y_sel, 0)?;
            }
            ys.reshape((b, t, h))
        }
    }
}

pub mod rotary {
    use candle::{DType, Device, Result, Tensor};
    use std::f32::consts::PI;

    const DEFAULT_TRUNCATE: bool = false;

    #[derive(Debug, Clone)]
    pub struct GptOssRopeConfig {
        pub head_dim: usize,
        pub max_position_embeddings: usize,
        pub rope_theta: f32,
        pub factor: f32,
        pub beta_fast: f32,
        pub beta_slow: f32,
        pub original_max_position_embeddings: usize,
        pub truncate: bool,
    }

    impl GptOssRopeConfig {
        pub fn new(
            head_dim: usize,
            max_position_embeddings: usize,
            rope_theta: f32,
            factor: f32,
            beta_fast: f32,
            beta_slow: f32,
            original_max_position_embeddings: usize,
        ) -> Self {
            Self {
                head_dim,
                max_position_embeddings,
                rope_theta,
                factor,
                beta_fast,
                beta_slow,
                original_max_position_embeddings,
                truncate: DEFAULT_TRUNCATE,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct GptOssRotaryEmbedding {
        sin: Tensor,
        cos: Tensor,
        attn_factor: f32,
    }

    impl GptOssRotaryEmbedding {
        pub fn new_yarn(dtype: DType, dev: &Device, cfg: &GptOssRopeConfig) -> Result<Self> {
            let dim = cfg.head_dim;
            let dim2 = dim / 2;

            let pos_freqs: Vec<f32> = (0..dim)
                .step_by(2)
                .map(|i| cfg.rope_theta.powf(i as f32 / dim as f32))
                .collect();
            let inv_freq_extrapolation: Vec<f32> = pos_freqs.iter().map(|&v| 1.0 / v).collect();
            let inv_freq_interpolation: Vec<f32> = inv_freq_extrapolation
                .iter()
                .map(|&v| v / cfg.factor)
                .collect();

            let (low, high) = yarn_find_correction_range(
                cfg.beta_fast,
                cfg.beta_slow,
                dim,
                cfg.rope_theta,
                cfg.original_max_position_embeddings,
                cfg.truncate,
            );

            let inv_freq_extrapolation_factor = {
                let ramp = yarn_linear_ramp_mask(low, high, dim2, dev)?;
                let ones = Tensor::ones(ramp.shape().dims(), DType::F32, dev)?;
                ones.broadcast_sub(&ramp)?
            };

            let inv_freq_extrapolation =
                Tensor::from_vec(inv_freq_extrapolation, (1, dim2), dev)?;
            let inv_freq_interpolation =
                Tensor::from_vec(inv_freq_interpolation, (1, dim2), dev)?;

            let ones = Tensor::ones(inv_freq_extrapolation_factor.shape().dims(), DType::F32, dev)?;
            let inv_freq = inv_freq_interpolation
                .broadcast_mul(&ones.broadcast_sub(&inv_freq_extrapolation_factor)?)?
                .broadcast_add(&inv_freq_extrapolation.broadcast_mul(&inv_freq_extrapolation_factor)?)?;

            let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
                .to_dtype(DType::F32)?
                .reshape((cfg.max_position_embeddings, 1))?;
            let freqs = t.matmul(&inv_freq)?;

            let attn_factor = yarn_get_mscale(cfg.factor);
            let mut sin = freqs.sin()?.to_dtype(dtype)?;
            let mut cos = freqs.cos()?.to_dtype(dtype)?;

            // Test toggle: place YaRN scaling either on cos/sin (Mode A) or on softmax (Mode B).
            // - CANDLE_YARN_MODE=cos     => multiply cos/sin by attn_factor (HF placement)
            // - CANDLE_YARN_MODE=softmax => leave cos/sin unchanged (current behavior)
            match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                Some("softmax") => { /* leave tables unscaled to emulate old behavior */ }
                _ => {
                    // Default (fixed): place mscale onto cos/sin tables.
                    let scale = Tensor::new(attn_factor, dev)?.to_dtype(dtype)?;
                    sin = sin.broadcast_mul(&scale)?;
                    cos = cos.broadcast_mul(&scale)?;
                }
            }

            Ok(Self { sin, cos, attn_factor })
        }

        pub fn apply_rotary_emb_qk(
            &self,
            q: &Tensor,
            k: &Tensor,
            seqlen_offset: usize,
        ) -> Result<(Tensor, Tensor)> {
            let (_b, _h, t, _d) = q.dims4()?;
            let cos = self.cos.narrow(0, seqlen_offset, t)?;
            let sin = self.sin.narrow(0, seqlen_offset, t)?;
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        }

        pub fn cos_table(&self) -> &Tensor { &self.cos }
        pub fn sin_table(&self) -> &Tensor { &self.sin }
        pub fn attention_factor(&self) -> f32 { self.attn_factor }
    }

    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2.0 * PI)).ln())
            / (2.0 * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
        truncate: bool,
    ) -> (f32, f32) {
        let mut low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings);
        let mut high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings);
        if truncate {
            low = low.floor();
            high = high.ceil();
        }
        (low.max(0.0), high.min(dim as f32 - 1.0))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if (min - max).abs() < f32::EPSILON { max += 0.001; }
        let idx = Tensor::arange(0f32, dim as f32, dev)?;
        let num = idx.broadcast_sub(&Tensor::new(min as f32, dev)?)?;
        let den = Tensor::new((max - min) as f32, dev)?;
        let linear = num.broadcast_div(&den)?;
        linear.clamp(0.0, 1.0)
    }

    pub fn yarn_get_mscale(scale: f32) -> f32 {
        if scale <= 1.0 { 1.0 } else { 0.1 * scale.ln() + 1.0 }
    }
}

pub mod model {
    use super::config::GptOssConfig;
    use super::experts::{ExpertMlp, GptOssExperts};
    use super::rotary::{GptOssRopeConfig, GptOssRotaryEmbedding};
    use crate::models::with_tracing::{linear, Embedding, RmsNorm};
    use candle::{DType, Device, Module, Result, Tensor, IndexOp, D};
    use candle_nn::VarBuilder;

    const DEFAULT_RMS_EPS: f64 = 1e-5;
    const ENV_KEEP_RMS_FP32: &str = "CANDLE_RMS_FP32";
    const ENV_DUMP_L1: &str = "CANDLE_DUMP_L1";

    #[derive(Debug, Clone)]
    pub struct GptOssAttentionWeights {
        pub q_proj: crate::models::with_tracing::Linear,
        pub k_proj: crate::models::with_tracing::Linear,
        pub v_proj: crate::models::with_tracing::Linear,
        pub o_proj: crate::models::with_tracing::Linear,
        pub sinks: Tensor,
    }

    #[derive(Debug, Clone)]
    pub struct GptOssLayerWeights {
        pub attn: GptOssAttentionWeights,
        pub input_layernorm: RmsNorm,
        pub post_attention_layernorm: RmsNorm,
        pub router: candle_nn::Linear,
        pub experts: GptOssExperts,
    }

    #[derive(Debug, Clone)]
    pub struct GptOssModel {
        pub cfg: GptOssConfig,
        pub embed: Embedding,
        pub norm: RmsNorm,
        pub layers: Vec<GptOssLayerWeights>,
        pub lm_head: candle_nn::Linear,
        pub rope: GptOssRotaryEmbedding,
        pub kv_caches: Vec<candle_nn::kv_cache::RotatingKvCache>,
    }

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
                    linear(hidden, cfg.num_attention_heads * head_dim, attn_vb.pp("q_proj"))?
                } else {
                    crate::models::with_tracing::linear_no_bias(
                        hidden,
                        cfg.num_attention_heads * head_dim,
                        attn_vb.pp("q_proj"),
                    )?
                };
                let k_proj = if attn_vb.contains_tensor("k_proj.bias") {
                    linear(hidden, cfg.num_key_value_heads * head_dim, attn_vb.pp("k_proj"))?
                } else {
                    crate::models::with_tracing::linear_no_bias(
                        hidden,
                        cfg.num_key_value_heads * head_dim,
                        attn_vb.pp("k_proj"),
                    )?
                };
                let v_proj = if attn_vb.contains_tensor("v_proj.bias") {
                    linear(hidden, cfg.num_key_value_heads * head_dim, attn_vb.pp("v_proj"))?
                } else {
                    crate::models::with_tracing::linear_no_bias(
                        hidden,
                        cfg.num_key_value_heads * head_dim,
                        attn_vb.pp("v_proj"),
                    )?
                };
                let o_proj = if attn_vb.contains_tensor("o_proj.bias") {
                    linear(cfg.num_attention_heads * head_dim, hidden, attn_vb.pp("o_proj"))?
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
                    Tensor::zeros(cfg.num_attention_heads, DType::BF16, &Device::Cpu)?
                        .to_device(&dev)?
                };
                let attn = GptOssAttentionWeights { q_proj, k_proj, v_proj, o_proj, sinks };

                // HF/GPT-OSS uses `mlp.router` with a bias for the MoE router.
                let router = candle_nn::linear(hidden, cfg.num_local_experts, l_vb.pp("mlp.router"))?;
                let experts = {
                    let mlp_vb = l_vb.pp("mlp");
                    let inter = cfg.intermediate_size;
                    let mut all: Vec<ExpertMlp> = Vec::with_capacity(cfg.num_local_experts);
                    for e in 0..cfg.num_local_experts {
                        let gate_up = match super::load_expert_linear_mxfp4_grouped(
                            hidden,
                            2 * inter,
                            true,
                            mlp_vb.clone(),
                            "experts.gate_up_proj",
                            e,
                            cfg.num_local_experts,
                        ) {
                            Ok(l) => l,
                            Err(_) => super::load_linear_maybe_mxfp4(
                                hidden,
                                2 * inter,
                                true,
                                mlp_vb.pp(&format!("experts.{e}")),
                                "gate_up_proj",
                            )?,
                        };
                        let down = match super::load_expert_linear_mxfp4_grouped(
                            inter,
                            hidden,
                            true,
                            mlp_vb.clone(),
                            "experts.down_proj",
                            e,
                            cfg.num_local_experts,
                        ) {
                            Ok(l) => l,
                            Err(_) => super::load_linear_maybe_mxfp4(
                                inter,
                                hidden,
                                true,
                                mlp_vb.pp(&format!("experts.{e}")),
                                "down_proj",
                            )?,
                        };
                        let limit = cfg.swiglu_limit.unwrap_or(7.0);
                        let alpha = 1.702f32;
                        all.push(ExpertMlp::new(gate_up, down, limit, alpha));
                    }
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

                layers.push(GptOssLayerWeights { attn, input_layernorm, post_attention_layernorm, router, experts });
            }

            let lm_head = candle_nn::linear_no_bias(hidden, cfg.vocab_size, vb_bf16.pp("lm_head"))?;
            let mut kv_caches = Vec::with_capacity(cfg.num_hidden_layers);
            let layer_types = cfg.effective_layer_types();
            for (i, _ly) in layer_types.iter().enumerate() {
                let attn_mode = super::select_attn_mode_for_layer(
                    &super::GptOssConfigMinimal {
                        num_hidden_layers: cfg.num_hidden_layers,
                        layer_types: layer_types.clone(),
                        max_position_embeddings: cfg.max_position_embeddings,
                        sliding_window: cfg.sliding_window,
                    },
                    i,
                );
                let window = match attn_mode {
                    super::AttnMode::Full => cfg.max_position_embeddings,
                    super::AttnMode::Sliding { left, .. } => left,
                };
                kv_caches.push(candle_nn::kv_cache::RotatingKvCache::new(2, window));
            }

            Ok(Self { cfg: cfg.clone(), embed, norm, layers, lm_head, rope, kv_caches })
        }

        pub fn forward_logits_minimal(&self, input_ids: &Tensor) -> Result<Tensor> {
            let xs = self.embed.forward(input_ids)?;
            let (b, t, h) = xs.dims3()?;
            let xs2 = xs.reshape(((), h))?.to_dtype(DType::F32)?;
            let w = self.lm_head.weight().to_dtype(DType::F32)?;
            let logits = xs2.matmul(&w.t()?)?;
            let logits = logits.reshape((b, t, self.cfg.vocab_size))?.to_dtype(DType::BF16)?;
            Ok(logits)
        }

        pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
            let mut xs = self.embed.forward(input_ids)?;
            let (b, t, hidden) = xs.dims3()?;
            let head_dim = self.cfg.head_dim();
            let n_q = self.cfg.num_attention_heads;
            let n_kv = self.cfg.num_key_value_heads;
            // Test toggle for YaRN scaling placement:
            // - CANDLE_YARN_MODE=cos     => use standard 1/sqrt(d) softmax scale (HF placement on cos/sin)
            // - CANDLE_YARN_MODE=softmax => include attention_factor in softmax scale (current behavior)
            let softmax_scale = {
                let base = 1.0f32 / (head_dim as f32).sqrt();
                match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                    Some("softmax") => self.rope.attention_factor() * base,
                    _ => base,
                }
            };

            let dump_l1 = matches!(
                std::env::var(ENV_DUMP_L1).ok().as_deref(),
                Some("1") | Some("true") | Some("TRUE")
            );
            for (i, layer) in self.layers.iter_mut().enumerate() {
                let x_norm = layer.input_layernorm.forward(&xs)?;
                if dump_l1 && i == 0 {
                    let last = x_norm.i((0, t - 1))?.to_dtype(DType::F32)?;
                    let v = last.to_vec1::<f32>()?;
                    let take = v.iter().take(8).copied().collect::<Vec<_>>();
                    let mean = v.iter().copied().sum::<f32>() / (v.len() as f32);
                    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (v.len() as f32);
                    eprintln!(
                        "[L1] post-norm last-token: len={} first8={:?} mean={:.6} std={:.6}",
                        v.len(), take, mean, var.sqrt()
                    );
                }

                let q = x_norm.apply(&layer.attn.q_proj)?;
                let k = x_norm.apply(&layer.attn.k_proj)?;
                let v = x_norm.apply(&layer.attn.v_proj)?;

                let q = q.reshape((b, t, n_q, head_dim))?;
                let k = k.reshape((b, t, n_kv, head_dim))?;
                let v = v.reshape((b, t, n_kv, head_dim))?;

                let q_bhtd = q.transpose(1, 2)?;
                let k_bhtd = k.transpose(1, 2)?;
                let (q_bhtd, k_bhtd) = self.rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, seqlen_offset)?;
                let q = q_bhtd.transpose(1, 2)?;
                let k_step = k_bhtd;
                let v_step = v.transpose(1, 2)?;

                let (k_all, v_all) = self.kv_caches[i].append(&k_step.contiguous()?, &v_step.contiguous()?)?;

                let n_rep = n_q / n_kv;
                let k_rep = crate::utils::repeat_kv(k_all.clone(), n_rep)?;
                let v_rep = crate::utils::repeat_kv(v_all.clone(), n_rep)?;
                let k_btkhd = k_rep.transpose(1, 2)?;
                let v_btkhd = v_rep.transpose(1, 2)?;

                let attn_mode = super::select_attn_mode_for_layer(
                    &super::GptOssConfigMinimal {
                        num_hidden_layers: self.cfg.num_hidden_layers,
                        layer_types: self.cfg.effective_layer_types(),
                        max_position_embeddings: self.cfg.max_position_embeddings,
                        sliding_window: self.cfg.sliding_window,
                    },
                    i,
                );

                let sinks = if matches!(
                    std::env::var("CANDLE_DISABLE_SINKS").ok().as_deref(),
                    Some("1") | Some("true") | Some("TRUE")
                ) {
                    None
                } else {
                    Some(&layer.attn.sinks)
                };

                #[cfg(feature = "flash-attn")]
                let y = {
                    let use_fa = !matches!(
                        std::env::var("CANDLE_DISABLE_FLASH").ok().as_deref(),
                        Some("1") | Some("true") | Some("TRUE")
                    );
                    if use_fa {
                        match attn_mode {
                            super::AttnMode::Full => super::flash_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                            super::AttnMode::Sliding { left, right } => super::flash_attn_windowed_with_sinks(
                                &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                            )?,
                        }
                    } else {
                        match attn_mode {
                            super::AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                            super::AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(
                                &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                            )?,
                        }
                    }
                };
                #[cfg(not(feature = "flash-attn"))]
                let y = {
                    match attn_mode {
                        super::AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                        super::AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(
                            &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                        )?,
                    }
                };

                let y = y.reshape((b, t, n_q * head_dim))?;
                let y = y.apply(&layer.attn.o_proj)?;
                xs = (xs + y)?;
                if dump_l1 && i == 0 {
                    let last = xs.i((0, t - 1))?.to_dtype(DType::F32)?;
                    let v = last.to_vec1::<f32>()?;
                    let take = v.iter().take(8).copied().collect::<Vec<_>>();
                    let mean = v.iter().copied().sum::<f32>() / (v.len() as f32);
                    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (v.len() as f32);
                    eprintln!(
                        "[L1] post-attn-residual last-token: len={} first8={:?} mean={:.6} std={:.6}",
                        v.len(), take, mean, var.sqrt()
                    );
                }

                let x_norm2 = layer.post_attention_layernorm.forward(&xs)?;
                let mlp_out = layer.experts.forward(&x_norm2)?;
                xs = (xs + mlp_out)?;
            }

            let xs = self.norm.forward(&xs)?;
            let xs2 = xs.reshape(((), hidden))?;
            let xs2_f = xs2.to_dtype(DType::F32)?;
            let w = self.lm_head.weight();
            let vocab = self.cfg.vocab_size;
            let chunk = 8192usize;
            let mut parts: Vec<Tensor> = Vec::new();
            let mut start = 0usize;
            while start < vocab {
                let len = (vocab - start).min(chunk);
                let w_chunk = w.narrow(0, start, len)?;
                let w_chunk_f = w_chunk.to_dtype(DType::F32)?;
                let logits_chunk = xs2_f.matmul(&w_chunk_f.t()?)?;
                parts.push(logits_chunk);
                start += len;
            }
            let logits_bt_v = Tensor::cat(&parts.iter().collect::<Vec<_>>(), D::Minus1)?;
            let logits = logits_bt_v.reshape((b, t, vocab))?;
            Ok(logits)
        }

        /// Runs the full forward pass up to (and including) the final RMSNorm and
        /// returns the last-token hidden vector (post-norm, pre-lm_head) as f32.
        /// Shape: (hidden_size)
        pub fn forward_last_hidden_post_norm(
            &mut self,
            input_ids: &Tensor,
            seqlen_offset: usize,
        ) -> Result<Tensor> {
            let mut xs = self.embed.forward(input_ids)?; // (b, t, h)
            let (b, t, _hidden) = xs.dims3()?;
            let head_dim = self.cfg.head_dim();
            let n_q = self.cfg.num_attention_heads;
            let n_kv = self.cfg.num_key_value_heads;
            let softmax_scale = {
                let base = 1.0f32 / (head_dim as f32).sqrt();
                match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                    Some("softmax") => self.rope.attention_factor() * base,
                    _ => base,
                }
            };

            for (i, layer) in self.layers.iter_mut().enumerate() {
                let x_norm = layer.input_layernorm.forward(&xs)?;

                let q = x_norm.apply(&layer.attn.q_proj)?;
                let k = x_norm.apply(&layer.attn.k_proj)?;
                let v = x_norm.apply(&layer.attn.v_proj)?;

                let q = q.reshape((b, t, n_q, head_dim))?;
                let k = k.reshape((b, t, n_kv, head_dim))?;
                let v = v.reshape((b, t, n_kv, head_dim))?;

                let q_bhtd = q.transpose(1, 2)?;
                let k_bhtd = k.transpose(1, 2)?;
                let (q_bhtd, k_bhtd) = self.rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, seqlen_offset)?;
                let q = q_bhtd.transpose(1, 2)?;
                let k_step = k_bhtd;
                let v_step = v.transpose(1, 2)?;

                // Append to KV cache and build repeated K/V for multi-query attention.
                let (k_all, v_all) = self.kv_caches[i].append(&k_step.contiguous()?, &v_step.contiguous()?)?;
                let n_rep = n_q / n_kv;
                let k_rep = crate::utils::repeat_kv(k_all.clone(), n_rep)?;
                let v_rep = crate::utils::repeat_kv(v_all.clone(), n_rep)?;
                let k_btkhd = k_rep.transpose(1, 2)?;
                let v_btkhd = v_rep.transpose(1, 2)?;

                let attn_mode = super::select_attn_mode_for_layer(
                    &super::GptOssConfigMinimal {
                        num_hidden_layers: self.cfg.num_hidden_layers,
                        layer_types: self.cfg.effective_layer_types(),
                        max_position_embeddings: self.cfg.max_position_embeddings,
                        sliding_window: self.cfg.sliding_window,
                    },
                    i,
                );

                let sinks = if matches!(
                    std::env::var("CANDLE_DISABLE_SINKS").ok().as_deref(),
                    Some("1") | Some("true") | Some("TRUE")
                ) {
                    None
                } else {
                    Some(&layer.attn.sinks)
                };

                #[cfg(feature = "flash-attn")]
                let y = {
                    let use_fa = !matches!(
                        std::env::var("CANDLE_DISABLE_FLASH").ok().as_deref(),
                        Some("1") | Some("true") | Some("TRUE")
                    );
                    if use_fa {
                        match attn_mode {
                            super::AttnMode::Full => super::flash_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                            super::AttnMode::Sliding { left, right } => super::flash_attn_windowed_with_sinks(
                                &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                            )?,
                        }
                    } else {
                        match attn_mode {
                            super::AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                            super::AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(
                                &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                            )?,
                        }
                    }
                };
                #[cfg(not(feature = "flash-attn"))]
                let y = {
                    match attn_mode {
                        super::AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                        super::AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(
                            &q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks,
                        )?,
                    }
                };

                let y = y.reshape((b, t, n_q * head_dim))?;
                let y = y.apply(&layer.attn.o_proj)?;
                xs = (xs + y)?;

                let x_norm2 = layer.post_attention_layernorm.forward(&xs)?;
                let mlp_out = layer.experts.forward(&x_norm2)?;
                xs = (xs + mlp_out)?;
            }

            let xs = self.norm.forward(&xs)?; // (b, t, h)
            let last = xs.i((0, t - 1))?; // (h)
            let last_f32 = last.to_dtype(candle::DType::F32)?;
            Ok(last_f32)
        }

        /// Collects the per-layer last-token states used for bisection at step 0.
        /// Returns a tensor of shape (1 + 3*L, H) float32 on CPU with rows:
        ///  [h0,
        ///   pre_attn_norm(l0), post_attn_resid(l0), post_mlp_resid(l0),
        ///   ... repeated for each layer ...]
        pub fn debug_collect_layer_states_last_token(
            &mut self,
            input_ids: &Tensor,
            seqlen_offset: usize,
        ) -> Result<Tensor> {
            let mut xs = self.embed.forward(input_ids)?; // (b, t, h)
            let (b, t, _hidden) = xs.dims3()?;
            let head_dim = self.cfg.head_dim();
            let n_q = self.cfg.num_attention_heads;
            let n_kv = self.cfg.num_key_value_heads;
            let softmax_scale = {
                let base = 1.0f32 / (head_dim as f32).sqrt();
                match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                    Some("softmax") => self.rope.attention_factor() * base,
                    _ => base,
                }
            };

            let mut rows: Vec<Tensor> = Vec::with_capacity(1 + 3 * self.cfg.num_hidden_layers);
            // h0
            rows.push(xs.i((0, t - 1))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);

            for (i, layer) in self.layers.iter_mut().enumerate() {
                // pre_attn_norm
                let x_norm = layer.input_layernorm.forward(&xs)?;
                rows.push(x_norm.i((0, t - 1))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);

                // attention
                let q = x_norm.apply(&layer.attn.q_proj)?;
                let k = x_norm.apply(&layer.attn.k_proj)?;
                let v = x_norm.apply(&layer.attn.v_proj)?;
                let q = q.reshape((b, t, n_q, head_dim))?;
                let k = k.reshape((b, t, n_kv, head_dim))?;
                let v = v.reshape((b, t, n_kv, head_dim))?;
                let q_bhtd = q.transpose(1, 2)?;
                let k_bhtd = k.transpose(1, 2)?;
                let (q_bhtd, k_bhtd) = self.rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, seqlen_offset)?;
                let q = q_bhtd.transpose(1, 2)?;
                let k_step = k_bhtd;
                let v_step = v.transpose(1, 2)?;
                let (k_all, v_all) = self.kv_caches[i].append(&k_step.contiguous()?, &v_step.contiguous()?)?;
                let n_rep = n_q / n_kv;
                let k_rep = crate::utils::repeat_kv(k_all.clone(), n_rep)?;
                let v_rep = crate::utils::repeat_kv(v_all.clone(), n_rep)?;
                let k_btkhd = k_rep.transpose(1, 2)?;
                let v_btkhd = v_rep.transpose(1, 2)?;
                let attn_mode = super::select_attn_mode_for_layer(
                    &super::GptOssConfigMinimal {
                        num_hidden_layers: self.cfg.num_hidden_layers,
                        layer_types: self.cfg.effective_layer_types(),
                        max_position_embeddings: self.cfg.max_position_embeddings,
                        sliding_window: self.cfg.sliding_window,
                    },
                    i,
                );
                let sinks = if matches!(
                    std::env::var("CANDLE_DISABLE_SINKS").ok().as_deref(),
                    Some("1") | Some("true") | Some("TRUE")
                ) { None } else { Some(&layer.attn.sinks) };
                #[cfg(feature = "flash-attn")]
                let y = {
                    let use_fa = !matches!(
                        std::env::var("CANDLE_DISABLE_FLASH").ok().as_deref(),
                        Some("1") | Some("true") | Some("TRUE")
                    );
                    if use_fa {
                        match attn_mode {
                            super::AttnMode::Full => super::flash_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                            super::AttnMode::Sliding { left, right } => super::flash_attn_windowed_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks)?,
                        }
                    } else {
                        match attn_mode {
                            super::AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                            super::AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks)?,
                        }
                    }
                };
                #[cfg(not(feature = "flash-attn"))]
                let y = {
                    match attn_mode {
                        super::AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                        super::AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks)?,
                    }
                };
                let y = y.reshape((b, t, n_q * head_dim))?;
                let y = y.apply(&layer.attn.o_proj)?;
                xs = (xs + y)?;

                // post_attn_resid
                rows.push(xs.i((0, t - 1))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);

                // MLP
                let x_norm2 = layer.post_attention_layernorm.forward(&xs)?;
                let mlp_out = layer.experts.forward(&x_norm2)?;
                xs = (xs + mlp_out)?;

                // post_mlp_resid
                rows.push(xs.i((0, t - 1))?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?);
            }

            // Stack rows along a new leading dimension -> (1+3L, H)
            let rows: Vec<&Tensor> = rows.iter().collect();
            let out = Tensor::stack(&rows, 0)?;
            Ok(out)
        }

        /// Debug helper for layer-0 attention: returns last-token Q/K/V after projections
        /// and Q/K after RoPE application (all as f32 on CPU) along with the softmax
        /// scale actually used and the selected attention mode parameters.
        /// Shapes:
        ///   - q_pre: (n_q, head_dim)
        ///   - k_pre: (n_kv, head_dim)
        ///   - v_pre: (n_kv, head_dim)
        ///   - q_rope: (n_q, head_dim)
        ///   - k_rope: (n_kv, head_dim)
        pub fn debug_l0_qkv_last_token(
            &mut self,
            input_ids: &Tensor,
            seqlen_offset: usize,
        ) -> Result<(
            Tensor, // q_pre
            Tensor, // k_pre
            Tensor, // v_pre
            Tensor, // q_rope
            Tensor, // k_rope
            f32,    // softmax_scale
            super::AttnMode,
        )> {
            let xs = self.embed.forward(input_ids)?; // (b,t,h)
            let (b, t, _h) = xs.dims3()?;
            let head_dim = self.cfg.head_dim();
            let n_q = self.cfg.num_attention_heads;
            let n_kv = self.cfg.num_key_value_heads;

            // Compute softmax scale according to YaRN placement.
            let softmax_scale = {
                let base = 1.0f32 / (head_dim as f32).sqrt();
                match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                    Some("softmax") => self.rope.attention_factor() * base,
                    _ => base,
                }
            };

            let layer = &mut self.layers[0];
            let x_norm = layer.input_layernorm.forward(&xs)?; // (b,t,h)

            let q_lin = x_norm.apply(&layer.attn.q_proj)?; // (b,t,n_q*hd)
            let k_lin = x_norm.apply(&layer.attn.k_proj)?; // (b,t,n_kv*hd)
            let v_lin = x_norm.apply(&layer.attn.v_proj)?; // (b,t,n_kv*hd)

            let q = q_lin.reshape((b, t, n_q, head_dim))?;
            let k = k_lin.reshape((b, t, n_kv, head_dim))?;
            let v = v_lin.reshape((b, t, n_kv, head_dim))?;

            // Last token slices pre-RoPE, shape (b, n_h, d)
            let q_last_bhd = q.i((0..b, t - 1, 0..n_q, 0..head_dim))?;
            let k_last_bhd = k.i((0..b, t - 1, 0..n_kv, 0..head_dim))?;
            let v_last_bhd = v.i((0..b, t - 1, 0..n_kv, 0..head_dim))?;

            // Transpose to (b, h, t, d) to apply RoPE as in the forward
            let q_bhtd = q.transpose(1, 2)?;
            let k_bhtd = k.transpose(1, 2)?;
            let (q_bhtd, k_bhtd) = self.rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, seqlen_offset)?;

            // Grab last token after RoPE: (b,h,d)
            let q_rope_last_bhd = q_bhtd.i((0..b, 0..n_q, t - 1, 0..head_dim))?;
            let k_rope_last_bhd = k_bhtd.i((0..b, 0..n_kv, t - 1, 0..head_dim))?;

            // Squeeze batch dimension and move to CPU f32 for easy inspection.
            let to_cpu_f32 = |t: Tensor| -> Result<Tensor> {
                t.squeeze(0)?.to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)
            };
            let q_pre = to_cpu_f32(q_last_bhd)?; // (n_q, d)
            let k_pre = to_cpu_f32(k_last_bhd)?; // (n_kv, d)
            let v_pre = to_cpu_f32(v_last_bhd)?; // (n_kv, d)
            let q_rope = to_cpu_f32(q_rope_last_bhd)?; // (n_q, d)
            let k_rope = to_cpu_f32(k_rope_last_bhd)?; // (n_kv, d)

            let attn_mode = super::select_attn_mode_for_layer(
                &super::GptOssConfigMinimal {
                    num_hidden_layers: self.cfg.num_hidden_layers,
                    layer_types: self.cfg.effective_layer_types(),
                    max_position_embeddings: self.cfg.max_position_embeddings,
                    sliding_window: self.cfg.sliding_window,
                },
                0,
            );

            Ok((q_pre, k_pre, v_pre, q_rope, k_rope, softmax_scale, attn_mode))
        }

        /// Debug: return the sinks vector for layer 0 as f32 on CPU.
        pub fn debug_l0_sinks(&self) -> Result<Tensor> {
            let s = &self.layers[0].attn.sinks;
            s.to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)
        }

        /// Debug: return (q,k,v) projection weights for layer 0 as f32 on CPU.
        pub fn debug_l0_qkv_weights(&self) -> Result<(Tensor, Tensor, Tensor)> {
            let q = self.layers[0].attn.q_proj.weight().to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)?;
            let k = self.layers[0].attn.k_proj.weight().to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)?;
            let v = self.layers[0].attn.v_proj.weight().to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)?;
            Ok((q, k, v))
        }

        /// Debug: compute attention output at layer 0 for the whole sequence
        /// and return the last-token vector both before and after the o_proj.
        /// Returns (pre_o_proj: (n_q*head_dim), post_o_proj: (hidden)) as f32 on CPU.
        pub fn debug_l0_attn_last_token(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
            let xs0 = self.embed.forward(input_ids)?; // (b,t,h)
            let (b, t, _h) = xs0.dims3()?;
            let head_dim = self.cfg.head_dim();
            let n_q = self.cfg.num_attention_heads;
            let n_kv = self.cfg.num_key_value_heads;
            let softmax_scale = {
                let base = 1.0f32 / (head_dim as f32).sqrt();
                match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                    Some("softmax") => self.rope.attention_factor() * base,
                    _ => base,
                }
            };
            let layer = &mut self.layers[0];
            let x_norm = layer.input_layernorm.forward(&xs0)?;
            let q = x_norm.apply(&layer.attn.q_proj)?;
            let k = x_norm.apply(&layer.attn.k_proj)?;
            let v = x_norm.apply(&layer.attn.v_proj)?;
            let q = q.reshape((b, t, n_q, head_dim))?;
            let k = k.reshape((b, t, n_kv, head_dim))?;
            let v = v.reshape((b, t, n_kv, head_dim))?;
            let q_bhtd = q.transpose(1, 2)?;
            let k_bhtd = k.transpose(1, 2)?;
            let (q_bhtd, k_bhtd) = self.rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, seqlen_offset)?;
            let q = q_bhtd.transpose(1, 2)?;
            let k_step = k_bhtd;
            let v_step = v.transpose(1, 2)?;
            let (k_all, v_all) = self.kv_caches[0].append(&k_step.contiguous()?, &v_step.contiguous()?)?;
            let n_rep = n_q / n_kv;
            let k_rep = crate::utils::repeat_kv(k_all.clone(), n_rep)?;
            let v_rep = crate::utils::repeat_kv(v_all.clone(), n_rep)?;
            let k_btkhd = k_rep.transpose(1, 2)?;
            let v_btkhd = v_rep.transpose(1, 2)?;
            let attn_mode = super::select_attn_mode_for_layer(
                &super::GptOssConfigMinimal {
                    num_hidden_layers: self.cfg.num_hidden_layers,
                    layer_types: self.cfg.effective_layer_types(),
                    max_position_embeddings: self.cfg.max_position_embeddings,
                    sliding_window: self.cfg.sliding_window,
                },
                0,
            );
            let sinks = Some(&layer.attn.sinks);
            #[cfg(feature = "flash-attn")]
            let y = {
                let use_fa = !matches!(
                    std::env::var("CANDLE_DISABLE_FLASH").ok().as_deref(),
                    Some("1") | Some("true") | Some("TRUE")
                );
                if use_fa {
                    match attn_mode {
                        super::AttnMode::Full => super::flash_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                        super::AttnMode::Sliding { left, right } => super::flash_attn_windowed_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks)?,
                    }
                } else {
                    match attn_mode {
                        super::AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                        super::AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks)?,
                    }
                }
            };
            #[cfg(not(feature = "flash-attn"))]
            let y = {
                match attn_mode {
                    super::AttnMode::Full => super::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
                    super::AttnMode::Sliding { left, right } => super::eager_attn_windowed_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks)?,
                }
            };
            let y_pre = y.reshape((b, t, n_q * head_dim))?; // before o_proj
            let y_post = y_pre.apply(&layer.attn.o_proj)?;
            let y_pre_last = y_pre.i((0, t - 1))?.to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)?;
            let y_post_last = y_post.i((0, t - 1))?.to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)?;
            Ok((y_pre_last, y_post_last))
        }

        /// Debug helper: returns attention logits row and effective softmax weights (with sinks
        /// renormalization applied) for the last token at layer 0 for the first `heads` heads.
        /// Shapes returned are (heads, klen). Dtype is f32 on CPU for determinism.
        pub fn debug_l0_attn_last_token_logits_weights(
            &mut self,
            input_ids: &Tensor,
            seqlen_offset: usize,
            heads: usize,
        ) -> Result<(Tensor, Tensor)> {
            let xs0 = self.embed.forward(input_ids)?; // (b,t,h)
            let (b, t, _h) = xs0.dims3()?;
            let head_dim = self.cfg.head_dim();
            let n_q = self.cfg.num_attention_heads;
            let n_kv = self.cfg.num_key_value_heads;
            let softmax_scale = {
                let base = 1.0f32 / (head_dim as f32).sqrt();
                match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                    Some("softmax") => self.rope.attention_factor() * base,
                    _ => base,
                }
            };
            let layer = &mut self.layers[0];

            // QKV + RoPE (matches main forward path)
            let x_norm = layer.input_layernorm.forward(&xs0)?;
            let q = x_norm.apply(&layer.attn.q_proj)?;
            let k = x_norm.apply(&layer.attn.k_proj)?;
            let v = x_norm.apply(&layer.attn.v_proj)?;
            let q = q.reshape((b, t, n_q, head_dim))?;
            let k = k.reshape((b, t, n_kv, head_dim))?;
            let v = v.reshape((b, t, n_kv, head_dim))?;
            let q_bhtd = q.transpose(1, 2)?; // (b,hq,t,d)
            let k_bhtd = k.transpose(1, 2)?; // (b,hkv,t,d)
            let (q_bhtd, k_bhtd) = self.rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, seqlen_offset)?;
            let q_bt_hqd = q_bhtd.transpose(1, 2)?; // (b,t,hq,d)
            let k_step = k_bhtd; // (b,hkv,t,d)
            let v_step = v.transpose(1, 2)?; // (b,hkv,t,d)

            // KV cache append and GQA replication
            let (k_all, _v_all) = self.kv_caches[0].append(&k_step.contiguous()?, &v_step.contiguous()?)?;
            let n_rep = n_q / n_kv;
            let k_rep = crate::utils::repeat_kv(k_all.clone(), n_rep)?; // (b,hq,t,d)
            // No need to replicate V for logits/weights dump
            let k_btkhd = k_rep.transpose(1, 2)?; // (b,t,hq,d)

            // Determine attention mode for mask semantics.
            let attn_mode = super::select_attn_mode_for_layer(
                &super::GptOssConfigMinimal {
                    num_hidden_layers: self.cfg.num_hidden_layers,
                    layer_types: self.cfg.effective_layer_types(),
                    max_position_embeddings: self.cfg.max_position_embeddings,
                    sliding_window: self.cfg.sliding_window,
                },
                0,
            );

            // Build logits = (b,h,q,k)
            let q_bhqd = q_bt_hqd.to_dtype(DType::F32)?.transpose(1, 2)?; // (b,hq,q,d)
            let k_bhkd = k_btkhd.to_dtype(DType::F32)?.transpose(1, 2)?; // (b,hq,k,d)
            // v_bhkd only needed when reconstructing attn_output; not required for logits/weights dump
            let mut logits = (q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * softmax_scale as f64)?; // (b,h,q,k)

            // Apply mask (causal or sliding)
            let (_, qlen, _, _) = q_bt_hqd.dims4()?;
            let (_, klen, _, _) = k_btkhd.dims4()?;
            logits = match attn_mode {
                super::AttnMode::Full => {
                    if qlen > 1 {
                        let mask: Vec<u8> = (0..qlen)
                            .flat_map(|i| (0..klen).map(move |j| u8::from(j > i)))
                            .collect();
                        let mask = Tensor::from_slice(&mask, (qlen, klen), logits.device())?;
                        super::masked_fill(&logits, &mask.broadcast_as((b, n_q, qlen, klen))?, f32::NEG_INFINITY)?
                    } else {
                        logits
                    }
                }
                super::AttnMode::Sliding { left, right } => {
                    let left = left;
                    let right = right;
                    let mask: Vec<u8> = (0..qlen)
                        .flat_map(|i| (0..klen).map(move |j| {
                            let i = i as isize;
                            let j = j as isize;
                            let l = left as isize;
                            let r = right as isize;
                            let allow = (j > i - l) && (j <= i + r);
                            u8::from(!allow)
                        }))
                        .collect();
                    let mask = Tensor::from_slice(&mask, (qlen, klen), logits.device())?;
                    super::masked_fill(&logits, &mask.broadcast_as((b, n_q, qlen, klen))?, f32::NEG_INFINITY)?
                }
            };

            // Softmax weights over keys, then apply sinks renormalization to match HF semantics
            let att = candle_nn::ops::softmax_last_dim(&logits)?; // (b,h,q,k)

            // Compute scale per (b,h,q) using the same formula as forward
            let sinks = if matches!(
                std::env::var("CANDLE_DISABLE_SINKS").ok().as_deref(),
                Some("1") | Some("true") | Some("TRUE")
            ) {
                None
            } else {
                Some(&layer.attn.sinks)
            };

            let scores = if let Some(sinks_t) = sinks {
                let lse = logits.log_sum_exp(D::Minus1)?; // (b,h,q)
                let scale = super::sinks_scale_from_lse(&lse, sinks_t)?; // (b,h,q)
                let scale = scale.unsqueeze(D::Minus1)?; // (b,h,q,1)
                att.to_dtype(DType::F32)?.broadcast_mul(&scale)?
            } else {
                att.to_dtype(DType::F32)?
            }; // (b,h,q,k)

            // Slice last-token rows for the first `heads` heads.
            let dump_h = heads.min(n_q);
            let logits_last = logits.i((0, 0..dump_h, qlen - 1))?; // (dump_h, klen)
            let scores_last = scores.i((0, 0..dump_h, qlen - 1))?; // (dump_h, klen)

            // Move to CPU f32 for deterministic printing/comparison.
            let logits_last = logits_last.to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)?;
            let scores_last = scores_last.to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)?;
            Ok((logits_last, scores_last))
        }
    }
}

// ============================
// Config and layer selection
// ============================

#[derive(Debug, Clone, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GptOssLayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GptOssConfigMinimal {
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub layer_types: Vec<GptOssLayerType>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl GptOssConfigMinimal {
    pub fn effective_layer_types(&self) -> Vec<GptOssLayerType> {
        if self.layer_types.is_empty() {
            vec![GptOssLayerType::FullAttention; self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        }
    }

    pub fn sliding_window_size(&self) -> Option<usize> {
        self.sliding_window
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnMode {
    Full,
    Sliding { left: usize, right: usize },
}

/// Select attention mode for a given `layer_idx` from config.
/// - Full: standard causal attention.
/// - Sliding: windowed attention with left window = `sliding_window` and right window = 0.
pub fn select_attn_mode_for_layer(cfg: &GptOssConfigMinimal, layer_idx: usize) -> AttnMode {
    // Respect the per-layer attention type declared in config. When a sliding window is
    // configured and the layer is marked `SlidingAttention`, use a windowed causal mask
    // with `left = sliding_window` and `right = 0`. Otherwise, run full causal attention.
    let types = cfg.effective_layer_types();
    match types.get(layer_idx) {
        Some(GptOssLayerType::SlidingAttention) => match cfg.sliding_window_size() {
            Some(w) if w > 0 => AttnMode::Sliding { left: w, right: 0 },
            _ => AttnMode::Full,
        },
        _ => AttnMode::Full,
    }
}

// Constants/config
const MXFP4_BLOCK_ELEMS: usize = 32; // k=32 elements per block on the last dim.
const MXFP4_BLOCK_BYTES: usize = 16; // packed two nibbles per byte.

/// Try to resolve the MXFP4 pair names for a given base, supporting both dot and underscore
/// variants ("base.blocks"/"base.scales" or "base_blocks"/"base_scales"). Returns the fully
/// qualified names including any VarBuilder path.
fn detect_mxfp4_pair_names(
    vb: &candle_nn::VarBuilder,
    base: &str,
) -> Result<Option<(String, String)>> {
    // 1) Dot-separated variant
    let blocks_dot = format!("{base}.blocks");
    let scales_dot = format!("{base}.scales");
    if vb.contains_tensor(&blocks_dot) && vb.contains_tensor(&scales_dot) {
        return Ok(Some((blocks_dot, scales_dot)));
    }
    if vb.contains_tensor(&blocks_dot) ^ vb.contains_tensor(&scales_dot) {
        if vb.contains_tensor(&blocks_dot) {
            candle::bail!("found '{base}.blocks' but missing '{base}.scales'")
        } else {
            candle::bail!("found '{base}.scales' but missing '{base}.blocks'")
        }
    }
    // 2) Underscore-joined variant (helps with simplified unit-test backends)
    let blocks_us = format!("{base}_blocks");
    let scales_us = format!("{base}_scales");
    if vb.contains_tensor(&blocks_us) && vb.contains_tensor(&scales_us) {
        return Ok(Some((blocks_us, scales_us)));
    }
    if vb.contains_tensor(&blocks_us) ^ vb.contains_tensor(&scales_us) {
        if vb.contains_tensor(&blocks_us) {
            candle::bail!("found '{base}_blocks' but missing '{base}_scales'")
        } else {
            candle::bail!("found '{base}_scales' but missing '{base}_blocks'")
        }
    }
    Ok(None)
}

/// Load a Linear layer weight for base `base` where the weight is either stored as BF16
/// under `"{base}.weight"` (fallback) or as MXFP4 experts using paired `blocks`/`scales`.
///
/// - `in_dim`: input dimension of the linear layer
/// - `out_dim`: output dimension of the linear layer
/// - `bias`: whether to load a bias if present
/// - `vb`: VarBuilder pointing at the module path containing the base
///
/// Returns a `Linear` with BF16 weights (and bias if requested and present).
pub fn load_linear_maybe_mxfp4(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: candle_nn::VarBuilder,
    base: &str,
) -> Result<Linear> {
    // Detect paired MXFP4 tensors.
    if let Some((blocks_name, scales_name)) = detect_mxfp4_pair_names(&vb, base)? {
        // Validate shape alignment to block size.
        if in_dim % MXFP4_BLOCK_ELEMS != 0 {
            candle::bail!(
                "MXFP4 weight '{base}': in_dim must be multiple of {MXFP4_BLOCK_ELEMS}, got {in_dim}"
            )
        }
        let nblocks = in_dim / MXFP4_BLOCK_ELEMS;

        // Load blocks and scales as U8 tensors on the VarBuilder device.
        let vb_u8 = vb.to_dtype(DType::U8);
        let blocks = vb_u8.get((out_dim, nblocks, MXFP4_BLOCK_BYTES), &blocks_name)?;
        let scales = vb_u8.get((out_dim, nblocks), &scales_name)?;

        // Dequantize to BF16 on the appropriate device (CPU or CUDA).
        let mut weight = candle::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [out_dim, in_dim])?;
        if !weight.device().same_device(vb.device()) {
            weight = weight.to_device(vb.device())?;
        }

        // Optionally load bias (kept BF16 as-is). Support dot and underscore naming.
        let bias_t = if bias {
            let bias_dot = format!("{base}.bias");
            let bias_us = format!("{base}_bias");
            let vb_bf16 = vb.to_dtype(DType::BF16);
            if vb.contains_tensor(&bias_dot) {
                Some(vb_bf16.get(out_dim, &bias_dot)?)
            } else if vb.contains_tensor(&bias_us) {
                Some(vb_bf16.get(out_dim, &bias_us)?)
            } else {
                None
            }
        } else {
            None
        };

        return Ok(Linear::new(weight, bias_t));
    }

    // Fallback: standard BF16 linear loading under "{base}.weight" and optional bias.
    let vb_bf16 = vb.to_dtype(DType::BF16).pp(base);
    candle_nn::linear_b(in_dim, out_dim, bias, vb_bf16)
}

/// Load a single expert’s Linear from grouped MXFP4 tensors stored under a common base
/// such as "experts.gate_up_proj" where the underlying tensors have a leading expert
/// dimension, e.g. `blocks: [n_experts, out_dim, in_dim/32, 16]` and `scales: [n_experts, out_dim, in_dim/32]`.
///
/// - `vb` should point at the module scope where the grouped tensors reside (e.g., the layer's `mlp`).
/// - `base` is the common prefix (e.g., "experts.gate_up_proj" or "experts.down_proj").
/// - `expert_idx` selects which expert slice to load.
pub fn load_expert_linear_mxfp4_grouped(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: candle_nn::VarBuilder,
    base: &str,
    expert_idx: usize,
    n_experts: usize,
) -> Result<Linear> {
    // Try paired grouped names: either dot or underscore variants.
    let blocks_dot = format!("{base}_blocks");
    let scales_dot = format!("{base}_scales");
    let has_us = vb.contains_tensor(&blocks_dot) && vb.contains_tensor(&scales_dot);
    let (blocks_name, scales_name) = if has_us {
        (blocks_dot, scales_dot)
    } else {
        // Also support dot suffix within grouped context: base.blocks / base.scales
        let blocks_alt = format!("{base}.blocks");
        let scales_alt = format!("{base}.scales");
        if vb.contains_tensor(&blocks_alt) && vb.contains_tensor(&scales_alt) {
            (blocks_alt, scales_alt)
        } else {
            candle::bail!("grouped MXFP4 tensors not found for base '{base}'")
        }
    };

    if in_dim % MXFP4_BLOCK_ELEMS != 0 {
        candle::bail!(
            "MXFP4 grouped weight '{base}': in_dim must be multiple of {MXFP4_BLOCK_ELEMS}, got {in_dim}"
        )
    }
    let nblocks = in_dim / MXFP4_BLOCK_ELEMS;

    // Load U8 grouped tensors and select expert slice along the leading dimension.
    let vb_u8 = vb.to_dtype(DType::U8);
    let blocks_g = vb_u8.get((n_experts, out_dim, nblocks, MXFP4_BLOCK_BYTES), &blocks_name)?; // (E, out, nb, 16)
    let scales_g = vb_u8.get((n_experts, out_dim, nblocks), &scales_name)?; // (E, out, nb)
    if matches!(std::env::var("CANDLE_DEBUG_MXFP4_SHAPES").ok().as_deref(), Some("1") | Some("true") | Some("TRUE")) {
        if expert_idx == 0 {
            eprintln!(
                "[MXFP4] {}: blocks {:?}, scales {:?} (expect (E={}, out={}, nb={}, 16); (E={}, out={}, nb={}))",
                base,
                blocks_g.dims(),
                scales_g.dims(),
                n_experts,
                out_dim,
                nblocks,
                n_experts,
                out_dim,
                nblocks
            );
        }
    }
    let blocks = blocks_g.narrow(0, expert_idx, 1)?.squeeze(0)?; // (out, nb, 16)
    let scales = scales_g.narrow(0, expert_idx, 1)?.squeeze(0)?; // (out, nb)

    let mut weight = candle::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [out_dim, in_dim])?;
    if !weight.device().same_device(vb.device()) {
        weight = weight.to_device(vb.device())?;
    }

    // Optional grouped bias under e.g. "experts.gate_up_proj_bias" (shape [E, out]).
    let bias_t = if bias {
        let bias_us = format!("{base}_bias");
        let bias_dot = format!("{base}.bias");
        let vb_bf16 = vb.to_dtype(DType::BF16);
        if vb.contains_tensor(&bias_us) {
            Some(vb_bf16.get((n_experts, out_dim), &bias_us)?.narrow(0, expert_idx, 1)?.squeeze(0)?)
        } else if vb.contains_tensor(&bias_dot) {
            Some(vb_bf16.get((n_experts, out_dim), &bias_dot)?.narrow(0, expert_idx, 1)?.squeeze(0)?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(Linear::new(weight, bias_t))
}

// ============================
// Attention with sinks helpers
// ============================

/// Compute renormalization scale per (b,h,q) given FA LSE and head-wise sinks.
/// scale = exp(lse - logsumexp([lse, sink])) = 1 / (1 + exp(sink - lse))
fn sinks_scale_from_lse(lse: &Tensor, sinks: &Tensor) -> Result<Tensor> {
    // lse: (b, h, q) in f32 or bf16/f16, convert to f32 for stability
    let lse_f32 = lse.to_dtype(DType::F32)?;
    // sinks: per-head values; accept shape (h) or broadcastable to (b, h, q)
    // Convert to f32 and broadcast.
    let sinks_f32 = sinks.to_dtype(DType::F32)?;
    let (b, h, q) = lse_f32.dims3()?;
    // Expand sinks to (b, h, q)
    let sinks_bhq = sinks_f32
        .reshape((1, h, 1))?
        .broadcast_as((b, h, q))?;

    // combined_lse = logsumexp([lse, sinks], axis=-1 of the concat)
    let lse_exp = lse_f32.unsqueeze(D::Minus1)?; // (b,h,q,1)
    let sinks_exp = sinks_bhq.unsqueeze(D::Minus1)?; // (b,h,q,1)
    let cat = Tensor::cat(&[&lse_exp, &sinks_exp], D::Minus1)?; // (b,h,q,2)
    let combined_lse = cat.log_sum_exp(D::Minus1)?; // (b,h,q)
    let scale = (lse_f32 - &combined_lse)?.exp()?; // (b,h,q)
    Ok(scale)
}

/// Eager attention with optional sinks scaling (causal or not).
/// Inputs are in FA layout: (b, seq_len_q, n_heads, head_dim) etc.
pub fn eager_attn_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
    sinks: Option<&Tensor>, // per-head values (h)
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let (b, qlen, h, _d) = q.dims4()?;
    let (_, klen, _, _) = k.dims4()?;

    // Convert to f32 for stability.
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;

    // (b,h,q,d) @ (b,h,d,k) -> (b,h,q,k)
    let q_bhqd = q.transpose(1, 2)?;
    let k_bhkd = k.transpose(1, 2)?;
    let v_bhkd = v.transpose(1, 2)?;
    let logits = (
        q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * softmax_scale as f64
    )?;

    // Apply causal mask if requested.
    let logits = if causal && qlen > 1 {
        // mask shape (q,k) with j > i masked
        let mask: Vec<u8> = (0..qlen)
            .flat_map(|i| (0..klen).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (qlen, klen), q.device())?;
        masked_fill(&logits, &mask.broadcast_as((b, h, qlen, klen))?, f32::NEG_INFINITY)?
    } else {
        logits
    };

    let att = candle_nn::ops::softmax_last_dim(&logits)?; // (b,h,q,k)
    let o_bhqd = att.matmul(&v_bhkd.contiguous()?)?; // (b,h,q,d)
    let mut o_bqhd = o_bhqd.transpose(1, 2)?; // (b,q,h,d)

    // Apply sinks renormalization if provided.
    if let Some(sinks_t) = sinks {
        // lse along keys
        let lse = logits.log_sum_exp(D::Minus1)?; // (b,h,q)
        let scale = sinks_scale_from_lse(&lse, sinks_t)?; // (b,h,q)
        let scale = scale.transpose(1, 2)?.unsqueeze(D::Minus1)?; // (b,q,h,1)
        let (_, qlen, h, d) = o_bqhd.dims4()?;
        let scale = scale.broadcast_as((b, qlen, h, d))?; // match (b,q,h,d)
        o_bqhd = (o_bqhd.to_dtype(DType::F32)? * &scale)?.to_dtype(in_dtype)?;
    } else {
        o_bqhd = o_bqhd.to_dtype(in_dtype)?;
    }

    Ok(o_bqhd)
}

/// Eager attention with windowed masking and optional sinks scaling.
pub fn eager_attn_windowed_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let (b, qlen, h, _d) = q.dims4()?;
    let (_, klen, _, _) = k.dims4()?;
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let q_bhqd = q.transpose(1, 2)?;
    let k_bhkd = k.transpose(1, 2)?;
    let v_bhkd = v.transpose(1, 2)?;
    let logits = (
        q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * softmax_scale as f64
    )?; // (b,h,q,k)

    // Build windowed mask if needed.
    let logits = match (window_size_left, window_size_right) {
        (None, None) => logits,
        (wsl, wsr) => {
            let left = wsl.unwrap_or(qlen);
            let right = wsr.unwrap_or(klen);
            let mask: Vec<u8> = (0..qlen)
                .flat_map(|i| {
                    (0..klen).map(move |j| {
                        let i = i as isize;
                        let j = j as isize;
                        let l = left as isize;
                        let r = right as isize;
                        // Match HF sliding_window_overlay semantics: kv_idx > q_idx - sliding_window
                        // Combined with causal, the allowed band is (i - left, i + right] (left-exclusive, right-inclusive)
                        let allow = (j > i - l) && (j <= i + r);
                        u8::from(!allow)
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (qlen, klen), q.device())?;
            masked_fill(&logits, &mask.broadcast_as((b, h, qlen, klen))?, f32::NEG_INFINITY)?
        }
    };

    let att = candle_nn::ops::softmax_last_dim(&logits)?;
    let o_bhqd = att.matmul(&v_bhkd.contiguous()?)?;
    let mut o_bqhd = o_bhqd.transpose(1, 2)?;

    if let Some(sinks_t) = sinks {
        let lse = logits.log_sum_exp(D::Minus1)?; // (b,h,q)
        let scale = sinks_scale_from_lse(&lse, sinks_t)?; // (b,h,q)
        let scale = scale.transpose(1, 2)?.unsqueeze(D::Minus1)?; // (b,q,h,1)
        let (_, qlen, h, d) = o_bqhd.dims4()?;
        let scale = scale.broadcast_as((b, qlen, h, d))?;
        o_bqhd = (o_bqhd.to_dtype(DType::F32)? * &scale)?.to_dtype(in_dtype)?;
    } else {
        o_bqhd = o_bqhd.to_dtype(in_dtype)?;
    }

    Ok(o_bqhd)
}

#[cfg(feature = "flash-attn")]
/// Flash-attn path with sinks renormalization using FA-returned LSE.
pub fn flash_attn_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    match sinks {
        None => candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal),
        Some(s) => {
            let (o, lse) = candle_flash_attn::flash_attn_with_lse(q, k, v, softmax_scale, causal)?;
            let scale = sinks_scale_from_lse(&lse, s)?; // (b,h,q)
            // Broadcast to (b,q,h,1) to match o (b,q,h,d)
            let scale = scale.transpose(1, 2)?.unsqueeze(D::Minus1)?; // (b,q,h,1)
            let (b, qlen, h, d) = o.dims4()?;
            let scale = scale.broadcast_as((b, qlen, h, d))?; // (b,q,h,d)
            let out = (o.to_dtype(DType::F32)? * scale)?.to_dtype(o.dtype())?;
            Ok(out)
        }
    }
}

#[cfg(feature = "flash-attn")]
/// Windowed flash-attn path with sinks renormalization.
pub fn flash_attn_windowed_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    match sinks {
        None => candle_flash_attn::flash_attn_windowed(q, k, v, softmax_scale, window_size_left, window_size_right),
        Some(s) => {
            let (o, lse) = candle_flash_attn::flash_attn_windowed_with_lse(
                q,
                k,
                v,
                softmax_scale,
                window_size_left,
                window_size_right,
            )?;
            let scale = sinks_scale_from_lse(&lse, s)?; // (b,h,q)
            let scale = scale.transpose(1, 2)?.unsqueeze(D::Minus1)?; // (b,q,h,1)
            let (b, qlen, h, d) = o.dims4()?;
            let scale = scale.broadcast_as((b, qlen, h, d))?; // (b,q,h,d)
            let out = (o.to_dtype(DType::F32)? * scale)?.to_dtype(o.dtype())?;
            Ok(out)
        }
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    // Validate sinks_scale_from_lse matches 1 / (1 + exp(sink - lse)).
    #[test]
    fn t18_sinks_scale_formula() -> Result<()> {
        let dev = Device::Cpu;
        // lse: (b=1, h=2, q=3)
        let lse = Tensor::from_vec(
            vec![
                // h0
                0.0f32, 1.0, 2.0,
                // h1
                -1.0, 0.5, 3.0,
            ],
            (1, 2, 3),
            &dev,
        )?;
        // sinks: (h=2)
        let sinks = Tensor::from_vec(vec![0.5f32, -0.5], 2, &dev)?;
        let scale = sinks_scale_from_lse(&lse, &sinks)?; // (1,2,3)
        let got = scale.to_vec3::<f32>()?;
        // Compute reference
        let lse_host = lse.to_vec3::<f32>()?;
        for b in 0..1 {
            for h in 0..2 {
                for q in 0..3 {
                    let lse_v = lse_host[b][h][q];
                    let s = if h == 0 { 0.5 } else { -0.5 };
                    let ref_v = 1.0 / (1.0 + (s - lse_v).exp());
                    let diff = (ref_v - got[b][h][q]).abs();
                    assert!(diff < 1e-6, "mismatch at (b={b},h={h},q={q}): ref={ref_v} got={} diff={diff}", got[b][h][q]);
                }
            }
        }
        Ok(())
    }
}
