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

        let inv_freq_extrapolation = Tensor::from_vec(inv_freq_extrapolation, (1, dim2), dev)?;
        let inv_freq_interpolation = Tensor::from_vec(inv_freq_interpolation, (1, dim2), dev)?;

        let ones = Tensor::ones(
            inv_freq_extrapolation_factor.shape().dims(),
            DType::F32,
            dev,
        )?;
        let inv_freq = inv_freq_interpolation
            .broadcast_mul(&ones.broadcast_sub(&inv_freq_extrapolation_factor)?)?
            .broadcast_add(
                &inv_freq_extrapolation.broadcast_mul(&inv_freq_extrapolation_factor)?,
            )?;

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

        Ok(Self {
            sin,
            cos,
            attn_factor,
        })
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

    pub fn cos_table(&self) -> &Tensor {
        &self.cos
    }
    pub fn sin_table(&self) -> &Tensor {
        &self.sin
    }
    pub fn attention_factor(&self) -> f32 {
        self.attn_factor
    }
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
    if (min - max).abs() < f32::EPSILON {
        max += 0.001;
    }
    let idx = Tensor::arange(0f32, dim as f32, dev)?;
    let num = idx.broadcast_sub(&Tensor::new(min as f32, dev)?)?;
    let den = Tensor::new((max - min) as f32, dev)?;
    let linear = num.broadcast_div(&den)?;
    linear.clamp(0.0, 1.0)
}

pub fn yarn_get_mscale(scale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * scale.ln() + 1.0
    }
}
