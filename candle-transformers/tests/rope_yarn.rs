use candle::{DType, Device, Result, Tensor, D};
use candle_transformers::models::gpt_oss::rotary::{GptOssRopeConfig, GptOssRotaryEmbedding};

// Reference: mirrors HF YARN rope_scaling semantics without any double scaling.
fn yarn_reference_cos_sin(
    head_dim: usize,
    pos: usize,
    rope_theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position_embeddings: usize,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let dim2 = head_dim / 2;
    // base^(i/dim)
    let pos_freqs: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| rope_theta.powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq_extrapolation: Vec<f32> = pos_freqs.iter().map(|&v| 1.0 / v).collect();
    let inv_freq_interpolation: Vec<f32> =
        inv_freq_extrapolation.iter().map(|&v| v / factor).collect();

    fn find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32
            * (max_position_embeddings as f32 / (num_rot * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * base.ln())
    }
    let mut low = find_correction_dim(
        beta_fast,
        head_dim,
        rope_theta,
        original_max_position_embeddings,
    );
    let mut high = find_correction_dim(
        beta_slow,
        head_dim,
        rope_theta,
        original_max_position_embeddings,
    );
    low = low.max(0.0);
    high = high.min(head_dim as f32 - 1.0);
    let mut maxv = high;
    if (low - maxv).abs() < f32::EPSILON {
        maxv += 0.001;
    }
    let idx = Tensor::arange(0f32, dim2 as f32, dev)?;
    let num = idx.broadcast_sub(&Tensor::new(low as f32, dev)?)?;
    let den = Tensor::new((maxv - low) as f32, dev)?;
    let ramp = num.broadcast_div(&den)?.clamp(0.0, 1.0)?; // [0,1]

    let inv_extrap = Tensor::from_vec(inv_freq_extrapolation, (1, dim2), dev)?;
    let inv_interp = Tensor::from_vec(inv_freq_interpolation, (1, dim2), dev)?;
    let ones = Tensor::ones((1, dim2), DType::F32, dev)?;
    let extr = ones.broadcast_sub(&ramp)?; // extrapolation factor = 1 - ramp
    let inv_freq = inv_interp
        .broadcast_mul(&ones.broadcast_sub(&extr)?)?
        .broadcast_add(&inv_extrap.broadcast_mul(&extr)?)?; // (1,dim/2)

    // Per-position frequencies
    let t = Tensor::new(pos as f32, dev)?.reshape((1, 1))?; // (1,1)
    let freqs = t.matmul(&inv_freq)?; // (1,dim/2)
                                      // Apply HF-style YARN attention scaling directly to tables.
    let mscale = if factor <= 1.0 {
        1.0
    } else {
        0.1 * factor.ln() + 1.0
    };
    let sin = (freqs.sin()? * mscale as f64)?.to_dtype(DType::F32)?;
    let cos = (freqs.cos()? * mscale as f64)?.to_dtype(DType::F32)?;
    Ok((cos, sin))
}

fn apply_rotary_reference(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: (b, h, q, d), cos/sin: (1, d/2)
    let (_b, _h, _q, d) = x.dims4()?;
    let d2 = d / 2;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // (1,1,1,d/2)
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?; // (1,1,1,d/2)
    let x1 = x.narrow(D::Minus1, 0, d2)?;
    let x2 = x.narrow(D::Minus1, d2, d2)?;
    let first = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let second = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&first, &second], D::Minus1)
}

#[test]
fn rope_yarn_spot_positions_against_reference() -> Result<()> {
    let dev = Device::Cpu;
    // Match GPT-OSS defaults
    let head_dim = 64usize;
    let rope_theta = 150000.0f32;
    let factor = 32.0f32;
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;
    let orig = 4096usize;
    let positions = [0usize, 4095, 4096, 8191, 131071];

    // Use max position for table creation once
    let max_t = positions.iter().copied().max().unwrap() + 1;
    let rope_cfg = GptOssRopeConfig::new(
        head_dim, max_t, rope_theta, factor, beta_fast, beta_slow, orig,
    );
    let rope = GptOssRotaryEmbedding::new_yarn(DType::F32, &dev, &rope_cfg)?;

    // Fixed inputs
    let b = 1;
    let h = 1;
    let d = head_dim;
    let q_in = Tensor::arange(0f32, (b * h * 1 * d) as f32, &dev)?.reshape((b, h, 1, d))?; // (1,1,1,d)
    let k_in = (&q_in * 0.5f64)?; // deterministic different input

    for &pos in &positions {
        let (cos_ref, sin_ref) = yarn_reference_cos_sin(
            head_dim, pos, rope_theta, factor, beta_fast, beta_slow, orig, &dev,
        )?; // (1, d/2)

        let (q_out, k_out) = rope.apply_rotary_emb_qk(&q_in, &k_in, pos)?; // offset=pos, t=1
        let q_ref = apply_rotary_reference(&q_in, &cos_ref, &sin_ref)?;
        let k_ref = apply_rotary_reference(&k_in, &cos_ref, &sin_ref)?;

        let dq = (q_out - &q_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let dk = (k_out - &k_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(dq < 1e-4, "pos {}: max|dq|={} exceeds tolerance", pos, dq);
        assert!(dk < 1e-4, "pos {}: max|dk|={} exceeds tolerance", pos, dk);
    }

    Ok(())
}
