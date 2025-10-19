use candle::{DType, Device, Result, Tensor, D, IndexOp};
use candle_transformers::models::gpt_oss::rotary::{GptOssRopeConfig, GptOssRotaryEmbedding};

fn apply_rotary_reference(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: (b, h, q, d), cos/sin: (q, d/2)
    let (_b, _h, q, d) = x.dims4()?;
    let d2 = d / 2;
    let cos = cos
        .narrow(0, 0, q)?
        .unsqueeze(0)?
        .unsqueeze(1)?; // (1,1,q,d/2)
    let sin = sin
        .narrow(0, 0, q)?
        .unsqueeze(0)?
        .unsqueeze(1)?; // (1,1,q,d/2)
    let x1 = x.narrow(D::Minus1, 0, d2)?;
    let x2 = x.narrow(D::Minus1, d2, d2)?;
    let first = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let second = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&first, &second], D::Minus1)
}

fn make_reference_cos_sin(
    head_dim: usize,
    max_t: usize,
    rope_theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position_embeddings: usize,
    truncate: bool,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let dim = head_dim;
    let dim2 = dim / 2;
    let pos_freqs: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| rope_theta.powf(i as f32 / dim as f32))
        .collect();
    let inv_freq_extrapolation: Vec<f32> = pos_freqs.iter().map(|&v| 1.0 / v).collect();
    let inv_freq_interpolation: Vec<f32> = inv_freq_extrapolation.iter().map(|&v| v / factor).collect();

    fn find_correction_dim(num_rot: f32, dim: usize, base: f32, max_position_embeddings: usize) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * base.ln())
    }
    let mut low = find_correction_dim(beta_fast, dim, rope_theta, original_max_position_embeddings);
    let mut high = find_correction_dim(beta_slow, dim, rope_theta, original_max_position_embeddings);
    if truncate {
        low = low.floor();
        high = high.ceil();
    }
    low = low.max(0.0);
    high = high.min(dim as f32 - 1.0);

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
    let ones = Tensor::ones(ramp.shape().dims(), DType::F32, dev)?;
    let extr = ones.broadcast_sub(&ramp)?; // extrapolation factor = 1 - ramp
    let inv_freq = inv_interp
        .broadcast_mul(&ones.broadcast_sub(&extr)?)?
        .broadcast_add(&inv_extrap.broadcast_mul(&extr)?)?; // (1,dim/2)

    let t = Tensor::arange(0u32, max_t as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((max_t, 1))?;
    let freqs = t.matmul(&inv_freq)?; // (T, dim/2)

    // attention factor (mscale)
    let attention_scale = if factor <= 1.0 { 1.0 } else { 0.1 * factor.ln() + 1.0 };
    let sin = (freqs.sin()? * attention_scale as f64)?.to_dtype(DType::F32)?;
    let cos = (freqs.cos()? * attention_scale as f64)?.to_dtype(DType::F32)?;
    Ok((cos, sin))
}

#[test]
fn gpt_oss_yarn_correctness_small() -> Result<()> {
    let dev = Device::Cpu;
    let head_dim = 8usize;
    let max_t = 16usize;
    let rope_theta = 15000.0f32;
    let factor = 32.0f32;
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;
    let orig = 4096usize;

    // Inputs
    let b = 1;
    let h = 1;
    let q = 4;
    let d = head_dim;
    let q_in = Tensor::arange(0f32, (b * q * h * d) as f32, &dev)?.reshape((b, h, q, d))?;
    let k_in = (Tensor::arange(0f32, (b * q * h * d) as f32, &dev)?.reshape((b, h, q, d))? * 0.5f64)?;

    // Reference cos/sin
    let (cos_ref, sin_ref) = make_reference_cos_sin(
        head_dim,
        max_t,
        rope_theta,
        factor,
        beta_fast,
        beta_slow,
        orig,
        false,
        &dev,
    )?;

    // Apply reference
    let q_ref = apply_rotary_reference(&q_in, &cos_ref, &sin_ref)?;
    let k_ref = apply_rotary_reference(&k_in, &cos_ref, &sin_ref)?;

    // Our implementation
    let cfg = GptOssRopeConfig::new(head_dim, max_t, rope_theta, factor, beta_fast, beta_slow, orig);
    let rope = GptOssRotaryEmbedding::new_yarn(DType::F32, &dev, &cfg)?;
    let (q_out, k_out) = rope.apply_rotary_emb_qk(&q_in, &k_in, 0)?;

    // Quick diagnostics on cos/sin tables if mismatch
    let cos_e = rope.cos_table().narrow(0, 0, q)?;
    let sin_e = rope.sin_table().narrow(0, 0, q)?;
    let cos_r = cos_ref.narrow(0, 0, q)?;
    let sin_r = sin_ref.narrow(0, 0, q)?;
    let cos_diff = (cos_e.clone() - &cos_r)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let sin_diff = (sin_e.clone() - &sin_r)?.abs()?.max_all()?.to_scalar::<f32>()?;
    // Compare
    let diff_q = (q_ref - &q_out)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let diff_k = (k_ref - &k_out)?.abs()?.max_all()?.to_scalar::<f32>()?;
    // Print a small slice for debugging when failing
    if diff_q >= 1e-4 {
        // Also print the ramp and inv_freq reference for debugging
        let dim2 = (head_dim / 2) as usize;
        let pos_freqs: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_extrapolation: Vec<f32> = pos_freqs.iter().map(|&v| 1.0 / v).collect();
        let inv_freq_interpolation: Vec<f32> =
            inv_freq_extrapolation.iter().map(|&v| v / factor).collect();
        let mut low = {
            (head_dim as f32 * (orig as f32 / (beta_fast * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * rope_theta.ln())
        };
        let mut high = {
            (head_dim as f32 * (orig as f32 / (beta_slow * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * rope_theta.ln())
        };
        low = low.max(0.0);
        high = high.min(head_dim as f32 - 1.0);
        let mut maxv = high;
        if (low - maxv).abs() < f32::EPSILON {
            maxv += 0.001;
        }
        let idx = Tensor::arange(0f32, dim2 as f32, &dev)?;
        let num = idx.broadcast_sub(&Tensor::new(low as f32, &dev)?)?;
        let den = Tensor::new((maxv - low) as f32, &dev)?;
        let ramp = num.broadcast_div(&den)?.clamp(0.0, 1.0)?;
        println!("ramp={:?}", ramp.to_vec1::<f32>()?);
        let inv_extrap = Tensor::from_vec(inv_freq_extrapolation, (1, dim2), &dev)?;
        let inv_interp = Tensor::from_vec(inv_freq_interpolation, (1, dim2), &dev)?;
        let extr = Tensor::ones((1, dim2), DType::F32, &dev)?.broadcast_sub(&ramp)?;
        let inv_freq = inv_interp
            .broadcast_mul(&Tensor::ones((1, dim2), DType::F32, &dev)?.broadcast_sub(&extr)?)?
            .broadcast_add(&inv_extrap.broadcast_mul(&extr)?)?;
        println!("inv_freq_ref={:?}", inv_freq.to_vec2::<f32>()?);
        let c_e0 = cos_e.i((0, ..1))?.flatten_all()?;
        let c_r0 = cos_r.i((0, ..1))?.flatten_all()?;
        let s_e0 = sin_e.i((0, ..1))?.flatten_all()?;
        let s_r0 = sin_r.i((0, ..1))?.flatten_all()?;
        println!(
            "cos_e[0][:8]={:?}",
            c_e0.to_vec1::<f32>()?.into_iter().take(8).collect::<Vec<_>>()
        );
        println!(
            "cos_r[0][:8]={:?}",
            c_r0.to_vec1::<f32>()?.into_iter().take(8).collect::<Vec<_>>()
        );
        println!(
            "sin_e[0][:8]={:?}",
            s_e0.to_vec1::<f32>()?.into_iter().take(8).collect::<Vec<_>>()
        );
        println!(
            "sin_r[0][:8]={:?}",
            s_r0.to_vec1::<f32>()?.into_iter().take(8).collect::<Vec<_>>()
        );
        println!("cos_e (q x d2) = {:?}", cos_e.to_vec2::<f32>()?);
        println!("cos_r (q x d2) = {:?}", cos_r.to_vec2::<f32>()?);
        println!("sin_e (q x d2) = {:?}", sin_e.to_vec2::<f32>()?);
        println!("sin_r (q x d2) = {:?}", sin_r.to_vec2::<f32>()?);
    }
    assert!(diff_q < 1e-4, "max abs diff q = {} (cos max diff {}, sin max diff {})", diff_q, cos_diff, sin_diff);
    assert!(diff_k < 1e-4, "max abs diff k = {} (cos max diff {}, sin max diff {})", diff_k, cos_diff, sin_diff);
    Ok(())
}

#[test]
fn gpt_oss_yarn_long_positions_stability() -> Result<()> {
    let dev = Device::Cpu;
    let head_dim = 64usize;
    let max_t = 65536usize; // 64K
    let rope_theta = 150000.0f32;
    let factor = 32.0f32;
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;
    let orig = 4096usize;

    let cfg = GptOssRopeConfig::new(head_dim, max_t, rope_theta, factor, beta_fast, beta_slow, orig);
    let rope = GptOssRotaryEmbedding::new_yarn(DType::F32, &dev, &cfg)?;
    let cos = rope.cos_table().clone();
    let sin = rope.sin_table().clone();

    // Bounds and finiteness checks
    let cos_max = cos.abs()?.max_all()?.to_scalar::<f32>()?;
    let sin_max = sin.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(cos_max.is_finite() && sin_max.is_finite());
    // With YARN attention scaling, cos/sin magnitude can exceed 1. Clamp to a safe bound.
    assert!(cos_max <= 2.0 && sin_max <= 2.0);
    // Ensure not all zeros
    let cos_sum = cos.sum_all()?.to_scalar::<f32>()?;
    let sin_sum = sin.sum_all()?.to_scalar::<f32>()?;
    assert!(cos_sum != 0.0 || sin_sum != 0.0);
    Ok(())
}
