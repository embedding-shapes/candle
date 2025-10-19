use candle::{DType, Device, Result, Tensor};
use candle_nn::ops;
use candle_transformers::models::gpt_oss::rotary::{GptOssRopeConfig, GptOssRotaryEmbedding};

fn yarn_mscale(scale: f32) -> f32 {
    if scale <= 1.0 { 1.0 } else { 0.1 * scale.ln() + 1.0 }
}

#[test]
fn yarn_equivalence_scaled_tables_vs_scaled_softmax() -> Result<()> {
    let dev = Device::Cpu;
    let head_dim = 64usize;
    let max_t = 256usize;
    let rope_theta = 150000.0f32;
    let factor = 32.0f32; // extrapolation factor >1 -> mscale >1
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;
    let orig = 4096usize;

    // Build base RoPE (no amplitude scaling in tables)
    let cfg = GptOssRopeConfig::new(head_dim, max_t, rope_theta, factor, beta_fast, beta_slow, orig);
    let rope = GptOssRotaryEmbedding::new_yarn(DType::F32, &dev, &cfg)?;
    let (b, h, t, d) = (1usize, 1usize, 8usize, head_dim);

    // Deterministic q/k/v
    let q = (Tensor::arange(0u32, (b * h * t * d) as u32, &dev)?
        .to_dtype(DType::F32)?
        .reshape((b, h, t, d))? / 123.0)?;
    let k = (Tensor::arange(0u32, (b * h * t * d) as u32, &dev)?
        .to_dtype(DType::F32)?
        .reshape((b, h, t, d))? / 111.0)?;
    let v = (&k / 7.0)?;

    // Apply RoPE once; API expects (b,h,t,d)
    let (q1, k1) = rope.apply_rotary_emb_qk(&q, &k, 0)?; // (b,h,t,d)

    // Work in 2D: (t,d)
    let q1_2d = q1.squeeze(0)?.squeeze(0)?; // (t,d)
    let k1_2d = k1.squeeze(0)?.squeeze(0)?; // (t,d)
    let v_2d = v.squeeze(0)?.squeeze(0)?;   // (t,d)

    // Path A: emulate scaled tables by scaling q/k post-RoPE.
    let m = yarn_mscale(factor) as f32;
    let q_a = (&q1_2d * (m as f64))?;
    let k_a = (&k1_2d * (m as f64))?;
    let scale_a = 1.0f32 / (d as f32).sqrt();
    let scores_a = (q_a.matmul(&k_a.transpose(0, 1)?)? * (scale_a as f64))?;
    let probs_a = ops::softmax_last_dim(&scores_a)?; // (t,t)
    let out_a = probs_a.matmul(&v_2d)?; // (t,d)

    // Path B: unscaled tables, scale softmax by m^2.
    let scale_b = (1.0f32 / (d as f32).sqrt()) * (m * m);
    let scores_b = (q1_2d.matmul(&k1_2d.transpose(0, 1)?)? * (scale_b as f64))?;
    let probs_b = ops::softmax_last_dim(&scores_b)?;
    let out_b = probs_b.matmul(&v_2d)?;

    let diff = (out_a.to_dtype(DType::F32)? - out_b.to_dtype(DType::F32)?)?
        .abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
    assert!(diff < 1e-4, "equivalence broken, max diff = {}", diff);
    Ok(())
}
