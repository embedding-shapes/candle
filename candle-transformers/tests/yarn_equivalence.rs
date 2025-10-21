use candle::{DType, Device, Result, Tensor, D};

fn yarn_mscale(factor: f32) -> f32 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1 * factor.ln() + 1.0
    }
}

#[test]
fn yarn_scaling_equivalence_softmax_vs_sincos() -> Result<()> {
    let dev = Device::Cpu;
    let b = 1usize;
    let h = 2usize;
    let t = 4usize;
    let d = 64usize;
    let factor = 32.0f32; // typical GPT-OSS value
    let m = yarn_mscale(factor);

    // Random q,k in F32 for stability
    let q = Tensor::randn(0.0f32, 1.0, (b, h, t, d), &dev)?;
    let k = Tensor::randn(0.0f32, 1.0, (b, h, t, d), &dev)?;

    // Build unscaled sin/cos tables using the same recipe as the model (subset)
    use candle_transformers::models::gpt_oss::rotary::GptOssRopeConfig;
    use candle_transformers::models::gpt_oss::rotary::GptOssRotaryEmbedding;
    let rope_cfg = GptOssRopeConfig::new(d, 2048, 150000.0, factor, 32.0, 1.0, 4096);
    let rope = GptOssRotaryEmbedding::new_yarn(DType::F32, &dev, &rope_cfg)?;
    let sin = rope.sin_table().narrow(0, 0, t)?; // (t, d/2)
    let cos = rope.cos_table().narrow(0, 0, t)?;

    // Case A: scale sin/cos by m; scale = 1/sqrt(d)
    let sin_a = (&sin * m as f64)?;
    let cos_a = (&cos * m as f64)?;
    let q_a = candle_nn::rotary_emb::rope(&q, &cos_a, &sin_a)?; // (b,h,t,d)
    let k_a = candle_nn::rotary_emb::rope(&k, &cos_a, &sin_a)?;
    let scale_a = 1.0f32 / (d as f32).sqrt();
    let s_a = (q_a
        .contiguous()?
        .matmul(&k_a.transpose(2, 3)?.contiguous()?)?
        * scale_a as f64)?; // (b,h,t,t)

    // Case B: unscaled sin/cos; scale = 1/sqrt(d) * m^2
    let q_b = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
    let k_b = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
    let scale_b = (1.0f32 / (d as f32).sqrt()) * (m * m);
    let s_b = (q_b
        .contiguous()?
        .matmul(&k_b.transpose(2, 3)?.contiguous()?)?
        * scale_b as f64)?;

    // Compare softmax distributions
    let p_a = candle_nn::ops::softmax(&s_a, D::Minus1)?.to_dtype(DType::F32)?;
    let p_b = candle_nn::ops::softmax(&s_b, D::Minus1)?.to_dtype(DType::F32)?;
    let diff = (p_a - p_b)?.abs()?;
    let flat = diff.flatten_all()?;
    let max_diff = flat.max(D::Minus1)?.to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-4,
        "YARN equivalence broken: max_diff={}",
        max_diff
    );
    Ok(())
}
