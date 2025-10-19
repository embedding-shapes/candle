use candle::{DType, Device, Result, Tensor, D};
use candle_transformers::models::gpt_oss::eager_attn_with_sinks;

// Helper: compute baseline by explicitly concatenating a per-head sink logit,
// softmax over [tokens + sink], drop sink column, then @V.
fn baseline_with_sink(
    q: &Tensor, // (b, q, h, d)
    k: &Tensor, // (b, k, h, d)
    v: &Tensor, // (b, k, h, d)
    sinks: &Tensor, // (h)
    softmax_scale: f32,
) -> Result<Tensor> {
    // Move to f32 for numeric stability and compute logits in (b,h,q,k)
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let (b, qlen, h, _d) = q.dims4()?;
    let (_, klen, _, _) = k.dims4()?;

    let q_bhqd = q.transpose(1, 2)?; // (b,h,q,d)
    let k_bhkd = k.transpose(1, 2)?; // (b,h,k,d)
    let v_bhkd = v.transpose(1, 2)?; // (b,h,k,d)
    let logits = (
        q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * softmax_scale as f64
    )?; // (b,h,q,k)

    // Append sink per head and query: broadcast (h) -> (b,h,q,1)
    let sinks = sinks.to_dtype(DType::F32)?;
    let sink_bhq1 = sinks.reshape((1, h, 1, 1))?.broadcast_as((b, h, qlen, 1))?; // (b,h,q,1)
    let logits_with_sink = Tensor::cat(&[&logits, &sink_bhq1], D::Minus1)?.contiguous()?; // (b,h,q,k+1)

    // Softmax over last dim in f32, drop sink column
    let probs_with_sink = candle_nn::ops::softmax_last_dim(&logits_with_sink)?; // (b,h,q,k+1)
    let probs = probs_with_sink.narrow(D::Minus1, 0, klen)?.contiguous()?; // (b,h,q,k)

    // @V to produce (b,h,q,d), then transpose to (b,q,h,d)
    let out_bhqd = probs.matmul(&v_bhkd.contiguous()?)?; // (b,h,q,d)
    out_bhqd.transpose(1, 2)
}

fn rel_error(a: &Tensor, b: &Tensor) -> Result<f32> {
    let num = (a - b)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    let den = a.abs()?.sum_all()?.to_scalar::<f32>()?;
    Ok(if den == 0.0 { num } else { num / den })
}

#[test]
fn attn_with_sinks_two_tokens() -> Result<()> {
    let dev = Device::Cpu;
    // Small, numerically sensitive shapes
    let b = 1usize;
    let h = 2usize;
    let qlen = 2usize;
    let klen = 2usize;
    let d = 64usize;
    let scale = 1f32 / (d as f32).sqrt();

    // Deterministic small-valued tensors
    let q = (Tensor::arange(0f32, (b * qlen * h * d) as f32, &dev)? / 37.0)?
        .reshape((b, qlen, h, d))?;
    let k = (Tensor::arange(0f32, (b * klen * h * d) as f32, &dev)? / 41.0)?
        .reshape((b, klen, h, d))?;
    let v = (Tensor::arange(0f32, (b * klen * h * d) as f32, &dev)? / 53.0)?
        .reshape((b, klen, h, d))?;
    let sinks = Tensor::from_vec(vec![0.15f32, -0.05f32], h, &dev)?;

    let ref_out = baseline_with_sink(&q, &k, &v, &sinks, scale)?;
    let rust_out = eager_attn_with_sinks(&q, &k, &v, scale, /*causal=*/ false, Some(&sinks))?;

    let rel = rel_error(&ref_out.to_dtype(DType::F32)?, &rust_out.to_dtype(DType::F32)?)?;
    assert!(rel <= 1e-3, "relative error too large: {}", rel);
    Ok(())
}

#[test]
fn attn_with_sinks_three_tokens() -> Result<()> {
    let dev = Device::Cpu;
    let b = 1usize;
    let h = 3usize;
    let qlen = 3usize;
    let klen = 3usize;
    let d = 64usize;
    let scale = 1f32 / (d as f32).sqrt();

    let q = (Tensor::arange(0f32, (b * qlen * h * d) as f32, &dev)? / 29.0)?
        .reshape((b, qlen, h, d))?;
    let k = (Tensor::arange(0f32, (b * klen * h * d) as f32, &dev)? / 31.0)?
        .reshape((b, klen, h, d))?;
    let v = (Tensor::arange(0f32, (b * klen * h * d) as f32, &dev)? / 47.0)?
        .reshape((b, klen, h, d))?;
    let sinks = Tensor::from_vec(vec![0.10f32, 0.20f32, -0.10f32], h, &dev)?;

    let ref_out = baseline_with_sink(&q, &k, &v, &sinks, scale)?;
    let rust_out = eager_attn_with_sinks(&q, &k, &v, scale, /*causal=*/ false, Some(&sinks))?;

    let rel = rel_error(&ref_out.to_dtype(DType::F32)?, &rust_out.to_dtype(DType::F32)?)?;
    assert!(rel <= 1e-3, "relative error too large: {}", rel);
    Ok(())
}
