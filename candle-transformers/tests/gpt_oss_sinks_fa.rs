use candle::{Device, DType, Result, Tensor};
use candle_transformers::models::gpt_oss::{eager_attn_with_sinks, eager_attn_windowed_with_sinks};
#[cfg(feature = "flash-attn")]
use candle_transformers::models::gpt_oss::{flash_attn_with_sinks, flash_attn_windowed_with_sinks};

// T12: FA (+LSE sinks) vs eager sinks parity on tiny shapes
#[cfg(feature = "flash-attn")]
#[test]
fn gpt_oss_fa_lse_sinks_vs_eager_parity() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        // Skip if CUDA is not available
        return Ok(());
    }
    device.set_seed(42)?;

    // Tiny shapes
    let b = 1usize;
    let h = 2usize;
    let qlen = 3usize;
    let klen = 4usize;
    let d = 64usize; // FA head-dim multiple of 8
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };
    let scale = 1f32 / (d as f32).sqrt();

    // Build q/k/v in FA layout: (b, seqlen, heads, dim)
    let q_bqhd = Tensor::arange(0u32, (b * h * qlen * d) as u32, &device)?
        .to_dtype(dtype)?
        .reshape((b, qlen, h, d))?;
    let k_bkhd = Tensor::arange(0u32, (b * h * klen * d) as u32, &device)?
        .to_dtype(dtype)?
        .reshape((b, klen, h, d))?;
    let v_bkhd = (&k_bkhd.to_dtype(dtype)? / 50.)?;
    let q = (&q_bqhd / 30.)?;
    let k = (&k_bkhd / 40.)?;
    let v = v_bkhd.clone();

    // Per-head sinks (h)
    let sinks = Tensor::from_vec(vec![0.1f32, 0.2f32], h, &device)?;

    let o_fa = flash_attn_with_sinks(&q, &k, &v, scale, false, Some(&sinks))?;
    let o_eager = eager_attn_with_sinks(&q, &k, &v, scale, false, Some(&sinks))?;

    let diff = (o_fa.to_dtype(DType::F32)? - o_eager.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-2, "max diff = {diff}");
    Ok(())
}

// T13: Windowed FA (+LSE sinks) vs eager windowed parity with left window and right=0
#[cfg(feature = "flash-attn")]
#[test]
fn gpt_oss_fa_windowed_lse_sinks_vs_eager_parity() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        // Skip if CUDA is not available
        return Ok(());
    }
    device.set_seed(123)?;

    // Small shapes
    let b = 1usize;
    let h = 2usize;
    let qlen = 5usize;
    let klen = 5usize;
    let d = 64usize;
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };
    let scale = 1f32 / (d as f32).sqrt();

    let q_bqhd = Tensor::arange(0u32, (b * h * qlen * d) as u32, &device)?
        .to_dtype(dtype)?
        .reshape((b, qlen, h, d))?;
    let k_bkhd = Tensor::arange(0u32, (b * h * klen * d) as u32, &device)?
        .to_dtype(dtype)?
        .reshape((b, klen, h, d))?;
    let v_bkhd = (&k_bkhd.to_dtype(dtype)? / 80.)?;
    let q = (&q_bqhd / 20.)?;
    let k = (&k_bkhd / 25.)?;
    let v = v_bkhd.clone();

    let sinks = Tensor::from_vec(vec![0.05f32, 0.15f32], h, &device)?;
    let left = Some(1usize);
    let right = Some(0usize);

    let o_fa = flash_attn_windowed_with_sinks(&q, &k, &v, scale, left, right, Some(&sinks))?;
    let o_eager = eager_attn_windowed_with_sinks(&q, &k, &v, scale, left, right, Some(&sinks))?;

    let diff = (o_fa.to_dtype(DType::F32)? - o_eager.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_vec0::<f32>()?;
    assert!(diff < 1e-2, "max diff = {diff}");
    Ok(())
}
