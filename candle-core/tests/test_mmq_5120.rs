#[cfg(feature = "cuda")]
#[test]
fn test_mmq_5120() -> candle_core::Result<()> {
    use candle_core::{Device, Tensor, DType};

    let device = Device::new_cuda(0)?;
    let rows = 1;
    let out_dim = 5120;  // QKV projection dimension
    let in_dim = 2880;   // Hidden dimension
    let nblocks = in_dim / 32;

    eprintln!("Testing MMQ kernel with QKV dimensions:");
    eprintln!("  rows={}, out_dim={}, in_dim={}, nblocks={}", rows, out_dim, in_dim, nblocks);

    // Create random test data
    let act = Tensor::randn(0f32, 1.0, (rows, in_dim), &device)?.to_dtype(DType::BF16)?;
    let blocks = Tensor::randn(0f32, 1.0, (out_dim, nblocks, 16), &device)?.to_dtype(DType::U8)?;
    let scales = Tensor::full(127u8, (out_dim, nblocks), &device)?;

    eprintln!("Attempting matmul with MMQ kernel...");
    std::env::set_var("CANDLE_MXFP4_USE_MMQ", "1");
    let result = candle_core::mxfp4::matmul_mxfp4_bf16(&act, &blocks, &scales)?;
    eprintln!("✓ Kernel succeeded!");
    eprintln!("  Output shape: {:?}", result.shape());
    
    let output = result.to_vec2::<half::bf16>()?;
    eprintln!("  First 5 output values: {:?}", &output[0][..5]);
    
    Ok(())
}
