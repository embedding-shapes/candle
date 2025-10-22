#[cfg(feature = "cuda")]
#[test]
fn test_mmq_rows21() -> candle_core::Result<()> {
    use candle_core::{Device, Tensor, DType};

    let device = Device::new_cuda(0)?;
    let rows = 21;  // This triggers the error!
    let out_dim = 5760;
    let in_dim = 2880;
    let nblocks = in_dim / 32;

    eprintln!("Testing MMQ kernel with rows=21:");
    eprintln!("  rows={}, out_dim={}, in_dim={}, nblocks={}", rows, out_dim, in_dim, nblocks);
    eprintln!("  grid_x = (21 + 2 - 1) / 2 = {}", (rows + 2 - 1) / 2);

    // Create random test data
    let act = Tensor::randn(0f32, 1.0, (rows, in_dim), &device)?.to_dtype(DType::BF16)?;
    let blocks = Tensor::randn(0f32, 1.0, (out_dim, nblocks, 16), &device)?.to_dtype(DType::U8)?;
    let scales = Tensor::full(127u8, (out_dim, nblocks), &device)?;

    eprintln!("Attempting matmul with MMQ kernel...");
    std::env::set_var("CANDLE_MXFP4_USE_MMQ", "1");
    std::env::set_var("CANDLE_MXFP4_DEBUG_MMQ", "1");
    
    match candle_core::mxfp4::matmul_mxfp4_bf16(&act, &blocks, &scales) {
        Ok(result) => {
            eprintln!("✓ Kernel succeeded!");
            eprintln!("  Output shape: {:?}", result.shape());
            
            let output = result.to_vec2::<half::bf16>()?;
            eprintln!("  First 5 output values: {:?}", &output[0][..5]);
        }
        Err(e) => {
            eprintln!("✗ Kernel failed: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}
