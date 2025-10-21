#[cfg(feature = "cuda")]
#[test]
fn test_mmq_launch_config() -> candle_core::Result<()> {
    use candle_core::{Device, Tensor, DType};
    use half::bf16;

    // Test with typical GPT-OSS dimensions
    let rows = 1;
    let out_dim = 2880;
    let in_dim = 2880;
    let nblocks = in_dim / 32; // 90 blocks

    eprintln!("Testing MMQ kernel launch with:");
    eprintln!("  rows={}, out_dim={}, in_dim={}, nblocks={}", rows, out_dim, in_dim, nblocks);

    // Create CUDA device
    let device = Device::new_cuda(0)?;

    // Create random activations [rows, in_dim] in BF16
    let act = Tensor::randn(0f32, 1f32, (rows, in_dim), &device)?.to_dtype(DType::BF16)?;

    // Create random MXFP4 weights [out_dim, nblocks, 16]
    let blocks = Tensor::randn(0f32, 1f32, (out_dim, nblocks, 16), &device)?.to_dtype(DType::U8)?;

    // Create random scales [out_dim, nblocks]
    let scales = Tensor::full(127u8, (out_dim, nblocks), &device)?;

    eprintln!("\nInput shapes:");
    eprintln!("  act: {:?}", act.shape());
    eprintln!("  blocks: {:?}", blocks.shape());
    eprintln!("  scales: {:?}", scales.shape());

    // MMQ configuration from mxfp4.rs
    const MMQ_X: usize = 64;
    const MMQ_Y: usize = 2;
    const NWARPS: usize = 4;

    let grid_x = (rows + MMQ_Y - 1) / MMQ_Y;
    let grid_y = (out_dim + MMQ_X - 1) / MMQ_X;

    eprintln!("\nLaunch configuration:");
    eprintln!("  MMQ_X={}, MMQ_Y={}", MMQ_X, MMQ_Y);
    eprintln!("  grid_dim=({}, {}, 1)", grid_x, grid_y);
    eprintln!("  block_dim=(32, {}, 1)", NWARPS);

    // Calculate shared memory based on kernel code
    const MMQ_TILE_NE_K: usize = 32;
    const MMQ_ITER_K: usize = 256;
    const BLOCKS_PER_ITER: usize = MMQ_ITER_K / 32;

    let weight_qs_shared_size = MMQ_X * (2 * MMQ_TILE_NE_K + 1) * std::mem::size_of::<i32>();
    let weight_scales_shared_size = MMQ_X * BLOCKS_PER_ITER * std::mem::size_of::<f32>();
    let total_shared_mem = weight_qs_shared_size + weight_scales_shared_size;

    eprintln!("\nShared memory calculation:");
    eprintln!("  weight_qs_shared: {} * {} * 4 = {} bytes", MMQ_X, 2*MMQ_TILE_NE_K+1, weight_qs_shared_size);
    eprintln!("  weight_scales_shared: {} * {} * 4 = {} bytes", MMQ_X, BLOCKS_PER_ITER, weight_scales_shared_size);
    eprintln!("  Total: {} bytes ({:.2} KB)", total_shared_mem, total_shared_mem as f32 / 1024.0);

    // Try to run the matmul with MMQ
    std::env::set_var("CANDLE_MXFP4_USE_MMQ", "1");

    eprintln!("\nAttempting matmul with MMQ kernel...");
    let result = candle_core::mxfp4::matmul_mxfp4_bf16(&act, &blocks, &scales)?;

    eprintln!("✓ Kernel launched successfully!");
    eprintln!("  Output shape: {:?}", result.shape());
    eprintln!("  Output dtype: {:?}", result.dtype());

    // Try to read a few values to ensure it actually ran
    let vals = result.to_vec2::<bf16>()?;
    eprintln!("  First 5 output values: {:?}", &vals[0][..5.min(vals[0].len())]);

    Ok(())
}
