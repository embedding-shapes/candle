// Test MXFP4 dequantization with 2 rows, 2 blocks
use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    // Data from Python: expert 3, first 2 rows, first 2 blocks
    let blocks_data: Vec<u8> = vec![
        // Row 0, Block 0
        66, 197, 60, 198, 140, 69, 17, 237, 137, 198, 9, 1, 149, 4, 176, 37,
        // Row 0, Block 1
        10, 9, 147, 184, 170, 147, 172, 46, 177, 136, 153, 24, 9, 201, 41, 144,
        // Row 1, Block 0
        52, 161, 150, 76, 251, 236, 43, 36, 82, 166, 210, 210, 210, 164, 156, 129,
        // Row 1, Block 1
        34, 205, 2, 166, 204, 10, 34, 244, 170, 2, 213, 46, 192, 164, 38, 74,
    ];
    let scales_data: Vec<u8> = vec![121, 122, 121, 121];

    println!("Testing 2x2 on CUDA...");
    let cuda_device = Device::new_cuda(0)?;

    // Create tensors: blocks [2, 2, 16], scales [2, 2]
    let blocks_cpu = Tensor::from_vec(blocks_data, (2, 2, 16), &Device::Cpu)?;
    let scales_cpu = Tensor::from_vec(scales_data, (2, 2), &Device::Cpu)?;

    let blocks = blocks_cpu.to_device(&cuda_device)?;
    let scales = scales_cpu.to_device(&cuda_device)?;

    println!(
        "  blocks shape: {:?}, device: {:?}",
        blocks.dims(),
        blocks.device()
    );
    println!(
        "  scales shape: {:?}, device: {:?}",
        scales.dims(),
        scales.device()
    );

    // Dequantize to shape [2, 64] (2 rows, 2 blocks * 32 elements = 64 columns)
    let dequant = candle_core::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [2, 64])?;

    println!("  dequant shape: {:?}", dequant.dims());

    // Get the values
    let values = dequant.to_vec2::<half::bf16>()?;

    println!("\nRow 0:");
    let row0: Vec<f32> = values[0].iter().map(|x| x.to_f32()).collect();
    println!("  First 8:  {:?}", &row0[0..8]);
    println!("  Second 8 (from block 1): {:?}", &row0[32..40]);

    println!("\nRow 1:");
    let row1: Vec<f32> = values[1].iter().map(|x| x.to_f32()).collect();
    println!("  First 8:  {:?}", &row1[0..8]);

    // Expected for row 0, first block (we already verified this)
    let expected_r0_b0: Vec<f32> = vec![
        0.015625, 0.03125, 0.046875, -0.03125, -0.03125, 0.0234375, 0.0625, -0.03125,
    ];

    println!("\nComparison Row 0, Block 0 (first 8):");
    for i in 0..8 {
        let diff = (row0[i] - expected_r0_b0[i]).abs();
        let status = if diff < 0.0001 { "✓" } else { "✗" };
        println!(
            "[{}] CUDA: {:.8}, Expected: {:.8}, diff: {:.8} {}",
            i, row0[i], expected_r0_b0[i], diff, status
        );
    }

    Ok(())
}
