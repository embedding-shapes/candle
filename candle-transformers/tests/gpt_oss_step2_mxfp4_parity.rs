//! Step 2: Verify Rust MXFP4 dequantization produces same values as Python
//!
//! Run with: cargo test --release --features cuda --test gpt_oss_step2_mxfp4_parity

use candle::{DType, Device, Result, Tensor};

#[test]
fn test_mxfp4_dequant_expert3_first_row() -> Result<()> {
    // Raw MXFP4 data from Python for Expert 3, out_dim=0, block=0
    // From /tmp/test_step1_python_weights.py output
    let block_bytes: [u8; 16] = [
        66, 197, 60, 198, 140, 69, 17, 237, 137, 198, 9, 1, 149, 4, 176, 37,
    ];
    let scale_byte: u8 = 121;

    // Expected dequantized values from Python (first 32 values = first row, first block)
    // From /tmp/test_step1_decode_fp4.py output
    let expected: [f32; 32] = [
        0.015625, 0.031250, 0.046875, -0.031250, -0.031250, 0.023438, 0.062500, -0.031250,
        -0.031250, -0.000000, 0.046875, 0.031250, 0.007812, 0.007812, -0.046875, -0.062500,
        -0.007812, -0.000000, 0.062500, -0.031250, -0.007812, 0.000000, 0.007812, 0.000000,
        0.046875, -0.007812, 0.031250, 0.000000, 0.000000, -0.023438, 0.046875, 0.015625,
    ];

    // Create tensors for blocks and scales
    // blocks shape: [1, 1, 16] (1 row, 1 block, 16 bytes)
    // scales shape: [1, 1] (1 row, 1 block)
    let dev = Device::cuda_if_available(0)?;

    let blocks_vec: Vec<u8> = block_bytes.to_vec();
    let blocks = Tensor::from_vec(blocks_vec, (1, 1, 16), &dev)?;

    let scales = Tensor::from_vec(vec![scale_byte], (1, 1), &dev)?;

    // Dequantize using internal implementation
    let dequantized = candle::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [1, 32])?;

    // Convert to f32 for comparison
    let result = dequantized.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let result_row = &result[0];

    println!("\n=== MXFP4 Dequantization Parity Test ===");
    println!("Expected (Python): {:?}", &expected[..8]);
    println!("Got (Rust):        {:?}", &result_row[..8]);

    // Compare with small epsilon for BF16 precision
    let epsilon = 1e-4;
    for i in 0..32 {
        let diff = (result_row[i] - expected[i]).abs();
        if diff > epsilon {
            panic!(
                "Mismatch at index {}: expected {}, got {}, diff={}",
                i, expected[i], result_row[i], diff
            );
        }
    }

    println!("✓ All 32 values match within epsilon={}", epsilon);
    Ok(())
}
