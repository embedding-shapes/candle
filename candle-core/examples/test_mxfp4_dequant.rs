// Minimal test to verify MXFP4 dequantization matches Python reference
use candle_core::{DType, Device, Result, Tensor};

fn main() -> Result<()> {
    // Expected values from Python test_single_block_dequant.py:
    // Expert 3, Row 0, Block 0:
    // Block bytes: [66, 197, 60, 198, 140, 69, 17, 237, 137, 198, 9, 1, 149, 4, 176, 37]
    // Scale (E8M0): 121
    // Expected dequantized: [0.015625, 0.03125, 0.046875, -0.03125, -0.03125, 0.0234375, 0.0625, -0.03125, ...]

    let block_bytes: Vec<u8> = vec![
        66, 197, 60, 198, 140, 69, 17, 237, 137, 198, 9, 1, 149, 4, 176, 37,
    ];
    let scale_byte: u8 = 121;

    // Create tensors: blocks [1, 1, 16], scales [1, 1]
    let blocks = Tensor::from_vec(block_bytes, (1, 1, 16), &Device::Cpu)?;
    let scales = Tensor::from_vec(vec![scale_byte], (1, 1), &Device::Cpu)?;

    // Dequantize to shape [1, 32]
    let dequant = candle_core::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [1, 32])?;

    // Get the values
    let values = dequant.to_vec2::<half::bf16>()?;
    let row0: Vec<f32> = values[0].iter().map(|x| x.to_f32()).collect();

    // Expected from Python
    let expected: Vec<f32> = vec![
        0.015625, 0.03125, 0.046875, -0.03125, -0.03125, 0.0234375, 0.0625, -0.03125, -0.03125,
        -0.0, 0.046875, 0.03125, 0.0078125, 0.0078125, -0.046875, -0.0625, -0.0078125, -0.0,
        0.0625, -0.03125, -0.0078125, 0.0, 0.0078125, 0.0, 0.046875, -0.0078125, 0.03125, 0.0, 0.0,
        -0.0234375, 0.046875, 0.015625,
    ];

    println!("Rust dequantized values:");
    println!("{:?}", row0);
    println!("\nExpected (Python):");
    println!("{:?}", expected);

    println!("\nComparison (first 8):");
    for i in 0..8 {
        let diff = (row0[i] - expected[i]).abs();
        let status = if diff < 0.0001 { "✓" } else { "✗" };
        println!(
            "[{}] Rust: {:.8}, Python: {:.8}, diff: {:.8} {}",
            i, row0[i], expected[i], diff, status
        );
    }

    // Check all values
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..32 {
        let diff = (row0[i] - expected[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.0001 {
            mismatches += 1;
            println!(
                "MISMATCH [{}]: Rust={:.8}, Python={:.8}, diff={:.8}",
                i, row0[i], expected[i], diff
            );
        }
    }

    println!("\nSummary:");
    println!("  Max difference: {:.8}", max_diff);
    println!("  Mismatches: {}/32", mismatches);

    if mismatches == 0 {
        println!("\n✓ PASS: All values match!");
    } else {
        println!("\n✗ FAIL: {} mismatches found", mismatches);
    }

    Ok(())
}
