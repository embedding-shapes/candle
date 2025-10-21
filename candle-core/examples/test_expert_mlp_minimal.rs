/// Minimal test: Load MXFP4 weights for a single expert and verify matmul
/// This avoids the 30-second full model load time
use candle_core::{DType, Device, Result, Tensor};
use std::path::PathBuf;

// Simple sigmoid without candle-nn
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let one_plus_exp = (exp_neg_x + 1.0)?;
    one_plus_exp.recip()
}

const HIDDEN_SIZE: usize = 2880;
const INTERMEDIATE_SIZE: usize = 5760;
const NUM_EXPERTS: usize = 32;
const MXFP4_BLOCK_SIZE: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 16;

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/user/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/model-00000-of-00002.safetensors");

    println!("=== Step 1: Load raw MXFP4 bytes for expert 3, layer 0, gate_up ===");

    // Load tensors directly from safetensors
    let tensors = candle_core::safetensors::load(&model_path, &Device::Cpu)?;

    let blocks_name = "model.layers.0.mlp.experts.gate_up_proj_blocks";
    let scales_name = "model.layers.0.mlp.experts.gate_up_proj_scales";

    let blocks_g = tensors.get(blocks_name).expect("blocks not found");
    let scales_g = tensors.get(scales_name).expect("scales not found");

    println!("Loaded blocks_g shape: {:?}", blocks_g.dims());
    println!("Loaded scales_g shape: {:?}", scales_g.dims());

    let expert_idx = 3;

    // Extract expert 3
    let blocks = blocks_g.narrow(0, expert_idx, 1)?.squeeze(0)?; // [out_dim, nblocks, 16]
    let scales = scales_g.narrow(0, expert_idx, 1)?.squeeze(0)?; // [out_dim, nblocks]

    println!("\nExpert 3 blocks shape: {:?}", blocks.dims());
    println!("Expert 3 scales shape: {:?}", scales.dims());

    // Get raw bytes for row 0, block 0
    let blocks_u8 = blocks.to_vec3::<u8>()?;
    let scales_u8 = scales.to_vec2::<u8>()?;

    println!("\n=== Step 2: Verify raw bytes match Python ===");
    println!("Row 0, Block 0 bytes: {:?}", &blocks_u8[0][0]);
    println!("Row 0, Block 0 scale: {}", scales_u8[0][0]);
    println!("Expected bytes:       [66, 197, 60, 198, 140, 69, 17, 237, 137, 198, 9, 1, 149, 4, 176, 37]");
    println!("Expected scale:       121");

    let bytes_match = blocks_u8[0][0] == [66, 197, 60, 198, 140, 69, 17, 237, 137, 198, 9, 1, 149, 4, 176, 37];
    let scale_match = scales_u8[0][0] == 121;

    if bytes_match && scale_match {
        println!("✓ Raw bytes MATCH Python");
    } else {
        println!("✗ Raw bytes MISMATCH!");
        return Ok(());
    }

    println!("\n=== Step 3: Dequantize to fp32 and compare ===");

    // Dequantize
    let weight = candle_core::mxfp4::dequant_mxfp4_to_bf16(
        &blocks,
        &scales,
        [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    )?;

    println!("Dequantized weight shape: {:?}, dtype: {:?}", weight.dims(), weight.dtype());

    // Convert to f32 for comparison
    let weight_f32 = weight.to_dtype(DType::F32)?;
    let weight_vec = weight_f32.to_vec2::<f32>()?;

    println!("\nRow 0, first 8 values:");
    println!("Rust:     {:?}", &weight_vec[0][0..8]);
    println!("Expected: [0.015625, 0.03125, 0.046875, -0.03125, -0.03125, 0.0234375, 0.0625, -0.03125]");

    let expected_row0: Vec<f32> = vec![
        0.015625, 0.03125, 0.046875, -0.03125, -0.03125, 0.0234375, 0.0625, -0.03125,
    ];

    let mut max_diff = 0.0f32;
    for i in 0..8 {
        let diff = (weight_vec[0][i] - expected_row0[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max diff in first 8: {:.8}", max_diff);

    if max_diff < 1e-6 {
        println!("✓ Dequantization MATCHES Python (fp32 tolerance)");
    } else {
        println!("✗ Dequantization DIVERGES from Python!");
        return Ok(());
    }

    println!("\n=== Step 4: Test matmul with probe inputs ===");

    // Probe 1: All ones
    let ones_input = Tensor::ones((1, HIDDEN_SIZE), DType::F32, &Device::Cpu)?;
    let ones_output = ones_input.matmul(&weight_f32.t()?)?;

    println!("\nProbe 1 (ones input):");
    println!("  Output shape: {:?}", ones_output.dims());
    let ones_out_vec = ones_output.to_vec2::<f32>()?;
    println!("  First 4 outputs: {:?}", &ones_out_vec[0][0..4]);

    // Probe 2: Simple test vector [1, 2, 3, 4, ..., 2880]
    let test_vec: Vec<f32> = (1..=HIDDEN_SIZE).map(|i| i as f32).collect();
    let test_input = Tensor::from_vec(test_vec, (1, HIDDEN_SIZE), &Device::Cpu)?;
    let test_output = test_input.matmul(&weight_f32.t()?)?;

    println!("\nProbe 2 (sequential input [1,2,3,...,2880]):");
    println!("  Output shape: {:?}", test_output.dims());
    let test_out_vec = test_output.to_vec2::<f32>()?;
    println!("  First 4 outputs: {:?}", &test_out_vec[0][0..4]);

    println!("\n=== Step 5: Verify split into gate/up ===");

    // The weight is [5760, 2880] = [out, in]
    // After matmul: [1, 2880] @ [2880, 5760] = [1, 5760]
    // We need to split into even (gate) and odd (up) indices

    let output = test_output.squeeze(0)?; // [5760]
    let out_vec = output.to_vec1::<f32>()?;

    let intermediate = INTERMEDIATE_SIZE / 2; // 2880
    let mut gate_vec = Vec::with_capacity(intermediate);
    let mut up_vec = Vec::with_capacity(intermediate);

    for i in 0..intermediate {
        gate_vec.push(out_vec[i * 2]);
        up_vec.push(out_vec[i * 2 + 1]);
    }

    println!("\nAfter split:");
    println!("  Gate first 4: {:?}", &gate_vec[0..4]);
    println!("  Up first 4:   {:?}", &up_vec[0..4]);

    println!("\n=== Step 6: Apply SwiGLU activation ===");

    let gate_t = Tensor::from_vec(gate_vec.clone(), (1, intermediate), &Device::Cpu)?;
    let up_t = Tensor::from_vec(up_vec.clone(), (1, intermediate), &Device::Cpu)?;

    let limit = 7.0f32;
    let alpha = 1.702f32;

    // Clamp
    let gate_clamped = gate_t.clamp(f32::NEG_INFINITY, limit)?;
    let up_clamped = up_t.clamp(-limit, limit)?;

    // SwiGLU: (up + 1) * (gate * sigmoid(alpha * gate))
    let gate_alpha = (gate_clamped * alpha)?;
    let sig = candle_nn::ops::sigmoid(&gate_alpha)?;
    let glu = (gate_clamped * sig)?;
    let up_plus = (up_clamped + 1.0)?;
    let swiglu_out = (up_plus * glu)?;

    println!("\nSwiGLU output shape: {:?}", swiglu_out.dims());
    let swiglu_vec = swiglu_out.to_vec2::<f32>()?;
    println!("  First 4 values: {:?}", &swiglu_vec[0][0..4]);

    println!("\n✓ All steps completed successfully!");
    println!("\nNext: Run this same computation in Python and compare all intermediate values.");

    Ok(())
}
