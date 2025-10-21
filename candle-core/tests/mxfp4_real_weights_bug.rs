#![cfg(feature = "cuda")]

use candle_core::safetensors::load;
use candle_core::{
    mxfp4::{dequant_mxfp4_to_bf16, dequant_mxfp4_to_bf16_cpu},
    Device, Result, Tensor,
};

/// Test MXFP4 dequantization with actual GPT-OSS-20B model weights.
/// This test demonstrates a bug where CUDA dequant produces incorrect values
/// while CPU dequant produces correct values for the same input.
#[test]
fn test_real_gpt_oss_weights_cpu_vs_cuda() -> Result<()> {
    let snapshot = "/home/user/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
    let model_file = format!("{}/model-00000-of-00002.safetensors", snapshot);

    // Load actual model weights
    let tensors = load(&model_file, &Device::Cpu)?;

    let blocks_all = tensors
        .get("model.layers.0.mlp.experts.gate_up_proj_blocks")
        .expect("blocks tensor not found");
    let scales_all = tensors
        .get("model.layers.0.mlp.experts.gate_up_proj_scales")
        .expect("scales tensor not found");

    // Extract expert 3 (the one we've been debugging)
    let blocks = blocks_all.narrow(0, 3, 1)?.squeeze(0)?; // [5760, 90, 16]
    let scales = scales_all.narrow(0, 3, 1)?.squeeze(0)?; // [5760, 90]

    println!("Blocks shape: {:?}", blocks.dims());
    println!("Scales shape: {:?}", scales.dims());

    // Get raw bytes for verification
    let blocks_vec = blocks.to_vec3::<u8>()?;
    let scales_vec = scales.to_vec2::<u8>()?;
    println!("First block bytes: {:?}", &blocks_vec[0][0]);
    println!("First scale: {}", scales_vec[0][0]);

    // Expected values (manually computed from the bytes)
    let expected_first_32 = vec![
        0.015625, 0.03125, 0.046875, -0.03125, -0.03125, 0.0234375, 0.0625, -0.03125, -0.03125,
        -0.0, 0.046875, 0.03125, 0.0078125, 0.0078125, -0.046875, -0.0625, -0.0078125, -0.0,
        0.0625, -0.03125, -0.0078125, 0.0, 0.0078125, 0.0, 0.046875, -0.0078125, 0.03125, 0.0, 0.0,
        -0.0234375, 0.046875, 0.015625,
    ];

    // CPU dequantization
    let weight_cpu = dequant_mxfp4_to_bf16_cpu(&blocks, &scales, [5760, 2880])?;
    let cpu_f32 = weight_cpu.to_dtype(candle_core::DType::F32)?;
    let cpu_vec = cpu_f32.to_vec2::<f32>()?;
    let cpu_row0 = &cpu_vec[0][..32];

    println!("\nCPU dequant row 0 [:32]:");
    println!("{:?}", cpu_row0);

    // CUDA dequantization
    let dev = Device::new_cuda(0)?;
    let blocks_gpu = blocks.to_device(&dev)?;
    let scales_gpu = scales.to_device(&dev)?;

    let weight_gpu = dequant_mxfp4_to_bf16(&blocks_gpu, &scales_gpu, [5760, 2880])?;
    let gpu_cpu = weight_gpu.to_device(&Device::Cpu)?;
    let gpu_f32 = gpu_cpu.to_dtype(candle_core::DType::F32)?;
    let gpu_vec = gpu_f32.to_vec2::<f32>()?;
    let gpu_row0 = &gpu_vec[0][..32];

    println!("\nGPU dequant row 0 [:32]:");
    println!("{:?}", gpu_row0);

    // Compare CPU against expected
    println!("\nCPU vs Expected:");
    let mut cpu_matches = true;
    for i in 0..32 {
        let diff = (cpu_row0[i] - expected_first_32[i]).abs();
        if diff > 1e-6 {
            println!(
                "  [{}] CPU: {:.6}, Expected: {:.6}, Diff: {:.6}",
                i, cpu_row0[i], expected_first_32[i], diff
            );
            cpu_matches = false;
        }
    }
    if cpu_matches {
        println!("  ✓ CPU matches expected values!");
    }

    // Compare GPU against expected
    println!("\nGPU vs Expected:");
    let mut gpu_matches = true;
    let mut mismatch_count = 0;
    for i in 0..32 {
        let diff = (gpu_row0[i] - expected_first_32[i]).abs();
        if diff > 1e-6 {
            if mismatch_count < 10 {
                // Only print first 10 mismatches
                println!(
                    "  [{}] GPU: {:.6}, Expected: {:.6}, Diff: {:.6}",
                    i, gpu_row0[i], expected_first_32[i], diff
                );
            }
            mismatch_count += 1;
            gpu_matches = false;
        }
    }
    if mismatch_count > 10 {
        println!("  ... and {} more mismatches", mismatch_count - 10);
    }
    if gpu_matches {
        println!("  ✓ GPU matches expected values!");
    } else {
        println!(
            "  ✗ GPU has {} mismatches out of 32 values!",
            mismatch_count
        );
    }

    // The test assertion
    assert!(cpu_matches, "CPU dequant should match expected values");
    assert!(
        gpu_matches,
        "GPU dequant should match expected values (BUG: this will fail)"
    );

    Ok(())
}
