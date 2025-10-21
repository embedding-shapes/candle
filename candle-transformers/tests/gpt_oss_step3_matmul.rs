//! Step 3: Test matrix multiplication and MLP forward with simple input [1, 0, 0, ...]
//!
//! Run with: cargo test --release --features cuda --test gpt_oss_step3_matmul -- --nocapture

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss;

#[test]
fn test_expert3_matmul_simple_input() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let model_path = "/home/user/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

    // Load VarBuilder
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[
                format!("{}/model-00000-of-00002.safetensors", model_path),
                format!("{}/model-00001-of-00002.safetensors", model_path),
            ],
            DType::BF16,
            &dev,
        )?
    };

    // Load Expert 3 gate_up_proj weights
    let layer0_vb = vb.pp("model.layers.0.mlp");
    let gate_up = gpt_oss::load_expert_linear_mxfp4_grouped(
        2880, // in_dim
        5760, // out_dim
        true, // has bias
        layer0_vb,
        "experts.gate_up_proj",
        3,  // expert_idx = 3
        32, // n_experts
    )?;

    // Create simple input: [1, 0, 0, ..., 0] (2880 elements)
    let mut input_vec = vec![0.0f32; 2880];
    input_vec[0] = 1.0;
    let input = Tensor::from_vec(input_vec, (1, 2880), &dev)?.to_dtype(DType::BF16)?;

    println!("\n=== Step 3: Matrix Multiplication Test ===");
    println!("Input shape: {:?}", input.dims());
    let input_f32 = input.to_dtype(DType::F32)?;
    let input_vec = input_f32.to_vec2::<f32>()?;
    println!("Input first 8: {:?}", &input_vec[0][..8]);

    // Forward through Linear (matmul + bias)
    let output = gate_up.forward(&input)?;

    println!("\nOutput shape: {:?}", output.dims());
    let output_f32 = output.to_dtype(DType::F32)?;
    let output_vec = output_f32.to_vec2::<f32>()?;
    println!("Output first 16: {:?}", &output_vec[0][..16]);

    // Print even and odd indices separately
    let even: Vec<f32> = (0..8).map(|i| output_vec[0][i * 2]).collect();
    let odd: Vec<f32> = (0..8).map(|i| output_vec[0][i * 2 + 1]).collect();
    println!("Even indices [0,2,4,6,8,10,12,14]: {:?}", even);
    println!("Odd indices [1,3,5,7,9,11,13,15]: {:?}", odd);

    // Python reference for input [1,0,0,...] through full MLP:
    // [0.054931640625, -0.0208740234375, -0.0849609375, -0.02783203125, ...]
    // But that's after full MLP (router + all experts + GLU + down_proj)
    // We're just testing gate_up_proj here, so values will be different

    println!("\n✓ Test completed - manual inspection needed for correctness");
    Ok(())
}
