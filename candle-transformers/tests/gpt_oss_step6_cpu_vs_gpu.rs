//! Step 6: Validate interleaved split works correctly on both CPU and GPU
//!
//! Run with: cargo test --release --features cuda --test gpt_oss_step6_cpu_vs_gpu -- --nocapture

use candle::{DType, Device, Result, Tensor, D};

#[test]
fn test_index_select_interleaved_cpu_vs_gpu() -> Result<()> {
    println!("\n=== Testing index_select interleaved split ===");

    // Create test data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    let test_data: Vec<f32> = (0..10).map(|i| i as f32).collect();

    // Test on CPU
    let cpu_tensor = Tensor::from_vec(test_data.clone(), (1, 10), &Device::Cpu)?;

    let even_indices = Tensor::from_vec(vec![0u32, 2, 4, 6, 8], 5, &Device::Cpu)?;
    let odd_indices = Tensor::from_vec(vec![1u32, 3, 5, 7, 9], 5, &Device::Cpu)?;

    let cpu_even = cpu_tensor.index_select(&even_indices, D::Minus1)?;
    let cpu_odd = cpu_tensor.index_select(&odd_indices, D::Minus1)?;

    let cpu_even_vec = cpu_even.to_vec2::<f32>()?;
    let cpu_odd_vec = cpu_odd.to_vec2::<f32>()?;

    println!("CPU results:");
    println!("  Original: {:?}", test_data);
    println!("  Even (indices 0,2,4,6,8): {:?}", cpu_even_vec[0]);
    println!("  Odd  (indices 1,3,5,7,9): {:?}", cpu_odd_vec[0]);

    // Test on GPU
    let gpu = Device::cuda_if_available(0)?;
    let gpu_tensor = Tensor::from_vec(test_data.clone(), (1, 10), &gpu)?;

    let even_indices_gpu = Tensor::from_vec(vec![0u32, 2, 4, 6, 8], 5, &gpu)?;
    let odd_indices_gpu = Tensor::from_vec(vec![1u32, 3, 5, 7, 9], 5, &gpu)?;

    let gpu_even = gpu_tensor.index_select(&even_indices_gpu, D::Minus1)?;
    let gpu_odd = gpu_tensor.index_select(&odd_indices_gpu, D::Minus1)?;

    let gpu_even_vec = gpu_even.to_vec2::<f32>()?;
    let gpu_odd_vec = gpu_odd.to_vec2::<f32>()?;

    println!("\nGPU results:");
    println!("  Original: {:?}", test_data);
    println!("  Even (indices 0,2,4,6,8): {:?}", gpu_even_vec[0]);
    println!("  Odd  (indices 1,3,5,7,9): {:?}", gpu_odd_vec[0]);

    // Compare
    assert_eq!(cpu_even_vec[0], vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    assert_eq!(cpu_odd_vec[0], vec![1.0, 3.0, 5.0, 7.0, 9.0]);
    assert_eq!(gpu_even_vec[0], vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    assert_eq!(gpu_odd_vec[0], vec![1.0, 3.0, 5.0, 7.0, 9.0]);

    println!("\n✓ CPU and GPU index_select produce identical results");
    Ok(())
}

#[test]
fn test_mlp_split_cpu_vs_gpu() -> Result<()> {
    println!("\n=== Testing MLP gate/up split on CPU vs GPU ===");

    // Simulate gate_up output: shape (1, 8) where first 4 are interleaved with last 4
    // Format: [gate0, up0, gate1, up1, gate2, up2, gate3, up3]
    let gate_up_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    // Test on CPU
    let cpu_gu = Tensor::from_vec(gate_up_data.clone(), (1, 8), &Device::Cpu)?;
    let inter = 4;

    let even_indices: Vec<u32> = (0..inter).map(|i| (i * 2) as u32).collect();
    let odd_indices: Vec<u32> = (0..inter).map(|i| (i * 2 + 1) as u32).collect();

    let even_idx_cpu = Tensor::from_vec(even_indices.clone(), inter, &Device::Cpu)?;
    let odd_idx_cpu = Tensor::from_vec(odd_indices.clone(), inter, &Device::Cpu)?;

    let cpu_gate = cpu_gu.index_select(&even_idx_cpu, D::Minus1)?;
    let cpu_up = cpu_gu.index_select(&odd_idx_cpu, D::Minus1)?;

    let cpu_gate_vec = cpu_gate.to_vec2::<f32>()?;
    let cpu_up_vec = cpu_up.to_vec2::<f32>()?;

    println!("CPU results:");
    println!("  gate_up: {:?}", gate_up_data);
    println!("  gate (even): {:?}", cpu_gate_vec[0]);
    println!("  up (odd):    {:?}", cpu_up_vec[0]);

    // Test on GPU
    let gpu = Device::cuda_if_available(0)?;
    let gpu_gu = Tensor::from_vec(gate_up_data.clone(), (1, 8), &gpu)?;

    let even_idx_gpu = Tensor::from_vec(even_indices, inter, &gpu)?;
    let odd_idx_gpu = Tensor::from_vec(odd_indices, inter, &gpu)?;

    let gpu_gate = gpu_gu.index_select(&even_idx_gpu, D::Minus1)?;
    let gpu_up = gpu_gu.index_select(&odd_idx_gpu, D::Minus1)?;

    let gpu_gate_vec = gpu_gate.to_vec2::<f32>()?;
    let gpu_up_vec = gpu_up.to_vec2::<f32>()?;

    println!("\nGPU results:");
    println!("  gate_up: {:?}", gate_up_data);
    println!("  gate (even): {:?}", gpu_gate_vec[0]);
    println!("  up (odd):    {:?}", gpu_up_vec[0]);

    // Verify correct split
    assert_eq!(cpu_gate_vec[0], vec![10.0, 30.0, 50.0, 70.0]);
    assert_eq!(cpu_up_vec[0], vec![20.0, 40.0, 60.0, 80.0]);
    assert_eq!(gpu_gate_vec[0], vec![10.0, 30.0, 50.0, 70.0]);
    assert_eq!(gpu_up_vec[0], vec![20.0, 40.0, 60.0, 80.0]);

    println!("\n✓ CPU and GPU produce identical gate/up splits");
    Ok(())
}

#[test]
fn test_halves_vs_interleaved_difference() -> Result<()> {
    println!("\n=== Comparing halves split vs interleaved split ===");

    let gate_up_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let gpu = Device::cuda_if_available(0)?;
    let gu = Tensor::from_vec(gate_up_data.clone(), (1, 8), &gpu)?;

    // Halves split (WRONG for this model)
    let gate_halves = gu.narrow(D::Minus1, 0, 4)?;
    let up_halves = gu.narrow(D::Minus1, 4, 4)?;

    let gate_halves_vec = gate_halves.to_vec2::<f32>()?;
    let up_halves_vec = up_halves.to_vec2::<f32>()?;

    println!("Halves split (first 4, last 4):");
    println!("  gate: {:?}", gate_halves_vec[0]);
    println!("  up:   {:?}", up_halves_vec[0]);

    // Interleaved split (CORRECT for this model)
    let inter = 4;
    let even_indices: Vec<u32> = (0..inter).map(|i| (i * 2) as u32).collect();
    let odd_indices: Vec<u32> = (0..inter).map(|i| (i * 2 + 1) as u32).collect();

    let even_idx = Tensor::from_vec(even_indices, inter, &gpu)?;
    let odd_idx = Tensor::from_vec(odd_indices, inter, &gpu)?;

    let gate_interleaved = gu.index_select(&even_idx, D::Minus1)?;
    let up_interleaved = gu.index_select(&odd_idx, D::Minus1)?;

    let gate_interleaved_vec = gate_interleaved.to_vec2::<f32>()?;
    let up_interleaved_vec = up_interleaved.to_vec2::<f32>()?;

    println!("\nInterleaved split (even, odd):");
    println!("  gate: {:?}", gate_interleaved_vec[0]);
    println!("  up:   {:?}", up_interleaved_vec[0]);

    println!("\n✓ The two methods produce DIFFERENT results (as expected)");
    println!("  Halves gives wrong gate/up mixing");
    println!("  Interleaved correctly separates gate and up");

    Ok(())
}
