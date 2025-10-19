use candle as candle_core;
use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::gpt_oss::load_linear_maybe_mxfp4;
use std::collections::HashMap;

fn make_blocks_scales(out_dim: usize, in_dim: usize) -> (Tensor, Tensor) {
    assert!(in_dim % 32 == 0);
    let nblocks = in_dim / 32;
    let dev = Device::Cpu;

    // Deterministic, simple patterns to keep test stable.
    let mut blocks_data = vec![0u8; out_dim * nblocks * 16];
    for (i, b) in blocks_data.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17).wrapping_add(23);
    }
    let mut scales_data = vec![0u8; out_dim * nblocks];
    for (i, s) in scales_data.iter_mut().enumerate() {
        // Keep modest exponents to avoid extreme magnitudes.
        *s = (i as u8) % 8; // 2^0 .. 2^7
    }
    let blocks = Tensor::from_vec(blocks_data, (out_dim, nblocks, 16), &dev).unwrap();
    let scales = Tensor::from_vec(scales_data, (out_dim, nblocks), &dev).unwrap();
    (blocks, scales)
}

#[test]
fn t7_name_resolution_linear_weight_shape_dtype() -> Result<()> {
    let dev = Device::Cpu;
    let (out_dim, in_dim) = (3, 64);
    let (blocks, scales) = make_blocks_scales(out_dim, in_dim);

    // Backend tensors with underscore naming to validate detection logic.
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("down_proj_blocks".to_string(), blocks);
    tensors.insert("down_proj_scales".to_string(), scales);

    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::BF16, &dev);
    let lin = load_linear_maybe_mxfp4(in_dim, out_dim, false, vb, "down_proj")?;

    let w = lin.weight();
    assert_eq!(w.dims(), &[out_dim, in_dim]);
    assert_eq!(w.dtype(), DType::BF16);
    Ok(())
}

#[test]
fn t8_mlp_one_layer_parity_vs_direct_dequant() -> Result<()> {
    let dev = Device::Cpu;
    let (out_dim, in_dim) = (5, 64);
    let (blocks, scales) = make_blocks_scales(out_dim, in_dim);

    // Expected BF16 weight from direct CPU dequantization.
    let w_ref = candle_core::mxfp4::dequant_mxfp4_to_bf16_cpu(&blocks, &scales, [out_dim, in_dim])?;

    // Provide tensors to VarBuilder and load via GPT-OSS loader.
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("down_proj_blocks".to_string(), blocks);
    tensors.insert("down_proj_scales".to_string(), scales);
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::BF16, &dev);
    let lin = load_linear_maybe_mxfp4(in_dim, out_dim, false, vb, "down_proj")?;
    let w_loaded = lin.weight().clone();

    // Random-ish input (deterministic pattern) in F32 (avoid CPU BF16 matmul unsupported).
    let mut x_data = vec![0f32; 2 * in_dim];
    for (i, v) in x_data.iter_mut().enumerate() {
        *v = ((i % 7) as f32) - 3.0;
    }
    let x = Tensor::from_vec(x_data, (2, in_dim), &dev)?;
    // Cast both weights to F32 for CPU matmul availability.
    let w_loaded_f = w_loaded.to_dtype(DType::F32)?;
    let w_ref_f = w_ref.to_dtype(DType::F32)?;
    // Compute outputs using both paths.
    let y_loaded_f = x.matmul(&w_loaded_f.t()?)?;
    let y_ref_f = x.matmul(&w_ref_f.t()?)?;
    let dl = y_loaded_f.sub(&y_ref_f)?;
    let max_abs_t = dl.abs()?.max_all()?;
    let max_abs = max_abs_t.to_scalar::<f32>()?;
    assert!(max_abs <= 1e-6, "max abs diff too large: {max_abs}");
    Ok(())
}
