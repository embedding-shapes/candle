#![allow(clippy::approx_constant)]

// GPU micro-check for MXFP4 nibble order and downstream mapping.
// Runs on CUDA, small deterministic inputs, no full model load.

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::WrapErr;
#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;
use half::bf16;

#[cfg(feature = "cuda")]
fn hex_bytes(v: &[u8]) -> String {
    v.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ")
}

#[cfg(feature = "cuda")]
fn decode_fp4_e2m1(n: u8) -> f32 {
    // s e e m (E2M1), bias=1
    let s = (n >> 3) & 0x1;
    let e = (n >> 1) & 0x3;
    let m = n & 0x1;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        let frac = (m as f32) * 0.5;
        // 2^(1-bias) == 2^0 == 1 for bias=1
        sign * frac
    } else {
        let frac = 1.0 + (m as f32) * 0.5;
        let exp = (e as i32) - 1;
        sign * f32::from_bits(((127 + exp) as u32) << 23) * frac
    }
}

#[cfg(feature = "cuda")]
fn l2_linf(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut l2 = 0f32;
    let mut linf = 0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        l2 += d * d;
        if d > linf { linf = d; }
    }
    (l2.sqrt(), linf)
}

#[test]
#[cfg(feature = "cuda")]
fn t_mxfp4_unpack_and_map() -> Result<()> {
    // Device
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;

    // Input buffer: 16 bytes with adversarial patterns
    let bytes: [u8; 16] = [
        0xF1, 0x1F, 0xAB, 0xBA, 0x00, 0xFF, 0x0F, 0xF0,
        0x5A, 0xA5, 0x3C, 0xC3, 0x7E, 0xE7, 0x12, 0x21,
    ];
    let n = bytes.len() as i32;

    // Expected hi/lo on host
    let mut exp_hi = vec![0u8; bytes.len()];
    let mut exp_lo = vec![0u8; bytes.len()];
    for i in 0..bytes.len() {
        exp_hi[i] = bytes[i] >> 4;
        exp_lo[i] = bytes[i] & 0x0F;
    }

    // Upload to device
    let d_in = dev.memcpy_stod(&bytes)?;
    let mut d_hi = unsafe { dev.alloc::<u8>(bytes.len())? };
    let mut d_lo = unsafe { dev.alloc::<u8>(bytes.len())? };

    // Load kernel and launch
    let func = dev.get_or_load_func("mxfp4_unpack", &candle_kernels::QUANTIZED)?;
    let block = 64u32;
    let grid = ((bytes.len() as u32 + block - 1) / block).max(1);
    let cfg = cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    let mut builder = func.builder();
    builder.arg(&d_in);
    builder.arg(&mut d_hi);
    builder.arg(&mut d_lo);
    candle_core::builder_arg!(builder, n);
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;

    // Download results
    let act_hi: Vec<u8> = dev.memcpy_dtov(&d_hi)?;
    let act_lo: Vec<u8> = dev.memcpy_dtov(&d_lo)?;

    // Logs: dtypes/devices/shapes
    println!("Input dtype u8 device cuda:0 shape [16] strides [1]");
    println!("hi/lo dtype u8 device cuda:0 shape [16] strides [1]");
    println!("Input bytes: {}", hex_bytes(&bytes));
    println!("Exp HI:      {}", hex_bytes(&exp_hi));
    println!("Act HI:      {}", hex_bytes(&act_hi));
    println!("Exp LO:      {}", hex_bytes(&exp_lo));
    println!("Act LO:      {}", hex_bytes(&act_lo));

    // Assert exact equality
    assert_eq!(act_hi, exp_hi, "HI nibble mismatch");
    assert_eq!(act_lo, exp_lo, "LO nibble mismatch");

    // Downstream mapping: fuse decode using CUDA dequant kernel and compare vs spec
    // Construct minimal MXFP4 tensors: rows=1, nblocks=1, cols=32
    let blocks = Tensor::from_vec(bytes.to_vec(), (1, 1, 16), &device)?; // U8 [1,1,16]
    let scales = Tensor::from_vec(vec![127u8], (1, 1), &device)?;        // E8M0=127 => scale=1.0
    let out = candle_core::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [1, 32])?;
    let out_cpu = out.to_device(&Device::Cpu)?;
    let out_bf16: Vec<bf16> = out_cpu.to_vec2::<bf16>()?.into_iter().next().unwrap(); // [1,32] -> Vec<bf16>
    let act: Vec<f32> = out_bf16.iter().map(|b| f32::from(*b)).collect();

    // Expected 32 values (low nibble at even idx, high at odd), scale=1
    let mut exp = vec![0f32; 32];
    for j in 0..16usize {
        let byte = bytes[j];
        let lo = byte & 0x0F;
        let hi = byte >> 4;
        exp[2 * j] = decode_fp4_e2m1(lo);
        exp[2 * j + 1] = decode_fp4_e2m1(hi);
    }

    // Logs and metrics
    println!("Dequant out dtype bf16 device cuda:0->cpu shape [1,32]");
    println!("Exp first 8:   {:?}", &exp[..8]);
    println!("Act first 8:   {:?}", &act[..8]);
    let (l2, linf) = l2_linf(&exp, &act);
    println!("Metrics: L2={:.6}, L∞={:.6} (bf16 rounding expected)", l2, linf);

    // BF16 rounding may introduce <= ~0.0078125 (1 ulp) error around magnitude 1.
    let tol = 0.01f32;
    if !(linf <= tol) {
        bail!("dequant mismatch: L∞={} > {}", linf, tol);
    }

    Ok(())
}

#[test]
#[cfg(not(feature = "cuda"))]
fn t_mxfp4_unpack_and_map_skip_without_cuda() {
    eprintln!("skipped: build without feature=\"cuda\"");
}
