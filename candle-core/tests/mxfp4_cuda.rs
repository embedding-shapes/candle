#![cfg(feature = "cuda")]

use candle_core::{mxfp4::{dequant_mxfp4_to_bf16, dequant_mxfp4_to_bf16_cpu}, Device, Result, Tensor};
use half::bf16;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

const K_BLOCK: usize = 32;

fn encode_fp4_e2m1(x: f32) -> u8 {
    let sign = if x < 0.0 { 1u8 } else { 0u8 };
    let ax = x.abs();
    const POS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut idx = 0usize;
    let mut min_d = f32::INFINITY;
    for (i, &p) in POS.iter().enumerate() {
        let d = (ax - p).abs();
        if d < min_d {
            min_d = d;
            idx = i;
        }
    }
    let (e_bits, m_bit): (u8, u8) = match idx {
        0 => (0, 0),
        1 => (0, 1),
        2 => (1, 0),
        3 => (1, 1),
        4 => (2, 0),
        5 => (2, 1),
        6 => (3, 0),
        _ => (3, 1),
    };
    (sign << 3) | (e_bits << 1) | m_bit
}

fn decode_fp4_e2m1(n: u8) -> f32 {
    let s = (n >> 3) & 0x1;
    let e = (n >> 1) & 0x3;
    let m = n & 0x1;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 { sign * (m as f32) * 0.5 } else { let frac = 1.0 + (m as f32) * 0.5; let exp = (e as i32) - 1; sign * (2f32).powi(exp) * frac }
}

fn quantize_block_e2m1(values: &[f32]) -> (u8, [u8; 16]) {
    assert_eq!(values.len(), K_BLOCK);
    if values.iter().all(|&v| v == 0.0) { return (0u8, [0u8; 16]); }
    let mut best_exp: i8 = 0;
    let mut best_err = f64::INFINITY;
    let mut best_packed = [0u8; 16];
    let max_abs = values.iter().map(|v| v.abs()).fold(0f32, |a,b| a.max(b));
    let mut center = if max_abs > 0.0 { (max_abs / 6.0).log2().floor() as i32 } else { 0 };
    center = center.clamp(-32, 31);
    let candidates: Vec<i8> = ((center - 8)..=(center + 8)).map(|e| e.clamp(-32, 31) as i8).collect();
    for exp in candidates {
        let scale = (2f32).powi(exp as i32);
        let mut packed = [0u8; 16];
        let mut sse = 0f64;
        for j in 0..16 {
            let i0 = 2*j;
            let i1 = i0+1;
            let q0 = values[i0] / scale;
            let q1 = values[i1] / scale;
            let c0 = encode_fp4_e2m1(q0);
            let c1 = encode_fp4_e2m1(q1);
            let v0 = decode_fp4_e2m1(c0) * scale;
            let v1 = decode_fp4_e2m1(c1) * scale;
            sse += (values[i0] as f64 - v0 as f64).powi(2);
            sse += (values[i1] as f64 - v1 as f64).powi(2);
            packed[j] = (c1 << 4) | (c0 & 0x0f);
        }
        if sse < best_err { best_err = sse; best_exp = exp; best_packed = packed; }
    }
    (best_exp as u8, best_packed)
}

#[test]
fn t5_cpu_gpu_parity_small() -> Result<()> {
    let rows = 8usize;
    let cols = 256usize;
    let nblocks = cols / K_BLOCK;
    // Synthetic matrix with stable seed
    let mut rng = StdRng::seed_from_u64(42);
    let mut w = vec![0f32; rows * cols];
    for v in &mut w { *v = rng.random::<f32>() * 4.0 - 2.0; }

    // Quantize per 32-col block
    let mut blocks = vec![0u8; rows * nblocks * 16];
    let mut scales = vec![0u8; rows * nblocks];
    for r in 0..rows {
        for b in 0..nblocks {
            let base = r * cols + b * K_BLOCK;
            let (se, packed) = quantize_block_e2m1(&w[base..base+K_BLOCK]);
            scales[r * nblocks + b] = se;
            let off = (r * nblocks + b) * 16;
            blocks[off..off+16].copy_from_slice(&packed);
        }
    }

    // CPU reference
    let blocks_t = Tensor::from_vec(blocks.clone(), (rows, nblocks, 16), &Device::Cpu)?;
    let scales_t = Tensor::from_vec(scales.clone(), (rows, nblocks), &Device::Cpu)?;
    let cpu_out = dequant_mxfp4_to_bf16_cpu(&blocks_t, &scales_t, [rows, cols])?;

    // GPU dequant and compare
    let dev = Device::new_cuda(0)?;
    let b_gpu = blocks_t.to_device(&dev)?;
    let s_gpu = scales_t.to_device(&dev)?;
    let gpu_out = dequant_mxfp4_to_bf16(&b_gpu, &s_gpu, [rows, cols])?;
    let gpu_cpu = gpu_out.to_device(&Device::Cpu)?;

    // Compare as BF16 bitwise equality
    let a = cpu_out.flatten_all()?.to_vec1::<bf16>()?;
    let b = gpu_cpu.flatten_all()?.to_vec1::<bf16>()?;
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert_eq!(a[i], b[i], "bf16 mismatch at {i}");
    }
    Ok(())
}

#[test]
fn t6_throughput_sanity_short() -> Result<()> {
    let rows = 64usize;
    let cols = 2048usize; // small to keep test fast
    let nblocks = cols / K_BLOCK;
    let mut rng = StdRng::seed_from_u64(7);
    let mut w = vec![0f32; rows * cols];
    for v in &mut w { *v = rng.random::<f32>() * 5.0 - 2.5; }
    let mut blocks = vec![0u8; rows * nblocks * 16];
    let mut scales = vec![0u8; rows * nblocks];
    for r in 0..rows {
        for b in 0..nblocks {
            let base = r * cols + b * K_BLOCK;
            let (se, packed) = quantize_block_e2m1(&w[base..base+K_BLOCK]);
            scales[r * nblocks + b] = se;
            let off = (r * nblocks + b) * 16;
            blocks[off..off+16].copy_from_slice(&packed);
        }
    }
    let blocks_t = Tensor::from_vec(blocks.clone(), (rows, nblocks, 16), &Device::Cpu)?;
    let scales_t = Tensor::from_vec(scales.clone(), (rows, nblocks), &Device::Cpu)?;

    // CPU timing
    let t0 = Instant::now();
    let _cpu_out = dequant_mxfp4_to_bf16_cpu(&blocks_t, &scales_t, [rows, cols])?;
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

    // GPU timing
    let dev = Device::new_cuda(0)?;
    let b_gpu = blocks_t.to_device(&dev)?;
    let s_gpu = scales_t.to_device(&dev)?;
    // Warmup
    let _ = dequant_mxfp4_to_bf16(&b_gpu, &s_gpu, [rows, cols])?;
    let t1 = Instant::now();
    let _gpu_out = dequant_mxfp4_to_bf16(&b_gpu, &s_gpu, [rows, cols])?;
    let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;

    // Print simple stats (not asserting ratios to avoid flakiness across HW)
    let bytes_in = (rows * nblocks * 16 + rows * nblocks) as f64; // blocks + scales
    let mb = bytes_in / (1024.0 * 1024.0);
    eprintln!("MXFP4 dequant CPU: {:.3} ms ({:.3} MB)", cpu_ms, mb);
    eprintln!("MXFP4 dequant GPU: {:.3} ms ({:.3} MB)", gpu_ms, mb);
    Ok(())
}
