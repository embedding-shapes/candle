#![cfg(feature = "cuda")]

use candle_core::{
    mxfp4::{dequant_mxfp4_to_bf16, matmul_mxfp4_bf16},
    Device, Result, Tensor,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const BLOCK_ELEMS: usize = 32;

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

fn quantize_block_e2m1(values: &[f32]) -> (u8, [u8; 16]) {
    assert_eq!(values.len(), BLOCK_ELEMS);
    if values.iter().all(|&v| v == 0.0) {
        return (0u8, [0u8; 16]);
    }
    let mut best_exp: i8 = 0;
    let mut best_err = f64::INFINITY;
    let mut best_packed = [0u8; 16];
    let max_abs = values.iter().map(|v| v.abs()).fold(0f32, |a, b| a.max(b));
    let mut center = if max_abs > 0.0 {
        (max_abs / 6.0).log2().floor() as i32
    } else {
        0
    };
    center = center.clamp(-32, 31);
    let candidates: Vec<i8> = ((center - 8)..=(center + 8))
        .map(|e| e.clamp(-32, 31) as i8)
        .collect();
    for exp in candidates {
        let scale = (2f32).powi(exp as i32);
        let mut packed = [0u8; 16];
        let mut sse = 0f64;
        for j in 0..16 {
            let i0 = 2 * j;
            let i1 = i0 + 1;
            let q0 = values[i0] / scale;
            let q1 = values[i1] / scale;
            let c0 = encode_fp4_e2m1(q0);
            let c1 = encode_fp4_e2m1(q1);
            let approx0 = decode_fp4_e2m1(c0) * scale;
            let approx1 = decode_fp4_e2m1(c1) * scale;
            sse += (values[i0] as f64 - approx0 as f64).powi(2);
            sse += (values[i1] as f64 - approx1 as f64).powi(2);
            packed[j] = (c1 << 4) | (c0 & 0x0f);
        }
        if sse < best_err {
            best_err = sse;
            best_exp = exp;
            best_packed = packed;
        }
    }
    ((best_exp as i32 + 127) as u8, best_packed)
}

fn decode_fp4_e2m1(n: u8) -> f32 {
    let s = (n >> 3) & 0x1;
    let e = (n >> 1) & 0x3;
    let m = n & 0x1;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        sign * (m as f32) * 0.5
    } else {
        let frac = 1.0 + (m as f32) * 0.5;
        let exp = (e as i32) - 1;
        sign * (2f32).powi(exp) * frac
    }
}

fn quantize_matrix(rows: usize, cols: usize, data: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(data.len(), rows * cols);
    let nblocks = cols / BLOCK_ELEMS;
    let mut blocks = vec![0u8; rows * nblocks * 16];
    let mut scales = vec![0u8; rows * nblocks];
    for r in 0..rows {
        for b in 0..nblocks {
            let start = r * cols + b * BLOCK_ELEMS;
            let (scale, packed) = quantize_block_e2m1(&data[start..start + BLOCK_ELEMS]);
            scales[r * nblocks + b] = scale;
            let dst = (r * nblocks + b) * 16;
            blocks[dst..dst + 16].copy_from_slice(&packed);
        }
    }
    (blocks, scales)
}

#[test]
#[ignore = "perf measurement, run manually"]
fn t_mxfp4_matmul_latency() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let rows = 1usize;
    let in_dim = 8192usize;
    let out_dim = 8192usize;
    let nblocks = in_dim / BLOCK_ELEMS;

    let mut rng = StdRng::seed_from_u64(2025);
    let mut act_vals = vec![0f32; rows * in_dim];
    for v in &mut act_vals {
        *v = rng.random_range(-3.0..3.0);
    }

    let mut weight_vals = vec![0f32; out_dim * in_dim];
    for v in &mut weight_vals {
        *v = rng.random_range(-2.5..2.5);
    }

    let (blocks_data, scales_data) = quantize_matrix(out_dim, in_dim, &weight_vals);

    let act = Tensor::from_vec(act_vals, (rows, in_dim), &Device::Cpu)?
        .to_dtype(candle_core::DType::BF16)?
        .to_device(&dev)?;

    let blocks = Tensor::from_vec(blocks_data, (out_dim, nblocks, 16), &dev)?;
    let scales = Tensor::from_vec(scales_data, (out_dim, nblocks), &dev)?;

    let weight_dequant = dequant_mxfp4_to_bf16(&blocks, &scales, [out_dim, in_dim])?;
    let reference = act.matmul(&weight_dequant.t()?)?;

    // Warm-up
    let _ = matmul_mxfp4_bf16(&act, &blocks, &scales)?;
    dev.synchronize()?;

    let iters = 20usize;
    let mut total = std::time::Duration::ZERO;
    for _ in 0..iters {
        let start = Instant::now();
        let fused = matmul_mxfp4_bf16(&act, &blocks, &scales)?;
        dev.synchronize()?;
        total += start.elapsed();
        drop(fused);
    }

    let fused = matmul_mxfp4_bf16(&act, &blocks, &scales)?;
    dev.synchronize()?;
    let fused_cpu = fused
        .to_dtype(candle_core::DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    let reference_cpu = reference
        .to_dtype(candle_core::DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;

    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    for (f_row, r_row) in fused_cpu.iter().zip(reference_cpu.iter()) {
        for (&f, &r) in f_row.iter().zip(r_row.iter()) {
            let diff = f - r;
            max_abs = max_abs.max(diff.abs());
            sum_sq += (diff as f64) * (diff as f64);
        }
    }
    let l2 = sum_sq.sqrt();

    println!(
        "iters={}, rows={}, in_dim={}, out_dim={}\nmean_us={:.1}\nmax_abs={:.3e} l2={:.3e}",
        iters,
        rows,
        in_dim,
        out_dim,
        total.as_secs_f64() * 1e6 / iters as f64,
        max_abs,
        l2
    );

    assert!(
        max_abs <= 0.1,
        "excessive diff between fused and dense matmul: max_abs={max_abs}"
    );
    Ok(())
}
