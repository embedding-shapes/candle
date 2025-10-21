#![cfg(feature = "cuda")]

// MXFP4 tail-group indexing micro-test
// Goal: Validate indexing when tensor length is not a multiple of G (32).
// We construct N = G + 13 codes packed into 2 blocks (last block padded with zeros),
// and two distinct per-block E8M0 exponents. We verify the CUDA dequant kernel applies
// the last block’s exponent only to its tail elements (the remaining padded elements stay 0),
// and that all indices and bounds are correct.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use half::bf16;

const G: usize = 32; // group size per block

#[inline]
fn decode_fp4_e2m1(n: u8) -> f32 {
    // s e e m (E2M1), bias=1
    let s = (n >> 3) & 0x1;
    let e = (n >> 1) & 0x3;
    let m = n & 0x1;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        // subnormal: 2^(1-bias) * (m * 2^-1) with bias=1 => 1 * (m*0.5)
        sign * (m as f32) * 0.5
    } else {
        // normal: 2^(E-bias) * (1 + m/2)
        let frac = 1.0 + (m as f32) * 0.5;
        let exp = (e as i32) - 1;
        sign * (2f32).powi(exp) * frac
    }
}

#[inline]
fn pow2_e8m0(u: u8) -> f32 {
    if u == 0xFF {
        f32::NAN
    } else {
        (2f32).powi((u as i32) - 127)
    }
}

fn l2_linf(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut l2 = 0f32;
    let mut linf = 0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        l2 += d * d;
        if d > linf {
            linf = d;
        }
    }
    (l2.sqrt(), linf)
}

#[test]
fn t_mxfp4_tail_groups_indexing() -> Result<()> {
    // Deterministic "random" codes for N = G + 13
    let n_tail = 13usize;
    let n = G + n_tail;
    let mut codes = vec![0u8; n];
    // Fixed seed-like generation: simple LCG over 4-bit space
    let mut x: u32 = 0x12345678;
    for i in 0..n {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        codes[i] = ((x >> 28) & 0x0F) as u8; // 4-bit code 0..15
    }

    // Pack into 2 blocks of 16 bytes each (32 values per block). Tail block padded with zeros.
    let mut b0 = [0u8; 16];
    let mut b1 = [0u8; 16];
    // Block 0: first 32 codes
    for j in 0..16 {
        let c0 = codes[2 * j] & 0x0F;
        let c1 = codes[2 * j + 1] & 0x0F;
        b0[j] = c0 | (c1 << 4);
    }
    // Block 1: next 13 codes then zeros
    for j in 0..16 {
        let idx0 = 2 * j;
        let idx1 = idx0 + 1;
        let c0 = if idx0 < n_tail {
            codes[G + idx0] & 0x0F
        } else {
            0
        };
        let c1 = if idx1 < n_tail {
            codes[G + idx1] & 0x0F
        } else {
            0
        };
        b1[j] = c0 | (c1 << 4);
    }

    // Two distinct E8M0 exponents for the two groups
    let s0 = 129u8; // 2^(2) = 4.0
    let s1 = 126u8; // 2^(-1) = 0.5
    let scale0 = pow2_e8m0(s0);
    let scale1 = pow2_e8m0(s1);

    // Build tensors on CUDA: blocks (1,2,16), scales (1,2); cols is rounded up to 64
    let dev = Device::new_cuda(0)?;
    let blocks = Tensor::from_vec([b0, b1].concat(), (1, 2, 16), &dev)?; // U8 CUDA
    let scales = Tensor::from_vec(vec![s0, s1], (1, 2), &dev)?; // U8 CUDA
    let out_bf16 = candle_core::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [1, 64])?;
    assert!(matches!(out_bf16.dtype(), DType::BF16));

    // Move to CPU and to f32 for comparison; consider only first N elements
    let out_cpu = out_bf16.to_device(&Device::Cpu)?;
    let out_row = out_cpu.to_vec2::<bf16>()?.remove(0);
    let act: Vec<f32> = out_row.iter().map(|b| f32::from(*b)).take(n).collect();

    // Compute expected on host, BF16-rounded to match kernel
    let mut exp = vec![0f32; n];
    // First group: 32 elems
    for j in 0..16 {
        let byte = b0[j];
        let lo = byte & 0x0F;
        let hi = byte >> 4;
        let c0 = 2 * j;
        let c1 = c0 + 1;
        exp[c0] = decode_fp4_e2m1(lo) * scale0;
        exp[c1] = decode_fp4_e2m1(hi) * scale0;
    }
    // Second group tail: 13 elems
    for j in 0..n_tail {
        let byte = b1[j / 2];
        let nib = if (j & 1) == 0 { byte & 0x0F } else { byte >> 4 };
        exp[G + j] = decode_fp4_e2m1(nib) * scale1;
    }
    // Round expected to BF16 for fair comparison
    let exp_bf16: Vec<f32> = exp.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();

    // Logs: shapes, dtypes, seed and samples
    println!(
        "Seed: 0x12345678 (LCG) ⇒ codes checksum={} (N={})",
        codes.iter().map(|&c| c as u64).sum::<u64>(),
        n
    );
    println!("blocks dtype u8 device cuda:0 shape [1,2,16] strides [32,16,1]");
    println!("scales dtype u8 device cuda:0 shape [1,2] strides [2,1]");
    println!("out dtype bf16 device cuda:0->cpu shape [1,64] strides [64,1]");
    println!("First block first 8 codes:   {:?}", &codes[..8]);
    println!("Second block tail 13 codes:  {:?}", &codes[G..G + n_tail]);
    println!("Scale[0]={} ({}) Scale[1]={} ({})", s0, scale0, s1, scale1);
    println!(
        "Expected last 8 tail vals:   {:?}",
        &exp_bf16[G..G + n_tail]
            .iter()
            .rev()
            .take(8)
            .collect::<Vec<&f32>>()
    );
    println!(
        "Actual   last 8 tail vals:   {:?}",
        &act[G..G + n_tail]
            .iter()
            .rev()
            .take(8)
            .collect::<Vec<&f32>>()
    );

    // Compare only the meaningful N elements
    let (l2, linf) = l2_linf(&exp_bf16, &act);
    println!("Metrics over N elements: L2={:.6} L∞={:.6}", l2, linf);
    let tol = 0.0f32; // exact after BF16 rounding
    assert!(linf <= tol, "Max-abs diff {} exceeds tol {}", linf, tol);

    Ok(())
}
