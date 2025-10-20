#![cfg(feature = "cuda")]

// MXFP4 blockwise scale extraction + application micro-test
// - One row, one block (G=32)
// - FP4 codes = [1,2,3,...,15] repeated twice, then [1,2]
// - Scale exponent e=129 => scale=2^(129-127)=4.0
// - Compare CUDA dequant vs spec decode (then rounded to BF16 to match kernel output)

use candle_core::{Device, Result, Tensor};
use half::bf16;

const G: usize = 32; // group size

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
    if u == 0xFF { f32::NAN } else { (2f32).powi((u as i32) - 127) }
}

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
fn t_mxfp4_block_scale_parity() -> Result<()> {
    // GPU device
    let dev = Device::new_cuda(0)?;

    // Construct FP4 codes sequence length 32 with values 1..15 repeating, then 1,2
    let mut codes = [0u8; G];
    for i in 0..G { codes[i] = 1 + (i % 15) as u8; }

    // Pack two nibbles per byte: even index -> low nibble, odd -> high nibble
    let mut packed = [0u8; 16];
    for j in 0..16 {
        let c0 = codes[2*j] & 0x0f;      // low nibble
        let c1 = codes[2*j + 1] & 0x0f;  // high nibble
        packed[j] = c0 | (c1 << 4);
    }

    // One row, one block, cols=32
    let rows = 1usize;
    let cols = G;
    let nblocks = 1usize;
    let blocks = Tensor::from_vec(packed.to_vec(), (rows, nblocks, 16), &dev)?; // U8 on CUDA
    let scales = Tensor::from_vec(vec![129u8], (rows, nblocks), &dev)?;         // E8M0=129 => 4.0

    // Run CUDA dequant to BF16
    let out = candle_core::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [rows, cols])?;
    let out_cpu = out.to_device(&Device::Cpu)?;
    let out_bf16 = out_cpu.to_vec2::<bf16>()?.remove(0);
    let act: Vec<f32> = out_bf16.iter().map(|b| f32::from(*b)).collect();

    // Spec reference: real = LUT(code) * 2^e, then BF16 rounding to match kernel
    let scale = pow2_e8m0(129u8);
    let mut exp_real = vec![0f32; G];
    for j in 0..16 {
        let byte = packed[j];
        let lo = byte & 0x0f;
        let hi = byte >> 4;
        let c0 = 2*j;
        let c1 = c0 + 1;
        exp_real[c0] = decode_fp4_e2m1(lo) * scale;
        exp_real[c1] = decode_fp4_e2m1(hi) * scale;
    }
    let exp_bf16: Vec<f32> = exp_real.iter().map(|v| bf16::from_f32(*v).to_f32()).collect();

    // Logs: dtypes/devices/shapes/strides and sample values
    println!("blocks dtype u8 device cuda:0 shape [1,1,16] strides [16,16,1]");
    println!("scales dtype u8 device cuda:0 shape [1,1] strides [1,1]");
    println!("out dtype bf16 device cuda:0->cpu shape [1,32] strides [32,1]");
    println!("FP4 codes (first 16): {:?}", &codes[..16]);
    println!("Scale exponent: 129 (scale=4.0)");
    println!("Expected first 8:   {:?}", &exp_bf16[..8]);
    println!("Actual first 8:     {:?}", &act[..8]);

    // Stats
    let mean = |v: &[f32]| v.iter().copied().sum::<f32>() / v.len() as f32;
    let std = |v: &[f32], m: f32| (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
    let (emin, emax) = exp_bf16.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &x| (mn.min(x), mx.max(x)));
    let (amin, amax) = act.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &x| (mn.min(x), mx.max(x)));
    let em = mean(&exp_bf16); let es = std(&exp_bf16, em);
    let am = mean(&act); let asd = std(&act, am);
    println!("Expected stats: mean={:.6} std={:.6} min={:.6} max={:.6}", em, es, emin, emax);
    println!("Actual   stats: mean={:.6} std={:.6} min={:.6} max={:.6}", am, asd, amin, amax);

    // Compare
    let (l2, linf) = l2_linf(&exp_bf16, &act);
    println!("Comparison: L2={:.6} L∞={:.6}", l2, linf);

    // BF16 parity: exact equality expected here since both sides rounded to BF16
    let tol = 0.0f32;
    assert!(linf <= tol, "Max-abs diff {} exceeds tol {}", linf, tol);

    Ok(())
}
