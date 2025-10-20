use candle_core::{mxfp4::dequant_mxfp4_to_bf16, Device, Result, Tensor};
use half::bf16;

// Spec reference: decode FP4 E2M1 nibble (s e e m, bias=1) to f32.
fn ref_decode_fp4_e2m1(code: u8) -> f32 {
    let s = (code >> 3) & 0x1;
    let e = (code >> 1) & 0x3;
    let m = code & 0x1;
    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        // Subnormal/zero: (-1)^s * 2^(1-bias) * (m * 2^-1), bias=1 => 1.0 * (m/2)
        sign * ((m as f32) * 0.5)
    } else {
        // Normal: (-1)^s * 2^(e-bias) * (1 + m/2)
        let exp = (e as i32) - 1; // bias=1
        let frac = 1.0 + (m as f32) * 0.5;
        sign * (2f32.powi(exp) * frac)
    }
}

#[test]
fn t_mxfp4_fp4_lut_parity_gpu() -> Result<()> {
    // Require CUDA; skip cleanly if not available.
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => {
            eprintln!("SKIP: no CUDA device available, LUT parity GPU test skipped");
            return Ok(());
        }
    };

    // Construct one block [rows=1, cols=32] from 16 bytes so that the low nibble cycles codes 0..15.
    // High nibble set to 0 for simplicity; scale=1.0 via E8M0 code 127.
    let mut block_bytes = [0u8; 16];
    for j in 0..16u8 { block_bytes[j as usize] = j; /* hi=0, lo=j */ }
    let blocks = Tensor::from_vec(block_bytes.to_vec(), (1, 1, 16), &dev)?; // U8 by default
    let scales = Tensor::from_vec(vec![127u8], (1, 1), &dev)?; // E8M0 -> 2^(127-127)=1

    // Launch fused dequant on GPU.
    let out_bf16 = dequant_mxfp4_to_bf16(&blocks, &scales, [1, 32])?; // BF16 on CUDA
    // Bring back to CPU and extract BF16 values and raw bits.
    let out_cpu = out_bf16.to_device(&Device::Cpu)?;
    let out_vals: Vec<bf16> = out_cpu.flatten_all()?.to_vec1::<bf16>()?; // length 32

    // Build expected LUT (codes 0..15) as BF16 using spec mapping, scale=1.
    let mut exp_vals: Vec<bf16> = Vec::with_capacity(16);
    let mut exp_bits: Vec<u16> = Vec::with_capacity(16);
    for code in 0u8..16u8 {
        let v = ref_decode_fp4_e2m1(code);
        let b = bf16::from_f32(v);
        exp_bits.push(b.to_bits());
        exp_vals.push(b);
    }

    // Extract the even positions (low nibbles) from GPU output -> these correspond to codes 0..15.
    let mut got_vals: Vec<bf16> = Vec::with_capacity(16);
    let mut got_bits: Vec<u16> = Vec::with_capacity(16);
    for i in 0..16 {
        let b = out_vals[2 * i];
        got_bits.push(b.to_bits());
        got_vals.push(b);
    }

    // Diagnostics: print inputs/outputs and metadata.
    println!("Device: CUDA, DType(out) = {:?}", out_bf16.dtype());
    println!("Blocks shape: [1,1,16], Scales shape: [1,1], Out shape: {:?}", out_bf16.dims());
    println!("Block bytes (lo=code, hi=0): {:?}", &block_bytes);
    // Show mapping table
    println!("LUT codes 0..15 -> BF16 bits (exp vs got, even indices):");
    for i in 0..16 {
        println!(
            "code {:2}: exp={:04x} got={:04x} exp_f={:?} got_f={:?}",
            i, exp_bits[i], got_bits[i], f32::from(exp_vals[i]), f32::from(got_vals[i])
        );
    }

    // Compare bitwise equality for all 16 entries.
    for i in 0..16 { assert_eq!(got_bits[i], exp_bits[i], "BF16 bits mismatch at code {}", i); }

    // Compute numeric metrics for completeness.
    let mut linf = 0f32;
    let mut l2 = 0f32;
    for i in 0..16 {
        let a = f32::from(got_vals[i]);
        let b = f32::from(exp_vals[i]);
        let d = (a - b).abs();
        linf = linf.max(d);
        l2 += d * d;
    }
    println!("Metrics: L∞ = {:.8}, L2 = {:.8}", linf, l2);

    Ok(())
}
