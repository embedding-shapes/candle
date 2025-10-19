use candle_core::{mxfp4::dequant_mxfp4_to_bf16_cpu, DType, Device, Result, Tensor};
use half::bf16;

const K_BLOCK: usize = 32;

fn quantize_block_e2m1(values: &[f32]) -> (u8, [u8; 16]) {
    // Search over a small exponent range to minimize SSE on this block.
    // Return: (scale_exp_e8m0, packed_16_bytes)
    assert_eq!(values.len(), K_BLOCK);
    if values.iter().all(|&v| v == 0.0) {
        return (0u8, [0u8; 16]);
    }

    let mut best_exp: i8 = 0;
    let mut best_err = f64::INFINITY;
    let mut best_packed = [0u8; 16];

    // Heuristic center exponent.
    let max_abs = values
        .iter()
        .map(|v| v.abs())
        .fold(0f32, |a, b| a.max(b));
    let mut center = if max_abs > 0.0 { (max_abs / 6.0).log2().floor() as i32 } else { 0 };
    if center < -32 {
        center = -32;
    }
    if center > 31 {
        center = 31;
    }

    // Candidate exponents around center.
    let candidates: Vec<i8> = ((center - 8)..=(center + 8))
        .map(|e| e.clamp(-32, 31) as i8)
        .collect();

    for exp in candidates {
        let scale = (2f32).powi(exp as i32);
        let mut packed = [0u8; 16];
        let mut sse = 0f64;
        for j in 0..16 {
            let idx0 = 2 * j;
            let idx1 = idx0 + 1;
            let q0 = (values[idx0] / scale) as f32;
            let q1 = (values[idx1] / scale) as f32;
            let code0 = encode_fp4_e2m1(q0);
            let code1 = encode_fp4_e2m1(q1);
            let v0 = decode_fp4_e2m1(code0) * scale;
            let v1 = decode_fp4_e2m1(code1) * scale;
            sse += f64::from((values[idx0] - v0).abs() as f32 as f64).powi(2);
            sse += f64::from((values[idx1] - v1).abs() as f32 as f64).powi(2);
            packed[j] = (code1 << 4) | (code0 & 0x0f);
        }
        if sse < best_err {
            best_err = sse;
            best_exp = exp;
            best_packed = packed;
        }
    }

    // MXFP4 E8M0 uses biased-u8 exponent with bias 127 and reserves 0xFF.
    // Our candidate range is clamped to [-32, 31], so adding 127 never yields 0xFF.
    ((best_exp as i32 + 127) as u8, best_packed)
}

#[inline]
fn encode_fp4_e2m1(x: f32) -> u8 {
    // Round-to-nearest for the set {0, 0.5, 1, 1.5, 2, 3, 4, 6}, with sign.
    let sign = if x < 0.0 { 1u8 } else { 0u8 };
    let ax = x.abs();
    // Map thresholds midpoints between representable values.
    // Order pairs by exponent then mantissa.
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
    // Convert idx (0..7) into E2M1 bits (without sign):
    // 0:(e=0,m=0), 1:(e=0,m=1), 2:(e=1,m=0), 3:(e=1,m=1), 4:(e=2,m=0), 5:(e=2,m=1), 6:(e=3,m=0), 7:(e=3,m=1)
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

#[inline]
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

#[test]
fn t3_reconstruction_small_blockwise() -> Result<()> {
    // Create a deterministic small matrix 4x64 with varied values.
    let rows = 4usize;
    let cols = 64usize;
    let mut w = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            // Smooth-ish distribution in roughly [-3, 3].
            let v = ((idx as f32) * 0.37).sin() * 2.2 + ((idx as f32) * 0.13).cos() * 0.8;
            w[idx] = v;
        }
    }

    // Quantize into MXFP4 blocks/scales.
    let nblocks = cols / K_BLOCK;
    let mut blocks = vec![0u8; rows * nblocks * 16];
    let mut scales = vec![0u8; rows * nblocks];
    for r in 0..rows {
        for b in 0..nblocks {
            let base = r * cols + b * K_BLOCK;
            let block_vals = &w[base..base + K_BLOCK];
            let (se, packed) = quantize_block_e2m1(block_vals);
            scales[r * nblocks + b] = se;
            let dst_off = (r * nblocks + b) * 16;
            blocks[dst_off..dst_off + 16].copy_from_slice(&packed);
        }
    }

    // Wrap as tensors.
    let blocks_t = Tensor::from_vec(blocks, (rows, nblocks, 16), &Device::Cpu)?;
    let scales_t = Tensor::from_vec(scales, (rows, nblocks), &Device::Cpu)?;
    let deq_bf16 = dequant_mxfp4_to_bf16_cpu(&blocks_t, &scales_t, [rows, cols])?;

    // Compare to original using MAE/MSE thresholds.
    let deq_f32 = deq_bf16.to_dtype(DType::F32)?;
    let deq = deq_f32.to_vec2::<f32>()?;
    let mut mae = 0f64;
    let mut mse = 0f64;
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let e = deq[r][c];
            let d = (w[idx] - e) as f64;
            mae += d.abs();
            mse += d * d;
        }
    }
    let count = (rows * cols) as f64;
    mae /= count;
    mse /= count;

    // Loose thresholds appropriate for FP4 block quantization.
    assert!(mae < 0.20, "MAE too large: {mae}");
    assert!(mse < 0.08, "MSE too large: {mse}");
    Ok(())
}

#[test]
fn t4_layout_last_dim_blocking_non_mult_64() -> Result<()> {
    // Shape 3x96 (96 is not a multiple of 64 but is multiple of 32), checks iteration order.
    let rows = 3usize;
    let cols = 96usize;
    let nblocks = cols / K_BLOCK; // 3

    // Build scales all zero => scale = 2^(-127)
    let scales = vec![0u8; rows * nblocks];
    // Build blocks so that each position has a known nibble code pattern.
    let mut blocks = vec![0u8; rows * nblocks * 16];

    for r in 0..rows {
        for b in 0..nblocks {
            for j in 0..16 {
                // Cycle the 8 positive magnitude codes for lo, and negative for hi.
                let pos_idx = (j % 8) as u8; // 0..7 maps to increasing values
                let (e_bits, m_bit) = match pos_idx {
                    0 => (0, 0),
                    1 => (0, 1),
                    2 => (1, 0),
                    3 => (1, 1),
                    4 => (2, 0),
                    5 => (2, 1),
                    6 => (3, 0),
                    _ => (3, 1),
                };
                let lo = (0 << 3) | (e_bits << 1) | m_bit; // positive
                let hi = (1 << 3) | (e_bits << 1) | m_bit; // negative
                let byte = (hi << 4) | lo;
                let off = (r * nblocks + b) * 16 + j;
                blocks[off] = byte;
            }
        }
    }

    let blocks_t = Tensor::from_vec(blocks.clone(), (rows, nblocks, 16), &Device::Cpu)?;
    let scales_t = Tensor::from_vec(scales.clone(), (rows, nblocks), &Device::Cpu)?;
    let out = dequant_mxfp4_to_bf16_cpu(&blocks_t, &scales_t, [rows, cols])?;
    assert_eq!(out.dtype(), DType::BF16);

    // Build expected BF16 matrix by direct decode on last-dim blocks.
    let mut expected: Vec<bf16> = vec![bf16::ZERO; rows * cols];
    for r in 0..rows {
        for b in 0..nblocks {
            // scales=0 => 2^(-127)
            let scale = (2f32).powi(-127);
            for j in 0..16 {
                let off = (r * nblocks + b) * 16 + j;
                let byte = blocks[off];
                let lo = byte & 0x0f; // column 2*j
                let hi = byte >> 4; // column 2*j+1
                let c0 = b * 32 + 2 * j;
                let c1 = c0 + 1;
                let v0 = decode_fp4_e2m1(lo) * scale;
                let v1 = decode_fp4_e2m1(hi) * scale;
                expected[r * cols + c0] = bf16::from_f32(v0);
                expected[r * cols + c1] = bf16::from_f32(v1);
            }
        }
    }

    let out_bf16 = out.flatten_all()?.to_vec1::<bf16>()?;
    assert_eq!(out_bf16.len(), expected.len());
    for i in 0..expected.len() {
        assert_eq!(out_bf16[i], expected[i], "Mismatch at idx {i}");
    }

    Ok(())
}
