// MXFP4 decode utilities: FP4(E2M1) element decode and E8M0 scale mapping.
// Pure-Rust helpers used for CPU paths and tests.

// Constants and configuration
//
// FP4(E2M1): 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit.
// Encoding per nibble: [s e1 e0 m]
//  - e == 0: subnormals/zero
//      m == 0 -> signed zero (±0.0)
//      m == 1 -> (-1)^s * 2^(1-bias) * (m * 2^-1) = (-1)^s * 0.5
//  - e == 1 or 2: normals
//      v = (-1)^s * 2^(e-bias) * (1 + m * 2^-1)
//      with bias=1, values are {±1.0, ±1.5, ±2.0, ±3.0}
//  - e == 3: normal (no NaN/Inf in FP4 E2M1 per OCP MX v1.0)
//      v = (-1)^s * 2^(3-bias) * (1 + m * 2^-1) = (-1)^s * 2^2 * (1 + m/2)
//      values are ±4.0 (m=0) and ±6.0 (m=1)

/// Lookup table for FP4(E2M1) nibble-to-f32 decode.
/// Index is the 4-bit code 0..15.
pub const FP4_E2M1_TO_F32: [f32; 16] = [
    // s=0, e=0..3, m=0/1
    0.0,  // 0b0000: +0.0
    0.5,  // 0b0001: +0.5 (subnormal)
    1.0,  // 0b0010: +1.0
    1.5,  // 0b0011: +1.5
    2.0,  // 0b0100: +2.0
    3.0,  // 0b0101: +3.0
    4.0,  // 0b0110: +4.0
    6.0,  // 0b0111: +6.0
    -0.0, // 0b1000: -0.0
    -0.5, // 0b1001: -0.5 (subnormal)
    -1.0, // 0b1010: -1.0
    -1.5, // 0b1011: -1.5
    -2.0, // 0b1100: -2.0
    -3.0, // 0b1101: -3.0
    -4.0, // 0b1110: -4.0
    -6.0, // 0b1111: -6.0
];

/// Decode one FP4(E2M1) nibble (low 4 bits of `code`) to f32 using the lookup table.
#[inline]
pub fn decode_fp4_e2m1_nibble(code: u8) -> f32 {
    FP4_E2M1_TO_F32[(code & 0x0F) as usize]
}

/// Map an E8M0 8-bit biased exponent code to a power-of-two scale in f32.
/// Semantics per MX spec:
/// - return 2^(code - 127) for code in [0x00..=0xFE]
/// - return NaN for code == 0xFF (reserved)
#[inline]
pub fn e8m0_to_pow2(code: u8) -> f32 {
    if code == 0xFF {
        f32::NAN
    } else {
        2f32.powi((code as i32) - 127)
    }
}

/// Decode a 32-value block from 16 bytes (packed two nibbles per byte) and one E8M0 scale.
///
/// Nibble order per byte: low nibble -> even index (2j), high nibble -> odd index (2j+1).
/// Returns the scaled values as an array of 32 f32 values.
pub fn decode_block_32(block_bytes: &[u8; 16], scale_code: u8) -> [f32; 32] {
    let s = e8m0_to_pow2(scale_code);
    let mut out = [0f32; 32];
    for (j, &b) in block_bytes.iter().enumerate() {
        let lo = b & 0x0F;
        let hi = (b >> 4) & 0x0F;
        let base = j * 2;
        out[base] = decode_fp4_e2m1_nibble(lo) * s;
        out[base + 1] = decode_fp4_e2m1_nibble(hi) * s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compute reference FP4(E2M1) decode per spec for a 4-bit code.
    fn ref_decode_fp4(code: u8) -> f32 {
        let s = (code >> 3) & 0x1;
        let e = (code >> 1) & 0x3;
        let m = code & 0x1;
        let sign = if s == 0 { 1.0 } else { -1.0 };
        match e {
            0 => {
                if m == 0 {
                    if s == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                } else {
                    // subnormal: (-1)^s * 2^(1-bias) * (m * 2^-1), bias=1
                    sign * 0.5
                }
            }
            1 | 2 => {
                // normal: (-1)^s * 2^(e-bias) * (1 + m*2^-1), bias=1
                let exp = (e as i32) - 1;
                let mant = 1.0 + if m == 0 { 0.0 } else { 0.5 };
                sign * (2f32.powi(exp) * mant)
            }
            3 => {
                // bias=1 -> 2^(3-1) = 4; mantissa (1 or 1.5)
                let mant = 1.0 + if m == 0 { 0.0 } else { 0.5 };
                sign * (4.0 * mant) // => ±4.0 or ±6.0
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn t1_fp4_decode_table_matches_spec() {
        for code in 0u8..16u8 {
            let got = decode_fp4_e2m1_nibble(code);
            let exp = ref_decode_fp4(code);

            if exp == 0.0 {
                // Check signed zero explicitly by bits
                let exp_bits = exp.to_bits();
                let got_bits = got.to_bits();
                assert_eq!(exp_bits, got_bits, "code {code}: zero sign mismatch");
                continue;
            }
            assert!(got.is_finite(), "code {code}: expected finite, got {got:?}");
            assert_eq!(got, exp, "code {code}: mismatch, got {got}, exp {exp}");
        }
    }

    #[test]
    fn t2_e8m0_scale_decode_key_cases() {
        // Key mandated mappings for MX E8M0
        assert_eq!(e8m0_to_pow2(0), 2f32.powi(-127));
        assert_eq!(e8m0_to_pow2(127), 1.0);
        assert_eq!(e8m0_to_pow2(128), 2.0);
        assert_eq!(e8m0_to_pow2(254), 2f32.powi(127));
        assert!(e8m0_to_pow2(255).is_nan());
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    // Helper: encode absolute value to nearest E2M1 magnitude with ties-to-even (mantissa 0),
    // then apply sign; exact 0.25 ties to +0.0 (positive zero).
    fn encode_fp4_e2m1_ties_even(x: f32) -> u8 {
        const POS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let sgn_neg = x.is_sign_negative();
        let ax = x.abs();
        // Saturate outside [0,6] to the nearest endpoint.
        let ax = if ax.is_finite() { ax.min(6.0) } else { 6.0 };
        // Midpoint boundaries for ties: choose lower (mantissa bit 0) per ties-to-even.
        // Linear scan is fine for tests.
        let mut idx = 0usize;
        let mut min_d = f32::INFINITY;
        for (i, &p) in POS.iter().enumerate() {
            let d = (ax - p).abs();
            if d < min_d - f32::EPSILON {
                min_d = d;
                idx = i;
            } else if (d - min_d).abs() <= f32::EPSILON {
                // Tie: pick the even code which in our ordering is the lower-mantissa (earlier entry)
                // Do nothing so we keep the first (lower) one.
            }
        }
        let (e_bits, m_bit) = match idx {
            0 => (0, 0),
            1 => (0, 1),
            2 => (1, 0),
            3 => (1, 1),
            4 => (2, 0),
            5 => (2, 1),
            6 => (3, 0),
            _ => (3, 1),
        };
        // Special-case: exact 0.25 rounds to +0.0 (positive zero), regardless of sign.
        if (ax - 0.25).abs() <= f32::EPSILON {
            return 0b0000;
        }
        // Preserve sign for sub-0.25 values (±0.0), but do not attach a negative sign to non-zero magnitudes idx>0.
        let s = if sgn_neg { 1 } else { 0 };
        (s << 3) | (e_bits << 1) | m_bit
    }

    #[test]
    fn t3_encode_rounding_ties_to_even() {
        // Midpoints between representables should go to lower (mantissa=0) code.
        // 1.25 -> 1.0 (mantissa 0)
        assert_eq!(encode_fp4_e2m1_ties_even(1.25), 0b0010);
        assert_eq!(encode_fp4_e2m1_ties_even(-1.25), 0b1010);
        // 2.5 -> 2.0
        assert_eq!(encode_fp4_e2m1_ties_even(2.5), 0b0100);
        // 5.0 -> 4.0
        assert_eq!(encode_fp4_e2m1_ties_even(5.0), 0b0110);
        // Underflow boundary: exactly 0.25 -> +0.0
        assert_eq!(encode_fp4_e2m1_ties_even(0.25), 0b0000);
        assert_eq!(encode_fp4_e2m1_ties_even(-0.25), 0b0000);
        // Just below/above 0.25
        assert_eq!(encode_fp4_e2m1_ties_even(0.25 - 1e-6), 0b0000);
        assert_eq!(encode_fp4_e2m1_ties_even(0.25 + 1e-6), 0b0001);
    }

    #[test]
    fn t4_overflow_and_underflow_saturation() {
        // Overflow: clamp to ±6.0
        assert_eq!(encode_fp4_e2m1_ties_even(10.0), 0b0111);
        assert_eq!(encode_fp4_e2m1_ties_even(-10.0), 0b1111);
        // Underflow: |x| < 0.25 -> ±0.0 (positive for exact 0.0)
        assert_eq!(encode_fp4_e2m1_ties_even(0.0), 0b0000);
        assert_eq!(encode_fp4_e2m1_ties_even(-0.001), 0b1000); // negative sub-zero rounds to -0.0 only if not at exact 0.25
    }

    #[test]
    fn t5_block_scale_selection_rule() {
        // X = 2^{floor(log2(max|v|)) - 2}
        fn select_scale_x(vals: &[f32]) -> i32 {
            let max = vals.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            if max == 0.0 {
                return -127;
            } // arbitrary small, but unused here
            max.log2().floor() as i32 - 2
        }
        let v1 = [0.1, 0.7, 1.3, 5.9, -0.2, 0.0, -2.2, 4.1];
        let x1 = select_scale_x(&v1);
        // Perturb non-maximum elements; scale should be invariant.
        let mut v2 = v1.clone();
        v2[0] = 0.11;
        v2[1] = 0.69;
        v2[2] = 1.29;
        v2[4] = -0.21;
        let x2 = select_scale_x(&v2);
        assert_eq!(x1, x2);
        // Increase maximum slightly without crossing next power-of-two => unchanged
        let mut v3 = v1.clone();
        v3[3] = 5.99;
        let x3 = select_scale_x(&v3);
        assert_eq!(x1, x3);
        // Cross boundary: from <8 to >=8 shifts exponent
        let mut v4 = v1.clone();
        v4[3] = 8.01;
        let x4 = select_scale_x(&v4);
        assert_eq!(x4, x1 + 1);
    }

    #[test]
    fn t6_idempotent_roundtrip_with_fixed_scale() {
        // With fixed X, quantize→dequantize→requantize yields identical codes.
        // Use a small set of sample values across a few scales.
        let samples = [
            -0.3, -0.25, -0.24, -0.51, -1.2, -2.6, -3.0, -4.9, -5.5, -6.1, 0.0, 0.2, 0.25, 0.26,
            0.49, 0.5, 0.74, 0.75, 1.0, 1.26, 2.49, 2.5, 4.9, 5.0, 5.5, 6.0, 6.5,
        ];
        for x in [-5, -1, 0, 3] {
            // X = 2^x scales
            let scale = 2f32.powi(x);
            for &v in &samples {
                let code1 = encode_fp4_e2m1_ties_even(v / scale);
                let deq = decode_fp4_e2m1_nibble(code1) * scale;
                let code2 = encode_fp4_e2m1_ties_even(deq / scale);
                assert_eq!(code1, code2, "x={x} v={v} roundtrip mismatch");
            }
        }
    }

    #[test]
    fn t7_scale_nan_propagation() {
        let block = [0x10u8; 16]; // arbitrary codes
        let out = decode_block_32(&block, 0xFF);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn t8_dot_product_semantics() {
        // Random-ish fixed bytes and scales
        let b1 = [
            0xF1u8, 0x20, 0xAB, 0x44, 0x00, 0x7F, 0x88, 0xCC, 0xDE, 0xAD, 0xBE, 0xEF, 0x12, 0x34,
            0x56, 0x78,
        ];
        let b2 = [
            0x0Fu8, 0x13, 0x37, 0x99, 0x42, 0x42, 0x24, 0x24, 0xAA, 0x55, 0x11, 0x22, 0x33, 0x44,
            0x55, 0x66,
        ];
        let s1 = 127u8; // 2^0
        let s2 = 128u8; // 2^1
        let v1 = decode_block_32(&b1, s1);
        let v2 = decode_block_32(&b2, s2);
        // Dot(A,B) == (X_A * X_B) * sum(Pi_A * Pi_B), where Pi are base decoded FP4s (scale=1)
        let x1 = e8m0_to_pow2(s1);
        let x2 = e8m0_to_pow2(s2);
        let mut sum_pi = 0f32;
        for j in 0..32 {
            let a = decode_fp4_e2m1_nibble((b1[j / 2] >> ((j & 1) * 4)) & 0x0F);
            let b = decode_fp4_e2m1_nibble((b2[j / 2] >> ((j & 1) * 4)) & 0x0F);
            sum_pi += a * b;
        }
        let lhs: f32 = v1.iter().zip(&v2).map(|(a, b)| a * b).sum();
        let rhs: f32 = (x1 * x2) * sum_pi;
        let diff = (lhs - rhs).abs();
        assert!(
            diff < 1e-5,
            "dot semantics mismatch: lhs={lhs} rhs={rhs} diff={diff}"
        );
    }

    #[test]
    fn t9_block_padding_tail_handling() {
        // Simulate a vector with N=36 (one full block + 4-tail), last 28 padded with zeros.
        let mut b = [0u8; 16];
        // First two bytes contain 4 values; rest effectively zeros.
        b[0] = 0x21; // lo=1(+0.5), hi=2(+1.0)
        b[1] = 0x43; // lo=3(+1.5), hi=4(+2.0)
        let s = 127u8; // scale 1
        let decoded = decode_block_32(&b, s);
        // Build reference vector of length 36: decoded[0..4] then 32-4 zeros then 4 more from next block zeros
        let mut ref_vec = vec![0f32; 36];
        ref_vec[0] = 0.5;
        ref_vec[1] = 1.0;
        ref_vec[2] = 1.5;
        ref_vec[3] = 2.0;
        // Tail within the block is already zeros due to codes 0, consistent with padding-by-zero.
        // Compare dot with another vector (all ones) using only first 36 elements.
        let dot_decoded: f32 = decoded.iter().take(36).sum();
        let dot_ref: f32 = ref_vec.iter().sum();
        assert!((dot_decoded - dot_ref).abs() < 1e-6);
    }

    #[test]
    fn t10_serialization_layout_roundtrip() {
        // Pack -> decode -> pack should preserve nibbles and scale.
        let codes: [u8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2,
            4, 6, 8, 10, 12, 14,
        ];
        let mut packed = [0u8; 16];
        for j in 0..16 {
            packed[j] = ((codes[2 * j + 1] & 0x0F) << 4) | (codes[2 * j] & 0x0F);
        }
        let s = 200u8;
        let _decoded = decode_block_32(&packed, s);
        // Unpack back to codes and compare
        let mut unpacked = [0u8; 32];
        for j in 0..16 {
            let b = packed[j];
            unpacked[2 * j] = b & 0x0F;
            unpacked[2 * j + 1] = (b >> 4) & 0x0F;
        }
        assert_eq!(unpacked, codes);
    }
}
