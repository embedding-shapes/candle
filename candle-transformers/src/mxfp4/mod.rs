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
//  - e == 3: special
//      m == 0 -> ±Inf
//      m == 1 -> NaN

/// Lookup table for FP4(E2M1) nibble-to-f32 decode.
/// Index is the 4-bit code 0..15.
pub const FP4_E2M1_TO_F32: [f32; 16] = [
    // s=0, e=0..3, m=0/1
    0.0,            // 0b0000: +0.0
    0.5,            // 0b0001: +0.5 (subnormal)
    1.0,            // 0b0010: +1.0
    1.5,            // 0b0011: +1.5
    2.0,            // 0b0100: +2.0
    3.0,            // 0b0101: +3.0
    f32::INFINITY,  // 0b0110: +Inf
    f32::NAN,       // 0b0111: NaN
    -0.0,           // 0b1000: -0.0
    -0.5,           // 0b1001: -0.5 (subnormal)
    -1.0,           // 0b1010: -1.0
    -1.5,           // 0b1011: -1.5
    -2.0,           // 0b1100: -2.0
    -3.0,           // 0b1101: -3.0
    f32::NEG_INFINITY, // 0b1110: -Inf
    f32::NAN,          // 0b1111: NaN (sign ignored)
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
    if code == 0xFF { f32::NAN } else { 2f32.powi((code as i32) - 127) }
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
                    if s == 0 { 0.0 } else { -0.0 }
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
                if m == 0 {
                    if s == 0 { f32::INFINITY } else { f32::NEG_INFINITY }
                } else {
                    f32::NAN
                }
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn t1_fp4_decode_table_matches_spec() {
        for code in 0u8..16u8 {
            let got = decode_fp4_e2m1_nibble(code);
            let exp = ref_decode_fp4(code);

            if exp.is_nan() {
                assert!(got.is_nan(), "code {code}: expected NaN, got {got:?}");
                continue;
            }
            if exp == 0.0 {
                // Check signed zero explicitly by bits
                let exp_bits = exp.to_bits();
                let got_bits = got.to_bits();
                assert_eq!(exp_bits, got_bits, "code {code}: zero sign mismatch");
                continue;
            }
            if exp.is_infinite() {
                assert!(got.is_infinite(), "code {code}: expected Inf, got {got:?}");
                assert_eq!(exp.is_sign_negative(), got.is_sign_negative(), "code {code}: Inf sign mismatch");
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
