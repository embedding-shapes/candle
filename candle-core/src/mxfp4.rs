//! MXFP4 (FP4 E2M1 + E8M0 block scale) utilities.
//!
//! Phase A CPU path: dequantize packed MXFP4 weights into a contiguous BF16 tensor.
//!
//! Layout assumptions (validated):
//! - `blocks`: [rows, blocks, 16] where each 16-byte block packs 32 FP4 values
//!   along the last dimension, two nibbles per byte (low then high).
//! - `scales`: [rows, blocks] where each u8 is an E8M0 exponent; real scale is 2^(u8 - 127),
//!   and 0xFF is reserved (maps to NaN) per MX spec.
//! - Final output shape is [rows, cols] with `cols = blocks * 32`.

use crate::{bail, DType, Device, Result, Tensor};
use half::bf16;

// Constants
const MXFP4_FP4_BIAS: i32 = 1; // IEEE754-style bias for E=2
const MXFP4_BLOCK_ELEMS: usize = 32; // k=32 elements per block
const MXFP4_BLOCK_BYTES: usize = 16; // 2 values per byte

// Shared shape validation used by CPU/CUDA paths.
fn validate_mxfp4_shapes(
    blocks: &Tensor,
    scales: &Tensor,
    full_shape: [usize; 2],
) -> Result<(usize, usize, usize)> {
    if blocks.dtype() != DType::U8 {
        bail!("mxfp4 blocks must be U8, got {:?}", blocks.dtype())
    }
    if scales.dtype() != DType::U8 {
        bail!("mxfp4 scales must be U8, got {:?}", scales.dtype())
    }
    let rows = full_shape[0];
    let cols = full_shape[1];
    if cols % MXFP4_BLOCK_ELEMS != 0 {
        bail!("mxfp4 cols must be multiple of 32, got {cols}")
    }
    let nblocks = cols / MXFP4_BLOCK_ELEMS;
    let bdims = blocks.dims();
    if bdims != [rows, nblocks, MXFP4_BLOCK_BYTES] {
        bail!(
            "mxfp4 blocks shape mismatch, expected [rows, cols/32, 16]=[{rows}, {nblocks}, 16], got {:?}",
            bdims
        )
    }
    let sdims = scales.dims();
    if sdims != [rows, nblocks] {
        bail!(
            "mxfp4 scales shape mismatch, expected [rows, cols/32]=[{rows}, {nblocks}], got {:?}",
            sdims
        )
    }
    Ok((rows, cols, nblocks))
}
// CPU-only dequant remains as-is below.

/// Decode a single 4-bit E2M1 (FP4) code into f32 according to:
/// - Normal (E>0): (-1)^S * 2^(E-bias) * (1 + M * 2^-1)
/// - Subnormal (E=0): (-1)^S * 2^(1-bias) * (M * 2^-1)
#[inline]
fn decode_fp4_e2m1(nibble: u8) -> f32 {
    let s = (nibble >> 3) & 0x1;
    let e = (nibble >> 1) & 0x3; // 2 bits
    let m = nibble & 0x1; // 1 bit mantissa

    let sign = if s == 1 { -1.0 } else { 1.0 };
    if e == 0 {
        // subnormal: 2^(1-bias) * (m * 2^-1)
        let frac = (m as f32) * 0.5;
        let exp = (1 - MXFP4_FP4_BIAS) as i32; // usually 0 for bias=1
        sign * (2f32).powi(exp) * frac
    } else {
        // normal: 2^(E-bias) * (1 + m * 2^-1)
        let frac = 1.0 + (m as f32) * 0.5;
        let exp = (e as i32) - MXFP4_FP4_BIAS;
        sign * (2f32).powi(exp) * frac
    }
}

/// Convert an E8M0 scale byte to its f32 power-of-two value using biased exponent semantics.
///
/// - For code != 0xFF: scale = 2^(code - 127)
/// - For code == 0xFF: scale = NaN (reserved code)
#[inline]
fn pow2_e8m0(scale_exp: u8) -> f32 {
    if scale_exp == 0xFF { f32::NAN } else { (2f32).powi((scale_exp as i32) - 127) }
}

/// Dequantize MXFP4 packed weights on CPU into a contiguous BF16 tensor with shape `[rows, cols]`.
///
/// - `blocks`: U8 tensor shaped `[rows, blocks, 16]` (two FP4 per byte, last-dim blocking)
/// - `scales`: U8 tensor shaped `[rows, blocks]` (E8M0 exponent per block)
/// - `full_shape`: `[rows, cols]` where `cols = blocks * 32`
///
/// Errors if shapes or dtypes do not match these invariants.
pub fn dequant_mxfp4_to_bf16_cpu(
    blocks: &Tensor,
    scales: &Tensor,
    full_shape: [usize; 2],
) -> Result<Tensor> {
    let (rows, cols, nblocks) = validate_mxfp4_shapes(blocks, scales, full_shape)?;

    // Materialize to CPU host vectors; accept non-contiguous tensors.
    let blocks_v = blocks.to_vec3::<u8>()?; // [rows][nblocks][16]
    let scales_v = scales.to_vec2::<u8>()?; // [rows][nblocks]

    // Output buffer, row-major BF16.
    let mut out: Vec<bf16> = vec![bf16::ZERO; rows * cols];

    for r in 0..rows {
        let row_off = r * cols;
        let row_blocks = &blocks_v[r];
        let row_scales = &scales_v[r];

        for b in 0..nblocks {
            let scale = pow2_e8m0(row_scales[b]);
            let packed = &row_blocks[b]; // 16 bytes

            // Within a block of 32 elements: two nibbles per byte → positions 2*j and 2*j+1.
            for j in 0..MXFP4_BLOCK_BYTES {
                let byte = packed[j];
                let lo = byte & 0x0f; // column 2*j
                let hi = byte >> 4; // column 2*j+1
                let c0 = b * MXFP4_BLOCK_ELEMS + (2 * j);
                let c1 = c0 + 1;

                let v0 = decode_fp4_e2m1(lo) * scale;
                let v1 = decode_fp4_e2m1(hi) * scale;
                out[row_off + c0] = bf16::from_f32(v0);
                out[row_off + c1] = bf16::from_f32(v1);
            }
        }
    }

    Tensor::from_vec(out, (rows, cols), &Device::Cpu)
}

/// CUDA fused dequant (if compiled with `feature = "cuda"`). Returns BF16 tensor on same CUDA device.
#[cfg(feature = "cuda")]
pub fn dequant_mxfp4_to_bf16_cuda(
    blocks: &Tensor,
    scales: &Tensor,
    full_shape: [usize; 2],
) -> Result<Tensor> {
    use crate::{cuda_backend::WrapErr, op::BackpropOp, storage::Storage, CudaDevice, CudaStorage};
    use cudarc::driver::PushKernelArg;
    let (rows, cols, nblocks) = validate_mxfp4_shapes(blocks, scales, full_shape)?;
    // Both inputs must be on CUDA and same device.
    if !matches!(blocks.device(), Device::Cuda(_)) || !blocks.device().same_device(scales.device()) {
        bail!("dequant_mxfp4_to_bf16_cuda expects both inputs on the same CUDA device")
    }
    let dev: &CudaDevice = blocks.device().as_cuda_device()?;
    let blocks_c = blocks.contiguous()?;
    let scales_c = scales.contiguous()?;
    let blocks_s = blocks_c.storage();
    let scales_s = scales_c.storage();
    let blocks_view = match &*blocks_s {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => bail!("expected CUDA storage for blocks"),
    };
    let scales_view = match &*scales_s {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => bail!("expected CUDA storage for scales"),
    };

    // Allocate output BF16 on device.
    let elem_count = rows * cols;
    let mut out_slice = unsafe { dev.alloc::<bf16>(elem_count)? };

    // Launch kernel: grid=(rows, nblocks, 1), block=(32,1,1)
    let func = dev.get_or_load_func("dequant_mxfp4_to_bf16", &candle_kernels::QUANTIZED)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (rows as u32, nblocks as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(blocks_view);
    builder.arg(scales_view);
    builder.arg(&mut out_slice);
    crate::builder_arg!(builder, rows as i32, nblocks as i32, cols as i32);
    unsafe { builder.launch(cfg) }.w()?;

    let out_storage = CudaStorage::wrap_cuda_slice(out_slice, dev.clone());
    let tensor = crate::tensor::from_storage(Storage::Cuda(out_storage), (rows, cols), BackpropOp::none(), false);
    Ok(tensor)
}

/// Dispatch dequantize based on device: CUDA if available, else CPU.
pub fn dequant_mxfp4_to_bf16(
    blocks: &Tensor,
    scales: &Tensor,
    full_shape: [usize; 2],
) -> Result<Tensor> {
    match (blocks.device(), scales.device()) {
        (Device::Cuda(_), d2) if blocks.device().same_device(d2) => {
            #[cfg(feature = "cuda")]
            {
                return dequant_mxfp4_to_bf16_cuda(blocks, scales, full_shape);
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("CUDA dequant requested but candle built without feature=\"cuda\"")
            }
        }
        _ => dequant_mxfp4_to_bf16_cpu(blocks, scales, full_shape),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_decode_basics() {
        // Spot-check a few codes to guard against sign / nibble mistakes.
        // 0b0_00_0 => +0.0
        assert_eq!(decode_fp4_e2m1(0b0000), 0.0);
        // 0b0_00_1 => +0.5 (subnormal)
        assert_eq!(decode_fp4_e2m1(0b0001), 0.5);
        // 0b0_01_0 => +1.0
        assert_eq!(decode_fp4_e2m1(0b0010), 1.0);
        // 0b0_01_1 => +1.5
        assert_eq!(decode_fp4_e2m1(0b0011), 1.5);
        // 0b0_10_0 => +2.0
        assert_eq!(decode_fp4_e2m1(0b0100), 2.0);
        // 0b0_10_1 => +3.0
        assert_eq!(decode_fp4_e2m1(0b0101), 3.0);
        // 0b0_11_0 => +4.0
        assert_eq!(decode_fp4_e2m1(0b0110), 4.0);
        // 0b0_11_1 => +6.0
        assert_eq!(decode_fp4_e2m1(0b0111), 6.0);
        // Negative variants
        assert_eq!(decode_fp4_e2m1(0b1000), -0.0);
        assert_eq!(decode_fp4_e2m1(0b1001), -0.5);
        assert_eq!(decode_fp4_e2m1(0b1010), -1.0);
        assert_eq!(decode_fp4_e2m1(0b1011), -1.5);
        assert_eq!(decode_fp4_e2m1(0b1100), -2.0);
        assert_eq!(decode_fp4_e2m1(0b1101), -3.0);
        assert_eq!(decode_fp4_e2m1(0b1110), -4.0);
        assert_eq!(decode_fp4_e2m1(0b1111), -6.0);
    }

    #[test]
    fn e8m0_scale_decode_key_cases() {
        // MX spec: biased exponent with bias 127, 0xFF reserved
        let cases: &[(u8, f32, bool)] = &[
            (0u8, 2f32.powi(-127), false),
            (127u8, 1.0, false),
            (128u8, 2.0, false),
            (254u8, 2f32.powi(127), false),
            (255u8, f32::NAN, true),
        ];
        for &(code, expect, is_nan) in cases {
            let got = pow2_e8m0(code);
            if is_nan {
                assert!(got.is_nan(), "code={code}: expected NaN, got {got}");
            } else {
                assert_eq!(got, expect, "code={code}: mismatch {got} vs {expect}");
            }
        }
    }
}
