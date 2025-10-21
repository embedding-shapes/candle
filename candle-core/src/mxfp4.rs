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
#[cfg(feature = "cuda")]
const BLOCK_Q8_1_BYTES: usize = 36; // two half scales + 32 int8 quants

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
    if scale_exp == 0xFF {
        f32::NAN
    } else {
        (2f32).powi((scale_exp as i32) - 127)
    }
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
    if !matches!(blocks.device(), Device::Cuda(_)) || !blocks.device().same_device(scales.device())
    {
        bail!("dequant_mxfp4_to_bf16_cuda expects both inputs on the same CUDA device")
    }
    let dev: &CudaDevice = blocks.device().as_cuda_device()?;

    // Materialize contiguous views (preserve any layout offset from prior narrow())
    let blocks_c = blocks.contiguous()?;
    let scales_c = scales.contiguous()?;

    let (blocks_storage, blocks_layout) = blocks_c.storage_and_layout();
    let (scales_storage, scales_layout) = scales_c.storage_and_layout();
    let blocks_offset = blocks_layout.start_offset();
    let scales_offset = scales_layout.start_offset();

    let blocks_view_base = match &*blocks_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => bail!("expected CUDA storage for blocks"),
    };
    let scales_view_base = match &*scales_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => bail!("expected CUDA storage for scales"),
    };
    // Respect the logical tensor offset so the kernel reads the intended sub-range.
    let blocks_view = blocks_view_base.slice(blocks_offset..);
    let scales_view = scales_view_base.slice(scales_offset..);
    let blocks_arg = &blocks_view;
    let scales_arg = &scales_view;

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
    builder.arg(blocks_arg);
    builder.arg(scales_arg);
    builder.arg(&mut out_slice);
    crate::builder_arg!(builder, rows as i32, nblocks as i32, cols as i32);
    unsafe { builder.launch(cfg) }.w()?;

    let out_storage = CudaStorage::wrap_cuda_slice(out_slice, dev.clone());
    let tensor = crate::tensor::from_storage(
        Storage::Cuda(out_storage),
        (rows, cols),
        BackpropOp::none(),
        false,
    );
    Ok(tensor)
}

/// Dispatch dequantize based on device: CUDA if available, else CPU.
pub fn dequant_mxfp4_to_bf16(
    blocks: &Tensor,
    scales: &Tensor,
    full_shape: [usize; 2],
) -> Result<Tensor> {
    // Diagnostic override: allow forcing CPU dequantization regardless of device.
    let force_cpu = matches!(
        std::env::var("CANDLE_DEQUANT_ON_CPU").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    );
    match (blocks.device(), scales.device()) {
        (Device::Cuda(_), d2) if blocks.device().same_device(d2) => {
            if force_cpu {
                return dequant_mxfp4_to_bf16_cpu(blocks, scales, full_shape);
            }
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

#[cfg(feature = "cuda")]
pub fn matmul_mxfp4_bf16_cuda(
    act: &Tensor,
    blocks: &Tensor,
    scales: &Tensor,
    rows: usize,
    out_dim: usize,
    nblocks: usize,
    in_dim: usize,
) -> Result<Tensor> {
    use crate::{cuda_backend::WrapErr, op::BackpropOp, storage::Storage, CudaDevice, CudaStorage};
    use cudarc::driver::PushKernelArg;

    let dev: &CudaDevice = act.device().as_cuda_device()?;

    let act_base = if act.is_contiguous() {
        act.clone()
    } else {
        act.contiguous()?
    };
    let (act_storage, act_layout) = act_base.storage_and_layout();
    let act_offset = act_layout.start_offset();
    let act_view_base = match &*act_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<bf16>()?,
        _ => bail!("expected CUDA storage for activations"),
    };
    let act_view = act_view_base.slice(act_offset..);
    let act_strides = act_layout.stride();
    if act_strides.len() != 2 || act_strides[1] != 1 {
        bail!("matmul_mxfp4_bf16_cuda expects row-major activations (stride[1] == 1)")
    }
    let act_row_stride = act_strides[0];
    if act_row_stride != in_dim {
        bail!("matmul_mxfp4_bf16_cuda expects contiguous rows (stride[0] == in_dim)")
    }

    let blocks_base = if blocks.is_contiguous() {
        blocks.clone()
    } else {
        blocks.contiguous()?
    };
    let (blocks_storage, blocks_layout) = blocks_base.storage_and_layout();
    let blocks_offset = blocks_layout.start_offset();
    let blocks_view_base = match &*blocks_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => bail!("expected CUDA storage for MXFP4 blocks"),
    };
    let blocks_view = blocks_view_base.slice(blocks_offset..);

    let scales_base = if scales.is_contiguous() {
        scales.clone()
    } else {
        scales.contiguous()?
    };
    let (scales_storage, scales_layout) = scales_base.storage_and_layout();
    let scales_offset = scales_layout.start_offset();
    let scales_view_base = match &*scales_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => bail!("expected CUDA storage for MXFP4 scales"),
    };
    let scales_view = scales_view_base.slice(scales_offset..);

    let mut out_slice = unsafe { dev.alloc::<bf16>(rows * out_dim)? };

    const TILE_COLS: usize = 16;
    const TILE_K_BLOCKS: usize = candle_kernels::MATMUL_MXFP4_TILE_K_BLOCKS;
    const WARPS_PER_BLOCK: usize = 4;
    let grid_y = (out_dim + TILE_COLS - 1) / TILE_COLS;

    let use_q8 = matches!(
        std::env::var("CANDLE_MXFP4_USE_Q8_ACT").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    );

    if use_q8 {
        use crate::quantized::cuda::{CUDA_QUANTIZE_BLOCK_SIZE, MATRIX_ROW_PADDING};

        let in_dim_padded =
            ((in_dim + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        let act_blocks_stride = in_dim_padded / MXFP4_BLOCK_ELEMS;
        let quantized_bytes = match rows
            .checked_mul(act_blocks_stride)
            .and_then(|v| v.checked_mul(BLOCK_Q8_1_BYTES))
        {
            Some(v) => v,
            None => bail!("overflow allocating quantized activations"),
        };
        let mut act_q8 = unsafe { dev.alloc::<u8>(quantized_bytes)? };

        let func_quant = dev.get_or_load_func("quantize_q8_1_bf16", &candle_kernels::QUANTIZED)?;
        let num_blocks = (in_dim_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
        let cfg_quant = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut quant_builder = func_quant.builder();
        quant_builder.arg(&act_view);
        quant_builder.arg(&mut act_q8);
        crate::builder_arg!(quant_builder, in_dim as i32, in_dim_padded as i32);
        unsafe { quant_builder.launch(cfg_quant) }.w()?;

        let func = dev.get_or_load_func("matmul_mxfp4_q8_1", &candle_kernels::QUANTIZED)?;
        let shared_mem_bytes = (TILE_K_BLOCKS * BLOCK_Q8_1_BYTES) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, grid_y as u32, 1),
            block_dim: (32, WARPS_PER_BLOCK as u32, 1),
            shared_mem_bytes,
        };
        let mut builder = func.builder();
        builder.arg(&act_q8);
        builder.arg(&blocks_view);
        builder.arg(&scales_view);
        builder.arg(&mut out_slice);
        crate::builder_arg!(
            builder,
            rows as i32,
            out_dim as i32,
            nblocks as i32,
            act_blocks_stride as i32,
            out_dim as i32
        );
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let func = dev.get_or_load_func("matmul_mxfp4_bf16", &candle_kernels::QUANTIZED)?;
        let shared_mem_bytes =
            (TILE_K_BLOCKS * MXFP4_BLOCK_ELEMS * core::mem::size_of::<f32>()) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (rows as u32, grid_y as u32, 1),
            block_dim: (32, WARPS_PER_BLOCK as u32, 1),
            shared_mem_bytes,
        };
        let mut builder = func.builder();
        builder.arg(&act_view);
        builder.arg(&blocks_view);
        builder.arg(&scales_view);
        builder.arg(&mut out_slice);
        crate::builder_arg!(
            builder,
            rows as i32,
            out_dim as i32,
            nblocks as i32,
            in_dim as i32,
            act_row_stride as i32,
            out_dim as i32
        );
        unsafe { builder.launch(cfg) }.w()?;
    }

    let out_storage = CudaStorage::wrap_cuda_slice(out_slice, dev.clone());
    let tensor = crate::tensor::from_storage(
        crate::storage::Storage::Cuda(out_storage),
        (rows, out_dim),
        BackpropOp::none(),
        false,
    );
    Ok(tensor)
}

#[cfg(feature = "cuda")]
pub fn matmul_mxfp4_bf16_mmq_cuda(
    act: &Tensor,
    blocks: &Tensor,
    scales: &Tensor,
    rows: usize,
    out_dim: usize,
    nblocks: usize,
    in_dim: usize,
) -> Result<Tensor> {
    use crate::{cuda_backend::WrapErr, op::BackpropOp, storage::Storage, CudaDevice, CudaStorage};
    use cudarc::driver::PushKernelArg;

    let dev: &CudaDevice = act.device().as_cuda_device()?;

    // Extract activation tensor (same as existing function)
    let act_base = if act.is_contiguous() {
        act.clone()
    } else {
        act.contiguous()?
    };
    let (act_storage, act_layout) = act_base.storage_and_layout();
    let act_offset = act_layout.start_offset();
    let act_view_base = match &*act_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<bf16>()?,
        _ => bail!("expected CUDA storage for activations"),
    };
    let act_view = act_view_base.slice(act_offset..);
    let act_strides = act_layout.stride();
    if act_strides.len() != 2 || act_strides[1] != 1 {
        bail!("matmul_mxfp4_bf16_mmq_cuda expects row-major activations (stride[1] == 1)")
    }
    let act_row_stride = act_strides[0];

    // Extract blocks tensor
    let blocks_base = if blocks.is_contiguous() {
        blocks.clone()
    } else {
        blocks.contiguous()?
    };
    let (blocks_storage, blocks_layout) = blocks_base.storage_and_layout();
    let blocks_offset = blocks_layout.start_offset();
    let blocks_view_base = match &*blocks_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => bail!("expected CUDA storage for MXFP4 blocks"),
    };
    let blocks_view = blocks_view_base.slice(blocks_offset..);

    // Extract scales tensor
    let scales_base = if scales.is_contiguous() {
        scales.clone()
    } else {
        scales.contiguous()?
    };
    let (scales_storage, scales_layout) = scales_base.storage_and_layout();
    let scales_offset = scales_layout.start_offset();
    let scales_view_base = match &*scales_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<u8>()?,
        _ => bail!("expected CUDA storage for MXFP4 scales"),
    };
    let scales_view = scales_view_base.slice(scales_offset..);

    // Allocate output buffer
    let mut out_slice = unsafe { dev.alloc::<bf16>(rows * out_dim)? };

    // Simplified MMQ configuration: 1 thread = 1 output
    const THREADS_PER_BLOCK: usize = 256;
    let grid_x = rows;
    let grid_y = (out_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Load and launch MMQ kernel
    let func = dev.get_or_load_func("matmul_mxfp4_bf16_mmq", &candle_kernels::QUANTIZED)?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&act_view);
    builder.arg(&blocks_view);
    builder.arg(&scales_view);
    builder.arg(&mut out_slice);
    crate::builder_arg!(
        builder,
        rows as i32,
        out_dim as i32,
        nblocks as i32,
        in_dim as i32,
        act_row_stride as i32,
        out_dim as i32  // out_row_stride
    );
    unsafe { builder.launch(cfg) }.w()?;

    // Wrap output in tensor
    let out_storage = CudaStorage::wrap_cuda_slice(out_slice, dev.clone());
    let tensor = crate::tensor::from_storage(
        crate::storage::Storage::Cuda(out_storage),
        (rows, out_dim),
        BackpropOp::none(),
        false,
    );
    Ok(tensor)
}

#[cfg(feature = "cuda")]
/// Fused expert activation for GPT-OSS
/// Input: gate_up [batch, 2*expert_dim] interleaved (even=gate, odd=up)
/// Output: [batch, expert_dim] = (up+1) * gate * sigmoid(alpha*gate) with asymmetric clamping
pub fn fused_expert_activation_cuda(
    gate_up: &Tensor,
    alpha: f32,
    limit: f32,
) -> Result<Tensor> {
    use crate::{cuda_backend::WrapErr, op::BackpropOp, storage::Storage, CudaStorage};
    use cudarc::driver::PushKernelArg;

    let (batch, two_expert_dim) = gate_up.dims2()?;
    let expert_dim = two_expert_dim / 2;

    let dev = gate_up.device().as_cuda_device()?;

    let gate_up_base = if gate_up.is_contiguous() {
        gate_up.clone()
    } else {
        gate_up.contiguous()?
    };
    let (gate_up_storage, gate_up_layout) = gate_up_base.storage_and_layout();
    let gate_up_offset = gate_up_layout.start_offset();
    let gate_up_view_base = match &*gate_up_storage {
        Storage::Cuda(s) => s.as_cuda_slice::<bf16>()?,
        _ => bail!("expected CUDA storage for gate_up"),
    };
    let gate_up_view = gate_up_view_base.slice(gate_up_offset..);

    let mut output = unsafe { dev.alloc::<half::bf16>(batch * expert_dim)? };

    let func = dev.get_or_load_func("fused_expert_activation_bf16", &candle_kernels::QUANTIZED)?;

    let threads_per_block = 256;
    let num_blocks = (batch * expert_dim + threads_per_block - 1) / threads_per_block;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&gate_up_view);
    builder.arg(&mut output);
    crate::builder_arg!(
        builder,
        batch as i32,
        expert_dim as i32,
        alpha,
        limit
    );
    unsafe { builder.launch(cfg) }.w()?;

    let out_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
    let tensor = crate::tensor::from_storage(
        crate::storage::Storage::Cuda(out_storage),
        (batch, expert_dim),
        crate::op::BackpropOp::none(),
        false,
    );
    Ok(tensor)
}

#[cfg(not(feature = "cuda"))]
pub fn fused_expert_activation_cuda(
    _gate_up: &Tensor,
    _alpha: f32,
    _limit: f32,
) -> Result<Tensor> {
    bail!("fused_expert_activation_cuda requires cuda feature")
}

pub fn matmul_mxfp4_bf16(act: &Tensor, blocks: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let (rows, in_dim) = act.dims2()?;
    let (out_dim, nblocks, bytes) = blocks.dims3()?;
    if bytes != MXFP4_BLOCK_BYTES {
        bail!("mxfp4 blocks last dimension must be 16 bytes, got {bytes}")
    }
    let (_, cols, _) = validate_mxfp4_shapes(blocks, scales, [out_dim, in_dim])?;
    if cols != in_dim {
        bail!("activation dim {in_dim} does not match MXFP4 cols {cols}");
    }
    let act_dev = act.device();
    let blocks_dev = blocks.device();
    let scales_dev = scales.device();

    if !act_dev.same_device(&blocks_dev) || !blocks_dev.same_device(&scales_dev) {
        bail!("activations, blocks, and scales must live on the same device for MXFP4 matmul")
    }

    match act_dev {
        Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            {
                // Check if MMQ kernel should be used (via environment variable)
                let use_mmq = matches!(
                    std::env::var("CANDLE_MXFP4_USE_MMQ").ok().as_deref(),
                    Some("1") | Some("true") | Some("TRUE")
                );

                if use_mmq {
                    return matmul_mxfp4_bf16_mmq_cuda(act, blocks, scales, rows, out_dim, nblocks, in_dim);
                } else {
                    return matmul_mxfp4_bf16_cuda(act, blocks, scales, rows, out_dim, nblocks, in_dim);
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("CUDA matmul requested but candle built without feature=\"cuda\"")
            }
        }
        Device::Cpu => {
            let weight = dequant_mxfp4_to_bf16(blocks, scales, [out_dim, in_dim])?;
            let output = act.matmul(&weight.t()?)?;
            Ok(output)
        }
        other => bail!("unsupported device for MXFP4 matmul: {other:?}"),
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

    // Encode helper used only in tests: round-to-nearest with ties-to-even
    // (even means mantissa bit == 0). Saturates beyond 6.0, underflows below 0.25 to +0.0.
    fn encode_fp4_e2m1_ties_even(x: f32) -> u8 {
        let sign = if x.is_sign_negative() { 1u8 } else { 0u8 };
        let ax = x.abs();
        // Representable magnitudes and their codes (s=0 variants).
        // Codes: (e_bits << 1) | m_bit, with bias=1, e in {0,1,2,3}, m in {0,1}
        const POS: &[(f32, u8)] = &[
            (0.0, 0b000), // e=0,m=0 (subnormal zero)
            (0.5, 0b001), // e=0,m=1 (subnormal half)
            (1.0, 0b010), // e=1,m=0
            (1.5, 0b011), // e=1,m=1
            (2.0, 0b100), // e=2,m=0
            (3.0, 0b101), // e=2,m=1
            (4.0, 0b110), // e=3,m=0
            (6.0, 0b111), // e=3,m=1
        ];
        // Fast-path saturation and underflow
        if ax < 0.25 {
            return sign << 3 /* +0.0 or -0.0, both decode to 0.0 */;
        }
        if ax >= 6.0 {
            return (sign << 3) | POS[7].1;
        }
        // Search nearest; on exact ties prefer the candidate with mantissa bit 0 (even)
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        let mut best_m_bit = 1u8;
        for (i, &(val, code)) in POS.iter().enumerate() {
            let d = (ax - val).abs();
            let m_bit = code & 0x1;
            if d < best_d {
                best = i;
                best_d = d;
                best_m_bit = m_bit;
            } else if (d - best_d).abs() <= 0.0 {
                // exact tie
                // prefer even mantissa (m_bit == 0)
                if m_bit == 0 && best_m_bit == 1 {
                    best = i;
                    best_m_bit = 0;
                }
            }
        }
        // If the chosen representable is +0.0, force sign to + (ties-to-even near zero)
        let code_mag = POS[best].1;
        let sign_bit = if code_mag == 0 { 0u8 } else { sign };
        (sign_bit << 3) | code_mag
    }

    #[test]
    fn fp4_encode_ties_to_even_key_midpoints() {
        // Midpoints between representables should round to the even (mantissa bit 0) code.
        // 0.25 is midpoint between 0.0 (m=0) and 0.5 (m=1) => +0.0
        assert_eq!(encode_fp4_e2m1_ties_even(0.25) & 0x7, 0b000);
        // 0.75 midpoint between 0.5 (m=1) and 1.0 (m=0) => 1.0
        assert_eq!(encode_fp4_e2m1_ties_even(0.75) & 0x7, 0b010);
        // 1.25 midpoint between 1.0 (m=0) and 1.5 (m=1) => 1.0
        assert_eq!(encode_fp4_e2m1_ties_even(1.25) & 0x7, 0b010);
        // 1.75 midpoint between 1.5 (m=1) and 2.0 (m=0) => 2.0
        assert_eq!(encode_fp4_e2m1_ties_even(1.75) & 0x7, 0b100);
        // 2.5 midpoint between 2.0 (m=0) and 3.0 (m=1) => 2.0
        assert_eq!(encode_fp4_e2m1_ties_even(2.5) & 0x7, 0b100);
        // 3.5 midpoint between 3.0 (m=1) and 4.0 (m=0) => 4.0
        assert_eq!(encode_fp4_e2m1_ties_even(3.5) & 0x7, 0b110);
        // 5.0 midpoint between 4.0 (m=0) and 6.0 (m=1) => 4.0
        assert_eq!(encode_fp4_e2m1_ties_even(5.0) & 0x7, 0b110);

        // Side checks just below/above midpoints
        assert_eq!(encode_fp4_e2m1_ties_even(0.75 - 1e-6) & 0x7, 0b001);
        assert_eq!(encode_fp4_e2m1_ties_even(0.75 + 1e-6) & 0x7, 0b010);
    }

    #[test]
    fn fp4_encode_overflow_and_underflow() {
        // Underflow: magnitude < 0.25 -> 0.0 (sign ignored for zero code selection)
        assert_eq!(encode_fp4_e2m1_ties_even(0.0) & 0x7, 0b000);
        assert_eq!(encode_fp4_e2m1_ties_even(0.24999999) & 0x7, 0b000);
        // Exactly 0.25 -> ties-to-even => +0.0 (mantissa 0)
        assert_eq!(encode_fp4_e2m1_ties_even(0.25) & 0x7, 0b000);
        // Overflow: magnitude > 6.0 -> clamp to 6.0
        assert_eq!(encode_fp4_e2m1_ties_even(6.000001) & 0x7, 0b111);
        assert_eq!(encode_fp4_e2m1_ties_even(1234.0) & 0x7, 0b111);
        // Signs preserved on non-zero magnitudes
        assert_eq!(encode_fp4_e2m1_ties_even(-1.0) >> 3, 1);
        assert_eq!(encode_fp4_e2m1_ties_even(1.0) >> 3, 0);
    }

    // Helper: block-scale selection rule per spec.
    // X = (largest power-of-two <= max|v|) / 2^2 = 2^(floor(log2(max|v|)) - 2)
    fn select_block_scale_pow2(vals: &[f32]) -> f32 {
        let max_abs = vals.iter().map(|v| v.abs()).fold(0f32, |a, b| a.max(b));
        if max_abs == 0.0 {
            return (2f32).powi(-127);
        } // arbitrary minimal scale
        let e = max_abs.log2().floor() as i32 - 2;
        (2f32).powi(e)
    }

    #[test]
    fn block_scale_selection_rule() {
        let v = [0.3f32, -1.75, 6.5, 0.0, 2.2, -0.9];
        let x = select_block_scale_pow2(&v);
        // max|v| = 6.5, floor_pow2=4, X=4/4=1
        assert!((x - 1.0).abs() < 1e-6, "scale mismatch: {x}");
        // Modify non-max element should not change X
        let mut v2 = v.clone();
        v2[0] = 0.31;
        let x2 = select_block_scale_pow2(&v2);
        assert!((x - x2).abs() < 1e-12);
    }

    #[test]
    fn idempotent_round_trip_fixed_scale() {
        // Fix scale X, quantize -> dequantize -> requantize reproduces codes.
        let vals = [
            -7.1, -6.0, -5.0, -3.7, -2.5, -1.75, -1.25, -0.75, -0.25, 0.0, 0.25, 0.3, 0.5, 0.75,
            1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.1,
        ];
        let x = 2.0f32; // power-of-two scale
                        // Quantize codes
        let mut codes = Vec::new();
        for &v in &vals {
            let q = v / x;
            codes.push(encode_fp4_e2m1_ties_even(q));
        }
        // Dequantize and requantize
        let mut rec = Vec::new();
        for &c in &codes {
            let v = decode_fp4_e2m1(c) * x;
            let c2 = encode_fp4_e2m1_ties_even(v / x);
            rec.push(c2);
        }
        assert_eq!(
            codes, rec,
            "codes not preserved under fixed-scale round-trip"
        );
    }

    #[test]
    fn scale_nan_propagation_all_elements_nan() -> Result<()> {
        // One row, one block, any bytes; scale 0xFF => NaN for all outputs
        let rows = 1usize;
        let cols = MXFP4_BLOCK_ELEMS;
        let nblocks = 1usize;
        let blocks = Tensor::from_vec(
            vec![0x22u8; MXFP4_BLOCK_BYTES],
            (rows, nblocks, MXFP4_BLOCK_BYTES),
            &Device::Cpu,
        )?;
        let scales = Tensor::from_vec(vec![0xFFu8], (rows, nblocks), &Device::Cpu)?;
        let out = dequant_mxfp4_to_bf16_cpu(&blocks, &scales, [rows, cols])?;
        let v = out.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        for i in 0..cols {
            assert!(v[0][i].is_nan(), "elem {i} expected NaN");
        }
        Ok(())
    }

    #[test]
    fn dot_product_semantics_match() {
        // Build two 32-element vectors in one block
        let mut a = [0f32; MXFP4_BLOCK_ELEMS];
        let mut b = [0f32; MXFP4_BLOCK_ELEMS];
        for i in 0..MXFP4_BLOCK_ELEMS {
            a[i] = ((i as f32) * 0.31).sin() * 3.2;
            b[i] = ((i as f32) * 0.17).cos() * 2.7;
        }
        // Select scales by rule and quantize codes
        let xa = select_block_scale_pow2(&a);
        let xb = select_block_scale_pow2(&b);
        let mut pa = [0u8; MXFP4_BLOCK_ELEMS];
        let mut pb = [0u8; MXFP4_BLOCK_ELEMS];
        for i in 0..MXFP4_BLOCK_ELEMS {
            pa[i] = encode_fp4_e2m1_ties_even(a[i] / xa);
            pb[i] = encode_fp4_e2m1_ties_even(b[i] / xb);
        }
        // Dot semantics: (Xa*Xb) * sum_i(Pa*Pb)
        let mut sum_pp = 0f32;
        for i in 0..MXFP4_BLOCK_ELEMS {
            sum_pp += decode_fp4_e2m1(pa[i]) * decode_fp4_e2m1(pb[i]);
        }
        let dot_q = xa * xb * sum_pp;
        // Reference: dequantize then f32 dot
        let mut a_dq = [0f32; MXFP4_BLOCK_ELEMS];
        let mut b_dq = [0f32; MXFP4_BLOCK_ELEMS];
        for i in 0..MXFP4_BLOCK_ELEMS {
            a_dq[i] = decode_fp4_e2m1(pa[i]) * xa;
            b_dq[i] = decode_fp4_e2m1(pb[i]) * xb;
        }
        let dot_ref: f32 = a_dq.iter().zip(b_dq.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (dot_q - dot_ref).abs() < 1e-6,
            "dot semantics mismatch: {} vs {}",
            dot_q,
            dot_ref
        );
    }

    #[test]
    fn last_dim_not_multiple_of_k_is_error() {
        // Validate we fail fast when cols is not divisible by k=32.
        let rows = 1usize;
        let cols = 48usize; // not divisible by 32
        let nblocks = 1usize; // mismatch on purpose
        let blocks = Tensor::from_vec(
            vec![0u8; MXFP4_BLOCK_BYTES],
            (rows, nblocks, MXFP4_BLOCK_BYTES),
            &Device::Cpu,
        )
        .unwrap();
        let scales = Tensor::from_vec(vec![0u8; nblocks], (rows, nblocks), &Device::Cpu).unwrap();
        let err = dequant_mxfp4_to_bf16_cpu(&blocks, &scales, [rows, cols]).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("must be multiple of 32"),
            "unexpected error: {}",
            msg
        );
    }
}
