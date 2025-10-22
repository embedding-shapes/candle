use candle::{DType, Result};
use candle_nn::Linear;

use super::{dequantize_mxfp4_linear, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS};

/// Load a single expert's Linear from grouped MXFP4 tensors stored under a common base
/// such as "experts.gate_up_proj" where the underlying tensors have a leading expert
/// dimension, e.g. `blocks: [n_experts, out_dim, in_dim/32, 16]` and `scales: [n_experts, out_dim, in_dim/32]`.
///
/// - `vb` should point at the module scope where the grouped tensors reside (e.g., the layer's `mlp`).
/// - `base` is the common prefix (e.g., "experts.gate_up_proj" or "experts.down_proj").
/// - `expert_idx` selects which expert slice to load.
pub fn load_expert_linear_mxfp4_grouped(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: candle_nn::VarBuilder,
    base: &str,
    expert_idx: usize,
    n_experts: usize,
) -> Result<Linear> {
    // Try paired grouped names: either dot or underscore variants.
    let blocks_dot = format!("{base}_blocks");
    let scales_dot = format!("{base}_scales");
    let has_us = vb.contains_tensor(&blocks_dot) && vb.contains_tensor(&scales_dot);
    let (blocks_name, scales_name) = if has_us {
        (blocks_dot, scales_dot)
    } else {
        // Also support dot suffix within grouped context: base.blocks / base.scales
        let blocks_alt = format!("{base}.blocks");
        let scales_alt = format!("{base}.scales");
        if vb.contains_tensor(&blocks_alt) && vb.contains_tensor(&scales_alt) {
            (blocks_alt, scales_alt)
        } else {
            candle::bail!("grouped MXFP4 tensors not found for base '{base}'")
        }
    };

    if in_dim % MXFP4_BLOCK_ELEMS != 0 {
        candle::bail!(
            "MXFP4 grouped weight '{base}': in_dim must be multiple of {MXFP4_BLOCK_ELEMS}, got {in_dim}"
        )
    }
    let nblocks = in_dim / MXFP4_BLOCK_ELEMS;

    // Load U8 grouped tensors and select expert slice along the leading dimension.
    let vb_u8 = vb.to_dtype(DType::U8);
    let blocks_g = vb_u8.get(
        (n_experts, out_dim, nblocks, MXFP4_BLOCK_BYTES),
        &blocks_name,
    )?; // (E, out, nb, 16)
    let scales_g = vb_u8.get((n_experts, out_dim, nblocks), &scales_name)?; // (E, out, nb)
    if matches!(
        std::env::var("CANDLE_DEBUG_MXFP4_SHAPES").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    ) {
        if expert_idx == 0 {
            eprintln!(
                "[MXFP4] {}: blocks {:?}, scales {:?} (expect (E={}, out={}, nb={}, 16); (E={}, out={}, nb={}))",
                base,
                blocks_g.dims(),
                scales_g.dims(),
                n_experts,
                out_dim,
                nblocks,
                n_experts,
                out_dim,
                nblocks
            );
        }
    }
    let mut blocks = blocks_g.narrow(0, expert_idx, 1)?.squeeze(0)?; // (out, nb, 16)
    let mut scales = scales_g.narrow(0, expert_idx, 1)?.squeeze(0)?; // (out, nb)

    if !blocks.device().same_device(vb.device()) {
        blocks = blocks.to_device(vb.device())?;
    }
    if !scales.device().same_device(vb.device()) {
        scales = scales.to_device(vb.device())?;
    }
    let blocks = blocks.contiguous()?;
    let scales = scales.contiguous()?;

    // Debug: Print raw block bytes for expert 3, row 0, block 0
    if matches!(std::env::var("CANDLE_DUMP_L1").ok().as_deref(), Some("1"))
        && expert_idx == 3
        && base.contains("gate_up")
    {
        eprintln!(
            "[MXFP4_RAW] blocks shape before dequant: {:?}, device: {:?}",
            blocks.dims(),
            blocks.device()
        );
        eprintln!(
            "[MXFP4_RAW] scales shape before dequant: {:?}, device: {:?}",
            scales.dims(),
            scales.device()
        );
        let blocks_u8 = blocks.to_vec3::<u8>()?;
        let scales_u8 = scales.to_vec2::<u8>()?;
        if blocks_u8.len() > 0 && blocks_u8[0].len() > 0 {
            eprintln!(
                "[MXFP4_RAW] Expert 3 row 0 block 0 bytes: {:?}",
                &blocks_u8[0][0]
            );
            eprintln!(
                "[MXFP4_RAW] Expert 3 row 0 block 0 scale: {}",
                scales_u8[0][0]
            );
        }
    }

    if matches!(std::env::var("CANDLE_DUMP_L1").ok().as_deref(), Some("1"))
        && expert_idx == 3
        && base.contains("gate_up")
    {
        let weight_debug = dequantize_mxfp4_linear(&blocks, &scales, out_dim, in_dim)?;
        eprintln!(
            "[MXFP4] Expert 3 gate_up_proj weight shape after dequant: {:?}",
            weight_debug.dims()
        );
        eprintln!(
            "[MXFP4] Expected: ({}, {}) [out_dim, in_dim] for candle::Linear",
            out_dim, in_dim
        );

        let w_f32 = weight_debug.to_dtype(DType::F32)?;
        let w_vec = w_f32.to_vec2::<f32>()?;
        if w_vec.len() >= 2 {
            eprintln!(
                "[MXFP4] Expert 3 row 0 [:32]: {:?}",
                &w_vec[0][..32.min(w_vec[0].len())]
            );
            eprintln!(
                "[MXFP4] Expert 3 row 1 [:32]: {:?}",
                &w_vec[1][..32.min(w_vec[1].len())]
            );
        }
    }

    // Optional grouped bias under e.g. "experts.gate_up_proj_bias" (shape [E, out]).
    let bias_t = if bias {
        let bias_us = format!("{base}_bias");
        let bias_dot = format!("{base}.bias");
        let vb_bf16 = vb.to_dtype(DType::BF16);
        if vb.contains_tensor(&bias_us) {
            Some(
                vb_bf16
                    .get((n_experts, out_dim), &bias_us)?
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?,
            )
        } else if vb.contains_tensor(&bias_dot) {
            Some(
                vb_bf16
                    .get((n_experts, out_dim), &bias_dot)?
                    .narrow(0, expert_idx, 1)?
                    .squeeze(0)?,
            )
        } else {
            None
        }
    } else {
        None
    };

    if matches!(std::env::var("CANDLE_DUMP_L1").ok().as_deref(), Some("1")) && expert_idx == 3 {
        if let Some(ref b) = bias_t {
            let b_f32 = b.to_dtype(DType::F32)?;
            let b_vec = b_f32.to_vec1::<f32>()?;
            eprintln!(
                "[BIAS] {} expert 3 bias [:8]: {:?}",
                base,
                &b_vec[..8.min(b_vec.len())]
            );
        } else {
            eprintln!("[BIAS] {} expert 3: NO BIAS LOADED", base);
        }
    }

    Ok(Linear::from_mxfp4(blocks, scales, in_dim, out_dim, bias_t))
}

/// Load ALL experts' Linear layers at once from grouped MXFP4 tensors, dequantizing in a single
/// batched operation. This is much faster than calling `load_expert_linear_mxfp4_grouped` E times
/// because it resolves names once and launches the dequantization kernel once instead of E times.
///
/// Returns a Vec of Linear layers, one per expert.
pub fn load_all_experts_linear_mxfp4_grouped(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: candle_nn::VarBuilder,
    base: &str,
    n_experts: usize,
) -> Result<Vec<Linear>> {
    // Resolve paired grouped names ONCE (not E times)
    let blocks_dot = format!("{base}_blocks");
    let scales_dot = format!("{base}_scales");
    let has_us = vb.contains_tensor(&blocks_dot) && vb.contains_tensor(&scales_dot);
    let (blocks_name, scales_name) = if has_us {
        (blocks_dot, scales_dot)
    } else {
        let blocks_alt = format!("{base}.blocks");
        let scales_alt = format!("{base}.scales");
        if vb.contains_tensor(&blocks_alt) && vb.contains_tensor(&scales_alt) {
            (blocks_alt, scales_alt)
        } else {
            candle::bail!("grouped MXFP4 tensors not found for base '{base}'")
        }
    };

    if in_dim % MXFP4_BLOCK_ELEMS != 0 {
        candle::bail!(
            "MXFP4 grouped weight '{base}': in_dim must be multiple of {MXFP4_BLOCK_ELEMS}, got {in_dim}"
        )
    }
    let nblocks = in_dim / MXFP4_BLOCK_ELEMS;

    // Load U8 grouped tensors ONCE
    let vb_u8 = vb.to_dtype(DType::U8);
    let mut blocks_g = vb_u8.get(
        (n_experts, out_dim, nblocks, MXFP4_BLOCK_BYTES),
        &blocks_name,
    )?; // (E, out, nb, 16)
    let mut scales_g = vb_u8.get((n_experts, out_dim, nblocks), &scales_name)?; // (E, out, nb)
    if !blocks_g.device().same_device(vb.device()) {
        blocks_g = blocks_g.to_device(vb.device())?;
    }
    if !scales_g.device().same_device(vb.device()) {
        scales_g = scales_g.to_device(vb.device())?;
    }
    let blocks_g = blocks_g.contiguous()?;
    let scales_g = scales_g.contiguous()?;

    // Load optional grouped bias ONCE if present
    let bias_all = if bias {
        let bias_us = format!("{base}_bias");
        let bias_dot = format!("{base}.bias");
        let vb_bf16 = vb.to_dtype(DType::BF16);
        if vb.contains_tensor(&bias_us) {
            Some(vb_bf16.get((n_experts, out_dim), &bias_us)?)
        } else if vb.contains_tensor(&bias_dot) {
            Some(vb_bf16.get((n_experts, out_dim), &bias_dot)?)
        } else {
            None
        }
    } else {
        None
    };

    // Slice into per-expert Linear layers
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let bias_t = if let Some(ref b) = bias_all {
            Some(b.narrow(0, e, 1)?.squeeze(0)?)
        } else {
            None
        };
        let blocks_e = blocks_g.narrow(0, e, 1)?.squeeze(0)?.contiguous()?;
        let scales_e = scales_g.narrow(0, e, 1)?.squeeze(0)?.contiguous()?;
        experts.push(Linear::from_mxfp4(
            blocks_e, scales_e, in_dim, out_dim, bias_t,
        ));
    }

    Ok(experts)
}
