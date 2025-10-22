use candle::{DType, Result};
use candle_nn::Linear;

use super::{MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS};

/// Try to resolve the MXFP4 pair names for a given base, supporting both dot and underscore
/// variants ("base.blocks"/"base.scales" or "base_blocks"/"base_scales"). Returns the fully
/// qualified names including any VarBuilder path.
pub fn detect_mxfp4_pair_names(
    vb: &candle_nn::VarBuilder,
    base: &str,
) -> Result<Option<(String, String)>> {
    // 1) Dot-separated variant
    let blocks_dot = format!("{base}.blocks");
    let scales_dot = format!("{base}.scales");
    if vb.contains_tensor(&blocks_dot) && vb.contains_tensor(&scales_dot) {
        return Ok(Some((blocks_dot, scales_dot)));
    }
    if vb.contains_tensor(&blocks_dot) ^ vb.contains_tensor(&scales_dot) {
        if vb.contains_tensor(&blocks_dot) {
            candle::bail!("found '{base}.blocks' but missing '{base}.scales'")
        } else {
            candle::bail!("found '{base}.scales' but missing '{base}.blocks'")
        }
    }
    // 2) Underscore-joined variant (helps with simplified unit-test backends)
    let blocks_us = format!("{base}_blocks");
    let scales_us = format!("{base}_scales");
    if vb.contains_tensor(&blocks_us) && vb.contains_tensor(&scales_us) {
        return Ok(Some((blocks_us, scales_us)));
    }
    if vb.contains_tensor(&blocks_us) ^ vb.contains_tensor(&scales_us) {
        if vb.contains_tensor(&blocks_us) {
            candle::bail!("found '{base}_blocks' but missing '{base}_scales'")
        } else {
            candle::bail!("found '{base}_scales' but missing '{base}_blocks'")
        }
    }
    Ok(None)
}

/// Load a Linear layer weight for base `base` where the weight is either stored as BF16
/// under `"{base}.weight"` (fallback) or as MXFP4 experts using paired `blocks`/`scales`.
///
/// - `in_dim`: input dimension of the linear layer
/// - `out_dim`: output dimension of the linear layer
/// - `bias`: whether to load a bias if present
/// - `vb`: VarBuilder pointing at the module path containing the base
///
/// Returns a `Linear` with BF16 weights (and bias if requested and present).
pub fn load_linear_maybe_mxfp4(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: candle_nn::VarBuilder,
    base: &str,
) -> Result<Linear> {
    // Detect paired MXFP4 tensors.
    if let Some((blocks_name, scales_name)) = detect_mxfp4_pair_names(&vb, base)? {
        // Validate shape alignment to block size.
        if in_dim % MXFP4_BLOCK_ELEMS != 0 {
            candle::bail!(
                "MXFP4 weight '{base}': in_dim must be multiple of {MXFP4_BLOCK_ELEMS}, got {in_dim}"
            )
        }
        let nblocks = in_dim / MXFP4_BLOCK_ELEMS;

        // Load blocks and scales as U8 tensors on the VarBuilder device.
        let vb_u8 = vb.to_dtype(DType::U8);
        let mut blocks = vb_u8.get((out_dim, nblocks, MXFP4_BLOCK_BYTES), &blocks_name)?;
        let mut scales = vb_u8.get((out_dim, nblocks), &scales_name)?;
        if !blocks.device().same_device(vb.device()) {
            blocks = blocks.to_device(vb.device())?;
        }
        if !scales.device().same_device(vb.device()) {
            scales = scales.to_device(vb.device())?;
        }
        let blocks = blocks.contiguous()?;
        let scales = scales.contiguous()?;
        // Optionally load bias (kept BF16 as-is). Support dot and underscore naming.
        let bias_t = if bias {
            let bias_dot = format!("{base}.bias");
            let bias_us = format!("{base}_bias");
            let vb_bf16 = vb.to_dtype(DType::BF16);
            if vb.contains_tensor(&bias_dot) {
                Some(vb_bf16.get(out_dim, &bias_dot)?)
            } else if vb.contains_tensor(&bias_us) {
                Some(vb_bf16.get(out_dim, &bias_us)?)
            } else {
                None
            }
        } else {
            None
        };

        return Ok(Linear::from_mxfp4(blocks, scales, in_dim, out_dim, bias_t));
    }

    // Fallback: standard BF16 linear loading under "{base}.weight" and optional bias.
    let vb_bf16 = vb.to_dtype(DType::BF16).pp(base);
    candle_nn::linear_b(in_dim, out_dim, bias, vb_bf16)
}
