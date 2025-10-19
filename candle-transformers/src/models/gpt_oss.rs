//! GPT-OSS specific helpers.
//!
//! This module implements the MXFP4 expert weight load-time detection and dequantization path.
//! It detects paired `*.blocks` and `*.scales` tensors under a VarBuilder path and reconstructs
//! a BF16 weight suitable for Candle's `Linear` layer. Other tensors remain BF16 as-is.

use candle::{DType, Result, Tensor, D};
use candle_nn::Linear;

// Submodule(s)
pub mod rotary;

// Constants/config
const MXFP4_BLOCK_ELEMS: usize = 32; // k=32 elements per block on the last dim.
const MXFP4_BLOCK_BYTES: usize = 16; // packed two nibbles per byte.

/// Try to resolve the MXFP4 pair names for a given base, supporting both dot and underscore
/// variants ("base.blocks"/"base.scales" or "base_blocks"/"base_scales"). Returns the fully
/// qualified names including any VarBuilder path.
fn detect_mxfp4_pair_names(vb: &candle_nn::VarBuilder, base: &str) -> Option<(String, String)> {
    // 1) Dot-separated variant
    let blocks_dot = format!("{base}.blocks");
    let scales_dot = format!("{base}.scales");
    if vb.contains_tensor(&blocks_dot) && vb.contains_tensor(&scales_dot) {
        return Some((blocks_dot, scales_dot));
    }
    // 2) Underscore-joined variant (helps with simplified unit-test backends)
    let blocks_us = format!("{base}_blocks");
    let scales_us = format!("{base}_scales");
    if vb.contains_tensor(&blocks_us) && vb.contains_tensor(&scales_us) {
        return Some((blocks_us, scales_us));
    }
    None
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
    if let Some((blocks_name, scales_name)) = detect_mxfp4_pair_names(&vb, base) {
        // Validate shape alignment to block size.
        if in_dim % MXFP4_BLOCK_ELEMS != 0 {
            candle::bail!(
                "MXFP4 weight '{base}': in_dim must be multiple of {MXFP4_BLOCK_ELEMS}, got {in_dim}"
            )
        }
        let nblocks = in_dim / MXFP4_BLOCK_ELEMS;

        // Load blocks and scales as U8 tensors on the VarBuilder device.
        let vb_u8 = vb.to_dtype(DType::U8);
        let blocks = vb_u8.get((out_dim, nblocks, MXFP4_BLOCK_BYTES), &blocks_name)?;
        let scales = vb_u8.get((out_dim, nblocks), &scales_name)?;

        // Dequantize to BF16 on the appropriate device (CPU or CUDA).
        let weight = candle::mxfp4::dequant_mxfp4_to_bf16(&blocks, &scales, [out_dim, in_dim])?;

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

        return Ok(Linear::new(weight, bias_t));
    }

    // Fallback: standard BF16 linear loading under "{base}.weight" and optional bias.
    let vb_bf16 = vb.to_dtype(DType::BF16).pp(base);
    candle_nn::linear_b(in_dim, out_dim, bias, vb_bf16)
}

// ============================
// Attention with sinks helpers
// ============================

/// Compute renormalization scale per (b,h,q) given FA LSE and head-wise sinks.
/// scale = exp(lse - logsumexp([lse, sink])) = 1 / (1 + exp(sink - lse))
fn sinks_scale_from_lse(lse: &Tensor, sinks: &Tensor) -> Result<Tensor> {
    // lse: (b, h, q) in f32 or bf16/f16, convert to f32 for stability
    let lse_f32 = lse.to_dtype(DType::F32)?;
    // sinks: per-head values; accept shape (h) or broadcastable to (b, h, q)
    // Convert to f32 and broadcast.
    let sinks_f32 = sinks.to_dtype(DType::F32)?;
    let (b, h, q) = lse_f32.dims3()?;
    // Expand sinks to (b, h, q)
    let sinks_bhq = sinks_f32
        .reshape((1, h, 1))?
        .broadcast_as((b, h, q))?;

    // combined_lse = logsumexp([lse, sinks], axis=-1 of the concat)
    let lse_exp = lse_f32.unsqueeze(D::Minus1)?; // (b,h,q,1)
    let sinks_exp = sinks_bhq.unsqueeze(D::Minus1)?; // (b,h,q,1)
    let cat = Tensor::cat(&[&lse_exp, &sinks_exp], D::Minus1)?; // (b,h,q,2)
    let combined_lse = cat.log_sum_exp(D::Minus1)?; // (b,h,q)
    let scale = (lse_f32 - &combined_lse)?.exp()?; // (b,h,q)
    Ok(scale)
}

/// Eager attention with optional sinks scaling (causal or not).
/// Inputs are in FA layout: (b, seq_len_q, n_heads, head_dim) etc.
pub fn eager_attn_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
    sinks: Option<&Tensor>, // per-head values (h)
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let (b, qlen, h, _d) = q.dims4()?;
    let (_, klen, _, _) = k.dims4()?;

    // Convert to f32 for stability.
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;

    // (b,h,q,d) @ (b,h,d,k) -> (b,h,q,k)
    let q_bhqd = q.transpose(1, 2)?;
    let k_bhkd = k.transpose(1, 2)?;
    let v_bhkd = v.transpose(1, 2)?;
    let logits = (
        q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * softmax_scale as f64
    )?;

    // Apply causal mask if requested.
    let logits = if causal && qlen > 1 {
        // mask shape (q,k) with j > i masked
        let mask: Vec<u8> = (0..qlen)
            .flat_map(|i| (0..klen).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (qlen, klen), q.device())?;
        masked_fill(&logits, &mask.broadcast_as((b, h, qlen, klen))?, f32::NEG_INFINITY)?
    } else {
        logits
    };

    let att = candle_nn::ops::softmax_last_dim(&logits)?; // (b,h,q,k)
    let o_bhqd = att.matmul(&v_bhkd.contiguous()?)?; // (b,h,q,d)
    let mut o_bqhd = o_bhqd.transpose(1, 2)?; // (b,q,h,d)

    // Apply sinks renormalization if provided.
    if let Some(sinks_t) = sinks {
        // lse along keys
        let lse = logits.log_sum_exp(D::Minus1)?; // (b,h,q)
        let scale = sinks_scale_from_lse(&lse, sinks_t)?; // (b,h,q)
        let scale = scale.transpose(1, 2)?.unsqueeze(D::Minus1)?; // (b,q,h,1)
        let (_, qlen, h, d) = o_bqhd.dims4()?;
        let scale = scale.broadcast_as((b, qlen, h, d))?; // match (b,q,h,d)
        o_bqhd = (o_bqhd.to_dtype(DType::F32)? * &scale)?.to_dtype(in_dtype)?;
    } else {
        o_bqhd = o_bqhd.to_dtype(in_dtype)?;
    }

    Ok(o_bqhd)
}

/// Eager attention with windowed masking and optional sinks scaling.
pub fn eager_attn_windowed_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let (b, qlen, h, _d) = q.dims4()?;
    let (_, klen, _, _) = k.dims4()?;
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let q_bhqd = q.transpose(1, 2)?;
    let k_bhkd = k.transpose(1, 2)?;
    let v_bhkd = v.transpose(1, 2)?;
    let logits = (
        q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * softmax_scale as f64
    )?; // (b,h,q,k)

    // Build windowed mask if needed.
    let logits = match (window_size_left, window_size_right) {
        (None, None) => logits,
        (wsl, wsr) => {
            let left = wsl.unwrap_or(qlen);
            let right = wsr.unwrap_or(klen);
            let mask: Vec<u8> = (0..qlen)
                .flat_map(|i| {
                    (0..klen).map(move |j| {
                        let i = i as isize;
                        let j = j as isize;
                        let l = left as isize;
                        let r = right as isize;
                        let allow = (j >= i - l) && (j <= i + r);
                        u8::from(!allow)
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (qlen, klen), q.device())?;
            masked_fill(&logits, &mask.broadcast_as((b, h, qlen, klen))?, f32::NEG_INFINITY)?
        }
    };

    let att = candle_nn::ops::softmax_last_dim(&logits)?;
    let o_bhqd = att.matmul(&v_bhkd.contiguous()?)?;
    let mut o_bqhd = o_bhqd.transpose(1, 2)?;

    if let Some(sinks_t) = sinks {
        let lse = logits.log_sum_exp(D::Minus1)?; // (b,h,q)
        let scale = sinks_scale_from_lse(&lse, sinks_t)?; // (b,h,q)
        let scale = scale.transpose(1, 2)?.unsqueeze(D::Minus1)?; // (b,q,h,1)
        let (_, qlen, h, d) = o_bqhd.dims4()?;
        let scale = scale.broadcast_as((b, qlen, h, d))?;
        o_bqhd = (o_bqhd.to_dtype(DType::F32)? * &scale)?.to_dtype(in_dtype)?;
    } else {
        o_bqhd = o_bqhd.to_dtype(in_dtype)?;
    }

    Ok(o_bqhd)
}

#[cfg(feature = "flash-attn")]
/// Flash-attn path with sinks renormalization using FA-returned LSE.
pub fn flash_attn_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    match sinks {
        None => candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal),
        Some(s) => {
            let (o, lse) = candle_flash_attn::flash_attn_with_lse(q, k, v, softmax_scale, causal)?;
            let scale = sinks_scale_from_lse(&lse, s)?; // (b,h,q)
            // Broadcast to (b,q,h,1) to match o (b,q,h,d)
            let scale = scale.transpose(1, 2)?.unsqueeze(D::Minus1)?; // (b,q,h,1)
            let (b, qlen, h, d) = o.dims4()?;
            let scale = scale.broadcast_as((b, qlen, h, d))?; // (b,q,h,d)
            let out = (o.to_dtype(DType::F32)? * scale)?.to_dtype(o.dtype())?;
            Ok(out)
        }
    }
}

#[cfg(feature = "flash-attn")]
/// Windowed flash-attn path with sinks renormalization.
pub fn flash_attn_windowed_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    match sinks {
        None => candle_flash_attn::flash_attn_windowed(q, k, v, softmax_scale, window_size_left, window_size_right),
        Some(s) => {
            let (o, lse) = candle_flash_attn::flash_attn_windowed_with_lse(
                q,
                k,
                v,
                softmax_scale,
                window_size_left,
                window_size_right,
            )?;
            let scale = sinks_scale_from_lse(&lse, s)?; // (b,h,q)
            let scale = scale.transpose(1, 2)?.unsqueeze(D::Minus1)?; // (b,q,h,1)
            let (b, qlen, h, d) = o.dims4()?;
            let scale = scale.broadcast_as((b, qlen, h, d))?; // (b,q,h,d)
            let out = (o.to_dtype(DType::F32)? * scale)?.to_dtype(o.dtype())?;
            Ok(out)
        }
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}
