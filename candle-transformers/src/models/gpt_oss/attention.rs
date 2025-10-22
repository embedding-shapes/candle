use candle::{DType, Result, Tensor, D};

/// Compute renormalization scale per (b,h,q) given FA LSE and head-wise sinks.
/// scale = exp(lse - logsumexp([lse, sink])) = 1 / (1 + exp(sink - lse))
pub fn sinks_scale_from_lse(lse: &Tensor, sinks: &Tensor) -> Result<Tensor> {
    // lse: (b, h, q) in f32 or bf16/f16, convert to f32 for stability
    let lse_f32 = lse.to_dtype(DType::F32)?;
    // sinks: per-head values; accept shape (h) or broadcastable to (b, h, q)
    // Convert to f32 and broadcast.
    let sinks_f32 = sinks.to_dtype(DType::F32)?;
    let (b, h, q) = lse_f32.dims3()?;
    // Expand sinks to (b, h, q)
    let sinks_bhq = sinks_f32.reshape((1, h, 1))?.broadcast_as((b, h, q))?;

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
    let logits = (q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * softmax_scale as f64)?;

    // Apply causal mask if requested.
    let logits = if causal && qlen > 1 {
        // mask shape (q,k) with j > i masked
        let mask: Vec<u8> = (0..qlen)
            .flat_map(|i| (0..klen).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask, (qlen, klen), q.device())?;
        masked_fill(
            &logits,
            &mask.broadcast_as((b, h, qlen, klen))?,
            f32::NEG_INFINITY,
        )?
    } else {
        logits
    };

    // Apply sinks renormalization if provided - match HF reference exactly.
    let o_bqhd = if let Some(sinks_t) = sinks {
        // Concatenate sinks to logits: combined_logits = cat([attn_weights, sinks], dim=-1)
        // sinks_t shape: (h), reshape to (1, h, 1, 1) and broadcast to (b, h, q, 1)
        let sinks_f32 = sinks_t.to_dtype(DType::F32)?;
        let sinks_bhq1 = sinks_f32
            .reshape((1, h, 1, 1))?
            .broadcast_as((b, h, qlen, 1))?; // (b, h, q, 1)
        let combined_logits = Tensor::cat(&[&logits, &sinks_bhq1], D::Minus1)?; // (b, h, q, k+1)

        // Max normalization: combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        let max_vals = combined_logits.max_keepdim(D::Minus1)?;
        let combined_logits = combined_logits.broadcast_sub(&max_vals)?;

        // Softmax on combined
        let probs = candle_nn::ops::softmax_last_dim(&combined_logits)?; // (b, h, q, k+1)

        // Drop the sink probability: scores = probs[..., :-1]
        let scores = probs.narrow(D::Minus1, 0, klen)?; // (b, h, q, k)

        let o_bhqd = scores.matmul(&v_bhkd.contiguous()?)?; // (b, h, q, d)
        o_bhqd.transpose(1, 2)?.to_dtype(in_dtype)? // (b, q, h, d)
    } else {
        let att = candle_nn::ops::softmax_last_dim(&logits)?; // (b, h, q, k)
        let o_bhqd = att.matmul(&v_bhkd.contiguous()?)?; // (b, h, q, d)
        o_bhqd.transpose(1, 2)?.to_dtype(in_dtype)? // (b, q, h, d)
    };

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
    let logits = (q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * softmax_scale as f64)?; // (b,h,q,k)

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
                        // Golden ref semantics: allow j in [i - left, i + right], i.e. left-inclusive, right-inclusive
                        // Combined with causal (right typically 0), this becomes [i - left, i].
                        let allow = (j >= i - l) && (j <= i + r);
                        u8::from(!allow)
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (qlen, klen), q.device())?;
            masked_fill(
                &logits,
                &mask.broadcast_as((b, h, qlen, klen))?,
                f32::NEG_INFINITY,
            )?
        }
    };

    // Apply sinks renormalization if provided - match HF reference exactly.
    let o_bqhd = if let Some(sinks_t) = sinks {
        // Concatenate sinks to logits: combined_logits = cat([attn_weights, sinks], dim=-1)
        let sinks_f32 = sinks_t.to_dtype(DType::F32)?;
        let sinks_bhq1 = sinks_f32
            .reshape((1, h, 1, 1))?
            .broadcast_as((b, h, qlen, 1))?; // (b, h, q, 1)
        let combined_logits = Tensor::cat(&[&logits, &sinks_bhq1], D::Minus1)?; // (b, h, q, k+1)

        // Max normalization
        let max_vals = combined_logits.max_keepdim(D::Minus1)?;
        let combined_logits = combined_logits.broadcast_sub(&max_vals)?;

        // Softmax on combined
        let probs = candle_nn::ops::softmax_last_dim(&combined_logits)?; // (b, h, q, k+1)

        // Drop the sink probability
        let scores = probs.narrow(D::Minus1, 0, klen)?; // (b, h, q, k)

        let o_bhqd = scores.matmul(&v_bhkd.contiguous()?)?; // (b, h, q, d)
        o_bhqd.transpose(1, 2)?.to_dtype(in_dtype)? // (b, q, h, d)
    } else {
        let att = candle_nn::ops::softmax_last_dim(&logits)?;
        let o_bhqd = att.matmul(&v_bhkd.contiguous()?)?;
        o_bhqd.transpose(1, 2)?.to_dtype(in_dtype)?
    };

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
        None => candle_flash_attn::flash_attn_windowed(
            q,
            k,
            v,
            softmax_scale,
            window_size_left,
            window_size_right,
        ),
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

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}
