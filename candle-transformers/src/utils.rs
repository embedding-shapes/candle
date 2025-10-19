//! Apply penalty and repeat_kv

use candle::{Result, Tensor};

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Repeat K/V heads for grouped-query attention (GQA).
///
/// Input shape is `(batch, num_kv_heads, seq_len, head_dim)` and the output
/// is `(batch, num_kv_heads * n_rep, seq_len, head_dim)` such that each KV head
/// is repeated contiguously `n_rep` times before moving to the next KV head.
///
/// This ensures the head mapping `kv_head_idx = q_head_idx / n_rep` holds, i.e.
/// Q heads `[0..n_rep-1] -> KV head 0`, `[n_rep..2*n_rep-1] -> KV head 1`, etc.
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, n_kv, t, d) = xs.dims4()?;
    // Introduce a replication axis after the head axis, broadcast, then
    // collapse (n_kv, n_rep) -> (n_kv * n_rep). Broadcasting preserves the
    // per-head grouping order required by GQA.
    let xs = xs.unsqueeze(2)?; // (b, n_kv, 1, t, d)
    let xs = xs.broadcast_as((b, n_kv, n_rep, t, d))?; // (b, n_kv, n_rep, t, d)
    xs.reshape((b, n_kv * n_rep, t, d)) // (b, n_kv*n_rep, t, d)
}
