#![cfg(feature = "flash-attn")]

use candle::{DType, Device, Result, Tensor};

// This test validates GQA head mapping on CUDA in two ways:
// 1) repeat_kv on (b, n_kv, t, d) produces contiguous repeats per KV head.
// 2) flash-attn with (K,V) replicated to n_q heads matches flash-attn with reduced heads,
//    and both reflect the expected grouping kv_head = q_head / n_rep.
#[test]
fn t_gqa_head_mapping_cuda_flashattn() -> Result<()> {
    // Small but realistic geometry: n_q=64, n_kv=8, head_dim=64, b=1, tq=2, tk=3.
    let dev = Device::new_cuda(0)?;
    let b = 1usize;
    let n_q = 64usize;
    let n_kv = 8usize;
    let n_rep = n_q / n_kv; // 8
    let d = 64usize; // multiple of 8 as required by FA
    let tq = 2usize;
    let tk = 3usize;

    // Build Q=0 so attention becomes uniform over time, independent of content.
    let q = Tensor::zeros((b, tq, n_q, d), DType::BF16, &dev)?;

    // Build K=0 as well to keep uniform attention.
    let k_kv = Tensor::zeros((b, tk, n_kv, d), DType::BF16, &dev)?;

    // Build V with a distinctive per-KV-head constant so outputs are easy to check.
    // For KV head h, all values are set to h as BF16.
    let mut v_host = vec![0f32; b * tk * n_kv * d];
    let mut idx = 0;
    for _ in 0..b {
        for _ in 0..tk {
            for h in 0..n_kv {
                for _ in 0..d {
                    v_host[idx] = h as f32;
                    idx += 1;
                }
            }
        }
    }
    let v_kv = Tensor::from_vec(v_host, (b, tk, n_kv, d), &dev)?.to_dtype(DType::BF16)?;

    // Path A: replicate (K,V) to n_q heads explicitly, then run FA.
    let k_bh = k_kv.transpose(1, 2)?; // (b,n_kv,tk,d)
    let v_bh = v_kv.transpose(1, 2)?; // (b,n_kv,tk,d)
    let k_rep = candle_transformers::utils::repeat_kv(k_bh, n_rep)?; // (b,n_q,tk,d)
    let v_rep = candle_transformers::utils::repeat_kv(v_bh, n_rep)?; // (b,n_q,tk,d)
    let k_rep = k_rep.transpose(1, 2)?; // (b,tk,n_q,d)
    let v_rep = v_rep.transpose(1, 2)?; // (b,tk,n_q,d)
    let y_rep = candle_flash_attn::flash_attn(&q, &k_rep, &v_rep, 1.0, false)?; // (b,tq,n_q,d)

    // Path B: rely on FA's internal GQA support with reduced heads in (K,V).
    let y_red = candle_flash_attn::flash_attn(&q, &k_kv, &v_kv, 1.0, false)?; // (b,tq,n_q,d)

    // Compare Path A vs B numerically on device
    let a = y_rep.to_dtype(DType::F32)?;
    let b_out = y_red.to_dtype(DType::F32)?;
    let max_diff = (a - &b_out)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_diff < 1e-2, "flash-attn outputs mismatch (replicated vs reduced): max_diff={}", max_diff);

    // Check grouping semantics on Path B directly: output head h must equal the KV head floor(h/n_rep).
    // With uniform attention over time and V constant per (kv head), the output per (b,t,h,*) should be that constant.
    let head_vals: Vec<f32> = (0..n_q).map(|h_ix| (h_ix / n_rep) as f32).collect();
    let expected = Tensor::from_vec(head_vals, (n_q,), &dev)?
        .reshape((1, 1, n_q, 1))?
        .broadcast_as((b, tq, n_q, d))?
        .to_dtype(DType::F32)?;
    let max_diff2 = (b_out - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_diff2 < 1e-2, "group mapping wrong: max_diff={} (should be <1e-2)", max_diff2);

    Ok(())
}
