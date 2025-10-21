use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use candle_transformers::models::gpt_oss::experts::{ExpertMlp, GptOssExperts};

// Reference implementation: mirror the original CPU-host routing logic to check parity.
fn reference_forward(experts: &GptOssExperts, xs: &Tensor) -> Result<Tensor> {
    let (b, t, h) = xs.dims3()?;
    let xs2 = xs.reshape(((), h))?;
    let logits = xs2.apply(&experts.router)?;
    let (topk_idx, probs) = experts.router_topk_softmax(&logits)?;

    let probs = probs.to_dtype(DType::F32)?;
    let probs_host = probs.to_vec2::<f32>()?;
    let idx_host = topk_idx.to_vec2::<u32>()?;

    let n_experts = experts.experts.len();
    let mut token_ids: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
    let mut token_wts: Vec<Vec<f32>> = vec![Vec::new(); n_experts];
    for (row, (row_probs, row_experts)) in probs_host.iter().zip(idx_host.iter()).enumerate() {
        for (&p, &e) in row_probs.iter().zip(row_experts.iter()) {
            token_ids[e as usize].push(row as u32);
            token_wts[e as usize].push(p);
        }
    }

    let mut ys = xs2.zeros_like()?;
    for (e_idx, expert) in experts.experts.iter().enumerate() {
        let ids = &token_ids[e_idx];
        if ids.is_empty() {
            continue;
        }
        let ids_t = Tensor::new(ids.as_slice(), xs2.device())?;
        let wts_t = Tensor::new(token_wts[e_idx].as_slice(), xs2.device())?
            .reshape(((), 1))?
            .to_dtype(xs2.dtype())?;
        let x_sel = xs2.index_select(&ids_t, 0)?;
        let y_sel = expert.forward(&x_sel)?;
        let y_sel = y_sel.broadcast_mul(&wts_t)?;
        ys = ys.index_add(&ids_t, &y_sel, 0)?;
    }
    ys.reshape((b, t, h))
}

#[test]
fn t60_moe_routing_gpu_matches_reference() -> Result<()> {
    // Deterministic seed via simple arithmetic sequence.
    let seed = 42u64;
    let dev = Device::new_cuda(0)?;

    // Tiny dimensions to keep test fast.
    let batch = 1usize;
    let tokens = 3usize;
    let hidden = 4usize;
    let inter = 4usize;
    let n_experts = 4usize;
    let top_k = 2usize;

    // Helper to generate deterministic values.
    let gen_val =
        |idx: usize| -> f32 { ((idx as u64 * 1664525 + seed) % 9973) as f32 * 1e-3 - 4.0 };

    // Router weights/bias.
    let mut router_w = Vec::with_capacity(n_experts * hidden);
    let mut router_b = Vec::with_capacity(n_experts);
    for out in 0..n_experts {
        for inn in 0..hidden {
            router_w.push(gen_val(out * hidden + inn));
        }
        router_b.push(gen_val(10_000 + out));
    }
    let router = Linear::new(
        Tensor::from_vec(router_w, (n_experts, hidden), &dev)?,
        Some(Tensor::from_vec(router_b, (n_experts,), &dev)?),
    );

    // Experts with simple, distinct weights.
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let mut gate_w = Vec::with_capacity(2 * inter * hidden);
        let mut gate_b = Vec::with_capacity(2 * inter);
        let mut down_w = Vec::with_capacity(hidden * inter);
        for i in 0..(2 * inter * hidden) {
            gate_w.push(gen_val(20_000 + e * 1_000 + i));
        }
        for i in 0..(2 * inter) {
            gate_b.push(gen_val(30_000 + e * 1_000 + i));
        }
        for i in 0..(hidden * inter) {
            down_w.push(gen_val(40_000 + e * 1_000 + i));
        }
        let gate_up = Linear::new(
            Tensor::from_vec(gate_w, (2 * inter, hidden), &dev)?,
            Some(Tensor::from_vec(gate_b, (2 * inter,), &dev)?),
        );
        let down = Linear::new(Tensor::from_vec(down_w, (hidden, inter), &dev)?, None);
        let limit = 7.0f32;
        let alpha = 1.702f32;
        experts.push(ExpertMlp::new(gate_up, down, limit, alpha));
    }

    let moe = GptOssExperts::new(router, experts, Some(top_k));

    // Input activations.
    let mut xs_data = Vec::with_capacity(batch * tokens * hidden);
    for i in 0..(batch * tokens * hidden) {
        xs_data.push(gen_val(100_000 + i));
    }
    let xs = Tensor::from_vec(xs_data, (batch, tokens, hidden), &dev)?;

    let gpu_out = moe.forward(&xs)?;
    let ref_out = reference_forward(&moe, &xs)?;

    // Diagnostics
    eprintln!("seed: {}", seed);
    eprintln!(
        "xs: dtype={:?}, device={:?}, shape={:?}, stride={:?}",
        xs.dtype(),
        xs.device(),
        xs.dims(),
        xs.stride()
    );
    eprintln!(
        "gpu_out: dtype={:?}, device={:?}, shape={:?}, stride={:?}",
        gpu_out.dtype(),
        gpu_out.device(),
        gpu_out.dims(),
        gpu_out.stride()
    );

    let gpu_vec = gpu_out.to_vec3::<f32>()?;
    let ref_vec = ref_out.to_vec3::<f32>()?;

    let mut linf = 0f32;
    let mut l2 = 0f32;
    for b in 0..batch {
        for t in 0..tokens {
            for h in 0..hidden {
                let d = ref_vec[b][t][h] - gpu_vec[b][t][h];
                linf = linf.max(d.abs());
                l2 += d * d;
            }
        }
    }
    l2 = l2.sqrt();

    eprintln!("L_inf={:.3e} L2={:.3e}", linf, l2);

    // Tight tolerance: both paths are exact same math in FP32.
    assert!(linf <= 5e-6, "max diff too large: {linf}");
    assert!(l2 <= 5e-6, "L2 diff too large: {l2}");

    Ok(())
}
