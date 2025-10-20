use candle::{DType, Device, Result, Tensor, D};
use candle_nn::Linear;

// Micro-check: router_topk_softmax selects largest-K per row and softmaxes them.
// Verifies GPU path and numeric parity against a simple Rust reference.
#[test]
fn gpt_oss_router_topk_softmax_largest() -> Result<()> {
    // Use CUDA if available; otherwise, this environment guarantees a GPU.
    let dev = if candle::Device::cuda_if_available(0).is_ok() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    // Construct fixed logits (N=3 tokens, E=6 experts), dtype f32.
    let logits_data: Vec<f32> = vec![
        // row 0
        0.1, -0.2, 0.3, 5.0, -1.2, 0.2,
        // row 1
        -2.0, 1.5, 0.0, -0.5, 3.0, 2.5,
        // row 2
        10.0,  9.9,  -5.0,  0.1,  0.2,  0.3,
    ];
    let n = 3usize;
    let e = 6usize;
    let logits = Tensor::from_slice(&logits_data, (n, e), &dev)?;

    // Build a dummy experts container with top_k=2. Router/expert weights unused here.
    let router_w = Tensor::zeros((e, 1), DType::F32, &dev)?; // out=e, in=1 (unused)
    let router = Linear::new(router_w, None);
    let experts: Vec<candle_transformers::models::gpt_oss::experts::ExpertMlp> = Vec::new();
    let moe = candle_transformers::models::gpt_oss::experts::GptOssExperts::new(router, experts, Some(2));

    // Run the function under test on GPU.
    let (idx_gpu, probs_gpu) = moe.router_topk_softmax(&logits)?; // idx: (n, k), probs: (n, k)

    // Reference: compute largest-2 indices per row and softmax over those values.
    let mut exp_idx: Vec<[usize; 2]> = Vec::with_capacity(n);
    let mut exp_probs: Vec<[f32; 2]> = Vec::with_capacity(n);
    for r in 0..n {
        let row = &logits_data[r * e..(r + 1) * e];
        // Find top-2 indices by value (descending), break ties by lower index for stability.
        let mut pairs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        pairs.sort_by(|(i1, v1), (i2, v2)| v2.partial_cmp(v1).unwrap_or(std::cmp::Ordering::Equal).then(i1.cmp(i2)));
        let i0 = pairs[0].0;
        let i1 = pairs[1].0;
        exp_idx.push([i0, i1]);
        let v0 = pairs[0].1;
        let v1 = pairs[1].1;
        let m = v0.max(v1);
        let z0 = (v0 - m).exp();
        let z1 = (v1 - m).exp();
        let z = z0 + z1;
        exp_probs.push([z0 / z, z1 / z]);
    }

    // Move GPU outputs to CPU f32 for inspection.
    let idx_vec = idx_gpu.to_dtype(DType::U32)?.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
    let probs_vec = probs_gpu.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

    // Logs: dtype/device, shapes, inputs and outputs.
    eprintln!("device={:?}, dtype={:?}", dev, logits.dtype());
    eprintln!("logits shape={:?}", logits.dims());
    eprintln!("logits rows: {:?}", logits_data);
    eprintln!("gpu idx: {:?}", idx_vec);
    eprintln!("gpu probs: {:?}", probs_vec);
    eprintln!("exp idx: {:?}", exp_idx);
    eprintln!("exp probs: {:?}", exp_probs);

    // Compare indices exactly and probabilities numerically.
    for r in 0..n {
        assert_eq!(idx_vec[r][0] as usize, exp_idx[r][0], "row {r} top-1 index mismatch");
        assert_eq!(idx_vec[r][1] as usize, exp_idx[r][1], "row {r} top-2 index mismatch");
        let d0 = (probs_vec[r][0] - exp_probs[r][0]).abs();
        let d1 = (probs_vec[r][1] - exp_probs[r][1]).abs();
        let l_inf = d0.max(d1);
        let l2 = (d0 * d0 + d1 * d1).sqrt();
        eprintln!("row {r} diffs: Linf={:.3e}, L2={:.3e}", l_inf, l2);
        assert!(l_inf < 1e-6, "row {r} Linf too large");
        assert!(l2 < 1e-6, "row {r} L2 too large");
    }

    Ok(())
}

