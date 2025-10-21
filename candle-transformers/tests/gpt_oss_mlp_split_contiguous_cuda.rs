use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use candle_transformers::models::gpt_oss::experts::ExpertMlp;

// Micro-check: fused gate_up split must use contiguous halves (gate, up)
// Run on CUDA with tiny shapes and deterministic values.
#[test]
fn t50_expert_mlp_split_contiguous_cuda() -> Result<()> {
    // Use CUDA:0 explicitly.
    let dev = Device::new_cuda(0)?;

    // Shapes: hidden = inter = 4 so we can use identity down-proj to observe fused result directly.
    let hidden = 4usize;
    let inter = 4usize;

    // gate_up: zero weights so output == bias. Bias arranged as contiguous halves [gate..., up...].
    // Choose small values to avoid saturation; set limit large and alpha=1 to simplify expected math.
    let gate_vals: [f32; 4] = [0.1, -0.2, 0.3, -0.4];
    let up_vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut bias = Vec::with_capacity(2 * inter);
    bias.extend_from_slice(&gate_vals);
    bias.extend_from_slice(&up_vals);

    let w_gu = Tensor::zeros((2 * inter, hidden), DType::F32, &dev)?; // (8,4)
    let b_gu = Tensor::from_vec(bias.clone(), (2 * inter,), &dev)?; // (8)
    let gate_up = Linear::new(w_gu, Some(b_gu));

    // down: identity (4x4) so final y == fused result
    let mut eye = vec![0f32; hidden * inter];
    for i in 0..hidden {
        eye[i * inter + i] = 1.0;
    }
    let w_dn = Tensor::from_vec(eye, (hidden, inter), &dev)?; // (4,4)
    let down = Linear::new(w_dn, None);

    let limit = 1000.0f32; // effectively disable clamp
    let alpha = 1.0f32; // simplify expected
    let mlp = ExpertMlp::new(gate_up, down, limit, alpha);

    // Input single token -> gate_up output equals bias on CUDA
    let xs = Tensor::zeros((1, hidden), DType::F32, &dev)?; // (1,4)
    let y = mlp.forward(&xs)?; // (1,4)

    // Expected per spec: fused = (up + 1) * gate * sigmoid(gate * alpha)
    let mut expected = vec![0f32; inter];
    for i in 0..inter {
        let g = gate_vals[i];
        let u = up_vals[i];
        let sig = 1.0f32 / (1.0 + (-(g * alpha)).exp());
        expected[i] = (u + 1.0) * g * sig;
    }

    // Fetch actual GPU result
    let got = y.to_vec2::<f32>()?; // (1,4)
    let got = &got[0];

    // Diagnostics: dtypes, devices, shapes, strides
    eprintln!("seed: fixed (deterministic constants)");
    eprintln!(
        "xs: dtype={:?}, device={:?}, shape={:?}, stride={:?}",
        xs.dtype(),
        xs.device(),
        xs.dims(),
        xs.stride()
    );
    eprintln!(
        "gate_up.bias (concat halves) checksum={:.6}",
        bias.iter().copied().sum::<f32>()
    );
    eprintln!(
        "y: dtype={:?}, device={:?}, shape={:?}, stride={:?}",
        y.dtype(),
        y.device(),
        y.dims(),
        y.stride()
    );
    eprintln!("expected: {:?}", expected);
    eprintln!("actual:   {:?}", got);

    // Metrics
    let mut linf = 0f32;
    let mut l2 = 0f32;
    for i in 0..inter {
        let d = expected[i] - got[i];
        linf = linf.max(d.abs());
        l2 += d * d;
    }
    l2 = l2.sqrt();
    eprintln!("errors: L_inf={:.3e}, L2={:.3e}", linf, l2);

    // Tolerances: exact FP32 math on small values ⇒ expect <= 1e-6
    assert!(linf <= 1e-6, "split/parity mismatch: L_inf={linf} > 1e-6");
    assert!(l2 <= 1e-6, "split/parity mismatch: L2={l2} > 1e-6");

    Ok(())
}
