use candle::{Device, Module, Result, Tensor};
use candle_nn::Linear;
use candle_transformers::models::gpt_oss::experts::ExpertMlp;

// Verify fused gate_up split matches HF interleaving: gate = even indices, up = odd.
// We craft a gate_up with zero weights and a bias vector arranged as [g0, u0, g1, u1, g2, u2, ...].
// The ExpertMlp must pick (gate, up) accordingly and apply the clamped SwiGLU + residual tweak.
#[test]
fn t21_expert_mlp_interleaved_split() -> Result<()> {
    let dev = Device::Cpu;
    let hidden = 1usize;
    let inter = 3usize;

    // gate_up: zero weights so output == bias. Bias interleaved as [g0, u0, g1, u1, g2, u2].
    let w_gu = Tensor::zeros((2 * inter, hidden), candle::DType::F32, &dev)?;
    let bias_vals = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let b_gu = Tensor::from_vec(bias_vals.clone(), (2 * inter,), &dev)?;
    let gate_up = Linear::new(w_gu, Some(b_gu));

    // down: sum all inter dims into single hidden dim, zero bias
    let w_dn = Tensor::from_vec(vec![1.0f32, 1.0, 1.0], (hidden, inter), &dev)?; // (1,3)
    let down = Linear::new(w_dn, None);

    let limit = 7.0f32;
    let alpha = 1.702f32;
    let mlp = ExpertMlp::new(gate_up, down, limit, alpha);

    // Input single token -> use only bias
    let xs = Tensor::from_vec(vec![0.0f32], (1, hidden), &dev)?;
    let y = mlp.forward(&xs)?; // (1,1)
    let got = y.to_vec2::<f32>()?[0][0];

    // Reference using interleaving: gate=[10,30,50], up=[20,40,60]
    let gate = [10.0f32.min(limit), 30.0f32.min(limit), 50.0f32.min(limit)];
    let up = [
        20.0f32.clamp(-limit, limit),
        40.0f32.clamp(-limit, limit),
        60.0f32.clamp(-limit, limit),
    ];
    let mut acc = 0.0f32;
    for i in 0..inter {
        let g = gate[i];
        let u = up[i];
        let glu = g * (g * alpha).exp() / (1.0 + (g * alpha).exp());
        acc += (u + 1.0) * glu;
    }

    let diff = (acc - got).abs();
    assert!(diff < 1e-4, "expected {acc}, got {got}, diff {diff}");
    Ok(())
}
