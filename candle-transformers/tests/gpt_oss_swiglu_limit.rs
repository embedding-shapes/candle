use candle::{Device, Module, Result, Tensor};
use candle_nn::Linear;
use candle_transformers::models::gpt_oss::experts::ExpertMlp;

// T20: ExpertMlp applies GPT-OSS clamped SwiGLU with residual tweak correctly
#[test]
fn t20_expert_mlp_swiglu_clamp_and_residual() -> Result<()> {
    let dev = Device::Cpu;
    let hidden = 1usize;
    let inter = 1usize;

    // gate_up: weight zeros, bias sets gate=+100, up=+100 → both should clamp to +/- limit
    let w_gu = Tensor::zeros((2 * inter, hidden), candle::DType::F32, &dev)?;
    let b_gu = Tensor::from_vec(vec![100.0f32, 100.0f32], (2 * inter,), &dev)?; // [gate_bias, up_bias]
    let gate_up = Linear::new(w_gu.clone(), Some(b_gu.clone()));

    // down: identity (1x1), zero bias
    let w_dn = Tensor::from_vec(vec![1.0f32], (hidden, inter), &dev)?;
    let down = Linear::new(w_dn.clone(), None);

    let limit = 7.0f32;
    let alpha = 1.702f32;
    let mlp = ExpertMlp::new(gate_up, down, limit, alpha);

    // Input single token
    let xs = Tensor::from_vec(vec![0.0f32], (1, hidden), &dev)?;
    let y = mlp.forward(&xs)?; // (1, hidden)
    let got = y.to_vec2::<f32>()?[0][0];

    // Expected: (up+1) * gate * sigmoid(gate*alpha) with gate=clamp(100, <=limit)=limit, up=clamp(100, [-limit,limit])=limit
    let gate = limit;
    let up = limit;
    let expected = (up + 1.0) * gate * (gate * alpha).exp() / (1.0 + (gate * alpha).exp());
    let diff = (expected - got).abs();
    assert!(diff < 1e-3, "expected {expected}, got {got}, diff {diff}");
    Ok(())
}
