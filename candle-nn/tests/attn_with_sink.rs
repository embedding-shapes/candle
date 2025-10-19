use candle::{Device, Result, Tensor, DType};

// T11 Eager sink correctness: Construct small q/k/v and a sink vector; confirm eager-with-sink
// output equals a baseline that explicitly concatenates sink to logits then drops it.
#[test]
fn eager_sink_correctness() -> Result<()> {
    let dev = Device::Cpu;

    // Shapes: (bh=2, tq=3, dk=4), (bh=2, tk=5, dk=4), (bh=2, tk=5, dv=6)
    let q = (Tensor::arange(0f32, 2.0 * 3.0 * 4.0, &dev)?.reshape((2, 3, 4))? / 23.0)?;
    let k = (Tensor::arange(0f32, 2.0 * 5.0 * 4.0, &dev)?.reshape((2, 5, 4))? / 17.0)?;
    let v = (Tensor::arange(0f32, 2.0 * 5.0 * 6.0, &dev)?.reshape((2, 5, 6))? / 11.0)?;

    // Per (bh, tq) sink logits.
    let sink = (Tensor::arange(0f32, 2.0 * 3.0, &dev)?.reshape((2, 3))? / 7.0)?;

    let dk: f64 = 4.0;
    let scale: f32 = (dk.sqrt().recip()) as f32;

    // Baseline: explicit concat of sink logit then softmax and drop sink column.
    let baseline = {
        let scores = (q.clone() * scale as f64)?.matmul(&k.clone().t()?)?; // (bh, tq, tk)
        let scores_with_sink = Tensor::cat(&[&scores, &sink.unsqueeze(2)?], 2)?; // (bh, tq, tk+1)
        let probs_with_sink = candle_nn::ops::softmax_last_dim(&scores_with_sink.to_dtype(DType::F32)?)?;
        let probs = probs_with_sink.narrow(2, 0, k.dim(1)?)?.to_dtype(q.dtype())?;
        probs.matmul(&v)
    }?;

    // Implementation under test.
    let out = candle_nn::ops::attention_with_sink(&q, &k, &v, scale, 1.0, &sink)?;

    // Compare numerically.
    let num = (&baseline - &out)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    let den = baseline.abs()?.sum_all()?.to_scalar::<f32>()?;
    let rel = if den == 0.0 { num } else { num / den };
    assert!(rel < 1e-6, "relative error {} too large", rel);

    Ok(())
}
