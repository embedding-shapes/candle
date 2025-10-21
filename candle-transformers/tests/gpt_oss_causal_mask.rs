use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::gpt_oss::eager_attn_with_sinks;

// T19: Eager causal mask correctness (triangular)
// Build a simple case where logits are uniform (q,k zeros),
// so softmax is uniform over allowed keys. Set v to encode key index
// so the output equals the mean of allowed key indices. This checks
// the mask is applied post QK^T and with correct shape.
#[test]
fn gpt_oss_eager_causal_mask_triangular() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    if !device.is_cuda() {
        // Still exercise on CPU if CUDA is unavailable.
    }

    let b = 1usize;
    let h = 1usize;
    let qlen = 3usize;
    let klen = 3usize;
    let d = 4usize;

    // q/k all zeros -> logits are all zeros pre-mask
    let q = Tensor::zeros((b, qlen, h, d), DType::F32, &device)?;
    let k = Tensor::zeros((b, klen, h, d), DType::F32, &device)?;

    // v encodes key index j as constant vector along d
    let mut v_vals = Vec::with_capacity(b * klen * h * d);
    for _b in 0..b {
        for j in 0..klen {
            for _h in 0..h {
                for _dd in 0..d {
                    v_vals.push(j as f32);
                }
            }
        }
    }
    let v = Tensor::from_vec(v_vals, (b, klen, h, d), &device)?;

    let scale = 1.0f32; // uniform logits remain uniform
    let out = eager_attn_with_sinks(&q, &k, &v, scale, true, None)?; // (b, q, h, d)
    let out = out.to_dtype(DType::F32)?;

    // Expected per-query means over allowed keys {0..i}
    // i=0 -> mean=0.0; i=1 -> mean=0.5; i=2 -> mean=1.0
    let expected = [0.0f32, 0.5, 1.0];
    // Reshape to (b, q, h*d) for easy host access.
    let out_host = out.reshape((b, qlen, h * d))?.to_vec3::<f32>()?; // [b][q][hd]
    for i in 0..qlen {
        for dd in 0..(h * d) {
            let got = out_host[0][i][dd];
            let diff = (got - expected[i]).abs();
            assert!(
                diff < 1e-4,
                "i={i} d={dd}: got={got} expected={}",
                expected[i]
            );
        }
    }

    Ok(())
}
