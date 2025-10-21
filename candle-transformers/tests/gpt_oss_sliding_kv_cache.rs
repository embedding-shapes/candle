use candle::IndexOp;
use candle::{Device, Result, Tensor, D};
use candle_transformers::models::gpt_oss::{
    eager_attn_windowed_with_sinks, select_attn_mode_for_layer, AttnMode, GptOssConfigMinimal,
    GptOssLayerType,
};

// T18: KV cache trimming for sliding-window layers and decode-step equivalence

#[test]
fn t18_kv_cache_trimming_and_decode_equiv() -> Result<()> {
    let dev = Device::Cpu;

    // Minimal config: 4 layers, alternating full/sliding, sliding_window=3
    let cfg = GptOssConfigMinimal {
        num_hidden_layers: 4,
        layer_types: vec![
            GptOssLayerType::FullAttention,
            GptOssLayerType::SlidingAttention,
            GptOssLayerType::FullAttention,
            GptOssLayerType::SlidingAttention,
        ],
        max_position_embeddings: 8192,
        sliding_window: Some(3),
    };
    // Verify selection helper behaves as expected.
    assert_eq!(select_attn_mode_for_layer(&cfg, 0), AttnMode::Full);
    assert_eq!(
        select_attn_mode_for_layer(&cfg, 1),
        AttnMode::Sliding { left: 3, right: 0 }
    );

    // Use RotatingKvCache directly to test trimming semantics on a sliding layer.
    let window = cfg.sliding_window.unwrap();
    let b = 1usize;
    let h = 2usize;
    let d = 4usize;
    let mut kv = candle_nn::kv_cache::RotatingKvCache::new(2, window);

    // Build a simple artificial stream: step s has all entries filled with value s (and 2*s for V)
    let steps = 7usize;
    let mut all_k = Vec::<Tensor>::new();
    let mut all_v = Vec::<Tensor>::new();
    for s in 0..steps {
        let val_k = s as f32;
        let val_v = (2 * s) as f32;
        let k_s = Tensor::full(val_k, (b, h, 1, d), &dev)?;
        let v_s = Tensor::full(val_v, (b, h, 1, d), &dev)?;
        all_k.push(k_s.clone());
        all_v.push(v_s.clone());
        let (kc, vc) = kv.append(&k_s, &v_s)?;

        // Current cache length trimmed to last `window` entries
        let klen = kc.dim(2)?;
        let expected_len = (s + 1).min(window);
        assert_eq!(klen, expected_len, "unexpected trimmed length at step {s}");

        // Content check (order-agnostic): min/max reflect the last `expected_len` step values
        let start = s + 1 - expected_len;
        let min_k = kc
            .reshape((kc.elem_count(),))?
            .min(D::Minus1)?
            .to_scalar::<f32>()?;
        let max_k = kc
            .reshape((kc.elem_count(),))?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        let min_v = vc
            .reshape((vc.elem_count(),))?
            .min(D::Minus1)?
            .to_scalar::<f32>()?;
        let max_v = vc
            .reshape((vc.elem_count(),))?
            .max(D::Minus1)?
            .to_scalar::<f32>()?;
        assert_eq!(min_k, start as f32, "k min mismatch at step {s}");
        assert_eq!(max_k, s as f32, "k max mismatch at step {s}");
        assert_eq!(min_v, (2 * start) as f32, "v min mismatch at step {s}");
        assert_eq!(max_v, (2 * s) as f32, "v max mismatch at step {s}");
    }

    // Decode-step equivalence for short sequences under windowed attention
    let head_dim = d;
    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let window = 3usize;
    let t = 5usize;

    // Construct deterministic per-token q/k/v
    let mut q_full = Vec::<Tensor>::new();
    let mut k_full = Vec::<Tensor>::new();
    let mut v_full = Vec::<Tensor>::new();
    let mut last_step_out: Option<Tensor> = None;
    for s in 0..t {
        let q_s_val = (s as f32) + 1.0;
        let k_s_val = (10 * s) as f32 + 0.5;
        let v_s_val = (100 * s) as f32 + 0.25;
        let q_s = Tensor::full(q_s_val, (b, 1, h, d), &dev)?;
        let k_s = Tensor::full(k_s_val, (b, 1, h, d), &dev)?;
        let v_s = Tensor::full(v_s_val, (b, 1, h, d), &dev)?;

        q_full.push(q_s.clone());
        k_full.push(k_s.clone());
        v_full.push(v_s.clone());

        // Trim to last `window` entries in chronological order.
        let start = s + 1 - (s + 1).min(window);
        let k_step = Tensor::cat(&k_full[start..=s], 1)?; // (b, t_step, h, d)
        let v_step = Tensor::cat(&v_full[start..=s], 1)?;
        let q_step = Tensor::cat(&q_full[start..=s], 1)?; // align i=t_step-1 for causal window
        let t_step = s - start + 1;
        let o_step_all = eager_attn_windowed_with_sinks(
            &q_step,
            &k_step,
            &v_step,
            softmax_scale,
            Some(window),
            Some(0),
            None,
        )?; // (b,t_step,h,d)
        let o_step = o_step_all
            .i((.., t_step - 1, .., ..))?
            .reshape((b, 1, h, d))?;
        if s == t - 1 {
            last_step_out = Some(o_step.clone());
        }
    }

    let q_full = Tensor::cat(&q_full, 1)?; // (b,t,h,d)
    let k_full = Tensor::cat(&k_full, 1)?;
    let v_full = Tensor::cat(&v_full, 1)?;
    let o_full = eager_attn_windowed_with_sinks(
        &q_full,
        &k_full,
        &v_full,
        softmax_scale,
        Some(window),
        Some(0),
        None,
    )?; // (b,t,h,d)
    let o_last_full = o_full.i((.., t - 1, .., ..))?.reshape((b, 1, h, d))?;
    let o_last_step = last_step_out.expect("last step output");
    let diff = (&o_last_step - &o_last_full)?
        .abs()?
        .reshape((o_last_step.elem_count(),))?
        .max(D::Minus1)?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "decode-step parity mismatch: max diff {diff}");

    Ok(())
}
