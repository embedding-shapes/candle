#[cfg(test)]
mod tests {
    use crate::models::gpt_oss::*;
    use candle::{Device, DType, IndexOp, Module, Result, Tensor, D};
    use candle_nn::{Linear, VarBuilder};
    use std::collections::HashMap;

    // Validate sinks_scale_from_lse matches 1 / (1 + exp(sink - lse)).
    #[test]
    fn t18_sinks_scale_formula() -> Result<()> {
        let dev = Device::Cpu;
        // lse: (b=1, h=2, q=3)
        let lse = Tensor::from_vec(
            vec![
                // h0
                0.0f32, 1.0, 2.0, // h1
                -1.0, 0.5, 3.0,
            ],
            (1, 2, 3),
            &dev,
        )?;
        // sinks: (h=2)
        let sinks = Tensor::from_vec(vec![0.5f32, -0.5], 2, &dev)?;
        let scale = sinks_scale_from_lse(&lse, &sinks)?; // (1,2,3)
        let got = scale.to_vec3::<f32>()?;
        // Compute reference
        let lse_host = lse.to_vec3::<f32>()?;
        for b in 0..1 {
            for h in 0..2 {
                for q in 0..3 {
                    let lse_v = lse_host[b][h][q];
                    let s = if h == 0 { 0.5 } else { -0.5 };
                    let ref_v = 1.0 / (1.0 + (s - lse_v).exp());
                    let diff = (ref_v - got[b][h][q]).abs();
                    assert!(
                        diff < 1e-6,
                        "mismatch at (b={b},h={h},q={q}): ref={ref_v} got={} diff={diff}",
                        got[b][h][q]
                    );
                }
            }
        }
        Ok(())
    }

    // Verify ExpertMlp GLU uses symmetric clamp [-limit, +limit] on both gate and up
    // and matches the golden formula ff = (up + 1) * (gate * sigmoid(alpha * gate)).
    // Runs on CUDA if available to exercise GPU path.
    #[test]
    fn t19_expert_glu_clamp_symmetry_gpu() -> Result<()> {
        // Select GPU; skip if CUDA not available.
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[skip] CUDA device not available; skipping GPU GLU clamp test");
                return Ok(());
            }
        };

        let dtype = DType::F32;
        let d = 4usize; // hidden = inter = 4
        let limit = 7.0f32;
        let alpha = 1.702f32;

        // Build gate_up as [I; I] (2D x D) and down as I (D x D)
        let mut w_gate_up = vec![0f32; 2 * d * d];
        for i in 0..d {
            // top identity block
            w_gate_up[i * d + i] = 1.0;
            // bottom identity block
            w_gate_up[(d + i) * d + i] = 1.0;
        }
        let w_gate_up = Tensor::from_vec(w_gate_up, (2 * d, d), &dev)?.to_dtype(dtype)?;
        let gate_up = Linear::new(w_gate_up, None);

        let mut w_down = vec![0f32; d * d];
        for i in 0..d {
            w_down[i * d + i] = 1.0;
        }
        let w_down = Tensor::from_vec(w_down, (d, d), &dev)?.to_dtype(dtype)?;
        let down = Linear::new(w_down, None);

        let expert = ExpertMlp::new(gate_up, down, limit, alpha);

        // Input pre-activations (x) so that gu = [x; x] after gate_up
        // Include values beyond clamp bounds to test symmetry.
        let x_host = vec![-10.0f32, -1.0, 0.0, 10.0];
        let xs = Tensor::from_vec(x_host.clone(), (1, d), &dev)?.to_dtype(dtype)?;

        // Forward through ExpertMlp
        let y = expert.forward(&xs)?; // (1, d)
        let y_f32 = y.to_dtype(DType::F32)?;
        let y_host = y_f32.to_vec2::<f32>()?;

        // Compute expected on host per golden reference
        let mut expected = vec![0f32; d];
        for i in 0..d {
            let gate = x_host[i].clamp(-limit, limit);
            let up = x_host[i].clamp(-limit, limit);
            let sig = 1.0f32 / (1.0f32 + (-alpha * gate).exp());
            let glu = gate * sig;
            let ff = (1.0 + up) * glu;
            expected[i] = ff;
        }

        // Metrics
        let mut linf = 0f32;
        let mut l2 = 0f32;
        for i in 0..d {
            let diff = (y_host[0][i] - expected[i]).abs();
            linf = linf.max(diff);
            l2 += diff * diff;
        }
        l2 = l2.sqrt();

        // Logging for evidence
        println!("device: cuda:0");
        println!("dtype: f32");
        println!("input x: {:?}", x_host);
        println!("expected ff: {:?}", expected);
        println!("actual out:   {:?}", y_host[0]);
        println!("errors: Linf={:.6}, L2={:.6}", linf, l2);

        // Tolerance: exact match in f32 math
        assert!(linf < 1e-6, "GLU clamp mismatch Linf={linf} L2={l2}");
        Ok(())
    }

    // Verify windowed mask uses left-inclusive range [i-left, i] on CUDA with simple uniform logits.
    // We set q=k=0 so logits are zeros for allowed positions and -inf elsewhere; softmax is uniform
    // over allowed positions. We encode v so that output equals the mean of allowed key indices,
    // then compare against the inclusive expectation.
    #[test]
    fn t20_window_mask_inclusive_left_gpu() -> Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[skip] CUDA device not available; skipping window mask test");
                return Ok(());
            }
        };

        // Shapes
        let b = 1usize;
        let h = 2usize;
        let qlen = 4usize;
        let klen = 4usize;
        let d = 8usize;

        // q,k zeros -> logits all zeros on allowed band
        let q = Tensor::zeros((b, qlen, h, d), DType::F32, &dev)?;
        let k = Tensor::zeros((b, klen, h, d), DType::F32, &dev)?;

        // v encodes key index in channel 0; other channels zero
        let mut v_host = vec![0f32; b * klen * h * d];
        for j in 0..klen {
            // position in flattened (b=1, k, h, d): index for head 0, channel 0
            let idx = ((0 * klen + j) * h + 0) * d + 0;
            v_host[idx] = j as f32;
        }
        let v = Tensor::from_vec(v_host.clone(), (b, klen, h, d), &dev)?;

        // Window: left=2, right=0
        let left = 2usize;
        let right = 0usize;
        let softmax_scale = 1.0f32; // irrelevant since logits are zeros
        let out = eager_attn_windowed_with_sinks(
            &q,
            &k,
            &v,
            softmax_scale,
            Some(left),
            Some(right),
            None,
        )?; // (b,q,h,d)

        let out_f32 = out.to_dtype(DType::F32)?;

        // Compute expected inclusive means for head 0, channel 0.
        let mut linf = 0f32;
        let mut l2 = 0f32;
        for i in 0..qlen {
            let j_start = i.saturating_sub(left);
            let j_end = i; // inclusive
            let n = (j_end - j_start + 1) as f32;
            let sum: f32 = (j_start..=j_end).map(|j| j as f32).sum();
            let expected = sum / n;
            let got = out_f32.i((0, i, 0, 0))?.to_scalar::<f32>()?;
            let diff = (got - expected).abs();
            linf = linf.max(diff);
            l2 += diff * diff;
        }
        l2 = l2.sqrt();

        // Logging for evidence
        println!("device: cuda:0");
        println!(
            "dtype: f32 shapes: q=({},{},{},{}), k=({},{},{},{}), v=({},{},{},{})",
            b, qlen, h, d, b, klen, h, d, b, klen, h, d
        );
        println!("window: left={}, right={}", left, right);
        // Print per-query expected vs actual for head 0, channel 0
        for i in 0..qlen {
            let j_start = i.saturating_sub(left);
            let j_end = i;
            let n = (j_end - j_start + 1) as f32;
            let sum: f32 = (j_start..=j_end).map(|j| j as f32).sum();
            let expected = sum / n;
            let got = out_f32.i((0, i, 0, 0))?.to_scalar::<f32>()?;
            println!(
                "q={} allowed=[{}..={}] expected_mean={:.6} got={:.6}",
                i, j_start, j_end, expected, got
            );
        }
        println!("errors: Linf={:.6} L2={:.6}", linf, l2);

        // Tight tolerance: exact in f32
        assert!(
            linf < 1e-6,
            "left boundary must be inclusive (Linf={linf} L2={l2})"
        );
        Ok(())
    }

    // Verify embedding scaling by sqrt(hidden_size) is applied at the very start of forward,
    // by inspecting the pre-attention last-token vector (h0) via the debug collector on GPU.
    // This avoids loading any layers by using num_hidden_layers=0.
    #[test]
    fn t21_embed_scale_matches_sqrt_hidden_gpu() -> Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[skip] CUDA device not available; skipping embed scale test");
                return Ok(());
            }
        };

        // Tiny config
        let vocab = 8usize;
        let hidden = 6usize;
        let cfg = GptOssConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_hidden_layers: 0,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(3),
            num_local_experts: 2,
            num_experts_per_tok: 1,
            intermediate_size: 4,
            max_position_embeddings: 16,
            rope_theta: Some(10000.0),
            rope_scaling: None,
            layer_types: vec![],
            sliding_window: None,
            rms_norm_eps: Some(1e-6),
            swiglu_limit: Some(7.0),
        };

        // Build VarBuilder with only the tensors required by GptOssModel::load for this cfg.
        // We need: model.embed_tokens.weight (vocab, hidden), model.norm.weight (hidden),
        // lm_head.weight (vocab, hidden). No layers present.
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        // Embeddings: simple increasing values per row.
        let mut emb: Vec<f32> = Vec::with_capacity(vocab * hidden);
        for i in 0..vocab {
            for j in 0..hidden {
                emb.push((i * hidden + j) as f32 / 100.0);
            }
        }
        ts.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::from_vec(emb.clone(), (vocab, hidden), &dev)?.to_dtype(DType::BF16)?,
        );
        // Final norm weight (unused in h0 but required by loader)
        ts.insert(
            "model.norm.weight".to_string(),
            Tensor::ones(hidden, DType::BF16, &dev)?,
        );
        // Tie lm_head to embeddings to satisfy loader paths (content irrelevant for this test)
        ts.insert(
            "lm_head.weight".to_string(),
            Tensor::from_vec(emb, (vocab, hidden), &dev)?.to_dtype(DType::BF16)?,
        );

        let vb = VarBuilder::from_tensors(ts, DType::BF16, &dev);
        let mut model = GptOssModel::load(vb, &cfg)?;

        // Single-token input id = 3
        let tok_id: u32 = 3;
        let input = Tensor::from_vec(vec![tok_id], (1, 1), &dev)?;
        let rows = model.debug_collect_layer_states_last_token(&input, 0)?; // (1, H)
        let h0 = rows.squeeze(0)?.to_dtype(DType::F32)?; // (H)

        // Expected: embedding row 3 scaled by sqrt(hidden)
        let embed_row = Tensor::arange(0u32, hidden as u32, &Device::Cpu)?
            .to_dtype(DType::F32)?
            .affine(1.0 / 100.0, (3 * hidden) as f64 / 100.0)?; // reconstruct same values
        let scale = (hidden as f64).sqrt();
        let expected = (&embed_row * scale)?;

        // Move h0 to CPU f32
        let got = h0.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let exp = expected.to_vec1::<f32>()?;

        // Metrics
        let mut linf = 0f32;
        let mut l2 = 0f32;
        for i in 0..hidden {
            let diff = (got[i] - exp[i]).abs();
            linf = linf.max(diff);
            l2 += diff * diff;
        }
        l2 = l2.sqrt();
        println!("device: cuda:0 dtype: bf16 hidden={}", hidden);
        println!("expected (first 6): {:?}", &exp);
        println!("actual   (first 6): {:?}", &got);
        println!("errors: Linf={:.6} L2={:.6}", linf, l2);
        // BF16 rounding yields ~3e-3 absolute differences; accept small tolerance.
        assert!(
            linf < 5e-3 && l2 < 1e-2,
            "embed scaling mismatch Linf={linf} L2={l2}"
        );
        Ok(())
    }

    // Validate eager attention with sinks matches the golden reference definition
    // (append per-head sink logits as an extra column, softmax over last dim,
    // then drop the sink column and apply to V).
    #[test]
    fn t22_eager_sinks_concat_parity_gpu() -> Result<()> {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("[skip] CUDA device not available; skipping eager sinks parity test");
                return Ok(());
            }
        };
        dev.set_seed(42)?;

        let b = 1usize;
        let h = 2usize;
        let qlen = 3usize;
        let klen = 4usize;
        let d = 16usize;
        let dtype = if dev.supports_bf16() {
            DType::BF16
        } else {
            DType::F16
        };
        let scale = 1f32 / (d as f32).sqrt();

        // q,k,v shaped as FA layout: (b, seqlen, heads, dim)
        let q = Tensor::arange(0u32, (b * qlen * h * d) as u32, &dev)?
            .to_dtype(dtype)?
            .reshape((b, qlen, h, d))?
            .affine(1.0 / 1000.0, 0.0)?;
        let k = Tensor::arange(0u32, (b * klen * h * d) as u32, &dev)?
            .to_dtype(dtype)?
            .reshape((b, klen, h, d))?
            .affine(1.0 / 800.0, 0.0)?;
        let v = (&k / 20.0)?; // make v correlated with k but smaller scale

        // sinks per head
        let sinks = Tensor::from_vec(vec![0.15f32, -0.05f32], h, &dev)?;

        // Actual eager path
        let out_eager = eager_attn_with_sinks(&q, &k, &v, scale, false, Some(&sinks))?; // (b,q,h,d)

        // Spec: concatenate sinks as extra column, softmax, drop last column
        // logits: (b,h,q,k)
        let q_bhqd = q.transpose(1, 2)?;
        let k_bhkd = k.transpose(1, 2)?;
        let logits = (q_bhqd
            .contiguous()?
            .matmul(&k_bhkd.transpose(2, 3)?.contiguous()?)?
            * scale as f64)?;
        let bq = logits.dims4()?.0;
        let sinks_broadcast = sinks
            .to_dtype(logits.dtype())?
            .reshape((1, h, 1, 1))?
            .broadcast_as((bq, h, qlen, 1))?;
        let combined = Tensor::cat(&[&logits, &sinks_broadcast], D::Minus1)?; // (b,h,q,k+1)
        let maxv = combined.max_keepdim(D::Minus1)?;
        let probs = combined.broadcast_sub(&maxv)?.exp()?;
        let z = probs.sum_keepdim(D::Minus1)?;
        let probs = probs.broadcast_div(&z)?; // (b,h,q,k+1)
        let scores = probs.narrow(D::Minus1, 0, klen)?; // drop sink
        let o_bhqd = scores
            .contiguous()?
            .matmul(&v.transpose(1, 2)?.contiguous()?)?; // (b,h,q,d)
        let out_spec = o_bhqd.transpose(1, 2)?; // (b,q,h,d)

        // Move to f32 for comparison
        let eager_f32 = out_eager.to_dtype(DType::F32)?;
        let spec_f32 = out_spec.to_dtype(DType::F32)?;

        // Metrics
        let diff = (eager_f32 - spec_f32)?.abs()?;
        let linf = diff.flatten_all()?.max(0)?.to_vec0::<f32>()?;
        let l2 = {
            let v = diff.flatten_all()?.to_vec1::<f32>()?;
            v.iter().map(|x| x * x).sum::<f32>().sqrt()
        };

        // Logging
        let (b0, q0, h0, d0) = out_eager.dims4()?;
        println!(
            "device: cuda:0 dtype: {}",
            match dtype {
                DType::BF16 => "bf16",
                DType::F16 => "f16",
                _ => "other",
            }
        );
        println!("shapes: b={}, q={}, h={}, d={}", b0, q0, h0, d0);
        println!("errors: Linf={:.8} L2={:.8}", linf, l2);

        // Tight enough for f32 math; casting from bf16/f16 introduces tiny error margins.
        assert!(
            linf < 4e-5 && l2 < 2e-4,
            "eager sinks != spec (Linf={linf} L2={l2})"
        );
        Ok(())
    }
}
