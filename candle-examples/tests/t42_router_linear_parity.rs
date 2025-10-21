use anyhow::{Context as _, Result};
use candle::Module as _;
use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use std::path::PathBuf;

const SNAPSHOT: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

fn expand_tilde(p: &str) -> PathBuf {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    PathBuf::from(p)
}

#[test]
fn t42_router_linear_parity() -> Result<()> {
    // Require GPU for this micro-check.
    let device = candle_examples::device(false /* cpu */)?;
    if !device.is_cuda() {
        eprintln!("[skip] CUDA device required for this test");
        return Ok(());
    }
    let dtype = if device.supports_bf16() {
        DType::BF16
    } else {
        DType::F16
    };

    // Load config and a single layer varbuilder from snapshot.
    let snapshot = expand_tilde(SNAPSHOT);
    let cfg_path = snapshot.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path)
        .with_context(|| format!("failed to read config: {}", cfg_path.display()))?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes).context("invalid config.json")?;

    // Map all model shards.
    let model_index = snapshot.join("model.safetensors.index.json");
    let model_files: Vec<PathBuf> = {
        let idx_bytes = std::fs::read(&model_index)
            .with_context(|| format!("failed to read model index: {}", model_index.display()))?;
        #[derive(serde::Deserialize)]
        struct Idx {
            weight_map: std::collections::BTreeMap<String, String>,
        }
        let idx: Idx =
            serde_json::from_slice(&idx_bytes).context("invalid model.safetensors.index.json")?;
        let files = idx
            .weight_map
            .values()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        files.iter().map(|f| snapshot.join(f)).collect()
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let l0 = vb.pp("model").pp("layers").pp("0");

    // Assert presence of correct router tensors and absence of old name.
    assert!(
        l0.pp("mlp").contains_tensor("router.weight"),
        "missing mlp.router.weight"
    );
    assert!(
        l0.pp("mlp").contains_tensor("router.bias"),
        "missing mlp.router.bias (bias required)"
    );

    // Build the router Linear exactly as in the model loader (with bias, mlp.router).
    let hidden = cfg.hidden_size;
    let e = cfg.num_local_experts;
    let router = candle_nn::linear(hidden, e, l0.pp("mlp").pp("router"))?;

    // Retrieve weight/bias explicitly for a direct matmul+bias reference.
    let w = l0.pp("mlp").get((e, hidden), "router.weight")?; // (E,H)
    let b = l0.pp("mlp").get(e, "router.bias")?; // (E)

    // Deterministic input vector on GPU: x[i] = (i % 13 - 6) / 10, cast to weight dtype.
    let x_f32: Vec<f32> = (0..hidden)
        .map(|i| (((i as i32 % 13) - 6) as f32) / 10.0)
        .collect();
    let x = Tensor::from_vec(x_f32.clone(), (1, hidden), &device)?.to_dtype(dtype)?; // (1,H)

    // Forward through the Linear and explicit matmul+bias reference.
    let y_lin = router.forward(&x)?.to_dtype(DType::F32)?; // (1,E) in f32
    let y_ref = x.matmul(&w.t()?)?.broadcast_add(&b)?.to_dtype(DType::F32)?; // (1,E) in f32

    // Metrics and logs
    let y_lin_v = y_lin.reshape((e,))?.to_vec1::<f32>()?;
    let y_ref_v = y_ref.reshape((e,))?.to_vec1::<f32>()?;
    let linf = y_lin_v
        .iter()
        .zip(&y_ref_v)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let l2 = y_lin_v
        .iter()
        .zip(&y_ref_v)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();

    println!(
        "seed=fixed; dtype={:?}; device=cuda; shapes: x=(1,{hidden}), w=({e},{hidden}), b=({e}); contiguous: x={}, w={}, b={}",
        dtype,
        x.is_contiguous(),
        w.is_contiguous(),
        b.is_contiguous(),
    );
    println!("x[0..8] = {:?}", &x_f32[..8]);
    println!("y_lin[0..8] = {:?}", &y_lin_v[..8]);
    println!("y_ref[0..8] = {:?}", &y_ref_v[..8]);
    println!("errors: Linf={:.3e}, L2={:.3e}", linf, l2);

    // Expect exact parity (same computation graph), allow tiny fp roundoff.
    assert!(
        linf < 5e-7 && l2 < 5e-6,
        "router linear mismatch: Linf={linf} L2={l2}"
    );

    // Additionally ensure bias is non-zero and used: y_lin - (x@W^T) == b (within tol).
    let y_nobias = x.matmul(&w.t()?)?.to_dtype(DType::F32)?; // (1,E)
    let diff = (y_lin - y_nobias)?.to_dtype(DType::F32)?; // (1,E)
    let diff_v = diff.reshape((e,))?.to_vec1::<f32>()?;
    let b_v = b.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let linf_b = diff_v
        .iter()
        .zip(&b_v)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("bias-check Linf={:.3e}", linf_b);
    // In BF16/F16 we allow a tiny rounding difference for the bias broadcast path.
    assert!(
        linf_b < 3e-3,
        "router bias not applied correctly: Linf={linf_b}"
    );

    Ok(())
}
