use anyhow::{Context as _, Result};
use candle::{DType, Device, IndexOp, Tensor, Module};
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;
use std::path::PathBuf;
use std::process::Command;

const SNAPSHOT: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

fn expand_tilde(p: &str) -> PathBuf {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") { return Path::new(&home).join(rest); }
    }
    PathBuf::from(p)
}

fn cuda_device_or_skip() -> Result<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Ok(d),
        Err(_) => {
            eprintln!("cuda device not available — skipping");
            Err(anyhow::anyhow!("skip"))
        }
    }
}

#[test]
fn t29_block0_attention_parity_with_sliding_and_sinks() -> Result<()> {
    if std::env::var("RUN_GPT_OSS_ATTENTION_PARITY").ok().as_deref() != Some("1") {
        eprintln!("[skip] set RUN_GPT_OSS_ATTENTION_PARITY=1 to run block-0 attention parity test");
        return Ok(());
    }

    // Force eager attention to avoid kernel variability.
    std::env::set_var("CANDLE_DISABLE_FLASH", "1");

    // GPU device and dtype
    let device = match cuda_device_or_skip() { Ok(d) => d, Err(_) => return Ok(()) };
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };

    // Load config and weights
    let snapshot = expand_tilde(SNAPSHOT);
    let cfg_bytes = std::fs::read(snapshot.join("config.json"))?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes).context("invalid config.json")?;
    let model_index = snapshot.join("model.safetensors.index.json");
    let model_files: Vec<PathBuf> = {
        #[derive(serde::Deserialize)]
        struct Idx { weight_map: std::collections::BTreeMap<String, String> }
        let idx_bytes = std::fs::read(&model_index)?;
        let idx: Idx = serde_json::from_slice(&idx_bytes)?;
        let files = idx.weight_map.values().cloned().collect::<std::collections::BTreeSet<_>>();
        files.iter().map(|f| snapshot.join(f)).collect()
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let mut model = GptOssModel::load(vb, &cfg).context("load GPT-OSS model")?;

    // Run Python dumper to get subgraph inputs and expected outputs
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let script_path = repo_root.join("scripts").join("dump_block0_attn_20b.py");
    let output = Command::new("uv")
        .args(["run", "python", script_path.to_str().unwrap()])
        .env("PYTHONUNBUFFERED", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .context("failed to run python dumper")?;
    if !output.status.success() { anyhow::bail!("python failed: {}", String::from_utf8_lossy(&output.stderr)); }
    let out = String::from_utf8_lossy(&output.stdout);
    #[derive(serde::Deserialize)]
    struct PyDump {
        input_ids: Vec<i64>,
        x_in: Vec<Vec<f32>>,        // (T,H)
        q_pre: Vec<Vec<Vec<f32>>>,  // (n_q,T,d)
        k_pre: Vec<Vec<Vec<f32>>>,  // (n_kv,T,d)
        v_pre: Vec<Vec<Vec<f32>>>,
        cos: Vec<Vec<f32>>,         // (T, d/2)
        sin: Vec<Vec<f32>>,         // (T, d/2)
        mask_row_counts: Vec<i64>,
        sinks: Vec<f32>,
        A0_last: Vec<f32>,
        hidden_size: usize,
        head_dim: usize,
        n_q: usize,
        n_kv: usize,
    }
    let py: PyDump = serde_json::from_str(&out).context("invalid python json")?;

    let b = 1usize;
    let t = py.input_ids.len();
    let hdim = py.hidden_size;
    let n_q = py.n_q;
    let n_kv = py.n_kv;
    let d = py.head_dim;

    // Prepare x_in from Python-dumped pre-norm input to avoid RMSNorm drift
    let x_in = Tensor::from_vec(py.x_in.clone().into_iter().flatten().collect::<Vec<_>>(), (b, t, hdim), &device)?
        .to_dtype(dtype)?;
    let layer = &mut model.layers[0];
    let rope = &model.rope;

    // prenorm
    let x_norm = layer.input_layernorm.forward(&x_in)?; // (b,t,h)

    // projections
    let q = x_norm.apply(&layer.attn.q_proj)?; // (b,t,n_q*d)
    let k = x_norm.apply(&layer.attn.k_proj)?; // (b,t,n_kv*d)
    let v = x_norm.apply(&layer.attn.v_proj)?;

    let q = q.reshape((b, t, n_q, d))?;
    let k = k.reshape((b, t, n_kv, d))?;
    let v = v.reshape((b, t, n_kv, d))?;

    // RoPE (use Candle's YaRN)
    let q_bhtd = q.transpose(1, 2)?;
    let k_bhtd = k.transpose(1, 2)?;
    let (q_bhtd, k_bhtd) = rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, 0 /* offset */)?;
    let q = q_bhtd.transpose(1, 2)?;          // (b,t,n_q,d)
    let k_step = k_bhtd;                      // (b,n_kv,t,d)
    let v_step = v.transpose(1, 2)?;          // (b,n_kv,t,d)

    // No cache for prefill parity: just use current K/V and repeat for GQA
    let n_rep = n_q / n_kv;
    let k_rep = candle_transformers::utils::repeat_kv(k_step.clone(), n_rep)?; // (b,n_q,t,d)
    let v_rep = candle_transformers::utils::repeat_kv(v_step.clone(), n_rep)?;
    let k_btkhd = k_rep.transpose(1, 2)?; // (b,t,n_q,d)
    let v_btkhd = v_rep.transpose(1, 2)?;

    // Sliding/windowed attention mode for layer 0
    let attn_mode = candle_transformers::models::gpt_oss::select_attn_mode_for_layer(
        &candle_transformers::models::gpt_oss::GptOssConfigMinimal {
            num_hidden_layers: cfg.num_hidden_layers,
            layer_types: cfg.effective_layer_types(),
            max_position_embeddings: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
        },
        0,
    );
    let softmax_scale = 1.0f32 / (d as f32).sqrt();

    // Compute attention output via eager/windowed + sinks
    let sinks = Some(&layer.attn.sinks);
    let y = match attn_mode {
        candle_transformers::models::gpt_oss::AttnMode::Full => {
            candle_transformers::models::gpt_oss::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?
        }
        candle_transformers::models::gpt_oss::AttnMode::Sliding { left, right } => {
            candle_transformers::models::gpt_oss::eager_attn_windowed_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks)?
        }
    }; // (b,t,n_q,d)

    // Project and residual add
    let y = y.reshape((b, t, n_q * d))?;
    let y = y.apply(&layer.attn.o_proj)?; // (b,t,h)
    let xs = (x_in + y)?;

    // Validate mask row counts parity (diagnostic) by re-creating the windowed mask summarization
    // Count allowed keys per query row under our implementation
    fn mask_counts(qlen: usize, klen: usize, left: usize, right: usize) -> Vec<i64> {
        let mut v = Vec::with_capacity(qlen);
        for i in 0..qlen {
            let mut c = 0i64;
            for j in 0..klen {
                let i = i as isize; let j = j as isize; let l = left as isize; let r = right as isize;
                let allow = (j > i - l) && (j <= i + r); // mirrors eager_attn_windowed_with_sinks
                if allow { c += 1; }
            }
            v.push(c);
        }
        v
    }

    let mask_counts_rust = match attn_mode {
        candle_transformers::models::gpt_oss::AttnMode::Full => (0..t).map(|i| (i + 1) as i64).collect(),
        candle_transformers::models::gpt_oss::AttnMode::Sliding { left, right } => mask_counts(t, t, left, right),
    };
    assert_eq!(mask_counts_rust, py.mask_row_counts, "mask row counts mismatch");

    // Compare last-position hidden (post-attn residual) exactly
    let a0_last = xs.i((0, t - 1))?.to_dtype(DType::F32)?;
    let a0_last_v = a0_last.to_vec1::<f32>()?;
    assert_eq!(a0_last_v.len(), py.A0_last.len());
    let mut max_abs = 0f32;
    let mut l2 = 0f64;
    for (_i, (&a, &b)) in a0_last_v.iter().zip(py.A0_last.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        let dd = a as f64 - b as f64;
        l2 += dd * dd;
    }
    l2 = l2.sqrt();

    // Softmax row sums ~1 for last row, head 0 (recompute logits here deterministically)
    let q_f = q.to_dtype(DType::F32)?; let k_f = k_btkhd.to_dtype(DType::F32)?; let v_f = v_btkhd.to_dtype(DType::F32)?;
    let q_bhqd = q_f.transpose(1, 2)?; // (b,h,q,d)
    let k_bhkd = k_f.transpose(1, 2)?; // (b,h,k,d)
    let logits = (q_bhqd.contiguous()?.matmul(&k_bhkd.t()?.contiguous()?)? * (1.0f32 / (d as f32).sqrt()) as f64)?; // (b,h,q,k)
    // build windowed mask like in eager path
    let masked = match attn_mode {
        candle_transformers::models::gpt_oss::AttnMode::Full => logits,
        candle_transformers::models::gpt_oss::AttnMode::Sliding { left, right } => {
            let mut mask = Vec::<u8>::with_capacity(t * t);
            for i in 0..t { for j in 0..t {
                let i = i as isize; let j = j as isize; let l = left as isize; let r = right as isize;
                let allow = (j > i - l) && (j <= i + r);
                mask.push(u8::from(!allow));
            }}
            let mask_t = Tensor::from_slice(&mask, (t, t), logits.device())?;
            let (b, h, _q, _k) = logits.dims4()?;
            let mask4 = mask_t.broadcast_as((b, h, t, t))?;
            let on_true = Tensor::new(f32::NEG_INFINITY, logits.device())?.broadcast_as(mask4.shape().dims())?;
            mask4.where_cond(&on_true, &logits)?
        }
    };
    let att = candle_nn::ops::softmax_last_dim(&masked)?;
    let sums_last = att.i((0, 0, t - 1))?.sum_all()?.to_scalar::<f32>()?;

    eprintln!("[attn] dtype={:?} device={:?} t={} hdim={} n_q={} n_kv={} d={} max_abs={:.3e} l2={:.3e} row_sum_last={:.6}",
        dtype, device, t, hdim, n_q, n_kv, d, max_abs, l2, sums_last);
    assert!((sums_last - 1.0).abs() < 1e-6);
    if max_abs > 1e-5 { anyhow::bail!("A0_last max_abs={:.3e} l2={:.3e} exceeds tolerance", max_abs, l2); }

    Ok(())
}
