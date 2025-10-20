use anyhow::{Context as _, Result};
use candle::{DType, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;
use gpt_oss_tokenizer::render_then_encode;
use openai_harmony::chat::{Message, Role};
use std::path::PathBuf;
use std::process::Command;

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
fn t29_first_step_parity_mini() -> Result<()> {
    // Gate heavy GPU+Python check to keep default runs fast.
    if std::env::var("RUN_GPT_OSS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("[skip] set RUN_GPT_OSS_PARITY=1 to run t29_first_step_parity_mini");
        return Ok(());
    }

    // Resolve snapshot + tokenizer for channel id.
    let snapshot = expand_tilde(SNAPSHOT);
    let tok_path = snapshot.join("tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
    let channel_id = hf_tok
        .token_to_id("<|channel|>")
        .context("missing <|channel|> in tokenizer")? as usize;

    // Messages -> tokens
    let user = Message::from_role_and_content(Role::User, "Explain what MXFP4 quantization is".to_string());
    let input_ids: Vec<u32> = render_then_encode(&snapshot, &[user], true)
        .context("render_then_encode failed")?;
    assert!(!input_ids.is_empty(), "prompt tokens should not be empty");

    // Candle forward (GPU, BF16/F16)
    let device = candle_examples::device(false /* cpu */)?;
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };
    assert!(device.is_cuda(), "CUDA device required for this parity test");

    // Load config + weights
    let cfg_path = snapshot.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path)
        .with_context(|| format!("failed to read config: {}", cfg_path.display()))?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes).context("invalid config.json")?;
    let model_index = snapshot.join("model.safetensors.index.json");
    let model_files: Vec<PathBuf> = {
        let idx_bytes = std::fs::read(&model_index)
            .with_context(|| format!("failed to read model index: {}", model_index.display()))?;
        #[derive(serde::Deserialize)]
        struct Idx { weight_map: std::collections::BTreeMap<String, String> }
        let idx: Idx = serde_json::from_slice(&idx_bytes).context("invalid model.safetensors.index.json")?;
        let files = idx.weight_map.values().cloned().collect::<std::collections::BTreeSet<_>>();
        files.iter().map(|f| snapshot.join(f)).collect()
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    // Ensure safe RMSNorm compute (matches HF) for this test by default.
    std::env::set_var("CANDLE_RMS_FP32", "1");
    let mut model = GptOssModel::load(vb, &cfg).context("load GPT-OSS model")?;

    // Forward to get last-step probs
    let t = Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), &device)?;
    let logits = model.forward(&t, 0 /* offset */)?; // (1,T,V)
    let last = logits.i((0, input_ids.len() - 1))?.to_dtype(DType::F32)?;
    let probs = candle_nn::ops::softmax_last_dim(&last)?;
    let probs_v = probs.to_vec1::<f32>()?;
    let mut argmax = 0usize;
    let mut maxv = f32::NEG_INFINITY;
    for (i, &p) in probs_v.iter().enumerate() { if p > maxv { maxv = p; argmax = i; } }
    let p_channel = probs_v[channel_id];

    // Python HF compute of the same quantities (top-1 id, p_channel)
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let py_src = repo_root.join("py-transformers").join("src");
    let script = {
        let toks_json = serde_json::to_string(&input_ids).unwrap();
        format!(
            r#"import json, os, torch, sys
sys.path.insert(0, {py_src})
from transformers import AutoModelForCausalLM
snap = {snap}
tokens = {toks}
channel_id = {chid}
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=dtype)
model.to(device)
model.eval()
with torch.no_grad():
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model(input_ids=inp)
    last = out.logits[0, -1].float()
    probs = torch.softmax(last, dim=-1)
    top1_id = int(torch.argmax(probs).item())
    res = {{ 'top1_id': top1_id, 'p_channel': float(probs[channel_id]) }}
    print(json.dumps(res))
"#,
            py_src = format!("{:?}", py_src.display()),
            snap = format!("{:?}", snapshot.display()),
            toks = toks_json,
            chid = channel_id)
    };

    let output = Command::new("uv")
        .args(["run", "python", "-c", &script])
        .env("PYTHONUNBUFFERED", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .context("failed to run uv python for HF logits")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("python failed: {}", stderr);
    }
    let out = String::from_utf8_lossy(&output.stdout);
    #[derive(serde::Deserialize)]
    struct PyRes { top1_id: usize, p_channel: f32 }
    let py_res: PyRes = serde_json::from_str(&out).context("invalid python json")?;

    // Assertions
    assert_eq!(argmax, channel_id, "Candle top-1 must be <|channel|>");
    assert!(p_channel > 0.99, "Candle p_channel must be > 0.99, was {}", p_channel);
    assert_eq!(py_res.top1_id, channel_id, "HF top-1 must be <|channel|>");
    assert!(py_res.p_channel > 0.99, "HF p_channel must be > 0.99, was {}", py_res.p_channel);
    Ok(())
}
