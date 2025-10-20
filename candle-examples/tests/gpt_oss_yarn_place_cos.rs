use anyhow::{Context as _, Result};
use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;
use gpt_oss_tokenizer::render_then_encode;
use openai_harmony::chat::{Message, Role};

const SNAPSHOT: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const INDEX: &str = "model.safetensors.index.json";

fn expand_tilde(p: &str) -> std::io::Result<std::path::PathBuf> {
    if let Some(rest) = p.strip_prefix("~/") {
        Ok(std::env::var("HOME").map(std::path::PathBuf::from).unwrap().join(rest))
    } else {
        Ok(std::path::PathBuf::from(p))
    }
}

#[test]
fn gpt_oss_yarn_place_cos() -> Result<()> {
    // Mode selection must be set by the harness: CANDLE_YARN_MODE=cos
    let mode = std::env::var("CANDLE_YARN_MODE").unwrap_or_else(|_| "<unset>".to_string());
    eprintln!("CANDLE_YARN_MODE={} (expected 'cos')", mode);

    // GPU device + dtype
    let dev = candle_examples::device(false /* cpu */)?;
    assert!(dev.is_cuda(), "CUDA GPU required for this test");
    let dtype = DType::BF16;
    eprintln!("device={:?} dtype={:?}", dev, dtype);

    // Resolve local snapshot and files
    let snap = expand_tilde(SNAPSHOT)?;
    let cfg_bytes = std::fs::read(snap.join("config.json")).context("read config.json")?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes).context("invalid config.json")?;
    let model_files = candle_examples::hub_load_local_safetensors(&snap, INDEX)?;
    assert!(!model_files.is_empty(), "no sharded weights found");

    // Build model
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &dev)? };
    let mut model = GptOssModel::load(vb, &cfg).context("load model")?;

    // Build prompt tokens using Harmony template+tokenizer from snapshot
    let user = Message::from_role_and_content(Role::User, "What is YaRN RoPE?".to_string());
    let tokens = render_then_encode(&snap, &[user], true).context("render+encode")?;
    eprintln!("prompt_len={} last_id={}", tokens.len(), tokens.last().copied().unwrap_or(0));

    // Compute logits for last prompt token (step-0)
    let context_index = 0usize; // full context on first pass
    let t = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &dev)?;
    let logits = model.forward(&t, context_index)?; // (1, T, V)
    let last = logits.i((0, tokens.len() - 1))?.to_dtype(DType::F32)?; // (V)
    let probs = candle_nn::ops::softmax_last_dim(&last)?;
    let v = probs.to_vec1::<f32>()?;

    // Decode top-10 using HF tokenizer for clarity
    let tk = tokenizers::Tokenizer::from_file(snap.join("tokenizer.json")).expect("load tokenizer");
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
    let top = 10.min(idx.len());
    eprintln!("-- Mode=A (cos) top-10 next-token candidates --");
    for &i in &idx[..top] {
        let id = i as u32;
        let s = tk.decode(&[id], /*skip_special_tokens=*/ false).unwrap_or_else(|_| "<dec-err>".to_string());
        eprintln!("  id={:6} p={:.6} tok={}", id, v[i], s);
    }
    let channel_id = 200005usize; // <|channel|>
    eprintln!("p(<|channel|>)={:.6}", v[channel_id]);

    // Log exact scales and cos/sin amplitude (pos 0..7 means first 8 positions)
    let head_dim = cfg.head_dim();
    let base = 1.0f32 / (head_dim as f32).sqrt();
    let attn_factor = model.rope.attention_factor();
    let used_scale = match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
        Some("cos") => base,
        _ => attn_factor * base,
    };
    eprintln!("scales: head_dim={} base=1/sqrt(d)={:.8} attn_factor(mscale)={:.8} softmax_scale_used={:.8}", head_dim, base, attn_factor, used_scale);

    let cos = model.rope.cos_table().to_dtype(DType::F32)?;
    let sin = model.rope.sin_table().to_dtype(DType::F32)?;
    eprintln!("cos/sin mean|.| over first 8 positions (dim-avg):");
    for pos in 0..8usize {
        let c = cos.narrow(0, pos, 1)?.flatten_all()?.abs()?.mean_all()?.to_scalar::<f32>()?;
        let s = sin.narrow(0, pos, 1)?.flatten_all()?.abs()?.mean_all()?.to_scalar::<f32>()?;
        eprintln!("  pos={:<2} mean|cos|={:.6} mean|sin|={:.6}", pos, c, s);
    }

    // Assertions for Mode A: expect <|channel|> to be top-1.
    let argmax = idx[0];
    assert_eq!(argmax, channel_id, "Mode A (cos) must have <|channel|> as top-1");

    Ok(())
}
