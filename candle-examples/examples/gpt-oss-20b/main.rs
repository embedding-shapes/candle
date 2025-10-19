use anyhow::{bail, Context as _, Result};
use clap::Parser;

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;

use openai_harmony::{chat::{Message, Role}};
use gpt_oss_tokenizer::{render_then_encode, load_stop_token_ids};

// Constants
const DEFAULT_SNAPSHOT_DIR: &str =
    "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const MODEL_INDEX_FILE: &str = "model.safetensors.index.json";
// Special token strings used in Harmony/GPT-OSS formatting.
const ST_START: &str = "<|start|>";
const ST_MESSAGE: &str = "<|message|>";
const ST_END: &str = "<|end|>"; // not a stop criterion for sampling here
const ST_CALL: &str = "<|call|>"; // stop criterion
const ST_RETURN: &str = "<|return|>"; // primary stop criterion
const ST_CHANNEL: &str = "<|channel|>";

#[derive(Parser, Debug)]
#[command(author, version, about = "GPT-OSS-20B example (Harmony prompt formatting)")]
struct Args {
    /// The user prompt to format and tokenize via Harmony.
    #[arg(long)]
    prompt: String,

    /// Random seed for sampling (reserved for later model generation).
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Temperature for sampling (reserved for later model generation).
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// Nucleus sampling probability cutoff (reserved for later model generation).
    #[arg(long)]
    top_p: Option<f64>,

    /// Maximum new tokens to sample (reserved for later model generation).
    #[arg(long, default_value_t = 256)]
    sample_len: usize,

    /// Prefill the assistant header with "<|channel|>final<|message|>" before sampling.
    #[arg(long, default_value_t = true)]
    prefill: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Resolve the local HF snapshot path (expand leading ~ for convenience).
    let snapshot_dir = expand_tilde(DEFAULT_SNAPSHOT_DIR)?;

    // Select device: prefer CUDA as this example targets GPU + flash-attn.
    let device = candle_examples::device(false /* cpu */)?;
    if !device.is_cuda() {
        eprintln!("Warning: CUDA not available, running on {:?}", device);
    }
    let mut dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };
    if matches!(std::env::var("CANDLE_FORCE_BF16").ok().as_deref(), Some("1") | Some("true") | Some("TRUE")) {
        dtype = DType::BF16;
    }

    // Conversation: one user message. Render via the model's chat_template.jinja and
    // encode with tokenizer.json from the same snapshot. This guarantees exact parity.
    let user_msg = Message::from_role_and_content(Role::User, args.prompt.clone());
    let mut tokens: Vec<u32> = render_then_encode(&snapshot_dir, &[user_msg], true)
        .context("failed to render+encode with chat_template.jinja + tokenizer.json")?;

    if std::env::var("CANDLE_DEBUG_TOKS").ok().as_deref() == Some("1") {
        eprintln!("first 32 token ids: {:?}", &tokens.iter().take(32).collect::<Vec<_>>());
        let ids_u32: Vec<u32> = tokens.iter().take(32).copied().collect();
        let tok_path = snapshot_dir.join("tokenizer.json");
        match tokenizers::Tokenizer::from_file(&tok_path) {
            Ok(hf_tok) => {
                // Keep special tokens to inspect structural tags when debugging.
                match hf_tok.decode(&ids_u32, /*skip_special_tokens=*/ false) {
                    Ok(s) => eprintln!("first 32 decode: {}", s),
                    Err(_) => eprintln!("first 32 decode: <decode-error>"),
                }
            }
            Err(_) => eprintln!("first 32 decode: <tokenizer-load-error>"),
        }
    }

    // Load GPT-OSS config from the local snapshot.
    let cfg_path = snapshot_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path)
        .with_context(|| format!("failed to read config: {}", cfg_path.display()))?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes).context("invalid config.json")?;

    // Resolve safetensors shard files from the local index.
    let model_files = candle_examples::hub_load_local_safetensors(&snapshot_dir, MODEL_INDEX_FILE)
        .with_context(|| format!("failed to read index {MODEL_INDEX_FILE} under {}", snapshot_dir.display()))?;
    if model_files.is_empty() {
        bail!("no safetensors files found under {}", snapshot_dir.display());
    }

    // Map weights and instantiate the model.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let mut model = GptOssModel::load(vb, &cfg).context("failed to load GPT-OSS model weights")?;

    // Set up the logits processor / sampler.
    let sampling = if args.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match args.top_p {
            None => Sampling::All { temperature: args.temperature },
            Some(p) => Sampling::TopP { p, temperature: args.temperature },
        }
    };
    let mut sampler = LogitsProcessor::from_sampling(args.seed, sampling);

    // Stop tokens loaded from generation_config.json to match HF exactly.
    let stop_ids = load_stop_token_ids(&snapshot_dir)?; // includes eos list + pad
    let stop_tokens: std::collections::BTreeSet<u32> = stop_ids.into_iter().collect();

    // Decode loop using full forward with KV cache, RoPE (YARN), sinks and flash-attn.
    let mut index_pos = 0usize; // global position in sequence
    for step in 0..args.sample_len {
        let (context_size, context_index) = if step > 0 { (1usize, index_pos) } else { (tokens.len(), 0usize) };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let t = Tensor::from_vec(ctxt.to_vec(), (1, context_size), &device)?;
        // Optional debug: check lm_head-only projection from embeddings to see vocab alignment.
        if step == 0 && std::env::var("CANDLE_DEBUG_HEAD_ONLY").ok().as_deref() == Some("1") {
            let test_logits = model.forward_logits_minimal(&t)?; // (1, context_size, vocab)
            let last = test_logits.i((0, context_size - 1))?;
            let last = last.to_dtype(DType::F32)?;
            let probs = candle_nn::ops::softmax_last_dim(&last)?;
            let v = probs.to_vec1::<f32>()?;
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
            let topn = 5usize;
            let tok_path = snapshot_dir.join("tokenizer.json");
            if let Ok(tk) = tokenizers::Tokenizer::from_file(&tok_path) {
                eprintln!("[head-only] top-5 candidates:");
                for &i in &idx[..topn] {
                    let id = i as u32;
                    let s = tk.decode(&[id], /*skip_special_tokens=*/ false).unwrap_or_else(|_| "<dec-err>".to_string());
                    eprintln!("  id={:6} p={:.4} tok={}", id, v[i], s);
                }
                if let Some(ch_id) = tk.token_to_id("<|channel|>") {
                    eprintln!("  [head-only] '<|channel|>' id={} p={:.6}", ch_id, v[ch_id as usize]);
                }
            }
        }

        let logits = model.forward(&t, context_index)?; // (1, context_size, vocab)
        let last = logits.i((0, context_size - 1))?; // (vocab)
        if step == 0 && std::env::var("CANDLE_DEBUG_TOPK").ok().as_deref() == Some("1") {
            // Inspect the top-10 candidates for the first generated token
            let last_f32 = last.to_dtype(DType::F32)?;
            let probs = candle_nn::ops::softmax_last_dim(&last_f32)?;
            let v = probs.to_vec1::<f32>()?;
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
            let topn = 10usize.min(idx.len());
            let tok_path = snapshot_dir.join("tokenizer.json");
            if let Ok(tk) = tokenizers::Tokenizer::from_file(&tok_path) {
                eprintln!("top-10 next-token candidates:");
                for &i in &idx[..topn] {
                    let id = i as u32;
                    let s = tk.decode(&[id], /*skip_special_tokens=*/ false).unwrap_or_else(|_| "<dec-err>".to_string());
                    eprintln!("  id={:6} p={:.4} tok={}", id, v[i], s);
                }
                if let Some(ch_id) = tk.token_to_id("<|channel|>") {
                    eprintln!("  special '<|channel|>' id={} p={:.6}", ch_id, v[ch_id as usize]);
                } else {
                    eprintln!("  special '<|channel|>' not present in tokenizer");
                }
            }
        }
        let next = sampler.sample(&last)?;
        tokens.push(next);
        index_pos += context_size;
        if stop_tokens.contains(&next) { break; }
    }

    // Decode via tokenizer.json to extract assistant final content and print only that.
    let tok_path = snapshot_dir.join("tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
    // Preserve special tokens so we can locate the assistant final channel content.
    let decoded_full = hf_tok
        .decode(&tokens, /*skip_special_tokens=*/ false)
        .unwrap_or_else(|_| String::from("<decode-error>"));
    if let Some(reply) = extract_assistant_final_message(&decoded_full) {
        println!("{}", reply.trim());
    } else {
        // As a fallback, print the decoded raw text once.
        println!("{}", decoded_full.trim());
    }

    Ok(())
}

// Utilities
fn expand_tilde(p: &str) -> Result<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").context("$HOME not set for ~ expansion")?;
        Ok(Path::new(&home).join(rest))
    } else {
        Ok(PathBuf::from(p))
    }
}

// --- Helpers (prompt parsing from decoded string) ---

fn extract_assistant_final_message(decoded: &str) -> Option<String> {
    // Find the last assistant header; then ensure message marker; capture until a stop.
    let start_tag = format!("{}assistant", ST_START);
    let start_pos = decoded.rfind(&start_tag)?;
    let rest = &decoded[start_pos + start_tag.len()..];
    let after_channel = if let Some(pos) = rest.find(ST_CHANNEL) {
        let rest2 = &rest[pos + ST_CHANNEL.len()..];
        // Expect channel name (e.g., "final") immediately following; skip it.
        // Then expect message marker.
        if let Some(mpos) = rest2.find(ST_MESSAGE) {
            &rest2[mpos + ST_MESSAGE.len()..]
        } else {
            return None;
        }
    } else if let Some(mpos) = rest.find(ST_MESSAGE) {
        &rest[mpos + ST_MESSAGE.len()..]
    } else {
        return None;
    };
    // Stop at first of return/call/end or next start.
    let mut end_idx = after_channel.len();
    for stop in [ST_RETURN, ST_CALL, ST_END, ST_START] {
        if let Some(p) = after_channel.find(stop) {
            end_idx = end_idx.min(p);
        }
    }
    Some(after_channel[..end_idx].to_string())
}
