use anyhow::{bail, Context as _, Result};
use clap::Parser;

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;

use openai_harmony::{chat::{Message, Role}};
use gpt_oss_tokenizer::{render_then_encode, load_stop_token_ids, extract_final_assistant_text_from_decoded};

// Constants
const DEFAULT_SNAPSHOT_DIR: &str =
    "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const MODEL_INDEX_FILE: &str = "model.safetensors.index.json";
// Special tokens are handled via tokenizer decoding and the shared helper in gpt_oss_tokenizer.

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

    // Optionally prefill the assistant final header to start streaming the final content.
    // This inserts: <|channel|>final<|message|> right after the assistant header.
    let tok_path = snapshot_dir.join("tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
    if args.prefill {
        let (hdr, _unstable) = {
            // Using tokenizers crate API: replicate encode("<|channel|>final<|message|>", allowed)
            // The public API is slightly different; use direct helpers for parity.
            let ids = hf_tok.encode("<|channel|>final<|message|>", /*add_special_tokens=*/ false)
                .map_err(|e| anyhow::anyhow!("tokenizer.encode failed: {e}"))?;
            (ids.get_ids().to_vec(), ())
        };
        tokens.extend(hdr);
    }

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
    // Stream only the assistant final-channel content as it appears.
    let mut index_pos = 0usize; // global position in sequence
    let mut last_emitted_len = 0usize; // number of bytes already emitted from final-channel text
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
        // Streaming: decode and extract after each token; emit only the newly appended portion.
        if let Ok(decoded_full) = hf_tok.decode(&tokens, /*skip_special_tokens=*/ false) {
            if let Some(final_text) = extract_final_assistant_text_from_decoded(&decoded_full) {
                // Emit the delta only; guard against truncation/reset by clamping.
                let new_len = final_text.len();
                if new_len > last_emitted_len {
                    let delta = &final_text[last_emitted_len..];
                    print!("{}", delta);
                    use std::io::Write as _;
                    std::io::stdout().flush().ok();
                    last_emitted_len = new_len;
                }
            }
        }
        if stop_tokens.contains(&next) { break; }
    }

    // If we have streamed some content, terminate the line.
    if last_emitted_len > 0 {
        println!();
    } else {
        // Fallback: decode once and print extracted final portion if available,
        // else decode with skip_special_tokens to ensure we never print control tokens.
        if let Ok(decoded_full) = hf_tok.decode(&tokens, /*skip_special_tokens=*/ false) {
            if let Some(reply) = extract_final_assistant_text_from_decoded(&decoded_full) {
                println!("{}", reply.trim());
            } else {
                let sanitized = hf_tok
                    .decode(&tokens, /*skip_special_tokens=*/ true)
                    .unwrap_or_else(|_| String::from(""));
                println!("{}", sanitized.trim());
            }
        } else {
            let sanitized = hf_tok
                .decode(&tokens, /*skip_special_tokens=*/ true)
                .unwrap_or_else(|_| String::from(""));
            println!("{}", sanitized.trim());
        }
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

// No local extraction helpers; use gpt_oss_tokenizer::extract_final_assistant_text_from_decoded
