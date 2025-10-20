use anyhow::{bail, Context as _, Result};
use clap::Parser;

use candle::{DType, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;

use openai_harmony::{chat::{Message, Role}};
use gpt_oss_tokenizer::{render_then_encode, load_stop_token_ids, extract_final_assistant_text_from_decoded, allowed_specials_for_next};

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

    // Debug: device + dtype + snapshot
    if device.is_cuda() {
        println!("Device set to use cuda:0");
    } else {
        println!("Device set to use {:?}", device);
    }
    println!("dtype: {}", match dtype { DType::BF16 => "bf16", DType::F16 => "f16", _ => "other" });
    println!("Snapshot: {}", snapshot_dir.display());

    // Conversation: one user message. Render via the model's chat_template.jinja and
    // encode with tokenizer.json from the same snapshot. This guarantees exact parity.
    let user_msg = Message::from_role_and_content(Role::User, args.prompt.clone());
    println!(
        "messages: [{}]",
        format!("{{'role': 'user', 'content': '{}'}}", args.prompt.replace('\n', "\\n").replace('\'', "\\'"))
    );
    let mut tokens: Vec<u32> = render_then_encode(&snapshot_dir, &[user_msg], true)
        .context("failed to render+encode with chat_template.jinja + tokenizer.json")?;
    // Dump both the first 32 and the full set of prompt tokens for exact parity inspection.
    println!("first 32 token ids: {:?}", &tokens.iter().take(32).copied().collect::<Vec<_>>());
    println!("prompt token ids: {:?}", &tokens);
    let ids_u32: Vec<u32> = tokens.iter().take(32).copied().collect();
    let tok_path = snapshot_dir.join("tokenizer.json");
    // Decode and verify the tail of the prompt includes the assistant header markers.
    match tokenizers::Tokenizer::from_file(&tok_path) {
        Ok(hf_tok) => match hf_tok.decode(&ids_u32, /*skip_special_tokens=*/ false) {
            Ok(s) => println!("first 32 decode: {}", s),
            Err(_) => println!("first 32 decode: <decode-error>"),
        },
        Err(_) => println!("first 32 decode: <tokenizer-load-error>"),
    }
    if let Ok(hf_tok_full) = tokenizers::Tokenizer::from_file(&tok_path) {
        if let Ok(full_dec) = hf_tok_full.decode(&tokens, /*skip_special_tokens=*/ false) {
            let tail_check = "<|start|>assistant<|channel|>final<|message|>";
            let has_tail = full_dec.contains(tail_check);
            println!("prompt tail contains '{}': {}", tail_check, has_tail);
            println!("prompt decode: {}", full_dec);
        }
    }

    // Load GPT-OSS config from the local snapshot.
    let cfg_path = snapshot_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path)
        .with_context(|| format!("failed to read config: {}", cfg_path.display()))?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes).context("invalid config.json")?;
    // Debug: print a compact config summary similar to Python
    {
        let rope_str = if let Some(r) = cfg.rope_scaling.as_ref() {
            let ty = r.r#type.as_deref().unwrap_or("?");
            let f = r.factor.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string());
            let bf = r.beta_fast.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string());
            let bs = r.beta_slow.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string());
            let orig = r
                .original_max_position_embeddings
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".to_string());
            format!("{{'type': '{}', 'factor': {}, 'beta_fast': {}, 'beta_slow': {}, 'original_max_position_embeddings': {}}}", ty, f, bf, bs, orig)
        } else {
            "null".to_string()
        };
        println!(
            "Config: hidden_size={} layers={} heads={} kv_heads={} head_dim={} max_pos={} rope={} sliding_window={:?}",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            cfg.max_position_embeddings,
            rope_str,
            cfg.sliding_window,
        );
    }

    // Resolve safetensors shard files from the local index.
    let model_files = candle_examples::hub_load_local_safetensors(&snapshot_dir, MODEL_INDEX_FILE)
        .with_context(|| format!("failed to read index {MODEL_INDEX_FILE} under {}", snapshot_dir.display()))?;
    if model_files.is_empty() {
        bail!("no safetensors files found under {}", snapshot_dir.display());
    }
    // Debug: shard summary (count and first few filenames)
    {
        let first: Vec<_> = model_files
            .iter()
            .take(3)
            .map(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())
            .collect();
        println!("Shards: count={} first={:?}", model_files.len(), first);
    }

    // Map weights and instantiate the model.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let mut model = GptOssModel::load(vb, &cfg).context("failed to load GPT-OSS model weights")?;

    // Debug: check lm_head weights
    {
        let w = model.lm_head.weight();
        let row0 = w.i(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let row_ch = w.i(200005)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        eprintln!("lm_head[0, :8] = {:?}", &row0[..8]);
        eprintln!("lm_head[200005, :8] = {:?}", &row_ch[..8]);
    }

    // Load tokenizer for decoding and inspection.
    let tok_path = snapshot_dir.join("tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;

    // Build a conservative global set of Harmony specials that are suppressed by default
    // and re-enabled step-by-step via the Harmony grammar. This mirrors the unit tests.
    let mut globally_forbidden_special_ids: std::collections::BTreeSet<u32> = Default::default();
    for sym in [
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>",
        "<|start|>",
    ] {
        if let Some(id) = hf_tok.token_to_id(sym) { globally_forbidden_special_ids.insert(id); }
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
    // This includes <|return|> (200002), <|call|> (200012) and <|endoftext|> (199999 pad).
    // We do not stop on <|end|> (200007).
    let stop_ids = load_stop_token_ids(&snapshot_dir)?;
    println!("stop token ids: {:?}", stop_ids);
    if let Ok(hf_tok2) = tokenizers::Tokenizer::from_file(&tok_path) {
        let bos = hf_tok2.token_to_id("<|startoftext|>").unwrap_or(u32::MAX);
        let eos = hf_tok2.token_to_id("<|endoftext|>").unwrap_or(u32::MAX);
        let ret = hf_tok2.token_to_id("<|return|>").unwrap_or(u32::MAX);
        let call = hf_tok2.token_to_id("<|call|>").unwrap_or(u32::MAX);
        println!(
            "special ids: {{'<|startoftext|>': {}, '<|endoftext|>': {}, '<|return|>': {}, '<|call|>': {}}}",
            bos, eos, ret, call
        );
    }
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
        if step == 0 {
            // Debug: check hidden states at various points
            // Get hidden right before lm_head (after final norm)
            let hidden_post_norm = model.forward_last_hidden_post_norm(&t, context_index)?;
            let hn_vec = hidden_post_norm.to_vec1::<f32>()?;
            eprintln!("  Hidden (after final norm) last token [:8]: {:?}", &hn_vec[..8]);
        }
        if step == 0 {
            // Inspect the top-10 candidates for the first generated token
            let last_f32 = last.to_dtype(DType::F32)?;
            let logits_vec = last_f32.to_vec1::<f32>()?;
            let probs = candle_nn::ops::softmax_last_dim(&last_f32)?;
            let v = probs.to_vec1::<f32>()?;
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap());
            let topn = 10usize.min(idx.len());
            let tok_path = snapshot_dir.join("tokenizer.json");
            if let Ok(tk) = tokenizers::Tokenizer::from_file(&tok_path) {
                if let Some(ch_id) = tk.token_to_id("<|channel|>") {
                    println!("  RAW: '<|channel|>' id={} logit={:.6}", ch_id, logits_vec[ch_id as usize]);
                }
                println!("top-10 next-token candidates:");
                for &i in &idx[..topn] {
                    let id = i as u32;
                    let s = tk.decode(&[id], /*skip_special_tokens=*/ false).unwrap_or_else(|_| "<dec-err>".to_string());
                    println!("  id={:6} p={:.4} tok={}", id, v[i], s);
                }
                if let Some(ch_id) = tk.token_to_id("<|channel|>") {
                    println!("  special '<|channel|>' id={} p={:.6}", ch_id, v[ch_id as usize]);
                } else {
                    println!("  special '<|channel|>' not present in tokenizer");
                }
            }
        }
        // Decode the context so far and compute the Harmony-allowed specials for the next step.
        // Then apply a probability mask that disables disallowed specials, keeping only those
        // required by the grammar enabled for sampling.
        let decoded_so_far = hf_tok
            .decode(&tokens, /*skip_special_tokens=*/ false)
            .unwrap_or_else(|_| String::new());
        let allow_syms = allowed_specials_for_next(&decoded_so_far);
        let mut allowed_ids: std::collections::BTreeSet<u32> = Default::default();
        for s in allow_syms.iter() {
            if let Some(id) = hf_tok.token_to_id(s) { allowed_ids.insert(id); }
        }
        // Compute the set of disallowed special ids this step.
        let to_mask: Vec<u32> = globally_forbidden_special_ids
            .iter()
            .copied()
            .filter(|id| !allowed_ids.contains(id))
            .collect();
        // Harmony grammar: immediately after "<|start|>assistant" we must emit "<|channel|>",
        // and immediately after "<|channel|>" we must emit "<|message|>". At those two
        // boundary steps, disable all non-special tokens so sampling can only choose the
        // required special token. After "<|message|>", do not force specials; free text
        // is allowed (terminators remain allowed but not forced).
        let must_force_special = allow_syms.contains("<|channel|>") || allow_syms.contains("<|message|>");
        let next = sampler.sample_f(&last, |prs: &mut [f32]| {
            for id in &to_mask {
                let idx = *id as usize;
                if idx < prs.len() { prs[idx] = 0.0; }
            }
            if must_force_special {
                // Zero out all non-special tokens and any specials not explicitly allowed.
                // Retain only the allowed special ids in `allowed_ids`.
                for (i, p) in prs.iter_mut().enumerate() {
                    if !allowed_ids.contains(&(i as u32)) { *p = 0.0; }
                }
            }
        })?;
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

    // Always print the full Harmony string (request + response with control tokens)
    if let Ok(decoded_full) = hf_tok.decode(&tokens, /*skip_special_tokens=*/ false) {
        println!("\n{}", decoded_full);
    }

    // If we have streamed some content, terminate the line after raw print.
    if last_emitted_len == 0 {
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
