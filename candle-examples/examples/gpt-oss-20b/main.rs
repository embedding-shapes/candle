use anyhow::{bail, Context as _, Result};
use clap::Parser;

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;

use openai_harmony::{chat::{Message, Role}, load_harmony_encoding, HarmonyEncodingName};

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
    #[arg(long, default_value_t = 0.8)]
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

    // Load Harmony (for prompt format + tokenizer) and build the prompt tokens via Harmony.
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
        .context("failed to load Harmony encoding")?;
    let tok = encoding.tokenizer();

    // Conversation: one user message. Render completed history first.
    let user_msg = Message::from_role_and_content(Role::User, args.prompt.clone());
    let mut tokens: Vec<u32> = encoding
        .render_conversation([&user_msg], None)
        .context("failed to render conversation")?
        .into_iter()
        .collect();

    // Spec: insert a newline between <|end|> and the next <|start|>assistant.
    // Use Harmony tokenizer for ordinary encoding of "\n".
    tokens.extend(tok.encode_ordinary("\n"));

    // Append the assistant header start explicitly: <|start|>assistant
    tokens.extend(tok.encode_with_special_tokens("<|start|>"));
    tokens.extend(tok.encode_ordinary("assistant"));

    // Optionally prefill the assistant header channel and message marker.
    if args.prefill {
        let (hdr, _unstable) = tok.encode("<|channel|>final<|message|>", &tok.special_tokens());
        tokens.extend(hdr);
    }

    if std::env::var("CANDLE_DEBUG_TOKS").ok().as_deref() == Some("1") {
        eprintln!("first 32 token ids: {:?}", &tokens.iter().take(32).collect::<Vec<_>>());
        let ids_u32: Vec<u32> = tokens.iter().take(32).copied().collect();
        let dec = tok.decode_utf8(ids_u32.into_iter()).unwrap_or_else(|_| String::from("<decode-error>"));
        eprintln!("first 32 decode: {}", dec);
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

    // Stop only on <|return|> and <|call|> (Harmony authoritative IDs). Do not stop on <|end|>.
    let stop_set = encoding.stop_tokens_for_assistant_actions().context("failed to resolve Harmony stop tokens")?;
    let stop_tokens: std::collections::BTreeSet<u32> = stop_set.into_iter().collect();

    // Decode loop using full forward with KV cache, RoPE (YARN), sinks and flash-attn.
    let mut index_pos = 0usize; // global position in sequence
    for step in 0..args.sample_len {
        let (context_size, context_index) = if step > 0 { (1usize, index_pos) } else { (tokens.len(), 0usize) };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let t = Tensor::from_vec(ctxt.to_vec(), (1, context_size), &device)?;
        let logits = model.forward(&t, context_index)?; // (1, context_size, vocab)
        let last = logits.i((0, context_size - 1))?; // (vocab)
        let next = sampler.sample(&last)?;
        tokens.push(next);
        index_pos += context_size;
        if stop_tokens.contains(&next) { break; }
    }

    // Decode and print using Harmony’s tokenizer (raw view with specials).
    let decoded_full = tok
        .decode_utf8(tokens.iter().copied())
        .unwrap_or_else(|_| String::from("<decode-error>"));
    println!("{}", decoded_full);

    // Lightweight parsing from the decoded string: extract assistant final channel content.
    if let Some(reply) = extract_assistant_final_message(&decoded_full) {
        println!("{}", reply);
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
