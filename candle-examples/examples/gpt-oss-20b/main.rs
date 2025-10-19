use anyhow::{bail, Context as _, Result};
use clap::Parser;

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;

use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};
use tokenizers::{EncodeInput, Tokenizer};

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
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };

    // Load Harmony (for prompt format) and the HF tokenizer from the local snapshot (for ids).
    let _encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
        .context("failed to load Harmony encoding")?;

    // HF tokenizer.json is authoritative for this model; use it for encoding/decoding.
    let tokenizer_path = snapshot_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json at {}: {e}", tokenizer_path.display()))?;

    // Build the conversation text in Harmony format (one user message, then next assistant header).
    let prompt_text = build_harmony_prompt_text(&args.prompt, /*prefill_channel=*/args.prefill);
    // Encode using HF tokenizer.
    let enc = tokenizer
        .encode(EncodeInput::Single(prompt_text.clone().into()), /*add_special_tokens=*/false)
        .map_err(|e| anyhow::anyhow!("HF tokenizer encode failed: {e}"))?;
    let mut tokens: Vec<u32> = enc
        .get_ids()
        .iter()
        .map(|&id| id as u32)
        .collect();

    if std::env::var("CANDLE_DEBUG_TOKS").ok().as_deref() == Some("1") {
        eprintln!("first 32 token ids: {:?}", &tokens.iter().take(32).collect::<Vec<_>>());
        let ids_u32: Vec<u32> = tokens.iter().take(32).copied().collect();
        let dec = tokenizer
            .decode(&ids_u32, false)
            .unwrap_or_else(|_| String::from("<decode-error>"));
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

    // Prepare stop tokens by encoding the specials via the HF tokenizer to ensure id parity.
    let token_id_return = encode_single_special(&tokenizer, ST_RETURN)
        .context("failed to encode <|return|> with HF tokenizer")?;
    let token_id_call = encode_single_special(&tokenizer, ST_CALL)
        .context("failed to encode <|call|> with HF tokenizer")?;

    // Stop only on <|return|> and <|call|>. Do not stop on <|end|>.
    let stop_tokens: std::collections::BTreeSet<u32> =
        [token_id_return, token_id_call].into_iter().collect();

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

    // Decode and print the full token stream using the HF tokenizer (raw view).
    let ids_u32: Vec<u32> = tokens.clone();
    let decoded_full = tokenizer
        .decode(&ids_u32, /*skip_special_tokens=*/false)
        .map_err(|e| anyhow::anyhow!("HF tokenizer decode failed: {e}"))?;
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

// --- Helpers (prompt/text + tokenizer encoding) ---

fn build_harmony_prompt_text(user_prompt: &str, prefill_channel: bool) -> String {
    let mut s = String::new();
    // User message
    s.push_str(ST_START);
    s.push_str("user");
    s.push_str(ST_MESSAGE);
    s.push_str(user_prompt);
    s.push_str(ST_END);
    // Next assistant header
    s.push_str(ST_START);
    s.push_str("assistant");
    if prefill_channel {
        s.push_str(ST_CHANNEL);
        s.push_str("final");
        s.push_str(ST_MESSAGE);
    }
    s
}

fn encode_single_special(tokenizer: &Tokenizer, s: &str) -> Result<u32> {
    let enc = tokenizer
        .encode(EncodeInput::Single(s.to_string().into()), false)
        .map_err(|e| anyhow::anyhow!("HF tokenizer encode failed for {s}: {e}"))?;
    let ids = enc.get_ids();
    if ids.len() != 1 {
        bail!("expected single id for special token {s}, got {:?}", ids);
    }
    Ok(ids[0] as u32)
}

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
