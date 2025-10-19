use anyhow::{bail, Context as _, Result};
use clap::Parser;

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;

use openai_harmony::{
    chat::{Message, Role},
    load_harmony_encoding, HarmonyEncodingName,
};

// Constants
const DEFAULT_SNAPSHOT_DIR: &str =
    "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const MODEL_INDEX_FILE: &str = "model.safetensors.index.json";

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

    // Load Harmony encoding for GPT-OSS prompt formatting.
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
        .context("failed to load Harmony encoding")?;

    // Build the conversation with a single user message using Harmony’s schema.
    let user_msg = Message::from_role_and_content(Role::User, args.prompt.clone());

    // Format to completion tokens (adds the next assistant header), allowing special tokens.
    let input_ids = encoding
        .render_conversation_for_completion([&user_msg], Role::Assistant, None)
        .context("failed to render conversation for completion")?;

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
    let model = GptOssModel::load(vb, &cfg).context("failed to load GPT-OSS model weights")?;

    // Set up the logits processor / sampler.
    let mut sampler = LogitsProcessor::new(args.seed, Some(args.temperature), args.top_p);

    // Prepare input ids tensor on device.
    let mut tokens: Vec<u32> = input_ids.iter().copied().collect();
    let stop_tokens = encoding.stop_tokens().context("could not compute Harmony stop tokens")?;

    // Simple decode loop using the model’s minimal logits forward (embed -> lm_head).
    // This establishes the end-to-end wiring; subsequent steps will replace it with the
    // full transformer decode with attention, KV-cache, sinks, and flash-attn.
    for _ in 0..args.sample_len {
        let t = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &device)?;
        let logits = model.forward_logits_minimal(&t)?; // (1, T, vocab)
        let (_b, _t, _v) = logits.dims3()?;
        let last = logits.i((0, tokens.len() - 1))?; // (vocab)
        let next = sampler.sample(&last)?;
        tokens.push(next);
        if stop_tokens.contains(&next) {
            break;
        }
    }

    // Decode full sequence using Harmony tokenizer and extract assistant text.
    let decoded = match encoding.tokenizer().decode_utf8(tokens.iter().copied()) {
        Ok(s) => s,
        Err(_) => {
            // Fallback: lossy UTF-8 conversion to ensure we always print something.
            let bytes = encoding
                .tokenizer()
                .decode_bytes(tokens.iter().copied())
                .context("failed to decode token bytes with Harmony tokenizer")?;
            String::from_utf8_lossy(&bytes).into_owned()
        }
    };
    if let Some(reply) = extract_assistant_reply(&decoded) {
        println!("{}", reply);
    } else {
        // Fallback: print the decoded string if parsing fails.
        println!("{}", decoded);
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

fn extract_assistant_reply(decoded: &str) -> Option<String> {
    // Expect structure like: ... <|start|>assistant<|message|>...user content...<|end|>
    // Return the segment between the last "<|message|>" after an assistant start and the next "<|end|>".
    let start_assistant = decoded.rfind("<|start|>assistant")?;
    let after_start = &decoded[start_assistant..];
    let msg_pos = after_start.find("<|message|>")? + "<|message|>".len();
    let after_msg = &after_start[msg_pos..];
    let end_pos = after_msg.find("<|end|>")?;
    Some(after_msg[..end_pos].to_string())
}
