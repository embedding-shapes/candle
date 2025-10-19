use anyhow::{bail, Context as _, Result};
use clap::Parser;

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
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
const TOKEN_ID_RETURN: u32 = 200002; // <|return|>
const TOKEN_ID_CALL: u32 = 200012; // <|call|>

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

    // Prepare input ids tensor on device.
    let mut tokens: Vec<u32> = input_ids.iter().copied().collect();

    // Optional prefill to force correct first token/channel sequence.
    if args.prefill {
        append_assistant_header(&encoding, &mut tokens)?;
    }

    // Stop only on <|return|> and <|call|>. Do not stop on <|end|>.
    let stop_tokens: std::collections::BTreeSet<u32> =
        [TOKEN_ID_RETURN, TOKEN_ID_CALL].into_iter().collect();

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

    // Decode and print the full token stream without stripping anything (raw view).
    let decoded_full = match encoding.tokenizer().decode_utf8(tokens.iter().copied()) {
        Ok(s) => s,
        Err(_) => {
            let bytes = encoding
                .tokenizer()
                .decode_bytes(tokens.iter().copied())
                .context("failed to decode token bytes with Harmony tokenizer")?;
            String::from_utf8_lossy(&bytes).into_owned()
        }
    };
    println!("{}", decoded_full);

    // Ensure the trailing assistant header is complete (append header if needed),
    // then parse messages from tokens via Harmony to extract a structured assistant reply.
    ensure_complete_assistant_header(&encoding, &mut tokens)?;
    // This handles channels (<|channel|>) and message segmentation correctly.
    let messages = encoding
        .parse_messages_from_completion_tokens(tokens.iter().copied(), None)
        .context("failed to parse messages from completion tokens")?;
    // Prefer the last assistant message with channel=="final"; fall back to last assistant.
    let preferred = messages
        .iter()
        .rev()
        .find(|m| m.author.role == Role::Assistant && m.channel.as_deref() == Some("final"))
        .or_else(|| messages.iter().rev().find(|m| m.author.role == Role::Assistant));
    if let Some(msg) = preferred {
        let text = msg
            .content
            .iter()
            .filter_map(|c| match c {
                openai_harmony::chat::Content::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        println!("{}", text);
    } else {
        // Fallback: raw decode
        let decoded = match encoding.tokenizer().decode_utf8(tokens.iter().copied()) {
            Ok(s) => s,
            Err(_) => {
                let bytes = encoding
                    .tokenizer()
                    .decode_bytes(tokens.iter().copied())
                    .context("failed to decode token bytes with Harmony tokenizer")?;
                String::from_utf8_lossy(&bytes).into_owned()
            }
        };
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

fn append_assistant_header(encoding: &openai_harmony::HarmonyEncoding, tokens: &mut Vec<u32>) -> Result<()> {
    let allowed = encoding.tokenizer().special_tokens();
    let (hdr_tokens, _unstable) = encoding
        .tokenizer()
        .encode("<|channel|>final<|message|>", &allowed);
    tokens.extend(hdr_tokens.into_iter().map(|t| t as u32));
    Ok(())
}

fn ensure_complete_assistant_header(
    encoding: &openai_harmony::HarmonyEncoding,
    tokens: &mut Vec<u32>,
) -> Result<()> {
    let eid = |s: &str| -> Result<u32> {
        let ids = encoding.tokenizer().encode_with_special_tokens(s);
        if ids.len() != 1 { bail!("expected single id for {s}, got {:?}", ids); }
        Ok(ids[0])
    };
    let id_start = eid("<|start|>")?;
    let id_message = eid("<|message|>")?;
    if let Some(start_pos) = tokens.iter().rposition(|&t| t == id_start) {
        let has_message = tokens[start_pos + 1..].iter().any(|&t| t == id_message);
        if !has_message {
            append_assistant_header(encoding, tokens)?;
        }
    }
    Ok(())
}

// No longer needed with Harmony message parsing
