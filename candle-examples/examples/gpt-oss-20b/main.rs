use anyhow::{Context as _, Result};
use clap::Parser;

use openai_harmony::{
    chat::{Message, Role},
    load_harmony_encoding, HarmonyEncodingName,
};

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

    // Load Harmony encoding for GPT-OSS prompt formatting.
    let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
        .context("failed to load Harmony encoding")?;

    // Build the conversation with a single user message using Harmony’s schema.
    let user_msg = Message::from_role_and_content(Role::User, args.prompt.clone());

    // Format to completion tokens (adds the next assistant header), allowing special tokens.
    let input_ids = encoding
        .render_conversation_for_completion([&user_msg], Role::Assistant, None)
        .context("failed to render conversation for completion")?;

    // Decode back to a string using Harmony’s tokenizer (includes special tokens).
    let decoded = encoding
        .tokenizer()
        .decode_utf8(input_ids.iter().copied())
        .context("failed to decode tokens with Harmony tokenizer")?;

    // Show a concise summary so we can verify prompt formatting behavior now.
    println!("Harmony tokenizer: {}", encoding.tokenizer_name());
    println!("Input tokens: {}", input_ids.len());
    println!("Decoded (first 200 chars): {}", &decoded.chars().take(200).collect::<String>());

    // NOTE: Model loading and generation will be added in subsequent steps.
    // The generation loop will use these `input_ids`, a KV cache, and sampling
    // parameters (seed/temperature/top_p) and then decode outputs via the same tokenizer.

    Ok(())
}

