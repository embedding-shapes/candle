use anyhow::Result;
use candle::{DType, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::collections::BTreeSet;

use gpt_oss_tokenizer::allowed_specials_for_next;

const SNAPSHOT_DIR: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}/{}", home, rest);
        }
    }
    p.to_string()
}

fn build_global_forbidden(hf_tok: &tokenizers::Tokenizer) -> BTreeSet<u32> {
    let mut s = BTreeSet::new();
    for sym in [
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>",
        "<|start|>",
    ] {
        if let Some(id) = hf_tok.token_to_id(sym) {
            s.insert(id);
        }
    }
    s
}

#[test]
fn t26_harmony_gating_masks_disallowed_specials() -> Result<()> {
    // Load tokenizer for id lookups
    let tok_path = expand_tilde(&format!("{}/{}", SNAPSHOT_DIR, "tokenizer.json"));
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;

    // Context right after assistant header: only <|channel|> is allowed next.
    let decoded_so_far = "<|start|>assistant";
    let allow = allowed_specials_for_next(decoded_so_far);
    assert!(allow.contains("<|channel|>"));

    // Convert allowed specials to ids.
    let mut allowed_ids: BTreeSet<u32> = BTreeSet::new();
    for s in allow {
        if let Some(id) = hf_tok.token_to_id(s) {
            allowed_ids.insert(id);
        }
    }
    let global_forbid = build_global_forbidden(&hf_tok);
    let to_mask: Vec<u32> = global_forbid
        .iter()
        .copied()
        .filter(|id| !allowed_ids.contains(id))
        .collect();

    // Create trivial logits vector (all zeros) and capture probabilities post-softmax & mask.
    let vocab_len = (global_forbid.iter().copied().max().unwrap_or(0) + 8) as usize;
    let logits = Tensor::zeros(vocab_len, DType::F32, &candle::Device::Cpu)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::All { temperature: 1.0 });
    let mut captured: Option<Vec<f32>> = None;
    let _ = sampler.sample_f(&logits, |prs: &mut [f32]| {
        for id in &to_mask {
            let i = *id as usize;
            if i < prs.len() {
                prs[i] = 0.0;
            }
        }
        captured = Some(prs.to_vec());
    })?;

    let prs = captured.expect("captured probs");
    // Disallowed specials must be zero probability; the single allowed <|channel|> must remain > 0.
    for id in &to_mask {
        assert_eq!(
            prs[*id as usize], 0.0,
            "disallowed special id {} must be masked to 0",
            id
        );
    }
    if let Some(ch) = hf_tok.token_to_id("<|channel|>") {
        assert!(prs[ch as usize] > 0.0, "<|channel|> must remain unmasked");
    }
    Ok(())
}
