use anyhow::Result;
use gpt_oss_tokenizer::{allowed_specials_for_next, unmask_required_harmony_specials};
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};
use std::collections::BTreeSet;

// Helpers
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
        if let Some(id) = hf_tok.token_to_id(sym) { s.insert(id); }
    }
    s
}

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}/{}", home, rest);
        }
    }
    p.to_string()
}

#[test]
fn t01_allowed_after_assistant_requires_channel() -> Result<()> {
    let _enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;

    let ctx = "<|start|>system<|message|>x<|end|>\n<|start|>assistant";
    let allow = allowed_specials_for_next(ctx);
    assert!(allow.contains("<|channel|>"), "<|channel|> must be allowed after assistant header");

    // Simulate an overly aggressive global special-token suppression and ensure unmask restores <|channel|>.
    let tok_path = expand_tilde("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
    let global_forbid = build_global_forbidden(&hf_tok);

    let ch: usize = hf_tok.token_to_id("<|channel|>").expect("channel id") as usize;
    let vocab_len = (global_forbid.iter().copied().max().unwrap_or(0) + 8) as usize;
    let mut logits = vec![f32::NEG_INFINITY; vocab_len];
    // Put a distinctive value at the channel id and check it is preserved.
    logits[ch] = 0.1234f32;
    unmask_required_harmony_specials(&mut logits, &hf_tok, ctx, &global_forbid);
    assert!(logits[ch] > f32::NEG_INFINITY / 2.0, "channel logit must remain finite, not -inf");
    assert_eq!(logits[ch], 0.1234f32, "channel logit must not be altered by unmask");
    Ok(())
}

#[test]
fn t02_allowed_after_channel_requires_message() -> Result<()> {
    let _enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let ctx = "<|start|>assistant<|channel|>final";
    let allow = allowed_specials_for_next(ctx);
    assert!(allow.contains("<|message|>"), "<|message|> must be allowed after channel");

    // Apply the same unmask behavior and verify <|message|> is preserved.
    let tok_path = expand_tilde("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
    let id = hf_tok.token_to_id("<|message|>").expect("message id") as usize;
    let tmp_forbid = build_global_forbidden(&hf_tok);
    let vocab_len = tmp_forbid.iter().copied().max().unwrap_or(0) as usize + 8;
    let mut logits = vec![f32::NEG_INFINITY; vocab_len];
    logits[id] = 0.5678;
    let global_forbid = build_global_forbidden(&hf_tok);
    unmask_required_harmony_specials(&mut logits, &hf_tok, ctx, &global_forbid);
    assert_eq!(logits[id], 0.5678, "message logit must not be altered by unmask");
    Ok(())
}

#[test]
fn t03_after_message_keeps_terminators_unmasked() -> Result<()> {
    let _enc = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?;
    let ctx = "<|start|>assistant<|channel|>final<|message|>Hello";
    let allow = allowed_specials_for_next(ctx);
    for s in ["<|end|>", "<|return|>", "<|call|>"] {
        assert!(allow.contains(s), "{s} must be allowed after message body");
    }

    // Check logits remain untouched for each terminator id.
    let tok_path = expand_tilde("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;
    let global_forbid = build_global_forbidden(&hf_tok);

    for &(sym, val) in [("<|end|>", 1.1f32), ("<|return|>", -0.3f32), ("<|call|>", 0.0f32)].iter() {
        let id = hf_tok.token_to_id(sym).expect("sym id") as usize;
        let tmp_forbid = build_global_forbidden(&hf_tok);
        let mut logits = vec![f32::NEG_INFINITY; tmp_forbid.iter().copied().max().unwrap_or(0) as usize + 8];
        logits[id] = val;
        unmask_required_harmony_specials(&mut logits, &hf_tok, ctx, &global_forbid);
        assert_eq!(logits[id], val, "{sym} logit must not be altered by unmask");
    }
    Ok(())
}
