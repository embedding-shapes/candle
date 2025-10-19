use anyhow::{Context as _, Result};
use tokenizers::{EncodeInput, Tokenizer};

const SNAPSHOT_DIR: &str =
    "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

fn expand_tilde(p: &str) -> Result<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").context("$HOME not set for ~ expansion")?;
        Ok(Path::new(&home).join(rest))
    } else {
        Ok(PathBuf::from(p))
    }
}

#[test]
fn t24_hf_tokenizer_special_tokens_roundtrip() -> Result<()> {
    let snap = expand_tilde(SNAPSHOT_DIR)?;
    let tok_path = snap.join("tokenizer.json");
    if !tok_path.exists() {
        // Optional skip if snapshot not present.
        eprintln!("tokenizer.json not found at {} — skipping", tok_path.display());
        return Ok(());
    }
    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;

    for s in ["<|start|>", "<|message|>", "<|end|>", "<|return|>", "<|call|>", "<|channel|>"] {
        let enc = tokenizer
            .encode(EncodeInput::Single(s.to_string().into()), /*add_special_tokens=*/false)
            .map_err(|e| anyhow::anyhow!("encode failed for {s}: {e}"))?;
        let ids = enc.get_ids();
        assert_eq!(ids.len(), 1, "special {s} should encode to exactly 1 id, got {:?}", ids);
        let ids_u32 = [ids[0]];
        let dec = tokenizer
            .decode(&ids_u32, /*skip_special_tokens=*/false)
            .map_err(|e| anyhow::anyhow!("decode failed for {s}: {e}"))?;
        assert_eq!(dec, s, "round-trip mismatch for {s}");
    }
    Ok(())
}
