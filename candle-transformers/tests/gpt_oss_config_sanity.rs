use candle_transformers::models::gpt_oss::config::GptOssConfig;

const SNAPSHOT_DIR: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

fn expand_tilde(p: &str) -> std::io::Result<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Path::new(&home).join(rest))
    } else {
        Ok(PathBuf::from(p))
    }
}

#[test]
fn gpt_oss_config_sanity_from_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    let snap = expand_tilde(SNAPSHOT_DIR)?;
    let cfg_path = snap.join("config.json");
    if !cfg_path.exists() {
        eprintln!("config.json not found at {} — skipping", cfg_path.display());
        return Ok(());
    }
    let cfg_bytes = std::fs::read(&cfg_path)?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes)?;

    // Geometry checks
    let _hd = cfg.head_dim();
    assert!(cfg.num_attention_heads % cfg.num_key_value_heads == 0);

    // Sliding window sanity (if set): should be <= max_position_embeddings
    if let Some(w) = cfg.sliding_window {
        assert!(w <= cfg.max_position_embeddings);
    }

    // MoE invariants
    assert!(cfg.num_local_experts >= 1);
    assert!(cfg.num_experts_per_tok >= 1 && cfg.num_experts_per_tok <= cfg.num_local_experts);

    Ok(())
}
