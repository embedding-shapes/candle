use candle_transformers::models::gpt_oss::config::GptOssConfig;

fn expand_tilde(p: &str) -> std::path::PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        std::path::Path::new(&std::env::var("HOME").unwrap()).join(rest)
    } else {
        std::path::PathBuf::from(p)
    }
}

#[test]
fn gpt_oss_config_snapshot_sanity() -> Result<(), Box<dyn std::error::Error>> {
    // Load actual config.json from the snapshot.
    let base = expand_tilde("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee");
    let cfg_path = base.join("config.json");
    let bytes = std::fs::read(&cfg_path).map_err(|e| format!("reading {}: {}", cfg_path.display(), e))?;
    let cfg: GptOssConfig = serde_json::from_slice(&bytes)?;

    // Geometry invariants
    assert!(cfg.hidden_size % cfg.num_attention_heads == 0, "hidden must be multiple of n_heads");
    assert!(cfg.num_attention_heads % cfg.num_key_value_heads == 0, "n_heads must be multiple of n_kv_heads");
    // head_dim may be explicitly provided and not necessarily equal to hidden/n_heads for some configs.

    // Layer types length
    let eff = cfg.effective_layer_types();
    assert_eq!(eff.len(), cfg.num_hidden_layers, "layer_types length must match num_hidden_layers");

    // Sliding window sane
    if let Some(w) = cfg.sliding_window { assert!(w <= cfg.max_position_embeddings); }

    // MoE invariants
    assert!(cfg.num_local_experts > 0);
    assert!(cfg.num_experts_per_tok > 0);
    assert!(cfg.num_experts_per_tok <= cfg.num_local_experts);
    assert!(cfg.intermediate_size > 0);

    Ok(())
}
