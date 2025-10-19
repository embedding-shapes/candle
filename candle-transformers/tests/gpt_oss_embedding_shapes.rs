use candle::safetensors::MmapedFile;

fn expand_tilde(p: &str) -> std::path::PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        std::path::Path::new(&std::env::var("HOME").unwrap()).join(rest)
    } else {
        std::path::PathBuf::from(p)
    }
}

#[test]
fn embedding_and_lm_head_rows_match() -> Result<(), Box<dyn std::error::Error>> {
    // Resolve file containing both weights via the index (here, shard 2).
    let base = expand_tilde("~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee");
    let shard = base.join("model-00002-of-00002.safetensors");
    let mm = unsafe { MmapedFile::new(&shard) }?;
    let st = mm.deserialize().map_err(|e| format!("deserialize {}: {}", shard.display(), e))?;

    let e = st.tensor("model.embed_tokens.weight")?;
    let l = st.tensor("lm_head.weight")?;
    // Shapes are [vocab, hidden]
    let e_shape = e.shape();
    let l_shape = l.shape();
    assert_eq!(e_shape.len(), 2);
    assert_eq!(l_shape.len(), 2);
    assert_eq!(e_shape[0], l_shape[0], "embedding rows must match lm_head rows");

    // Both should be BF16
    let e_dt = format!("{:?}", e.dtype());
    let l_dt = format!("{:?}", l.dtype());
    assert_eq!(e_dt, "BF16");
    assert_eq!(l_dt, "BF16");
    Ok(())
}
