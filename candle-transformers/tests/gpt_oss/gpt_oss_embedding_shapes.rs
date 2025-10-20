use candle::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::config::GptOssConfig;

const SNAPSHOT_DIR: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const INDEX_FILE: &str = "model.safetensors.index.json";

fn expand_tilde(p: &str) -> std::io::Result<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Path::new(&home).join(rest))
    } else {
        Ok(PathBuf::from(p))
    }
}

fn hub_load_local_safetensors<P: AsRef<std::path::Path>>(
    path: P,
    json_file: &str,
) -> candle::Result<Vec<std::path::PathBuf>> {
    let path = path.as_ref();
    let jsfile = std::fs::File::open(path.join(json_file))?;
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files.into_iter().map(|v| path.join(v)).collect();
    Ok(safetensors_files)
}

#[test]
fn gpt_oss_embedding_rows_match_lm_head() -> Result<(), Box<dyn std::error::Error>> {
    let snap = expand_tilde(SNAPSHOT_DIR)?;
    let idx = snap.join(INDEX_FILE);
    if !idx.exists() {
        eprintln!("index not found at {} — skipping", idx.display());
        return Ok(());
    }

    // Load config for shapes and verify loading succeeds with those shapes.
    let cfg: GptOssConfig = serde_json::from_slice(&std::fs::read(snap.join("config.json"))?)?;

    let model_files = hub_load_local_safetensors(&snap, INDEX_FILE)?;
    if model_files.is_empty() {
        eprintln!("no shards — skipping");
        return Ok(());
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::BF16, &Device::Cpu)? };
    let emb = vb.pp("model.embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    let lm = vb.pp("lm_head").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    let (emb_rows, _) = emb.dims2()?;
    let (lm_rows, _) = lm.dims2()?;
    assert_eq!(emb_rows, lm_rows, "embedding and lm_head rows (vocab) must match");
    assert_eq!(emb.dtype(), DType::BF16);
    assert_eq!(lm.dtype(), DType::BF16);
    Ok(())
}
