use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;
use std::collections::HashMap;

fn minimal_weight_map(vocab: usize, hidden: usize, dev: &Device) -> HashMap<String, Tensor> {
    let mut tmap: HashMap<String, Tensor> = HashMap::new();
    // Embedding and norm
    let emb = Tensor::zeros((vocab, hidden), DType::BF16, dev).unwrap();
    let norm = Tensor::ones(hidden, DType::BF16, dev).unwrap();
    tmap.insert("model.embed_tokens.weight".to_string(), emb);
    tmap.insert("model.norm.weight".to_string(), norm);
    // Untied lm_head
    let lm = Tensor::zeros((vocab, hidden), DType::BF16, dev).unwrap();
    tmap.insert("lm_head.weight".to_string(), lm);
    tmap
}

fn build_minimal_cfg_with_rope_scaling_key(key: &str) -> String {
    // Build a minimal JSON config that exercises `rope_scaling` alias parsing.
    // We use 0 layers to avoid needing to populate per-layer weights.
    // The `rope_scaling` object only provides the type under the given key; all
    // other YARN parameters should fall back to defaults in the loader.
    format!(
        r#"{{
    "vocab_size": 32,
    "hidden_size": 64,
    "num_hidden_layers": 0,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "num_local_experts": 1,
    "num_experts_per_tok": 1,
    "intermediate_size": 16,
    "max_position_embeddings": 1024,
    "layer_types": [],
    "sliding_window": null,
    "rope_scaling": {{ "{}": "yarn" }}
}}"#,
        key
    )
}

fn load_model_from_cfg_json(cfg_json: &str) -> Result<()> {
    let cfg: GptOssConfig = serde_json::from_str(cfg_json).unwrap();
    // Build minimal VarBuilder with only embedding, norm and lm_head; 0 layers.
    let dev = {
        #[cfg(feature = "cuda")]
        { Device::new_cuda(0)? }
        #[cfg(not(feature = "cuda"))]
        { Device::Cpu }
    };
    let tmap = minimal_weight_map(cfg.vocab_size, cfg.hidden_size, &dev);
    let vb = candle_nn::VarBuilder::from_tensors(tmap, DType::BF16, &dev);
    let _model = GptOssModel::load(vb, &cfg)?;
    Ok(())
}

#[test]
fn t20_rope_scaling_alias_rope_type() -> Result<()> {
    let j = build_minimal_cfg_with_rope_scaling_key("rope_type");
    load_model_from_cfg_json(&j)
}

#[test]
fn t21_rope_scaling_alias_type() -> Result<()> {
    let j = build_minimal_cfg_with_rope_scaling_key("type");
    load_model_from_cfg_json(&j)
}
