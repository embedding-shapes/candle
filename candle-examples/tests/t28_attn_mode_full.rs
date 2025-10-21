use anyhow::Result;
use candle_transformers::models::gpt_oss::{select_attn_mode_for_layer, AttnMode};

#[test]
fn t28_attn_mode_full_always() -> Result<()> {
    // Even if the config advertises sliding layers, we align to HF eager behavior
    // by selecting full attention during inference.
    let cfg = candle_transformers::models::gpt_oss::GptOssConfigMinimal {
        num_hidden_layers: 4,
        layer_types: vec![
            candle_transformers::models::gpt_oss::GptOssLayerType::SlidingAttention,
            candle_transformers::models::gpt_oss::GptOssLayerType::FullAttention,
            candle_transformers::models::gpt_oss::GptOssLayerType::SlidingAttention,
            candle_transformers::models::gpt_oss::GptOssLayerType::FullAttention,
        ],
        max_position_embeddings: 1024,
        sliding_window: Some(128),
    };
    for i in 0..cfg.num_hidden_layers {
        let mode = select_attn_mode_for_layer(&cfg, i);
        assert!(matches!(mode, AttnMode::Full));
    }
    Ok(())
}
