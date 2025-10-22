// GPT-OSS-20B (with MXFP4 quantization) implementation.
//
// Supports:
// - YaRN extended RoPE
// - GQA (grouped query attention)
// - MoE with top-k routing
// - Attention sinks
// - Sliding window attention (optional)
// - MXFP4 quantized weights (both per-layer and grouped expert weights)

pub mod attention;
pub mod config;
pub mod experts;
pub mod layers;
pub mod model;
pub mod quantization;
pub mod rotary;

#[cfg(test)]
mod tests;

// Re-export config types
pub use config::{GptOssConfig, RopeScalingConfig};

// Re-export layer types
pub use layers::{select_attn_mode_for_layer, AttnMode, GptOssConfigMinimal, GptOssLayerType};

// Re-export attention functions
pub use attention::{
    eager_attn_with_sinks, eager_attn_windowed_with_sinks, masked_fill, sinks_scale_from_lse,
};

#[cfg(feature = "flash-attn")]
pub use attention::{flash_attn_with_sinks, flash_attn_windowed_with_sinks};

// Re-export expert types
pub use experts::{ExpertMlp, GptOssExperts};

// Re-export rotary types
pub use rotary::{yarn_get_mscale, GptOssRopeConfig, GptOssRotaryEmbedding};

// Re-export quantization functions
pub use quantization::{
    dequantize_mxfp4_linear, detect_mxfp4_pair_names, load_all_experts_linear_mxfp4_grouped,
    load_expert_linear_mxfp4_grouped, load_linear_maybe_mxfp4, MXFP4_BLOCK_BYTES,
    MXFP4_BLOCK_ELEMS,
};

// Re-export model types and functions
pub use model::{
    debug_collect_layer_states_last_token, debug_l0_attn_last_token,
    debug_l0_attn_last_token_logits_weights, debug_l0_qkv_last_token, debug_l0_qkv_weights,
    debug_l0_sinks, forward, forward_last_hidden_post_norm, forward_logits_minimal,
    GptOssAttentionWeights, GptOssLayerWeights, GptOssModel,
};
