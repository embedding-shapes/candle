use crate::models::with_tracing::{Embedding, RmsNorm};
use candle_nn::Linear;

use super::config::GptOssConfig;
use super::experts::GptOssExperts;
use super::rotary::GptOssRotaryEmbedding;

pub mod debug;
pub mod forward;
pub mod forward_variants;
pub mod loader;

// Re-export the forward methods
pub use forward::{forward, forward_logits_minimal};
pub use forward_variants::forward_last_hidden_post_norm;

// Re-export debug methods
pub use debug::{
    debug_collect_layer_states_last_token, debug_l0_attn_last_token,
    debug_l0_attn_last_token_logits_weights, debug_l0_qkv_last_token, debug_l0_qkv_weights,
    debug_l0_sinks,
};

#[derive(Debug, Clone)]
pub struct GptOssAttentionWeights {
    pub q_proj: crate::models::with_tracing::Linear,
    pub k_proj: crate::models::with_tracing::Linear,
    pub v_proj: crate::models::with_tracing::Linear,
    pub o_proj: crate::models::with_tracing::Linear,
    pub sinks: candle::Tensor,
}

#[derive(Debug, Clone)]
pub struct GptOssLayerWeights {
    pub attn: GptOssAttentionWeights,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub router: candle_nn::Linear,
    pub experts: GptOssExperts,
}

#[derive(Debug, Clone)]
pub struct GptOssModel {
    pub cfg: GptOssConfig,
    pub embed: Embedding,
    pub norm: RmsNorm,
    pub layers: Vec<GptOssLayerWeights>,
    pub lm_head: Linear,
    pub rope: GptOssRotaryEmbedding,
    pub kv_caches: Vec<candle_nn::kv_cache::RotatingKvCache>,
}
