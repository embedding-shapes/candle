use super::GptOssLayerType;

// GPT-OSS configuration matching the subset of Hugging Face fields we rely on for
// model assembly and weight loading. The structure is intentionally minimal and
// focused on wiring; behavior beyond loading is handled in other modules.

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default)]
    pub r#type: Option<String>, // typically "yarn"
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GptOssConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Some GPT-OSS configs include `head_dim`; when absent it defaults to hidden/heads.
    #[serde(default)]
    pub head_dim: Option<usize>,

    // MoE
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub intermediate_size: usize, // per-expert fused gate_up uses 2*intermediate_size

    // Positional
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    // Attention layout
    #[serde(default)]
    pub layer_types: Vec<GptOssLayerType>,
    #[serde(default)]
    pub sliding_window: Option<usize>,

    // Numerics / norms
    #[serde(default)]
    pub rms_norm_eps: Option<f64>,
}

impl GptOssConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or_else(|| self.hidden_size / self.num_attention_heads)
    }

    pub fn effective_layer_types(&self) -> Vec<GptOssLayerType> {
        if self.layer_types.is_empty() {
            vec![GptOssLayerType::FullAttention; self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        }
    }
}

