#[derive(Debug, Clone, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GptOssLayerType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct GptOssConfigMinimal {
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub layer_types: Vec<GptOssLayerType>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl GptOssConfigMinimal {
    pub fn effective_layer_types(&self) -> Vec<GptOssLayerType> {
        if self.layer_types.is_empty() {
            vec![GptOssLayerType::FullAttention; self.num_hidden_layers]
        } else {
            self.layer_types.clone()
        }
    }

    pub fn sliding_window_size(&self) -> Option<usize> {
        self.sliding_window
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnMode {
    Full,
    Sliding { left: usize, right: usize },
}

/// Select attention mode for a given `layer_idx` from config.
/// - Full: standard causal attention.
/// - Sliding: windowed attention with left window = `sliding_window` and right window = 0.
pub fn select_attn_mode_for_layer(cfg: &GptOssConfigMinimal, layer_idx: usize) -> AttnMode {
    // Respect the per-layer attention type declared in config. When a sliding window is
    // configured and the layer is marked `SlidingAttention`, use a windowed causal mask
    // with `left = sliding_window` and `right = 0`. Otherwise, run full causal attention.
    let types = cfg.effective_layer_types();
    match types.get(layer_idx) {
        Some(GptOssLayerType::SlidingAttention) => match cfg.sliding_window_size() {
            Some(w) if w > 0 => AttnMode::Sliding { left: w, right: 0 },
            _ => AttnMode::Full,
        },
        _ => AttnMode::Full,
    }
}
