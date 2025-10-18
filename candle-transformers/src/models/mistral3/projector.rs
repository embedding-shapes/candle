use candle::{Module, Result, Tensor};
use candle_nn::{linear_b, rms_norm, Linear, RmsNorm, VarBuilder};

use super::config::{Mistral3Config, VisionFeatureLayer};

// Constants/configurable defaults placed below imports as per AGENTS.md
const DEFAULT_EPS_FALLBACK: f64 = 1e-5;

#[derive(Debug, Clone)]
pub struct Mistral3PatchMerger {
    spatial_merge_size: usize,
    patch_size: usize,
    merging_layer: Linear, // maps (d * s * s) -> d (bias = false)
}

impl Mistral3PatchMerger {
    pub fn new(cfg: &Mistral3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_v = cfg.vision_config.inner.hidden_size;
        let s = cfg.spatial_merge_size;
        let merging_layer = linear_b(hidden_v * s * s, hidden_v, false, vb.pp("merging_layer"))?;
        Ok(Self {
            spatial_merge_size: s,
            patch_size: cfg.vision_config.inner.patch_size,
            merging_layer,
        })
    }

    pub fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    /// Forward pass matching the PyTorch `Mistral3PatchMerger` logic.
    ///
    /// - `image_features` shape: (total_tokens_across_images, hidden_v).
    /// - `image_sizes` is a slice of (H, W) in pixels for each image.
    ///
    /// It reduces token count by merging non-overlapping s×s windows (s = spatial_merge_size),
    /// concatenating features inside each window and applying a linear projection d*s*s -> d.
    pub fn forward(&self, image_features: &Tensor, image_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let d = image_features.dim(1)?;
        let s = self.spatial_merge_size;

        // Derive grid sizes in patch units for each image.
        let grid_sizes: Vec<(usize, usize)> = image_sizes
            .iter()
            .map(|(h, w)| (h / self.patch_size, w / self.patch_size))
            .collect();
        let tokens_per_image: Vec<usize> = grid_sizes.iter().map(|(h, w)| h * w).collect();

        // Split the long (sum_i t_i, d) into per image chunks along dim=0.
        let mut chunks: Vec<Tensor> = Vec::with_capacity(tokens_per_image.len());
        let mut start = 0usize;
        for sz in tokens_per_image.iter().copied() {
            chunks.push(image_features.narrow(0, start, sz)?);
            start += sz;
        }

        let mut merged: Vec<Tensor> = Vec::with_capacity(chunks.len());
        for (image_index, image_tokens) in chunks.into_iter().enumerate() {
            let (h, w) = grid_sizes[image_index];
            // Safety: tests and typical usage ensure divisibility by s.
            let h_blocks = h / s;
            let w_blocks = w / s;

            // Reshape to (h, w, d) then to (1, d, h, w).
            let img = image_tokens.reshape((h, w, d))?;
            let img = img.permute((2, 0, 1))?; // (d, h, w)
            let img = img.unsqueeze(0)?; // (1, d, h, w)

            // Space-to-depth (non-overlapping windows):
            // (1, d, h, w) -> (1, d, s, s, h//s, w//s) -> (1, d*s*s, h//s, w//s)
            let y = img
                .reshape((1, d, h_blocks, s, w_blocks, s))?
                .permute((0, 1, 3, 5, 2, 4))? // (1, d, s, s, h_blocks, w_blocks)
                .reshape((1, d * s * s, h_blocks, w_blocks))?;

            // (1, d*s*s, h_blocks, w_blocks) -> (h_blocks * w_blocks, d*s*s)
            let y = y.permute((0, 2, 3, 1))?.reshape((h_blocks * w_blocks, d * s * s))?;
            // Apply the merging linear layer on each window.
            let y = y.apply(&self.merging_layer)?; // (h_blocks * w_blocks, d)
            merged.push(y);
        }

        Tensor::cat(&merged, 0)
    }
}

#[derive(Debug, Clone)]
pub struct Mistral3MultiModalProjector {
    norm: RmsNorm,                // RMSNorm over vision hidden dim
    pub patch_merger: Mistral3PatchMerger,
    linear_1: Linear,             // (hidden_v * num_feature_layers) -> hidden_t
    act: candle_nn::Activation,   // projector_hidden_act
    linear_2: Linear,             // hidden_t -> hidden_t
}

impl Mistral3MultiModalProjector {
    pub fn new(cfg: &Mistral3Config, vb: VarBuilder) -> Result<Self> {
        let eps = cfg.text_config.inner.rms_norm_eps;
        let eps = if eps.is_finite() { eps } else { DEFAULT_EPS_FALLBACK };
        let hidden_v = cfg.vision_config.inner.hidden_size;
        let hidden_t = cfg.text_config.inner.hidden_size;
        let num_feature_layers = match &cfg.vision_feature_layer {
            VisionFeatureLayer::Single(_) => 1,
            VisionFeatureLayer::List(v) => v.len(),
        };

        let norm = rms_norm(hidden_v, eps, vb.pp("norm"))?;
        let patch_merger = Mistral3PatchMerger::new(cfg, vb.pp("patch_merger"))?;
        let linear_1 = linear_b(
            hidden_v * num_feature_layers,
            hidden_t,
            cfg.multimodal_projector_bias,
            vb.pp("linear_1"),
        )?;
        let linear_2 = linear_b(
            hidden_t,
            hidden_t,
            cfg.multimodal_projector_bias,
            vb.pp("linear_2"),
        )?;

        Ok(Self {
            norm,
            patch_merger,
            linear_1,
            act: cfg.projector_hidden_act,
            linear_2,
        })
    }

    /// Forward pass, matching `Mistral3MultiModalProjector` in Transformers.
    ///
    /// - `image_features`: (total_tokens, hidden_v) or (total_tokens, hidden_v * num_feature_layers)
    ///   when concatenating multiple feature layers.
    /// - `image_sizes`: list of (H, W) pixel sizes.
    pub fn forward(&self, image_features: &Tensor, image_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let xs = self.norm.forward(image_features)?;
        let xs = self.patch_merger.forward(&xs, image_sizes)?;
        xs.apply(&self.linear_1)?.apply(&self.act)?.apply(&self.linear_2)
    }
}

impl Module for Mistral3MultiModalProjector {
    fn forward(&self, _xs: &Tensor) -> Result<Tensor> {
        // Not used, as we need image_sizes; keep a clear error to avoid misuse.
        candle::bail!("Use forward(&self, image_features, image_sizes) instead of Module::forward")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::VarMap;

    fn mk_cfg(
        hidden_v: usize,
        hidden_t: usize,
        patch_size: usize,
        spatial_merge_size: usize,
    ) -> Mistral3Config {
        use crate::models::mistral::Config as TxtCfg;
        use crate::models::pixtral::vision_model::Config as VisCfg;

        let text_config = TxtCfg {
            vocab_size: 128,
            hidden_size: hidden_t,
            intermediate_size: hidden_t * 4,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            head_dim: None,
            num_key_value_heads: 2,
            hidden_act: candle_nn::Activation::Silu,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            sliding_window: Some(64),
            use_flash_attn: false,
        };
        let vision_config = VisCfg {
            hidden_size: hidden_v,
            num_channels: 3,
            image_size: 16,
            patch_size,
            rope_theta: 10000.0,
            intermediate_size: hidden_v * 4,
            num_hidden_layers: 2,
            head_dim: None,
            num_attention_heads: 2,
            hidden_act: candle_nn::Activation::Silu,
        };

        Mistral3Config {
            model_type: "mistral3".to_string(),
            image_token_index: 10,
            projector_hidden_act: candle_nn::Activation::Gelu,
            multimodal_projector_bias: false,
            spatial_merge_size,
            vision_feature_layer: VisionFeatureLayer::Single(-1),
            text_config: super::super::config::TextSubConfig {
                inner: text_config,
                model_type: Some("mistral".into()),
            },
            vision_config: super::super::config::VisionSubConfig {
                inner: vision_config,
                model_type: Some("pixtral".into()),
            },
        }
    }

    #[test]
    fn patch_merger_shape_4x4_s2() -> Result<()> {
        let dev = Device::Cpu;
        let hidden_v = 8usize;
        // s = 2 merges 2x2 windows → (4x4) -> (2x2) = 4 tokens per image
        let cfg = mk_cfg(hidden_v, 16, 1, 2);
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let merger = Mistral3PatchMerger::new(&cfg, vb.pp("patch_merger"))?;

        // Two images of 4x4 patches, tokens per image = 16
        let tokens_per_image = 16usize;
        let images = 2usize;
        let total_tokens = tokens_per_image * images;
        let xs = Tensor::zeros((total_tokens, hidden_v), DType::F32, &dev)?;
        let image_sizes = vec![(4, 4), (4, 4)]; // in pixels, patch_size=1 → h=4, w=4

        let ys = merger.forward(&xs, &image_sizes)?;
        assert_eq!(ys.dims2()?, (images * 4, hidden_v));
        Ok(())
    }

    #[test]
    fn patch_merger_value_order_4x4_s2() -> Result<()> {
        // Validate exact unfold/merge ordering against an easily-audited pattern.
        // Use d=1, s=2, patch_size=1. The merging layer reduces 4 values -> 1 via a fixed
        // weight pattern [1, 10, 100, 1000] so ordering errors are obvious.
        let dev = Device::Cpu;
        let hidden_v = 1usize; // d
        let cfg = mk_cfg(hidden_v, 4, 1, 2);

        // Set merging_layer weights explicitly.
        let mut tensors = std::collections::HashMap::new();
        // Weight shape: (d, d*s*s) = (1, 4)
        let w = Tensor::new(&[[1f32, 10.0, 100.0, 1000.0]], &dev)?;
        tensors.insert("patch_merger.merging_layer.weight".to_string(), w);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let merger = Mistral3PatchMerger::new(&cfg, vb.pp("patch_merger"))?;

        // One image with 4x4 tokens laid out row-major with values equal to their linear index.
        // Tokens per image = 16, each token has d=1.
        let h = 4usize;
        let w = 4usize;
        let mut vals = Vec::with_capacity(h * w);
        for i in 0..(h * w) {
            vals.push(i as f32);
        }
        let xs = Tensor::from_vec(vals, (h * w, hidden_v), &dev)?;
        let ys = merger.forward(&xs, &[(h, w)])?; // (4, 1)
        assert_eq!(ys.dims2()?, (4, 1));

        // Expected windows (2x2) in row-major block order:
        // block(0,0): [[0,1],[4,5]] -> 0*1 + 1*10 + 4*100 + 5*1000 = 0 + 10 + 400 + 5000 = 5410
        // block(0,1): [[2,3],[6,7]] -> 2 + 30 + 600 + 7000 = 7632
        // block(1,0): [[8,9],[12,13]] -> 8 + 90 + 1200 + 13000 = 143
        //   Correction: 8 + 90 + 1200 + 13000 = 143 + 151? Let's compute exactly → 8 + 90 + 1200 + 13000 = 143? Wrong.
        //   8 + 90 = 98; 98 + 1200 = 1298; 1298 + 13000 = 14298.
        // block(1,1): [[10,11],[14,15]] -> 10 + 110 + 1400 + 15000 = 16520
        let expected = vec![5410f32, 7632.0, 14298.0, 16520.0];
        let got = ys.squeeze(1)?.to_vec1::<f32>()?;
        assert_eq!(got, expected);
        Ok(())
    }

    #[test]
    fn patch_merger_non_square_h4_w2_s2() -> Result<()> {
        // Non-square grid: h=4, w=2 (patch_size=1) with s=2 should yield h/2 * w/2 = 2 * 1 = 2 tokens.
        let dev = Device::Cpu;
        let cfg = mk_cfg(3, 8, 1, 2);
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let merger = Mistral3PatchMerger::new(&cfg, vb.pp("patch_merger"))?;

        let h = 4usize;
        let w = 2usize;
        let tokens = h * w;
        let xs = Tensor::zeros((tokens, 3), DType::F32, &dev)?;
        let ys = merger.forward(&xs, &[(h, w)])?;
        assert_eq!(ys.dims2()?, (2, 3));
        Ok(())
    }

    #[test]
    fn projector_forward_shapes_and_activation() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 6usize;
        // Choose s=1 so PatchMerger is identity on token count and linear projects d->d
        let mut cfg = mk_cfg(hidden, hidden, 1, 1);
        cfg.projector_hidden_act = candle_nn::Activation::Relu;
        cfg.multimodal_projector_bias = false;

        // Build a var map to make the linear layers behave like identity.
        let mut tensors = std::collections::HashMap::new();
        // norm.weight = ones
        let norm_w = Tensor::ones(hidden, DType::F32, &dev)?;
        tensors.insert("multi_modal_projector.norm.weight".to_string(), norm_w);
        // patch_merger.merging_layer.weight = identity (d x d)
        let mut merge_w = Tensor::zeros((hidden, hidden), DType::F32, &dev)?;
        for i in 0..hidden {
            merge_w = merge_w.slice_assign(&[i..i + 1, i..i + 1], &Tensor::new(&[[1f32]], &dev)?)?;
        }
        tensors.insert(
            "multi_modal_projector.patch_merger.merging_layer.weight".to_string(),
            merge_w,
        );
        // linear_1.weight = identity (hidden_t x hidden_v)
        let mut l1_w = Tensor::zeros((hidden, hidden), DType::F32, &dev)?;
        for i in 0..hidden {
            l1_w = l1_w.slice_assign(&[i..i + 1, i..i + 1], &Tensor::new(&[[1f32]], &dev)?)?;
        }
        tensors.insert("multi_modal_projector.linear_1.weight".to_string(), l1_w);
        // linear_2.weight = identity
        let mut l2_w = Tensor::zeros((hidden, hidden), DType::F32, &dev)?;
        for i in 0..hidden {
            l2_w = l2_w.slice_assign(&[i..i + 1, i..i + 1], &Tensor::new(&[[1f32]], &dev)?)?;
        }
        tensors.insert("multi_modal_projector.linear_2.weight".to_string(), l2_w);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let projector = Mistral3MultiModalProjector::new(&cfg, vb.pp("multi_modal_projector"))?;

        // One image 2x2 tokens -> tokens_per_image = 4
        let image_sizes = vec![(2, 2)];
        let data: Vec<f32> = vec![
            // 4 tokens, each with `hidden` dims; mix negatives to trigger ReLU
            -1.0, -0.5, 0.0, 0.1, 1.0, 2.0,
            -2.0, 0.0, 3.0, -3.0, 0.5, -0.1,
            0.2, -0.2, 0.3, -0.3, 0.4, -0.4,
            -0.9, 0.9, -0.8, 0.8, -0.7, 0.7,
        ];
        let xs = Tensor::from_vec(data, (4, hidden), &dev)?;
        let ys = projector.forward(&xs, &image_sizes)?;
        // Shape: still 4 tokens by `hidden_t` (== hidden).
        assert_eq!(ys.dims2()?, (4, hidden));

        // Activation invoked: with identity weights and RMSNorm, negatives should be clamped by ReLU.
        let ys_v: Vec<f32> = ys.flatten_all()?.to_vec1()?;
        assert!(ys_v.iter().all(|v| *v >= 0.0));
        Ok(())
    }
}
