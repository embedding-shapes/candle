use candle::{IndexOp, Module, Result, Tensor, DType};
use candle_nn::VarBuilder;

use super::config::{Mistral3Config, VisionFeatureLayer};
use super::projector::Mistral3MultiModalProjector;

// External models
use crate::models::mistral;
use crate::models::pixtral::vision_model;

// Constants/configurable defaults placed below imports as per AGENTS.md

#[derive(Debug, Clone, Default)]
pub struct Mistral3Cache {
    pub image_processed: bool,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub vision_tower: vision_model::Model,
    pub language_model: mistral::Model,
    pub multi_modal_projector: Mistral3MultiModalProjector,
    pub image_token_id: usize,
    pub patch_size: usize,
    pub dtype: DType,
}

impl Model {
    pub fn new(cfg: &Mistral3Config, vb: VarBuilder) -> Result<Self> {
        let language_model = mistral::Model::new(&cfg.text_config.inner, vb.pp("language_model"))?;

        // Vision + projector use F32 to match Pixtral implementation
        let vision_tower = vision_model::Model::new(
            &cfg.vision_config.inner,
            vb.pp("vision_tower").to_dtype(candle::DType::F32),
        )?;
        let multi_modal_projector =
            Mistral3MultiModalProjector::new(cfg, vb.pp("multi_modal_projector").to_dtype(candle::DType::F32))?;

        Ok(Self {
            vision_tower,
            language_model,
            multi_modal_projector,
            image_token_id: cfg.image_token_index,
            patch_size: cfg.vision_config.inner.patch_size,
            dtype: vb.dtype(),
        })
    }

    /// Forward pixels through the vision tower and projector.
    ///
    /// For Magistral, `vision_feature_layer` is -1 (last hidden state). If multiple layers
    /// are ever requested, concatenate along the feature dim before projection. Currently,
    /// Pixtral vision tower does not expose hidden states, so only `Single(-1)` is supported.
    ///
    /// Returns (num_images, hidden_t) by mean-pooling merged tokens per image to provide a
    /// single embedding per [IMG] placeholder.
    pub fn get_image_features(
        &self,
        pixel_values: &Tensor,
        image_sizes: &[(u32, u32)],
        feature_layer: &VisionFeatureLayer,
    ) -> Result<Tensor> {
        if !matches!(feature_layer, VisionFeatureLayer::Single(_)) {
            candle::bail!("Pixtral vision tower only supports a single feature layer (-1)");
        }

        // (B, C, H, W) -> (B, P, hidden_v)
        let feats = self.vision_tower.forward(pixel_values)?;
        let (b, p, h_v) = feats.dims3()?;
        if image_sizes.len() != b {
            candle::bail!(
                "image_sizes length ({}) must equal batch size ({})",
                image_sizes.len(),
                b
            );
        }

        // Flatten to feed projector (concat per image before merge)
        let flat = feats.reshape((b * p, h_v))?;

        // Projector handles: RMSNorm -> PatchMerger -> MLP to text hidden size
        let image_sizes_usize: Vec<(usize, usize)> = image_sizes
            .iter()
            .map(|(h, w)| (*h as usize, *w as usize))
            .collect();
        let projected = self
            .multi_modal_projector
            .forward(&flat, &image_sizes_usize)?; // (sum merged tokens, hidden_t)

        // Compute merged tokens per image and mean-pool to get 1 embedding per image.
        let s = self.multi_modal_projector.patch_merger.spatial_merge_size();
        let mut per_image: Vec<Tensor> = Vec::with_capacity(b);
        let mut start = 0usize;
        for (h_px, w_px) in image_sizes_usize.iter().copied() {
            // Convert pixel sizes to patch grid, then to merged grid size
            let gh = h_px / self.patch_size;
            let gw = w_px / self.patch_size;
            let mh = gh / s;
            let mw = gw / s;
            let t_i = mh * mw;
            if t_i == 0 {
                candle::bail!(
                    "invalid merged token count: image ({h_px}x{w_px}), patch_size {}, s {}",
                    self.patch_size,
                    s
                );
            }
            let seg = projected.narrow(0, start, t_i)?; // (t_i, hidden_t)
            per_image.push(seg.mean(0)?); // (hidden_t)
            start += t_i;
        }
        Tensor::stack(&per_image, 0) // (B, hidden_t)
    }

    /// Find positions of [IMG] tokens in input_ids.
    pub fn find_image_token_positions(
        input_ids: &Tensor,
        image_token_id: usize,
    ) -> Result<Vec<(usize, usize)>> {
        // Normalize dtype to i64 then scan.
        let input_ids = if input_ids.dtype() == candle::DType::U32 {
            input_ids.to_dtype(candle::DType::I64)?
        } else {
            input_ids.clone()
        };
        let ids = input_ids.to_vec2::<i64>()?;
        let mut positions = Vec::new();
        for (b, seq) in ids.iter().enumerate() {
            for (i, &tok) in seq.iter().enumerate() {
                if tok as usize == image_token_id {
                    positions.push((b, i));
                }
            }
        }
        Ok(positions)
    }

    /// Replace [IMG] tokens with corresponding image embeddings.
    pub fn replace_image_tokens(
        inputs_embeds: &Tensor,
        image_embeds: &Tensor,
        positions: &[(usize, usize)],
    ) -> Result<Tensor> {
        if positions.is_empty() {
            return Ok(inputs_embeds.clone());
        }

        let (b, s, h) = inputs_embeds.dims3()?;
        let need = positions.len();
        let (have, eh) = image_embeds.dims2()?;
        if eh != h {
            candle::bail!(
                "hidden size mismatch: inputs {}, image_embeds {}",
                h,
                eh
            );
        }
        if have < need {
            candle::bail!(
                "not enough image embeddings: have {}, need {}",
                have,
                need
            );
        }

        // Align dtypes/devices: cast image_embeds to inputs dtype
        let image_embeds = if image_embeds.dtype() != inputs_embeds.dtype() {
            image_embeds.to_dtype(inputs_embeds.dtype())?
        } else {
            image_embeds.clone()
        };

        let mut result = inputs_embeds.clone();
        for (idx, &(bb, ii)) in positions.iter().enumerate() {
            if bb >= b || ii >= s {
                candle::bail!(
                    "invalid image position: ({}, {}) for ({}, {}, {})",
                    bb,
                    ii,
                    b,
                    s,
                    h
                );
            }
            let emb = image_embeds.i(idx)?; // (h)

            // Mask for the specific position
            let mut mask = vec![0f32; b * s];
            mask[bb * s + ii] = 1.0;
            let mask = Tensor::new(mask.as_slice(), inputs_embeds.device())?
                .reshape((b, s, 1))?
                .to_dtype(inputs_embeds.dtype())?;

            let emb_brd = emb
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as((b, s, h))?;

            let inv = (1.0 - &mask)?;
            result = (result.broadcast_mul(&inv)? + emb_brd.broadcast_mul(&mask)?)?;
        }

        Ok(result)
    }

    /// Forward full multimodal step: embed tokens, optionally insert image embeddings,
    /// then run the language model using embeddings and a sequence position offset.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_sizes: Option<&[(u32, u32)]>,
        cache: &mut Mistral3Cache,
        index_pos: usize,
        feature_layer: &VisionFeatureLayer,
    ) -> Result<Tensor> {
        // Text embeddings from language model's embedding table
        let mut inputs_embeds = self.language_model.embed_tokens().forward(input_ids)?;

        // Insert image embeddings on the first step when provided
        if let (Some(pixels), Some(sizes)) = (pixel_values, image_sizes) {
            if !cache.image_processed {
                let image_embeds = self.get_image_features(pixels, sizes, feature_layer)?; // (B, hidden_t)
                let positions = Self::find_image_token_positions(input_ids, self.image_token_id)?;

                // Require 1 [IMG] placeholder per image
                if positions.len() != image_embeds.dim(0)? {
                    candle::bail!(
                        "mismatch: {} [IMG] tokens vs {} image embeddings",
                        positions.len(),
                        image_embeds.dim(0)?
                    );
                }

                inputs_embeds = Self::replace_image_tokens(&inputs_embeds, &image_embeds, &positions)?;
                cache.image_processed = true;
            }
        }

        // Language model forward from embeddings
        self.language_model.forward_embeds(&inputs_embeds, None, index_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};
    use candle_nn::VarMap;

    fn tiny_cfg() -> Mistral3Config {
        use crate::models::mistral::Config as TxtCfg;
        use crate::models::pixtral::vision_model::Config as VisCfg;

        let text_config = TxtCfg {
            vocab_size: 128,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            head_dim: None,
            num_key_value_heads: 4,
            hidden_act: candle_nn::Activation::Silu,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            sliding_window: Some(64),
            use_flash_attn: false,
        };
        let vision_config = VisCfg {
            hidden_size: 16,
            num_channels: 3,
            image_size: 8,
            patch_size: 2,
            rope_theta: 10000.0,
            intermediate_size: 32,
            num_hidden_layers: 2,
            head_dim: None,
            num_attention_heads: 2,
            hidden_act: candle_nn::Activation::Silu,
        };
        Mistral3Config {
            model_type: "mistral3".into(),
            image_token_index: 10,
            projector_hidden_act: candle_nn::Activation::Gelu,
            multimodal_projector_bias: false,
            spatial_merge_size: 1, // keep token counts simple
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
    fn positions_and_replacement_work() -> Result<()> {
        let dev = Device::Cpu;
        // inputs_embeds: (2, 5, 4)
        let inputs = Tensor::arange(0f32, 2.0 * 5.0 * 4.0, &dev)?
            .reshape((2, 5, 4))?;

        // positions: (0,1) and (1,3)
        let positions = vec![(0usize, 1usize), (1usize, 3usize)];

        // image_embeds: 2 x 4
        let image_embeds = Tensor::from_vec(vec![
            1.0, 1.1, 1.2, 1.3, // for (0,1)
            2.0, 2.1, 2.2, 2.3, // for (1,3)
        ], (2, 4), &dev)?;

        let replaced = Model::replace_image_tokens(&inputs, &image_embeds, &positions)?;
        // Verify shape unchanged
        assert_eq!(replaced.dims3()?, (2, 5, 4));

        // Check specific positions were replaced
        let v01 = replaced.i((0, 1))?.to_vec1::<f32>()?;
        assert_eq!(v01, vec![1.0, 1.1, 1.2, 1.3]);
        let v13 = replaced.i((1, 3))?.to_vec1::<f32>()?;
        assert_eq!(v13, vec![2.0, 2.1, 2.2, 2.3]);

        // Unchanged position sample
        let orig_00 = inputs.i((0, 0))?.to_vec1::<f32>()?;
        let rep_00 = replaced.i((0, 0))?.to_vec1::<f32>()?;
        assert_eq!(orig_00, rep_00);
        Ok(())
    }

    #[test]
    fn find_positions_from_ids() -> Result<()> {
        let dev = Device::Cpu;
        // Build input_ids with [IMG]=10 placed at (0,1) and (1,3)
        let input_ids = Tensor::from_vec(
            vec![
                // batch 0
                5i64, 10, 6, 7, 8,
                // batch 1
                9, 3, 2, 10, 1,
            ],
            (2, 5),
            &dev,
        )?;
        let positions = Model::find_image_token_positions(&input_ids, 10)?;
        assert_eq!(positions, vec![(0, 1), (1, 3)]);
        Ok(())
    }

    #[test]
    fn text_only_forward_returns_logits() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let cfg = tiny_cfg();
        let mut model = Model::new(&cfg, vb)?;

        // input_ids with no [IMG] tokens
        let input_ids = Tensor::from_vec(vec![1i64, 2, 3, 4], (1, 4), &dev)?;
        let mut cache = Mistral3Cache::default();
        let logits = model.forward(&input_ids, None, None, &mut cache, 0, &cfg.vision_feature_layer)?;

        let dims = logits.dims();
        assert_eq!(*dims.last().unwrap(), cfg.text_config.inner.vocab_size);
        Ok(())
    }
}
