use candle::{IndexOp, Module, Result, Tensor};

use super::GptOssModel;

impl GptOssModel {
    /// Runs the full forward pass up to (and including) the final RMSNorm and
    /// returns the last-token hidden vector (post-norm, pre-lm_head) as f32.
    /// Shape: (hidden_size)
    pub fn forward_last_hidden_post_norm(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut xs = self.embed.forward(input_ids)?; // (b, t, h)
        let (b, t, _hidden) = xs.dims3()?;
        let head_dim = self.cfg.head_dim();
        let n_q = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        let softmax_scale = {
            let base = 1.0f32 / (head_dim as f32).sqrt();
            match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                Some("softmax") => self.rope.attention_factor() * base,
                _ => base,
            }
        };

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let x_norm = layer.input_layernorm.forward(&xs)?;

            let q = x_norm.apply(&layer.attn.q_proj)?;
            let k = x_norm.apply(&layer.attn.k_proj)?;
            let v = x_norm.apply(&layer.attn.v_proj)?;

            let q = q.reshape((b, t, n_q, head_dim))?;
            let k = k.reshape((b, t, n_kv, head_dim))?;
            let v = v.reshape((b, t, n_kv, head_dim))?;

            let q_bhtd = q.transpose(1, 2)?;
            let k_bhtd = k.transpose(1, 2)?;
            let (q_bhtd, k_bhtd) =
                self.rope
                    .apply_rotary_emb_qk(&q_bhtd, &k_bhtd, seqlen_offset)?;
            let q = q_bhtd.transpose(1, 2)?;
            let k_step = k_bhtd;
            let v_step = v.transpose(1, 2)?;

            // Append to KV cache and build repeated K/V for multi-query attention.
            let (k_all, v_all) =
                self.kv_caches[i].append(&k_step.contiguous()?, &v_step.contiguous()?)?;
            let n_rep = n_q / n_kv;
            let k_rep = crate::utils::repeat_kv(k_all.clone(), n_rep)?;
            let v_rep = crate::utils::repeat_kv(v_all.clone(), n_rep)?;
            let k_btkhd = k_rep.transpose(1, 2)?;
            let v_btkhd = v_rep.transpose(1, 2)?;

            let attn_mode = crate::models::gpt_oss::select_attn_mode_for_layer(
                &crate::models::gpt_oss::GptOssConfigMinimal {
                    num_hidden_layers: self.cfg.num_hidden_layers,
                    layer_types: self.cfg.effective_layer_types(),
                    max_position_embeddings: self.cfg.max_position_embeddings,
                    sliding_window: self.cfg.sliding_window,
                },
                i,
            );

            let sinks = if matches!(
                std::env::var("CANDLE_DISABLE_SINKS").ok().as_deref(),
                Some("1") | Some("true") | Some("TRUE")
            ) {
                None
            } else {
                Some(&layer.attn.sinks)
            };

            #[cfg(feature = "flash-attn")]
            let y = {
                let use_fa = !matches!(
                    std::env::var("CANDLE_DISABLE_FLASH").ok().as_deref(),
                    Some("1") | Some("true") | Some("TRUE")
                );
                if use_fa {
                    match attn_mode {
                        crate::models::gpt_oss::AttnMode::Full => {
                            crate::models::gpt_oss::flash_attn_with_sinks(
                                &q,
                                &k_btkhd,
                                &v_btkhd,
                                softmax_scale,
                                t > 1,
                                sinks,
                            )?
                        }
                        crate::models::gpt_oss::AttnMode::Sliding { left, right } => {
                            crate::models::gpt_oss::flash_attn_windowed_with_sinks(
                                &q,
                                &k_btkhd,
                                &v_btkhd,
                                softmax_scale,
                                Some(left),
                                Some(right),
                                sinks,
                            )?
                        }
                    }
                } else {
                    match attn_mode {
                        crate::models::gpt_oss::AttnMode::Full => {
                            crate::models::gpt_oss::eager_attn_with_sinks(
                                &q,
                                &k_btkhd,
                                &v_btkhd,
                                softmax_scale,
                                t > 1,
                                sinks,
                            )?
                        }
                        crate::models::gpt_oss::AttnMode::Sliding { left, right } => {
                            crate::models::gpt_oss::eager_attn_windowed_with_sinks(
                                &q,
                                &k_btkhd,
                                &v_btkhd,
                                softmax_scale,
                                Some(left),
                                Some(right),
                                sinks,
                            )?
                        }
                    }
                }
            };
            #[cfg(not(feature = "flash-attn"))]
            let y = {
                match attn_mode {
                    crate::models::gpt_oss::AttnMode::Full => {
                        crate::models::gpt_oss::eager_attn_with_sinks(
                            &q,
                            &k_btkhd,
                            &v_btkhd,
                            softmax_scale,
                            t > 1,
                            sinks,
                        )?
                    }
                    crate::models::gpt_oss::AttnMode::Sliding { left, right } => {
                        crate::models::gpt_oss::eager_attn_windowed_with_sinks(
                            &q,
                            &k_btkhd,
                            &v_btkhd,
                            softmax_scale,
                            Some(left),
                            Some(right),
                            sinks,
                        )?
                    }
                }
            };

            let y = y.reshape((b, t, n_q * head_dim))?;
            let y = y.apply(&layer.attn.o_proj)?;
            xs = (xs + y)?;

            let x_norm2 = layer.post_attention_layernorm.forward(&xs)?;
            let mlp_out = layer.experts.forward(&x_norm2)?;
            xs = (xs + mlp_out)?;
        }

        let xs = self.norm.forward(&xs)?; // (b, t, h)
        let last = xs.i((0, t - 1))?; // (h)
        let last_f32 = last.to_dtype(candle::DType::F32)?;
        Ok(last_f32)
    }
}

// Re-export as standalone function to match original API
pub fn forward_last_hidden_post_norm(
    model: &mut GptOssModel,
    input_ids: &Tensor,
    seqlen_offset: usize,
) -> Result<Tensor> {
    model.forward_last_hidden_post_norm(input_ids, seqlen_offset)
}
