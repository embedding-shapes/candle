use candle::{DType, IndexOp, Module, Result, Tensor, D};

use super::GptOssModel;

const ENV_DUMP_L1: &str = "CANDLE_DUMP_L1";

impl GptOssModel {
    pub fn forward_logits_minimal(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embed.forward(input_ids)?;
        let (b, t, h) = xs.dims3()?;
        let xs2 = xs.reshape(((), h))?.to_dtype(DType::F32)?;
        let w = self.lm_head.weight().to_dtype(DType::F32)?;
        let logits = xs2.matmul(&w.t()?)?;
        let logits = logits
            .reshape((b, t, self.cfg.vocab_size))?
            .to_dtype(DType::BF16)?;
        Ok(logits)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let mut xs = self.embed.forward(input_ids)?;
        let (b, t, hidden) = xs.dims3()?;
        let head_dim = self.cfg.head_dim();
        let n_q = self.cfg.num_attention_heads;
        let n_kv = self.cfg.num_key_value_heads;
        // Test toggle for YaRN scaling placement:
        // - CANDLE_YARN_MODE=cos     => use standard 1/sqrt(d) softmax scale (HF placement on cos/sin)
        // - CANDLE_YARN_MODE=softmax => include attention_factor in softmax scale (current behavior)
        let softmax_scale = {
            let base = 1.0f32 / (head_dim as f32).sqrt();
            match std::env::var("CANDLE_YARN_MODE").ok().as_deref() {
                Some("softmax") => self.rope.attention_factor() * base,
                _ => base,
            }
        };

        let dump_l1 = matches!(
            std::env::var(ENV_DUMP_L1).ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        );
        if dump_l1 {
            let xs_pre = xs.i((0, t - 1))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            eprintln!("[L1] xs (before layer 0) last [:8]: {:?}", &xs_pre[..8]);
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let x_norm = layer.input_layernorm.forward(&xs)?;
            if dump_l1 && i == 0 {
                let last = x_norm.i((0, t - 1))?.to_dtype(DType::F32)?;
                let v = last.to_vec1::<f32>()?;
                let take = v.iter().take(8).copied().collect::<Vec<_>>();
                let mean = v.iter().copied().sum::<f32>() / (v.len() as f32);
                let var =
                    v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (v.len() as f32);
                eprintln!(
                    "[L1] post-norm last-token: len={} first8={:?} mean={:.6} std={:.6}",
                    v.len(),
                    take,
                    mean,
                    var.sqrt()
                );
            }

            let q = x_norm.apply(&layer.attn.q_proj)?;
            let k = x_norm.apply(&layer.attn.k_proj)?;
            let v = x_norm.apply(&layer.attn.v_proj)?;
            if dump_l1 && i == 0 {
                let q_last = q.i((0, t - 1))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let k_last = k.i((0, t - 1))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let v_last = v.i((0, t - 1))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                eprintln!("[L1] q_proj last [:8]: {:?}", &q_last[..8]);
                eprintln!("[L1] k_proj last [:8]: {:?}", &k_last[..8]);
                eprintln!("[L1] v_proj last [:8]: {:?}", &v_last[..8]);
            }

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
            if dump_l1 && i == 0 {
                let y_last = y.i((0, t - 1))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                eprintln!("[L1] o_proj last [:8]: {:?}", &y_last[..8]);
            }
            xs = (xs + y)?;
            if dump_l1 && i == 0 {
                let last = xs.i((0, t - 1))?.to_dtype(DType::F32)?;
                let v = last.to_vec1::<f32>()?;
                let take = v.iter().take(8).copied().collect::<Vec<_>>();
                let mean = v.iter().copied().sum::<f32>() / (v.len() as f32);
                let var =
                    v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (v.len() as f32);
                eprintln!(
                    "[L1] post-attn-residual last-token: len={} first8={:?} mean={:.6} std={:.6}",
                    v.len(), take, mean, var.sqrt()
                );
            }

            let x_norm2 = layer.post_attention_layernorm.forward(&xs)?;
            if dump_l1 && i == 0 {
                let norm2_last = x_norm2
                    .i((0, t - 1))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                eprintln!("[L1] post-attn-norm last [:8]: {:?}", &norm2_last[..8]);
            }
            let mlp_out = layer.experts.forward(&x_norm2)?;
            if dump_l1 && i == 0 {
                let mlp_last = mlp_out
                    .i((0, t - 1))?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                eprintln!("[L1] mlp_out last [:8]: {:?}", &mlp_last[..8]);
            }
            xs = (xs + mlp_out)?;
            if dump_l1 && i == 0 {
                let last = xs.i((0, t - 1))?.to_dtype(DType::F32)?;
                let v = last.to_vec1::<f32>()?;
                eprintln!("[L1] end-of-layer-0 last-token [:8]: {:?}", &v[..8]);
            }
        }

        let xs = self.norm.forward(&xs)?;
        let xs2 = xs.reshape(((), hidden))?;
        let xs2_f = xs2.to_dtype(DType::F32)?;
        let w = self.lm_head.weight();
        let vocab = self.cfg.vocab_size;
        let chunk = 8192usize;
        let mut parts: Vec<Tensor> = Vec::new();
        let mut start = 0usize;
        while start < vocab {
            let len = (vocab - start).min(chunk);
            let w_chunk = w.narrow(0, start, len)?;
            let w_chunk_f = w_chunk.to_dtype(DType::F32)?;
            let logits_chunk = xs2_f.matmul(&w_chunk_f.t()?)?;
            parts.push(logits_chunk);
            start += len;
        }
        let logits_bt_v = Tensor::cat(&parts.iter().collect::<Vec<_>>(), D::Minus1)?;
        let logits = logits_bt_v.reshape((b, t, vocab))?;
        Ok(logits)
    }
}

// Re-export as standalone functions to match original API
pub fn forward(model: &mut GptOssModel, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
    model.forward(input_ids, seqlen_offset)
}

pub fn forward_logits_minimal(model: &GptOssModel, input_ids: &Tensor) -> Result<Tensor> {
    model.forward_logits_minimal(input_ids)
}
