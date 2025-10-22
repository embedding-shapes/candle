use crate::models::deepseek2::TopKLastDimOp;
use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{ops, Linear};

// Configurable constants
const DEFAULT_TOP_K: usize = 4;

#[derive(Debug, Clone)]
pub struct ExpertMlp {
    pub gate_up: Linear,
    pub down: Linear,
    pub limit: f32,
    pub alpha: f32,
}

impl ExpertMlp {
    pub fn new(gate_up: Linear, down: Linear, limit: f32, alpha: f32) -> Self {
        Self {
            gate_up,
            down,
            limit,
            alpha,
        }
    }
}

impl Module for ExpertMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gu = xs.apply(&self.gate_up)?;

        // Use fused CUDA kernel for activation if available
        let fused = if gu.device().is_cuda() {
            candle::mxfp4::fused_expert_activation_cuda(&gu, self.alpha, self.limit)?
        } else {
            // Fallback to original implementation for non-CUDA
            let gu = gu.to_dtype(DType::F32)?;
            let (_n, two_inter) = gu.dims2()?;
            let inter = two_inter / 2;

            let device = gu.device();
            let even_indices: Vec<u32> = (0..inter).map(|i| (i * 2) as u32).collect();
            let odd_indices: Vec<u32> = (0..inter).map(|i| (i * 2 + 1) as u32).collect();

            let even_idx_tensor = Tensor::from_vec(even_indices, inter, device)?;
            let odd_idx_tensor = Tensor::from_vec(odd_indices, inter, device)?;

            let mut gate = gu.index_select(&even_idx_tensor, D::Minus1)?;
            let mut up = gu.index_select(&odd_idx_tensor, D::Minus1)?;

            gate = gate.clamp(f32::NEG_INFINITY, self.limit)?;
            up = up.clamp(-self.limit, self.limit)?;

            let alpha_t = Tensor::new(self.alpha, xs.device())?.to_dtype(DType::F32)?;
            let gate_alpha = gate.broadcast_mul(&alpha_t)?;
            let sig = ops::sigmoid(&gate_alpha)?;
            let glu = gate.broadcast_mul(&sig)?;

            let one_t = Tensor::new(1.0f32, xs.device())?.to_dtype(DType::F32)?;
            let up_plus = up.broadcast_add(&one_t)?;
            let fused = up_plus.broadcast_mul(&glu)?;
            fused.to_dtype(xs.dtype())?
        };

        let dump_l1 = std::env::var("CANDLE_DUMP_L1").ok().as_deref() == Some("1");

        if dump_l1 && xs.dims2()?.0 > 0 {
            let fused_f32 = fused.to_dtype(DType::F32)?;
            let fused_vec = fused_f32.to_vec2::<f32>()?;
            if !fused_vec.is_empty() {
                eprintln!(
                    "[L1 Expert] fused (before down) first 8: {:?}",
                    &fused_vec[0][..8.min(fused_vec[0].len())]
                );
            }
        }

        let result = fused.apply(&self.down)?;

        if dump_l1 && xs.dims2()?.0 > 0 {
            let result_f32 = result.to_dtype(DType::F32)?;
            let result_vec = result_f32.to_vec2::<f32>()?;
            if !result_vec.is_empty() {
                eprintln!(
                    "[L1 Expert] result (after down) first 8: {:?}",
                    &result_vec[0][..8.min(result_vec[0].len())]
                );
            }
        }

        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub struct GptOssExperts {
    pub router: Linear,
    pub experts: Vec<ExpertMlp>,
    pub num_experts_per_tok: usize,
}

impl GptOssExperts {
    pub fn new(
        router: Linear,
        experts: Vec<ExpertMlp>,
        num_experts_per_tok: Option<usize>,
    ) -> Self {
        let k_env = std::env::var("CANDLE_MOE_TOPK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let k = k_env.or(num_experts_per_tok).unwrap_or(DEFAULT_TOP_K);
        Self {
            router,
            experts,
            num_experts_per_tok: k,
        }
    }

    pub fn router_topk_softmax(&self, logits: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.num_experts_per_tok;
        // Match golden: take largest K and softmax only over those values.
        let o = logits.contiguous()?.topk(k)?;
        let top_probs = candle_nn::ops::softmax_last_dim(&o.values.to_dtype(DType::F32)?)?;
        Ok((o.indices, top_probs))
    }
}

impl GptOssExperts {
    /// GPU-only forward pass - no CPU synchronization
    /// Uses token-centric routing for better performance on small batches
    fn forward_fused(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, h) = xs.dims3()?;
        let xs2 = xs.reshape(((), h))?;
        let logits = xs2.apply(&self.router)?;

        let (topk_idx, probs) = self.router_topk_softmax(&logits)?;

        // topk_idx: [batch_seq, k] - expert indices for each token
        // probs: [batch_seq, k] - routing weights

        let batch_seq = b * t;
        let k = self.num_experts_per_tok;
        let mut ys = xs2.zeros_like()?;

        // Transfer routing info once (smaller than old approach)
        let topk_idx_vec = topk_idx.to_vec2::<u32>()?;  // [batch_seq, k]
        let probs_vec = probs.to_vec2::<f32>()?;  // [batch_seq, k]

        // Build expert->tokens mapping on CPU (fast for small batch)
        use std::collections::HashMap;
        let mut expert_tokens: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();

        for tok in 0..batch_seq {
            for k_i in 0..k {
                let expert_id = topk_idx_vec[tok][k_i] as usize;
                let weight = probs_vec[tok][k_i];
                expert_tokens
                    .entry(expert_id)
                    .or_insert_with(Vec::new)
                    .push((tok, weight));
            }
        }

        // Process each expert that has tokens assigned
        for (expert_id, token_list) in expert_tokens.iter() {
            if token_list.is_empty() {
                continue;
            }

            // Extract token indices and weights
            let token_ids: Vec<u32> = token_list.iter().map(|(t, _)| *t as u32).collect();
            let weights: Vec<f32> = token_list.iter().map(|(_, w)| *w).collect();

            let count = token_ids.len();

            // Transfer to GPU
            let ids_tensor = Tensor::from_vec(token_ids, count, xs2.device())?;
            let weights_tensor = Tensor::from_vec(weights, count, xs2.device())?
                .to_dtype(xs2.dtype())?;  // Match dtype with activations

            // Select inputs
            let selected_inputs = xs2.index_select(&ids_tensor, 0)?;

            // Forward through expert
            let expert_outputs = self.experts[*expert_id].forward(&selected_inputs)?;

            // Apply routing weights
            let weighted_outputs = expert_outputs.broadcast_mul(&weights_tensor.unsqueeze(1)?)?;

            // Accumulate to result
            ys = ys.index_add(&ids_tensor, &weighted_outputs, 0)?;
        }

        let result = ys.reshape((b, t, h))?;
        Ok(result)
    }
}

impl Module for GptOssExperts {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Use fused GPU-only path
        return self.forward_fused(xs);

        /* OLD CPU-sync path - keeping for reference but commented out
        let (b, t, h) = xs.dims3()?;
        let xs2 = xs.reshape(((), h))?;
        let logits = xs2.apply(&self.router)?;

        let (topk_idx, probs) = self.router_topk_softmax(&logits)?;

        // Optimized routing: minimize CPU↔GPU transfers
        // Key: sync once, then keep all computation on GPU
        let n_experts = self.experts.len();

        // ONE-TIME D2H sync for routing info
        let probs = probs.to_dtype(DType::F32)?;
        let probs_host = probs.to_vec2::<f32>()?;
        let idx_host = topk_idx.to_vec2::<u32>()?;

        // CPU-side organization (fast)
        let mut token_ids: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
        let mut token_wts: Vec<Vec<f32>> = vec![Vec::new(); n_experts];
        for (row, (row_probs, row_experts)) in
            probs_host.iter().zip(idx_host.iter()).enumerate()
        {
            for (&p, &e) in row_probs.iter().zip(row_experts.iter()) {
                token_ids[e as usize].push(row as u32);
                token_wts[e as usize].push(p);
            }
        }

        // Count assignments per expert
        let counts: Vec<usize> = token_ids.iter().map(|ids| ids.len()).collect();
        let total_assignments: usize = counts.iter().sum();

        // Flatten into single arrays for BATCHED H2D transfer (much faster!)
        let mut flat_ids = Vec::with_capacity(total_assignments);
        let mut flat_wts = Vec::with_capacity(total_assignments);
        for e_idx in 0..n_experts {
            flat_ids.extend_from_slice(&token_ids[e_idx]);
            flat_wts.extend_from_slice(&token_wts[e_idx]);
        }

        // SINGLE H2D transfer for all indices and weights
        let ids_all = if total_assignments == 0 {
            Tensor::zeros((0,), DType::U32, xs2.device())?
        } else {
            Tensor::from_vec(flat_ids, total_assignments, xs2.device())?
        };

        let weights_all = if total_assignments == 0 {
            Tensor::zeros((0, 1), xs2.dtype(), xs2.device())?
        } else {
            Tensor::from_vec(flat_wts, total_assignments, xs2.device())?
                .to_dtype(xs2.dtype())?
                .reshape((total_assignments, 1))?
        };

        // GPU-resident expert processing
        let mut ys = xs2.zeros_like()?;
        let mut offset = 0usize;

        for (e_idx, expert) in self.experts.iter().enumerate() {
            let count = counts[e_idx];
            if count == 0 {
                continue;
            }

            // Extract this expert's slice from the batched tensors (NO H2D!)
            let ids_t = ids_all.narrow(0, offset, count)?;
            let wts_t = weights_all.narrow(0, offset, count)?;

            // All ops GPU-only
            let x_sel = xs2.index_select(&ids_t, 0)?;
            let y_sel = expert.forward(&x_sel)?;
            let y_sel = y_sel.broadcast_mul(&wts_t)?;
            ys = ys.index_add(&ids_t, &y_sel, 0)?;

            offset += count;
        }

        let result = ys.reshape((b, t, h))?;
        Ok(result)
        */
    }
}
