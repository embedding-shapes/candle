use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{ops, Activation, Linear};

// Configurable constants
const DEFAULT_TOP_K: usize = 4;

#[derive(Debug, Clone)]
pub struct ExpertMlp {
    // Fused gate_up: outputs 2 * intermediate, split into gate and up
    pub gate_up: Linear,
    pub down: Linear,
    pub act: Activation,
}

impl ExpertMlp {
    pub fn new(gate_up: Linear, down: Linear, act: Activation) -> Self {
        Self { gate_up, down, act }
    }
}

impl Module for ExpertMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (n, hidden)
        let gu = xs.apply(&self.gate_up)?; // (n, 2*inter)
        let (_n, two_inter) = gu.dims2()?;
        let inter = two_inter / 2;
        let gate = gu.narrow(D::Minus1, 0, inter)?; // (n, inter)
        let up = gu.narrow(D::Minus1, inter, inter)?; // (n, inter)
        let lhs = ops::silu(&gate)?; // (n, inter)
        let fused = (lhs * up)?; // (n, inter)
        fused.apply(&self.down)
    }
}

#[derive(Debug, Clone)]
pub struct GptOssExperts {
    pub router: Linear, // hidden -> num_experts
    pub experts: Vec<ExpertMlp>,
    pub num_experts_per_tok: usize, // typically 4
}

impl GptOssExperts {
    pub fn new(router: Linear, experts: Vec<ExpertMlp>, num_experts_per_tok: Option<usize>) -> Self {
        let k = num_experts_per_tok.unwrap_or(DEFAULT_TOP_K);
        Self { router, experts, num_experts_per_tok: k }
    }

    // Returns (topk_indices: (n, k) u32, probs: (n, k) f32) using softmax over selected logits.
    pub fn router_topk_softmax(&self, logits: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.num_experts_per_tok;
        // Indices of top-k by descending logits
        let topk_idx = logits
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, k)?
            .contiguous()?; // (n, k) u32
        // Gather selected logits and softmax over them only
        let selected = logits.gather(&topk_idx, D::Minus1)?; // (n, k)
        let probs = ops::softmax_last_dim(&selected.to_dtype(DType::F32)?)?; // (n, k)
        Ok((topk_idx, probs))
    }
}

impl Module for GptOssExperts {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (b, t, hidden)
        let (b, t, h) = xs.dims3()?;
        let xs2 = xs.reshape(((), h))?; // (bt, hidden)
        let logits = xs2.apply(&self.router)?; // (bt, n_experts)

        let (topk_idx, probs) = self.router_topk_softmax(&logits)?; // (bt,k), (bt,k)

        // Build per-expert token buckets on host for simple, fast inference path
        let probs = probs.to_dtype(DType::F32)?;
        let probs_host = probs.to_vec2::<f32>()?; // (bt, k)
        let idx_host = topk_idx.to_vec2::<u32>()?; // (bt, k)

        let n_experts = self.experts.len();
        let mut token_ids: Vec<Vec<u32>> = vec![Vec::new(); n_experts];
        let mut token_wts: Vec<Vec<f32>> = vec![Vec::new(); n_experts];
        for (row, (row_probs, row_experts)) in probs_host.iter().zip(idx_host.iter()).enumerate() {
            for (&p, &e) in row_probs.iter().zip(row_experts.iter()) {
                token_ids[e as usize].push(row as u32);
                token_wts[e as usize].push(p);
            }
        }

        let mut ys = xs2.zeros_like()?; // (bt, hidden)
        for (e_idx, expert) in self.experts.iter().enumerate() {
            let ids = &token_ids[e_idx];
            if ids.is_empty() {
                continue;
            }
            let ids_t = Tensor::new(ids.as_slice(), xs2.device())?; // (m)
            let wts_t = Tensor::new(token_wts[e_idx].as_slice(), xs2.device())?
                .reshape(((), 1))?
                .to_dtype(xs2.dtype())?; // (m,1)
            let x_sel = xs2.index_select(&ids_t, 0)?; // (m, h)
            let y_sel = expert.forward(&x_sel)?; // (m, h)
            let y_sel = y_sel.broadcast_mul(&wts_t)?; // (m, h)
            ys = ys.index_add(&ids_t, &y_sel, 0)?;
        }
        ys.reshape((b, t, h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    fn make_linear(in_d: usize, out_d: usize, dev: &Device) -> Linear {
        // Deterministic small weights for test stability
        let w: Vec<f32> = (0..(out_d * in_d))
            .map(|i| ((i as i32 % 7) as f32 - 3.0) / (in_d as f32))
            .collect();
        let w = Tensor::from_vec(w, (out_d, in_d), dev).unwrap();
        Linear::new(w, None)
    }

    #[test]
    fn t16_router_semantics_topk_softmax() -> Result<()> {
        let dev = Device::Cpu;
        let n = 5usize; // tokens
        let e = 7usize; // experts
        let k = 3usize; // top-k
        let logits = Tensor::randn(0f32, 1f32, (n, e), &dev)?;

        // Router with dummy weights, we only use router_topk_softmax here
        let dummy_router = make_linear(4, e, &dev);
        let experts = vec![];
        let moe = GptOssExperts::new(dummy_router, experts, Some(k));
        let (idx_t, probs_t) = moe.router_topk_softmax(&logits)?;

        let idx = idx_t.to_vec2::<u32>()?; // (n,k)
        let probs = probs_t.to_vec2::<f32>()?; // (n,k)

        // Check each row: indices correspond to top-k logits, and probs sum to 1.
        let logits_host = logits.to_vec2::<f32>()?;
        for i in 0..n {
            let mut pairs: Vec<(f32, usize)> = logits_host[i]
                .iter()
                .copied()
                .enumerate()
                .map(|(j, v)| (v, j))
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // desc
            let top = &pairs[..k];
            // Indices match
            for (j, (_, exp_idx)) in top.iter().enumerate() {
                assert_eq!(idx[i][j] as usize, *exp_idx, "top-k index mismatch at row {i}, pos {j}");
            }
            // Probabilities are softmax over selected logits
            let selected: Vec<f32> = top.iter().map(|(v, _)| *v).collect();
            let max_v = selected
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = selected.iter().map(|v| (v - max_v).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            for j in 0..k {
                let p = exps[j] / sum_e;
                let got = probs[i][j];
                assert!((p - got).abs() < 1e-5, "prob mismatch at row {i}, pos {j}: exp={p}, got={got}");
            }
            let sum_p: f32 = probs[i].iter().sum();
            assert!((sum_p - 1.0).abs() < 1e-6, "probs do not sum to 1 at row {i}: {sum_p}");
        }

        Ok(())
    }

    #[test]
    fn t17_mlp_block_tiny_moe_matches_eager() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 4usize;
        let inter = 4usize;
        let n_experts = 3usize;
        let k = 2usize;

        // Build experts with deterministic weights
        let mut experts = Vec::new();
        for _ in 0..n_experts {
            let gate_up = make_linear(hidden, 2 * inter, &dev);
            let down = make_linear(inter, hidden, &dev);
            experts.push(ExpertMlp::new(gate_up, down, Activation::Silu));
        }
        // Router: hidden -> n_experts
        let router = make_linear(hidden, n_experts, &dev);
        let moe = GptOssExperts::new(router, experts.clone(), Some(k));

        // Input: 2 tokens
        let xs = Tensor::from_vec(vec![
            0.1f32, -0.2, 0.3, 0.4, // tok 0
            -0.5, 0.6, -0.7, 0.8,   // tok 1
        ], (1, 2, hidden), &dev)?;

        // Eager reference: per-token compute router, top-k over logits, softmax within top-k,
        // then sum weighted expert outputs for selected experts.
        let (b, t, h) = xs.dims3()?;
        let xs2 = xs.reshape(((), h))?; // (2,4)
        let ref_logits = xs2.apply(&moe.router)?; // (2,3)
        let ref_idx = ref_logits
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, k)?
            .contiguous()?; // (2,k)
        let ref_sel = ref_logits.gather(&ref_idx, D::Minus1)?; // (2,k)
        let ref_probs = ops::softmax_last_dim(&ref_sel.to_dtype(DType::F32)?)?; // (2,k)
        let ref_idx = ref_idx.to_vec2::<u32>()?;
        let ref_probs = ref_probs.to_vec2::<f32>()?;

        // Compute eager sum
        let mut ys_ref = xs2.zeros_like()?; // (2,4)
        for row in 0..(b * t) {
            let x_row = xs2.narrow(0, row, 1)?; // (1,4)
            let mut acc = x_row.zeros_like()?;
            for j in 0..k {
                let e_idx = ref_idx[row][j] as usize;
                let p = ref_probs[row][j];
                let y = experts[e_idx].forward(&x_row)?; // (1,4)
                let p_t = Tensor::from_vec(vec![p], (1, 1), y.device())?;
                acc = (acc + y.broadcast_mul(&p_t)?)?;
            }
            let id = Tensor::from_vec(vec![row as u32], 1, &dev)?;
            ys_ref = ys_ref.index_add(&id, &acc, 0)?;
        }
        ys_ref = ys_ref.reshape((b, t, h))?;

        let ys = moe.forward(&xs)?;
        let diff = (&ys - &ys_ref)?.abs()?;
        let flat = diff.reshape((diff.elem_count(),))?;
        let max_diff = flat.max(D::Minus1)?.to_scalar::<f32>()?;
        assert!(max_diff < 1e-5, "max diff too large: {max_diff}");
        Ok(())
    }
}
