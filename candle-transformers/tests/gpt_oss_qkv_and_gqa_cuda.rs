use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::config::GptOssConfig;

use std::collections::BTreeMap;

const SNAPSHOT_DIR: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const INDEX_FILE: &str = "model.safetensors.index.json";

fn expand_tilde(p: &str) -> std::io::Result<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Path::new(&home).join(rest))
    } else {
        Ok(PathBuf::from(p))
    }
}

#[cfg(feature = "cuda")]
#[test]
fn gpt_oss_qkv_projection_orientation_layer0_cuda() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    eprintln!("device={:?}", dev);

    // Resolve snapshot and index; skip if not present locally.
    let snap = expand_tilde(SNAPSHOT_DIR).unwrap();
    let idx = snap.join(INDEX_FILE);
    if !idx.exists() {
        eprintln!("index not found at {} — skipping", idx.display());
        return Ok(());
    }

    // Load config for expected dims and heads.
    let cfg: GptOssConfig = serde_json::from_slice(&std::fs::read(snap.join("config.json"))?).unwrap();
    eprintln!("config: hidden={} n_q={} n_kv={} head_dim={}", cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim());
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim();
    let q_out = cfg.num_attention_heads * head_dim;
    let kv_out = cfg.num_key_value_heads * head_dim;

    // Build VarBuilder over local shards in BF16 on CPU (mmap), move tensors as needed.
    #[derive(serde::Deserialize)]
    struct Idx { weight_map: BTreeMap<String, String> }
    let idx_parsed: Idx = serde_json::from_slice(&std::fs::read(&idx).unwrap()).unwrap();
    let mut files_set = std::collections::BTreeSet::new();
    for f in idx_parsed.weight_map.values() { files_set.insert(f.clone()); }
    let mut model_files: Vec<std::path::PathBuf> = files_set.into_iter().map(|f| snap.join(f)).collect();
    model_files.sort();
    let vb_cpu = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::BF16, &Device::Cpu)? };

    // Load layer 0 q/k/v projection weights
    let l0 = 0usize;
    let pp = |name: &str| vb_cpu.pp(&format!("model.layers.{l0}.self_attn.{name}"));
    let wq_cpu = pp("q_proj").get((q_out, hidden), "weight")?;
    let wk_cpu = pp("k_proj").get((kv_out, hidden), "weight")?;
    let wv_cpu = pp("v_proj").get((kv_out, hidden), "weight")?;
    eprintln!("wq: dtype={:?} dev={:?} shape={:?} stride={:?}", wq_cpu.dtype(), wq_cpu.device(), wq_cpu.dims(), wq_cpu.stride());
    eprintln!("wk: dtype={:?} dev={:?} shape={:?} stride={:?}", wk_cpu.dtype(), wk_cpu.device(), wk_cpu.dims(), wk_cpu.stride());
    eprintln!("wv: dtype={:?} dev={:?} shape={:?} stride={:?}", wv_cpu.dtype(), wv_cpu.device(), wv_cpu.dims(), wv_cpu.stride());
    assert_eq!(wq_cpu.dtype(), DType::BF16);
    assert_eq!(wk_cpu.dtype(), DType::BF16);
    assert_eq!(wv_cpu.dtype(), DType::BF16);
    assert_eq!(wq_cpu.dims(), &[q_out, hidden]);
    assert_eq!(wk_cpu.dims(), &[kv_out, hidden]);
    assert_eq!(wv_cpu.dims(), &[kv_out, hidden]);

    // Deterministic 8 one-hot indices in [0, hidden)
    let seed: u64 = 42;
    eprintln!("seed={}", seed);
    let mut idxs = Vec::with_capacity(8);
    let mut x = seed;
    for _ in 0..8 { x = x.wrapping_mul(2862933555777941757).wrapping_add(3037000493); idxs.push((x as usize) % hidden); }
    eprintln!("one_hot_indices={:?}", idxs);

    // Move weights to GPU in F32 for strict arithmetic equality on one-hot
    let wq = wq_cpu.to_device(&dev)?.to_dtype(DType::F32)?;
    let wk = wk_cpu.to_device(&dev)?.to_dtype(DType::F32)?;
    let wv = wv_cpu.to_device(&dev)?.to_dtype(DType::F32)?;

    for &i in &idxs {
        // x = one_hot(i) in R^{hidden}
        let mut host = vec![0f32; hidden]; host[i] = 1.0;
        let x = Tensor::from_vec(host, (1, hidden), &dev)?;
        eprintln!("x: dtype={:?} dev={:?} shape={:?} stride={:?} i={}", x.dtype(), x.device(), x.dims(), x.stride(), i);

        // q_candle = x @ Wq.T; expected = column_i(Wq)
        let q_candle = x.matmul(&wq.t()?)?; // (1, q_out)
        let q_col = wq_cpu.to_device(&dev)?.to_dtype(DType::F32)?.i((.., i))?; // (q_out)
        let q_c = q_candle.squeeze(0)?.to_dtype(DType::F32)?;
        let q_c_dtype = q_c.dtype();
        let q_c_dims = q_c.dims();
        let q_e = q_col.to_dtype(DType::F32)?;
        let q_diff = (&q_c - &q_e)?.abs()?;
        let q_linf = q_diff.max_all()?.to_scalar::<f32>()?;
        let q_l2 = q_diff.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        eprintln!("Q: i={} L_inf={:.3e} L2={:.3e} dtype={:?}→{:?} dims_c={:?} dims_e={:?}", i, q_linf, q_l2, wq_cpu.dtype(), q_c_dtype, q_c_dims, q_e.dims());
        assert!(q_linf <= 1e-6, "q orientation mismatch at i={}: L_inf={}", i, q_linf);

        // k
        let k_candle = x.matmul(&wk.t()?)?; // (1, kv_out)
        let k_col = wk_cpu.to_device(&dev)?.to_dtype(DType::F32)?.i((.., i))?; // (kv_out)
        let k_c = k_candle.squeeze(0)?.to_dtype(DType::F32)?;
        let k_e = k_col.to_dtype(DType::F32)?;
        let k_diff = (k_c - &k_e)?.abs()?;
        let k_linf = k_diff.max_all()?.to_scalar::<f32>()?;
        let k_l2 = k_diff.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        eprintln!("K: i={} L_inf={:.3e} L2={:.3e}", i, k_linf, k_l2);
        assert!(k_linf <= 1e-6, "k orientation mismatch at i={}: L_inf={}", i, k_linf);

        // v
        let v_candle = x.matmul(&wv.t()?)?; // (1, kv_out)
        let v_col = wv_cpu.to_device(&dev)?.to_dtype(DType::F32)?.i((.., i))?; // (kv_out)
        let v_c = v_candle.squeeze(0)?.to_dtype(DType::F32)?;
        let v_e = v_col.to_dtype(DType::F32)?;
        let v_diff = (v_c - &v_e)?.abs()?;
        let v_linf = v_diff.max_all()?.to_scalar::<f32>()?;
        let v_l2 = v_diff.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        eprintln!("V: i={} L_inf={:.3e} L2={:.3e}", i, v_linf, v_l2);
        assert!(v_linf <= 1e-6, "v orientation mismatch at i={}: L_inf={}", i, v_linf);
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[test]
fn gpt_oss_qkv_projection_orientation_layer0_cuda_skipped() { eprintln!("skipped: build without 'cuda' feature"); }

#[cfg(feature = "cuda")]
#[test]
fn gpt_oss_gqa_mapping_64_to_8_cuda() -> Result<()> {
    // Geometry: Hq=64, Hkv=8, n_rep=8, b=1, tk=2, d=5
    let dev = Device::new_cuda(0)?;
    let b = 1usize;
    let n_q = 64usize;
    let n_kv = 8usize;
    let n_rep = n_q / n_kv; // 8
    let tk = 2usize;
    let d = 5usize;

    // Build K/V in (b, n_kv, tk, d) with distinctive per-head constants for visibility
    let mut k_host = vec![0f32; b * n_kv * tk * d];
    let mut v_host = vec![0f32; b * n_kv * tk * d];
    let mut idx = 0;
    for _ in 0..b {
        for h in 0..n_kv {
            for _ in 0..tk {
                for _ in 0..d {
                    k_host[idx] = (100 + h) as f32; // 100+h
                    v_host[idx] = (200 + h) as f32; // 200+h
                    idx += 1;
                }
            }
        }
    }
    let k_bhd = Tensor::from_vec(k_host, (b, n_kv, tk, d), &dev)?.to_dtype(DType::BF16)?;
    let v_bhd = Tensor::from_vec(v_host, (b, n_kv, tk, d), &dev)?.to_dtype(DType::BF16)?;
    eprintln!("k_bhd: dtype={:?} dev={:?} shape={:?} stride={:?}", k_bhd.dtype(), k_bhd.device(), k_bhd.dims(), k_bhd.stride());
    eprintln!("v_bhd: dtype={:?} dev={:?} shape={:?} stride={:?}", v_bhd.dtype(), v_bhd.device(), v_bhd.dims(), v_bhd.stride());

    // Repeat to Q heads
    let k_rep = candle_transformers::utils::repeat_kv(k_bhd.clone(), n_rep)?; // (b, n_q, tk, d)
    let v_rep = candle_transformers::utils::repeat_kv(v_bhd.clone(), n_rep)?; // (b, n_q, tk, d)

    // For each Q head h, expected kv_idx = floor(h/8). Check the entire slice equals the source kv head.
    for h in 0..n_q {
        let kv_idx = h / n_rep;
        assert_eq!(kv_idx, (h as usize) / 8);

        // Extract slices (tk, d) at head h and head kv_idx
        let got_k = k_rep.i((0, h, .., ..))?.to_dtype(DType::F32)?;
        let exp_k = k_bhd.i((0, kv_idx, .., ..))?.to_dtype(DType::F32)?;
        let dk = (got_k - &exp_k)?.abs()?;
        let dk_linf = dk.max_all()?.to_scalar::<f32>()?;
        let got_v = v_rep.i((0, h, .., ..))?.to_dtype(DType::F32)?;
        let exp_v = v_bhd.i((0, kv_idx, .., ..))?.to_dtype(DType::F32)?;
        let dv = (got_v - &exp_v)?.abs()?;
        let dv_linf = dv.max_all()?.to_scalar::<f32>()?;
        if h < 3 || h > n_q - 3 { eprintln!("h={} kv_idx={} dk_inf={:.3e} dv_inf={:.3e}", h, kv_idx, dk_linf, dv_linf); }
        assert!(dk_linf <= 1e-6, "K mapping mismatch at head {} -> kv {}: L_inf={}", h, kv_idx, dk_linf);
        assert!(dv_linf <= 1e-6, "V mapping mismatch at head {} -> kv {}: L_inf={}", h, kv_idx, dv_linf);
    }

    // Also validate reshape contracts: q.view(Hq,d), k.view(Hkv,d), v.view(Hkv,d) are consistent with the mapping
    // (We do a lightweight sanity by constructing dummy q/k/v flat vectors and checking shape products)
    let hdim = d; // toy head dim for this micro-check
    let q_flat = Tensor::zeros((b, 1, n_q * hdim), DType::BF16, &dev)?;
    let k_flat = Tensor::zeros((b, 1, n_kv * hdim), DType::BF16, &dev)?;
    let v_flat = Tensor::zeros((b, 1, n_kv * hdim), DType::BF16, &dev)?;
    let _q_view = q_flat.reshape((b, 1, n_q, hdim))?;
    let _k_view = k_flat.reshape((b, 1, n_kv, hdim))?;
    let _v_view = v_flat.reshape((b, 1, n_kv, hdim))?;
    eprintln!("reshape contracts ok: q=({},{}), k=({},{}), v=({},{}), n_rep={}", n_q, hdim, n_kv, hdim, n_kv, hdim, n_rep);

    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[test]
fn gpt_oss_gqa_mapping_64_to_8_cuda_skipped() { eprintln!("skipped: build without 'cuda' feature"); }
