use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
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
fn gpt_oss_lm_head_contract_cuda() -> Result<()> {
    // GPU device as required.
    let dev = Device::new_cuda(0)?;
    eprintln!("device={:?}", dev);

    // Resolve snapshot and index; skip if not present locally.
    let snap = expand_tilde(SNAPSHOT_DIR).unwrap();
    let idx = snap.join(INDEX_FILE);
    if !idx.exists() {
        eprintln!("index not found at {} — skipping", idx.display());
        return Ok(());
    }

    // Load config for expected dims.
    let cfg: GptOssConfig = serde_json::from_slice(&std::fs::read(snap.join("config.json"))?).unwrap();
    assert_eq!(cfg.vocab_size, 201_088);
    assert_eq!(cfg.hidden_size, 2_880);

    // Build VarBuilder over local shards in BF16 on CPU (mmap, no copies), then move tensors to GPU when needed.
    // Build shard file list from model.safetensors.index.json weight_map entries.
    #[derive(serde::Deserialize)]
    struct Idx { weight_map: BTreeMap<String, String> }
    let idx_bytes = std::fs::read(&idx).unwrap();
    let idx_parsed: Idx = serde_json::from_slice(&idx_bytes).unwrap();
    let mut files_set = std::collections::BTreeSet::new();
    for f in idx_parsed.weight_map.values() { files_set.insert(f.clone()); }
    let mut model_files: Vec<std::path::PathBuf> = files_set.into_iter().map(|f| snap.join(f)).collect();
    model_files.sort();
    assert!(!model_files.is_empty(), "no shards resolved from index");
    let vb_cpu = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::BF16, &Device::Cpu)? };

    // 1) Load only lm_head.weight (+bias if present) and assert shape/dtype
    let w_cpu = vb_cpu.pp("lm_head").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    let b_cpu = if vb_cpu.pp("lm_head").contains_tensor("bias") {
        Some(vb_cpu.pp("lm_head").get(cfg.vocab_size, "bias")?)
    } else {
        None
    };
    let (rows, cols) = w_cpu.dims2()?;
    eprintln!("lm_head.weight: dtype={:?} device={:?} shape=({}, {}) stride={:?}", w_cpu.dtype(), w_cpu.device(), rows, cols, w_cpu.stride());
    assert_eq!(w_cpu.dtype(), DType::BF16);
    assert_eq!((rows, cols), (cfg.vocab_size, cfg.hidden_size));
    if let Some(b) = &b_cpu {
        eprintln!("lm_head.bias: dtype={:?} device={:?} shape={:?}", b.dtype(), b.device(), b.dims());
        assert_eq!(b.dtype(), DType::BF16);
        assert_eq!(b.dims(), &[cfg.vocab_size]);
    }

    // 2) One-hot orientation check on GPU with fixed seed
    let seed: u64 = 1337;
    eprintln!("seed={}", seed);
    // Select 8 deterministic indices within [0, hidden_size)
    let mut idxs = Vec::with_capacity(8);
    let mut x = seed;
    for _ in 0..8 { x = x.wrapping_mul(6364136223846793005).wrapping_add(1); idxs.push((x as usize) % cfg.hidden_size); }
    eprintln!("one_hot_indices={:?}", idxs);

    // Move weights to GPU, operate in F32 for strict numeric parity on kernel semantics (contract, not precision).
    let w = w_cpu.to_device(&dev)?.to_dtype(DType::F32)?;
    let b = match b_cpu { Some(bc) => Some(bc.to_device(&dev)?.to_dtype(DType::F32)?), None => None };
    let lin = Linear::new(w.clone(), b.clone());

    // For each index i: x one-hot -> expect y == (W.col(i) + b)
    for &i in &idxs {
        let mut x_host = vec![0f32; cfg.hidden_size];
        x_host[i] = 1.0;
        let x = Tensor::from_vec(x_host.clone(), (1, cfg.hidden_size), &dev)?;
        eprintln!(
            "x: dtype={:?} device={:?} shape={:?} stride={:?} i={}",
            x.dtype(), x.device(), x.dims(), x.stride(), i
        );

        let y_candle = lin.forward(&x)?; // (1, vocab)
        let y_manual = {
            let mm = x.matmul(&w.t()?)?; // (1, vocab)
            match &b { Some(bb) => (mm + bb)?, None => mm }
        };

        // Also build the explicit expected column_i(W) (+ b)
        let col_i = w_cpu.to_dtype(DType::F32)?.to_device(&dev)?.i((.., i))?; // (vocab)
        let col_i = match &b { Some(bb) => (col_i + bb)?, None => col_i };

        let y_c = y_candle.to_dtype(DType::F32)?.squeeze(0)?;
        let y_m = y_manual.to_dtype(DType::F32)?.squeeze(0)?;

        // Compute metrics
        let y_c_v = y_c.to_vec1::<f32>()?;
        let y_m_v = y_m.to_vec1::<f32>()?;
        let col_v = col_i.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut linf_cm = 0f32; // candle vs manual
        let mut l2_cm = 0f32;
        let mut linf_cc = 0f32; // candle vs column
        let mut l2_cc = 0f32;
        for j in 0..cfg.vocab_size {
            let d1 = (y_c_v[j] - y_m_v[j]).abs();
            if d1 > linf_cm { linf_cm = d1; }
            l2_cm += d1 * d1;
            let d2 = (y_c_v[j] - col_v[j]).abs();
            if d2 > linf_cc { linf_cc = d2; }
            l2_cc += d2 * d2;
        }
        l2_cm = l2_cm.sqrt();
        l2_cc = l2_cc.sqrt();
        eprintln!("i={} L_inf(candle,manual)={:.3e} L2={:.3e} L_inf(candle,column)={:.3e} L2={:.3e}", i, linf_cm, l2_cm, linf_cc, l2_cc);

        // Tight tolerance: exact contract on one-hot in F32
        assert!(linf_cm <= 1e-6, "candle vs manual mismatch at i={}: L_inf={}", i, linf_cm);
        assert!(linf_cc <= 1e-6, "orientation mismatch at i={}: L_inf(candle,column)={}", i, linf_cc);
    }

    // 3) Binding check (structural): ensure both names exist in the index → untied weights present.
    assert!(idx_parsed.weight_map.contains_key("lm_head.weight"), "missing lm_head.weight in index");
    assert!(idx_parsed.weight_map.contains_key("model.embed_tokens.weight"), "missing model.embed_tokens.weight in index");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[test]
fn gpt_oss_lm_head_contract_cuda_skipped() { eprintln!("skipped: build without 'cuda' feature"); }
