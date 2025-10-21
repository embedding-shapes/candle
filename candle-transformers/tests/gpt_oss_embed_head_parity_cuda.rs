use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use std::collections::BTreeMap;

const SNAPSHOT_DIR: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const INDEX_FILE: &str = "model.safetensors.index.json";

fn expand_tilde(p: &str) -> std::io::Result<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        let home =
            std::env::var("HOME").map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Path::new(&home).join(rest))
    } else {
        Ok(PathBuf::from(p))
    }
}

#[cfg(feature = "cuda")]
#[test]
fn gpt_oss_embed_head_parity_cuda() -> Result<()> {
    // GPU device as required by the verification harness.
    let dev = Device::new_cuda(0)?;
    eprintln!("device={:?}", dev);

    // Resolve snapshot and index; skip if not present locally.
    let snap = expand_tilde(SNAPSHOT_DIR).unwrap();
    let idx = snap.join(INDEX_FILE);
    if !idx.exists() {
        eprintln!("index not found at {} — skipping", idx.display());
        return Ok(());
    }

    // Load config and assert expected model dims from spec.
    let cfg: GptOssConfig =
        serde_json::from_slice(&std::fs::read(snap.join("config.json"))?).unwrap();
    assert_eq!(cfg.vocab_size, 201_088);
    assert_eq!(cfg.hidden_size, 2_880);

    // Build shard file list from index (local only, no network) and mmap them.
    #[derive(serde::Deserialize)]
    struct Idx {
        weight_map: BTreeMap<String, String>,
    }
    let idx_bytes = std::fs::read(&idx).unwrap();
    let idx_parsed: Idx = serde_json::from_slice(&idx_bytes).unwrap();
    let mut files_set = std::collections::BTreeSet::new();
    for f in idx_parsed.weight_map.values() {
        files_set.insert(f.clone());
    }
    let mut model_files: Vec<std::path::PathBuf> =
        files_set.into_iter().map(|f| snap.join(f)).collect();
    model_files.sort();
    assert!(!model_files.is_empty(), "no shards resolved from index");
    let vb_cpu =
        unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::BF16, &Device::Cpu)? };

    // 1) Load only embed_tokens.weight and lm_head.weight
    let emb_cpu = vb_cpu
        .pp("model.embed_tokens")
        .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    let lm_cpu = vb_cpu
        .pp("lm_head")
        .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    let (er, ec) = emb_cpu.dims2()?;
    let (lr, lc) = lm_cpu.dims2()?;
    eprintln!(
        "embed_tokens.weight: dtype={:?} device={:?} shape=({}, {}) stride={:?}",
        emb_cpu.dtype(),
        emb_cpu.device(),
        er,
        ec,
        emb_cpu.stride()
    );
    eprintln!(
        "lm_head.weight:       dtype={:?} device={:?} shape=({}, {}) stride={:?}",
        lm_cpu.dtype(),
        lm_cpu.device(),
        lr,
        lc,
        lm_cpu.stride()
    );
    assert_eq!(emb_cpu.dtype(), DType::BF16);
    assert_eq!(lm_cpu.dtype(), DType::BF16);
    assert_eq!((er, ec), (201_088, 2_880));
    assert_eq!((lr, lc), (201_088, 2_880));

    // Distinct storage (untied) check: underlying storage objects must differ.
    let (s1, _l1) = emb_cpu.storage_and_layout();
    let (s2, _l2) = lm_cpu.storage_and_layout();
    let same_storage = std::ptr::eq(&*s1 as *const _, &*s2 as *const _);
    assert!(
        !same_storage,
        "embed and lm_head must be untied (distinct storages)"
    );

    // 2) Pick 16 token IDs, include the specified specials.
    let sample_ids: [usize; 16] = [
        0, 1, 2, 3, 42, 100, 777, 12345, 199999, // normal + pad
        200005, 200006, 200008, // <|channel|>, <|start|>, <|message|>
        200010, 200012, 201000, 201087, // more specials, last id
    ];
    eprintln!("token_ids={:?}", sample_ids);

    // For each id t: row from emb and row from lm_head must be finite with non-zero norm.
    for &t in &sample_ids {
        assert!(t < cfg.vocab_size, "token id {} out of range", t);
        let e_row = emb_cpu.i((t, ..))?.to_dtype(DType::F32)?;
        let h_row = lm_cpu.i((t, ..))?.to_dtype(DType::F32)?;
        let e_vec = e_row.to_vec1::<f32>()?;
        let h_vec = h_row.to_vec1::<f32>()?;
        let mut e_sum = 0f32;
        let mut h_sum = 0f32;
        let mut e_all_finite = true;
        let mut h_all_finite = true;
        for i in 0..ec {
            let ev = e_vec[i];
            let hv = h_vec[i];
            if !ev.is_finite() {
                e_all_finite = false;
            }
            if !hv.is_finite() {
                h_all_finite = false;
            }
            e_sum += ev * ev;
            h_sum += hv * hv;
        }
        let e_l2 = e_sum.sqrt();
        let h_l2 = h_sum.sqrt();
        eprintln!(
            "t={} ||emb[t]||2={:.6} ||lm[t]||2={:.6} finite: emb={} lm={}",
            t, e_l2, h_l2, e_all_finite, h_all_finite
        );
        assert!(e_all_finite, "embed row contains non-finite at t={}", t);
        assert!(h_all_finite, "lm_head row contains non-finite at t={}", t);
        assert!(e_l2 > 0.0, "embed row has zero norm at t={}", t);
        assert!(h_l2 > 0.0, "lm_head row has zero norm at t={}", t);
    }

    // 3) Orientation sanity: Linear forward vs x @ W^T
    // Choose 8 deterministic hidden indices based on a fixed seed.
    let seed: u64 = 42424242;
    eprintln!("seed={}", seed);
    let mut idxs = Vec::with_capacity(8);
    let mut x = seed;
    for _ in 0..8 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        idxs.push((x as usize) % cfg.hidden_size);
    }
    eprintln!("one_hot_indices={:?}", idxs);

    // Move lm_head to GPU, use F32 to check exact contract numerically.
    let w = lm_cpu.to_device(&dev)?.to_dtype(DType::F32)?; // (vocab, hidden)
    let lin = Linear::new(w.clone(), None);

    for &i in &idxs {
        let mut x_host = vec![0f32; cfg.hidden_size];
        x_host[i] = 1.0;
        let x = Tensor::from_vec(x_host, (1, cfg.hidden_size), &dev)?;
        eprintln!(
            "x: dtype={:?} device={:?} shape={:?} stride={:?} i={}",
            x.dtype(),
            x.device(),
            x.dims(),
            x.stride(),
            i
        );

        let y_lin = lin.forward(&x)?; // (1, vocab)
        let y_ref = x.matmul(&w.t()?)?; // (1, vocab)

        // Compare on GPU, F32
        let y_lin_f = y_lin.to_dtype(DType::F32)?.squeeze(0)?;
        let y_ref_f = y_ref.to_dtype(DType::F32)?.squeeze(0)?;
        let y_l = y_lin_f.to_vec1::<f32>()?;
        let y_r = y_ref_f.to_vec1::<f32>()?;
        let mut linf = 0f32;
        let mut l2 = 0f32;
        for j in 0..cfg.vocab_size {
            let d = (y_l[j] - y_r[j]).abs();
            if d > linf {
                linf = d;
            }
            l2 += d * d;
        }
        l2 = l2.sqrt();
        eprintln!("i={} Linear vs ref: L_inf={:.3e} L2={:.3e}", i, linf, l2);
        assert!(
            linf <= 1e-6,
            "orientation mismatch at i={} (L_inf={})",
            i,
            linf
        );
    }

    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[test]
fn gpt_oss_embed_head_parity_cuda_skipped() {
    eprintln!("skipped: build without 'cuda' feature");
}
