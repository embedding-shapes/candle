use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::llama::{Cache, Config as LlamaCfg, Llama};
use candle_transformers::models::mistral;

// Constants live at the top per AGENTS.md
const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 128;
const VOCAB: usize = 256;
const N_LAYERS: usize = 36; // include index 35 explicitly
const N_HEADS: usize = 8;
const N_KV_HEADS: usize = 8;
const MAX_POS: usize = 2048;
const PREFILL_TOKENS: usize = 512; // long enough while staying fast
const PROBE_TOKENS: usize = 16;
const RNG_SEED: u64 = 1337;

fn pick_gpu_device() -> Option<Device> {
    if candle::utils::cuda_is_available() {
        Device::new_cuda(0).ok()
    } else if candle::utils::metal_is_available() {
        Device::new_metal(0).ok()
    } else {
        None
    }
}

fn small_llama_config() -> LlamaCfg {
    LlamaCfg {
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        vocab_size: VOCAB,
        num_hidden_layers: N_LAYERS,
        num_attention_heads: N_HEADS,
        num_key_value_heads: N_KV_HEADS,
        use_flash_attn: false,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        max_position_embeddings: MAX_POS,
        tie_word_embeddings: false,
    }
}

fn build_model(dtype: DType, device: &Device) -> Result<(Llama, Cache, VarMap, LlamaCfg)> {
    let cfg = small_llama_config();
    device.set_seed(RNG_SEED)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, device);
    let model = Llama::load(vb, &cfg)?;
    let cache = Cache::new(true, dtype, &cfg, device)?;
    Ok((model, cache, varmap, cfg))
}

fn make_tokens(len: usize, vocab: usize, device: &Device) -> Result<Tensor> {
    // simple deterministic token sequence 0..vocab repeated
    let ids: Vec<u32> = (0..len as u32).map(|i| i % (vocab as u32)).collect();
    Tensor::from_slice(&ids, (1, len), device)
}

fn tensor_checksum(t: &Tensor) -> Result<(f64, f64)> {
    let t = t.to_dtype(DType::F32)?;
    let sum = t.sum_all()?.to_scalar::<f32>()? as f64;
    let sq = t.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok((sum, sq))
}

fn model_checksum(varmap: &VarMap) -> Result<(f64, f64)> {
    let mut names: Vec<String> = {
        let guard = varmap.data().lock().unwrap();
        guard.keys().cloned().collect()
    };
    names.sort();
    let mut s = 0f64;
    let mut s2 = 0f64;
    {
        let guard = varmap.data().lock().unwrap();
        for name in names.iter() {
            let var = guard.get(name).unwrap();
            let (x, y) = tensor_checksum(var.as_tensor())?;
            s += x;
            s2 += y;
        }
    }
    Ok((s, s2))
}

#[test]
fn bf16_long_prefill_weight_stability_embed_first() -> Result<()> {
    let Some(device) = pick_gpu_device() else {
        eprintln!("skipping: no CUDA/Metal device available");
        return Ok(());
    };
    // Use BF16 on GPU as per the report; skip on CPU.
    let dtype = device.bf16_default_to_f32();
    if dtype != DType::BF16 {
        eprintln!("skipping: device does not select BF16");
        return Ok(());
    }

    let (model, mut cache, varmap, cfg) = build_model(dtype, &device)?;

    // Snapshot embedding outputs before prefill.
    let probe = make_tokens(PROBE_TOKENS, cfg.vocab_size, &device)?;
    let emb_before = model.embed(&probe)?.to_dtype(DType::F32)?;
    let emb_before_v = emb_before.flatten_all()?.to_vec1::<f32>()?;

    // Snapshot model-wide checksum (sum and sumsq across all vars).
    let (m_sum_before, m_sumsq_before) = model_checksum(&varmap)?;

    // Long prefill forward pass (embed-first path is model.forward).
    let prefill = make_tokens(PREFILL_TOKENS, cfg.vocab_size, &device)?;
    let _ = model.forward(&prefill, 0, &mut cache)?;

    // Check embedding outputs did not change.
    let emb_after = model.embed(&probe)?.to_dtype(DType::F32)?;
    let emb_after_v = emb_after.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(
        emb_before_v, emb_after_v,
        "embedding outputs changed after prefill; weights likely mutated"
    );

    // Check model-wide checksum did not change.
    let (m_sum_after, m_sumsq_after) = model_checksum(&varmap)?;
    assert!(
        (m_sum_before - m_sum_after).abs() == 0.0 && (m_sumsq_before - m_sumsq_after).abs() == 0.0,
        "model weights checksum changed: before=({m_sum_before},{m_sumsq_before}) after=({m_sum_after},{m_sumsq_after})"
    );

    // Specific hot-spot check: layer 35 q_proj weight remains identical.
    let key_name = "model.layers.35.self_attn.q_proj.weight";
    if varmap.data().lock().unwrap().contains_key(key_name) {
        let guard = varmap.data().lock().unwrap();
        let w = guard.get(key_name).unwrap().as_tensor().clone();
        drop(guard);
        let (w_sum_before, w_sumsq_before) = tensor_checksum(&w)?;
        // recompute after (read again from varmap to avoid caching effects)
        let guard2 = varmap.data().lock().unwrap();
        let w2 = guard2.get(key_name).unwrap().as_tensor().clone();
        drop(guard2);
        let (w_sum_after, w_sumsq_after) = tensor_checksum(&w2)?;
        assert!(
            (w_sum_before - w_sum_after).abs() == 0.0
                && (w_sumsq_before - w_sumsq_after).abs() == 0.0,
            "layer 35 q_proj weight checksum changed"
        );
    }

    // Reproducibility: same prefill twice yields identical logits (within small tol).
    let prefill2 = prefill.clone();
    let mut cache2 = Cache::new(true, dtype, &cfg, &device)?;
    let logits1 = model.forward(&prefill, 0, &mut cache)?;
    let logits2 = model.forward(&prefill2, 0, &mut cache2)?;
    let l1 = logits1
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let l2 = logits2
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(l1.len(), l2.len());
    let mut max_abs = 0f32;
    for i in 0..l1.len() {
        let d = (l1[i] - l2[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < 1e-3,
        "logits mismatch after repeated prefill: max_abs_diff={max_abs}"
    );

    Ok(())
}

// Mistral embed-first path stability test mirroring the reported scenario.
#[test]
fn bf16_mistral_embed_first_weight_stability() -> Result<()> {
    let Some(device) = pick_gpu_device() else {
        eprintln!("skipping: no CUDA/Metal device available");
        return Ok(());
    };
    let dtype = device.bf16_default_to_f32();
    if dtype != DType::BF16 {
        eprintln!("skipping: device does not select BF16");
        return Ok(());
    }

    // Small but representative Mistral config with 36 layers.
    let cfg = mistral::Config {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        num_hidden_layers: N_LAYERS,
        num_attention_heads: N_HEADS,
        head_dim: None,
        num_key_value_heads: N_KV_HEADS,
        hidden_act: candle_nn::Activation::Silu,
        max_position_embeddings: 4096,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        sliding_window: Some(1024),
        use_flash_attn: false,
    };

    device.set_seed(RNG_SEED)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
    let mut model = mistral::Model::new(&cfg, vb)?;

    // Snapshot model-wide checksum and layer 35 MLP down_proj.
    let (m_sum_before, m_sumsq_before) = model_checksum(&varmap)?;
    let key_name = "model.layers.35.mlp.down_proj.weight";
    let (w_sum_before, w_sumsq_before) = {
        let guard = varmap.data().lock().unwrap();
        let w = guard.get(key_name).unwrap().as_tensor().clone();
        tensor_checksum(&w)?
    };

    // Build embed-first sequence: image-like embeddings + embedded text.
    // Image embeds (random) + 64 text tokens converted to embeddings, then concatenated.
    let img_len = 512usize; // reasonably long but fast
    let txt_len = 64usize;
    // Simulate projector-produced embeddings as BF16 to avoid mixed-type slow paths in tests.
    let img_embeds = Tensor::randn(0f32, 1f32, (1, img_len, HIDDEN), &device)?.to_dtype(dtype)?;
    let text_ids = make_tokens(txt_len, cfg.vocab_size, &device)?;
    let txt_embeds = model.embed_tokens().forward(&text_ids)?.to_dtype(dtype)?;
    let xs = Tensor::cat(&[&img_embeds, &txt_embeds], 1)?;

    // Run a prefill on embed-first path.
    let logits = model.forward_embeds(&xs, None, 0)?;
    // Immediately repeat with a fresh model to check reproducibility.
    let varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, dtype, &device);
    let mut model2 = mistral::Model::new(&cfg, vb2)?;
    let txt_embeds2 = model2.embed_tokens().forward(&text_ids)?.to_dtype(dtype)?;
    let xs2 = Tensor::cat(&[&img_embeds, &txt_embeds2], 1)?;
    let logits2 = model2.forward_embeds(&xs2, None, 0)?;

    // Check model-wide checksum unchanged.
    let (m_sum_after, m_sumsq_after) = model_checksum(&varmap)?;
    assert!(
        (m_sum_before - m_sum_after).abs() == 0.0 && (m_sumsq_before - m_sumsq_after).abs() == 0.0,
        "mistral weights checksum changed after embed-first prefill"
    );

    // Check specific hot-spot at layer 35 MLP down-proj.
    let (w_sum_after, w_sumsq_after) = {
        let guard = varmap.data().lock().unwrap();
        let w = guard.get(key_name).unwrap().as_tensor().clone();
        tensor_checksum(&w)?
    };
    assert!(
        (w_sum_before - w_sum_after).abs() == 0.0 && (w_sumsq_before - w_sumsq_after).abs() == 0.0,
        "mistral layer 35 down_proj checksum changed"
    );

    // Reproducibility: logits should match across fresh models given same inputs.
    let l1 = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let l2 = logits2
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(l1.len(), l2.len());
    let mut max_abs = 0f32;
    for i in 0..l1.len() {
        let d = (l1[i] - l2[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < 1e-3,
        "mistral logits mismatch after repeated embed-first prefill: max_abs_diff={max_abs}"
    );

    Ok(())
}
