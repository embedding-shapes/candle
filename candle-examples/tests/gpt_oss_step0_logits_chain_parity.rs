use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;
use gpt_oss_tokenizer::{allowed_specials_for_next, load_stop_token_ids, render_then_encode};
use openai_harmony::chat::{Message, Role};

// Constants / config
const SNAPSHOT: &str =
    "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const PROMPT: &str = "Explain what MXFP4 quantization is";

fn expand_tilde(p: &str) -> std::path::PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        std::path::Path::new(&std::env::var("HOME").unwrap()).join(rest)
    } else {
        std::path::PathBuf::from(p)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

fn cuda_device_or_skip() -> Result<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Ok(d),
        Err(_) => {
            eprintln!("cuda device not available — skipping");
            // Return early by using anyhow::bail inside the test if needed; the caller will handle.
            Err(anyhow::anyhow!("skip"))
        }
    }
}

fn hub_load_local_safetensors<P: AsRef<std::path::Path>>(
    path: P,
    json_file: &str,
) -> candle::Result<Vec<std::path::PathBuf>> {
    let path = path.as_ref();
    let jsfile = std::fs::File::open(path.join(json_file))?;
    let json: serde_json::Value = serde_json::from_reader(&jsfile).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file);
        }
    }
    let safetensors_files: Vec<_> = safetensors_files.into_iter().map(|v| path.join(v)).collect();
    Ok(safetensors_files)
}

#[test]
fn gpt_oss_step0_logits_chain_parity() -> Result<()> {
    // Device + dtype match the example (prefer CUDA + bf16). Skip if no CUDA.
    let device = match cuda_device_or_skip() {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    let mut dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };
    if matches!(std::env::var("CANDLE_FORCE_BF16").ok().as_deref(), Some("1") | Some("true") | Some("TRUE")) {
        dtype = DType::BF16;
    }
    eprintln!("device={:?} dtype={:?}", device, dtype);

    // Resolve snapshot and tokenizer paths.
    let snap = expand_tilde(SNAPSHOT);
    let tok_path = snap.join("tokenizer.json");
    let hf_tok = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer.json: {e}"))?;

    // Render prompt via Harmony + chat_template from the snapshot.
    let msg = Message::from_role_and_content(Role::User, PROMPT.to_string());
    let input_ids = render_then_encode(&snap, &[msg], true)?;
    // Debug inputs
    eprintln!("seed=0 prompt='{}'", PROMPT);
    eprintln!("prompt_ids_len={} first32={:?}", input_ids.len(), &input_ids.iter().take(32).collect::<Vec<_>>());
    let decoded = hf_tok.decode(&input_ids, /*skip_special_tokens=*/ false).unwrap();
    eprintln!("decoded_tail_last_64={}", &decoded[decoded.len().saturating_sub(64)..]);
    assert!(decoded.contains("<|start|>assistant"), "render must end with <|start|>assistant");

    // Load config + model weights exactly like the example.
    let cfg_bytes = std::fs::read(snap.join("config.json"))?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes)?;
    let model_files = hub_load_local_safetensors(&snap, "model.safetensors.index.json")?;
    assert!(!model_files.is_empty(), "no safetensors shards found under snapshot");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let mut model = GptOssModel::load(vb, &cfg)?;

    // Build inputs tensor and forward once (prefill). Use seqlen_offset=0 as per spec.
    let t = Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), &device)?;
    let logits = model.forward(&t, 0)?; // (1, T, V)
    let (b, tlen, vocab) = logits.dims3()?;
    assert_eq!(b, 1);
    assert_eq!(vocab, cfg.vocab_size);
    let last = logits.i((0, tlen - 1))?; // (V)
    let last_f32 = last.to_dtype(DType::F32)?;
    let raw: Vec<f32> = last_f32.to_vec1()?;
    eprintln!("raw dtype=F32 device={:?} shape=({})", device, raw.len());

    // Top-10 raw (logits) and argmax before any masking.
    let mut idx: Vec<usize> = (0..raw.len()).collect();
    idx.sort_by(|&i, &j| raw[j].partial_cmp(&raw[i]).unwrap());
    let top10_raw: Vec<(usize, f32)> = idx.iter().take(10).map(|&i| (i, raw[i])).collect();
    let id_raw = idx[0];
    eprintln!("top10_raw={:?}", top10_raw);

    // Build the global forbid set identical to the example.
    let mut global_forbid: std::collections::BTreeSet<u32> = Default::default();
    for s in [
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>",
        "<|start|>",
    ] {
        if let Some(id) = hf_tok.token_to_id(s) { global_forbid.insert(id); }
    }
    let id_channel = hf_tok.token_to_id("<|channel|>").unwrap() as usize; // 200005
    let id_message = hf_tok.token_to_id("<|message|>").unwrap() as usize; // 200008
    let id_return = hf_tok.token_to_id("<|return|>").unwrap() as usize; // 200002
    let id_call = hf_tok.token_to_id("<|call|>").unwrap() as usize; // 200012
    let id_end = hf_tok.token_to_id("<|end|>").unwrap() as usize; // 200007

    // Build probabilities before masking to probe invariants.
    let logits_t = Tensor::from_vec(raw.clone(), raw.len(), &device)?;
    let probs_pre = candle_nn::ops::softmax_last_dim(&logits_t)?;
    let probs_pre_v: Vec<f32> = probs_pre.to_vec1()?;

    // FSM state detection via decoded string
    let allow_syms = allowed_specials_for_next(&decoded);
    let mut allow_ids: std::collections::BTreeSet<u32> = Default::default();
    for s in allow_syms.iter() {
        if let Some(id) = hf_tok.token_to_id(s) { allow_ids.insert(id); }
    }
    let to_mask: Vec<u32> = global_forbid.iter().copied().filter(|id| !allow_ids.contains(id)).collect();
    eprintln!(
        "fsm_state=AfterAssistant allow_size={} forbid_size={} allow_has_channel={} allow_syms={:?}",
        allow_ids.len(), global_forbid.len(), allow_ids.contains(&(id_channel as u32)), allow_syms
    );

    // Per-processor trace: temperature (1.0), then probability mask.
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::All { temperature: 1.0 });
    let mut captured_post: Option<Vec<f32>> = None;
    let _ = sampler.sample_f(&last_f32, |prs: &mut [f32]| {
        for id in &to_mask {
            let i = *id as usize;
            if i < prs.len() { prs[i] = 0.0; }
        }
        captured_post = Some(prs.to_vec());
    })?;
    let post = captured_post.unwrap();
    let mut idx2: Vec<usize> = (0..post.len()).collect();
    idx2.sort_by(|&i, &j| post[j].partial_cmp(&post[i]).unwrap());
    let top10_post: Vec<(usize, f32)> = idx2.iter().take(10).map(|&i| (i, post[i])).collect();
    let id_post = idx2[0];
    eprintln!("top10_post={:?}", top10_post);

    // Log trace for the single critical id
    eprintln!(
        "trace=[('temp', {:.6}, {}), ('mask', {:.6}, {})]",
        probs_pre_v[id_channel], argmax(&probs_pre_v), post[id_channel], id_post
    );
    eprintln!(
        "id_channel={} pre={:.6} post={:.6} masked?={} | id_message={} post={:.6} | id_return={} post={:.6} | id_call={} post={:.6} | id_end={} post={:.6}",
        id_channel, probs_pre_v[id_channel], post[id_channel], post[id_channel] == 0.0,
        id_message, post[id_message], id_return, post[id_return], id_call, post[id_call], id_end, post[id_end]
    );
    // Normal token invariants
    for &tid in &[11usize, 13usize, 25usize, 220usize] {
        assert_eq!(post[tid], probs_pre_v[tid], "normal token {} must be unchanged by special masking", tid);
    }

    // Stop token policy parity check
    let stop_ids = load_stop_token_ids(&snap)?;
    eprintln!("stop_token_ids={:?}", stop_ids);
    assert!(stop_ids.contains(&(id_return as u32)) && stop_ids.contains(&(id_call as u32)), "stop set must contain <|return|> and <|call|>");

    // Assertions per the spec
    assert_eq!(id_raw, id_channel, "raw argmax must be <|channel|> (id {})", id_channel);
    assert!(post[id_channel].is_finite() && post[id_channel] > 0.0, "channel must remain unmasked and finite");
    assert_eq!(id_post, id_channel, "post-mask argmax must remain <|channel|>");
    assert_eq!(post[id_message], 0.0, "<|message|> must be masked at step 0");
    assert_eq!(post[id_return], 0.0, "<|return|> must be masked at step 0");
    assert_eq!(post[id_call], 0.0, "<|call|> must be masked at step 0");
    assert_eq!(post[id_end], 0.0, "<|end|> must be masked at step 0");

    Ok(())
}
