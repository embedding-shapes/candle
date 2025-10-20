use anyhow::{Context as _, Result};
use candle::{DType, IndexOp, Tensor, D};
use candle::Module as _;
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::utils;
use candle_transformers::models::deepseek2::TopKLastDimOp as _;
use candle_transformers::models::gpt_oss::model::GptOssModel;
use candle_transformers::models::gpt_oss::{select_attn_mode_for_layer, AttnMode, GptOssConfigMinimal};
use gpt_oss_tokenizer::render_then_encode;
use openai_harmony::chat::{Message, Role};
use std::path::PathBuf;
use std::process::Command;

const SNAPSHOT: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

fn expand_tilde(p: &str) -> PathBuf {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    PathBuf::from(p)
}

#[test]
fn t41_bisect_first_layer_embed_and_layer0_out() -> Result<()> {
    // Gate heavy GPU+Python check.
    if std::env::var("RUN_GPT_OSS_PARITY").ok().as_deref() != Some("1") {
        eprintln!("[skip] set RUN_GPT_OSS_PARITY=1 to run t41_first_layer_bisect");
        return Ok(());
    }

    // Resolve snapshot and messages -> tokens.
    let snapshot = expand_tilde(SNAPSHOT);
    let user = Message::from_role_and_content(Role::User, "Explain what MXFP4 quantization is".to_string());
    let input_ids: Vec<u32> = render_then_encode(&snapshot, &[user], true)
        .context("render_then_encode failed")?;
    assert!(input_ids.len() >= 4, "unexpectedly short prompt");

    // Candle device/dtype and model load
    let device = candle_examples::device(false /* cpu */)?;
    assert!(device.is_cuda(), "CUDA device required for this parity test");
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };

    let cfg_path = snapshot.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path)
        .with_context(|| format!("failed to read config: {}", cfg_path.display()))?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes).context("invalid config.json")?;

    let model_index = snapshot.join("model.safetensors.index.json");
    let model_files: Vec<PathBuf> = {
        let idx_bytes = std::fs::read(&model_index)
            .with_context(|| format!("failed to read model index: {}", model_index.display()))?;
        #[derive(serde::Deserialize)]
        struct Idx { weight_map: std::collections::BTreeMap<String, String> }
        let idx: Idx = serde_json::from_slice(&idx_bytes).context("invalid model.safetensors.index.json")?;
        let files = idx.weight_map.values().cloned().collect::<std::collections::BTreeSet<_>>();
        files.iter().map(|f| snapshot.join(f)).collect()
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };

    // Force safe RMS compute in f32 to match HF stability.
    std::env::set_var("CANDLE_RMS_FP32", "1");
    let mut model = GptOssModel::load(vb, &cfg).context("load GPT-OSS model")?;

    // Embed only
    let t_ids = Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), &device)?;
    let xs_embed = model.embed.forward(&t_ids)?; // (1,T,H)
    let (b, t, h) = xs_embed.dims3()?;
    assert_eq!(b, 1);
    let last_embed = xs_embed.i((0, t - 1))?.to_dtype(DType::F32)?; // (H)
    let last_embed_v = last_embed.to_vec1::<f32>()?;

    // One full layer forward (layer 0) mirroring model.forward logic.
    let head_dim = cfg.head_dim();
    let n_q = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Pre-attn norm
    let layer0 = &mut model.layers[0];
    let x_norm = layer0.input_layernorm.forward(&xs_embed)?; // (1,T,H)
    // QKV projections
    let q = x_norm.apply(&layer0.attn.q_proj)?; // (1,T,n_q*hd)
    let k = x_norm.apply(&layer0.attn.k_proj)?; // (1,T,n_kv*hd)
    let v = x_norm.apply(&layer0.attn.v_proj)?; // (1,T,n_kv*hd)
    let q = q.reshape((1, t, n_q, head_dim))?;
    let k = k.reshape((1, t, n_kv, head_dim))?;
    let v = v.reshape((1, t, n_kv, head_dim))?;
    // RoPE
    let q_bhtd = q.transpose(1, 2)?; // (1,n_q,t,d)
    let k_bhtd = k.transpose(1, 2)?; // (1,n_kv,t,d)
    let (q_bhtd, k_bhtd) = model.rope.apply_rotary_emb_qk(&q_bhtd, &k_bhtd, 0usize)?;
    let q = q_bhtd.transpose(1, 2)?; // (1,t,n_q,d)
    let k_step = k_bhtd; // (1,n_kv,t,d)
    let v_step = v.transpose(1, 2)?; // (1,n_kv,t,d)
    // KV cache append for layer 0
    let (k_all, v_all) = model.kv_caches[0].append(&k_step.contiguous()?, &v_step.contiguous()?)?; // (1,n_kv,tk,d)
    // Repeat for GQA
    let n_rep = n_q / n_kv;
    let k_rep = utils::repeat_kv(k_all.clone(), n_rep)?; // (1,n_q,tk,d)
    let v_rep = utils::repeat_kv(v_all.clone(), n_rep)?; // (1,n_q,tk,d)
    let k_btkhd = k_rep.transpose(1, 2)?; // (1,tk,n_q,d)
    let v_btkhd = v_rep.transpose(1, 2)?; // (1,tk,n_q,d)

    // Attention mode
    let attn_mode = select_attn_mode_for_layer(
        &GptOssConfigMinimal {
            num_hidden_layers: cfg.num_hidden_layers,
            layer_types: cfg.effective_layer_types(),
            max_position_embeddings: cfg.max_position_embeddings,
            sliding_window: cfg.sliding_window,
        },
        0,
    );

    // Eager attention with sinks to keep deterministic path in tests.
    let sinks = Some(&layer0.attn.sinks);
    let y = match attn_mode {
        AttnMode::Full => candle_transformers::models::gpt_oss::eager_attn_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, t > 1, sinks)?,
        AttnMode::Sliding { left, right } => candle_transformers::models::gpt_oss::eager_attn_windowed_with_sinks(&q, &k_btkhd, &v_btkhd, softmax_scale, Some(left), Some(right), sinks)?,
    }; // (1,t,n_q,d)
    let y = y.reshape((1, t, n_q * head_dim))?;
    let y = y.apply(&layer0.attn.o_proj)?; // (1,t,H)
    let xs1 = (&xs_embed + &y)?; // residual
    // Post-attn norm + MoE
    let x_norm2 = layer0.post_attention_layernorm.forward(&xs1)?; // (1,t,H)
    let mlp_out = layer0.experts.forward(&x_norm2)?; // (1,t,H)
    let xs_after = (&xs1 + &mlp_out)?; // (1,t,H)
    let last_after = xs_after.i((0, t - 1))?.to_dtype(DType::F32)?;
    let last_after_v = last_after.to_vec1::<f32>()?;

    // Python HF: capture
    //  - embed last token (hidden_states[0][-1])
    //  - post-attention pre-MLP residual (via pre-hook on post_attention_layernorm)
    //  - first-layer output last token (hidden_states[1][-1])
    let script = {
        let toks_json = serde_json::to_string(&input_ids).unwrap();
        format!(
            r#"import json, torch
from transformers import AutoModelForCausalLM
snap = {snap}
tokens = {toks}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=dtype, local_files_only=True)
model.to(device)
model.eval()
cap = {{}}
def pre_hook(module, inputs):
    x = inputs[0].detach().float().cpu()
    cap['post_attn_in'] = x
    return None
h = model.model.layers[0].post_attention_layernorm.register_forward_pre_hook(pre_hook)
with torch.no_grad():
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model(input_ids=inp, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states  # tuple(len = layers+1)
    emb_last = hs[0][0, -1].float().cpu().tolist()
    layer0_last = hs[1][0, -1].float().cpu().tolist()
    post_attn_in_last = cap['post_attn_in'][0, -1].tolist()
    print(json.dumps({{"emb_last": emb_last, "post_attn_in_last": post_attn_in_last, "layer0_last": layer0_last}}))
    
h.remove()
"#,
            snap = format!("{:?}", snapshot.display()),
            toks = toks_json,
        )
    };
    let output = Command::new("uv")
        .args(["run", "python", "-c", &script])
        .env("PYTHONUNBUFFERED", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .context("failed to run uv python for HF hidden_states")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("python failed: {}", stderr);
    }
    let out = String::from_utf8_lossy(&output.stdout);
    #[derive(serde::Deserialize)]
    struct PyRes { emb_last: Vec<f32>, post_attn_in_last: Vec<f32>, layer0_last: Vec<f32> }
    let py: PyRes = serde_json::from_str(&out).context("invalid python json for hidden_states")?;

    // Quick stats compare: mean/std and first 8 elems.
    fn stats(v: &[f32]) -> (f32, f32) { let m = v.iter().copied().sum::<f32>()/(v.len() as f32); let var = v.iter().map(|x| (x-m)*(x-m)).sum::<f32>()/(v.len() as f32); (m, var.sqrt()) }
    let (m_c_emb, s_c_emb) = stats(&last_embed_v);
    let (m_p_emb, s_p_emb) = stats(&py.emb_last);
    // Candle post-attn pre-MLP
    let xs1_last = xs1.i((0, t - 1))?.to_dtype(DType::F32)?;
    let xs1_last_v = xs1_last.to_vec1::<f32>()?;
    let (m_c_l0pre, s_c_l0pre) = stats(&xs1_last_v);
    // Candle after full layer0
    let (m_c_l0, s_c_l0) = stats(&last_after_v);
    let (m_p_l0, s_p_l0) = stats(&py.layer0_last);
    let (m_p_l0pre, s_p_l0pre) = stats(&py.post_attn_in_last);

    // Candle router on last token of layer0 input
    let x_last = xs1_last.reshape((1, h))?; // (1,H) in F32
    let x_last_bf = x_last.to_dtype(dtype)?; // match weights dtype
    let router_logits = x_last_bf.apply(&layer0.router)?; // (1,E) in BF16/F16
    let router_logits = router_logits.to_dtype(DType::F32)?; // stable for topk+softmax
    let tk = router_logits.topk(4)?;
    let top_probs = candle_nn::ops::softmax_last_dim(&tk.values)?;
    let idx_rust = tk.indices.reshape((4,))?.to_vec1::<u32>()?;
    let probs_rust = top_probs.reshape((4,))?.to_vec1::<f32>()?;

    // Tolerances (these are quite strict but allow tiny numeric drift)
    assert!((m_c_emb - m_p_emb).abs() < 1e-3, "embed mean mismatch: rust={m_c_emb} py={m_p_emb}");
    assert!((s_c_emb - s_p_emb).abs() < 1e-3, "embed std mismatch: rust={s_c_emb} py={s_p_emb}");
    // Embed parity
    assert!((m_c_l0pre - m_p_l0pre).abs() < 5e-3, "post-attn-in mean mismatch: rust={m_c_l0pre} py={m_p_l0pre}");
    assert!((s_c_l0pre - s_p_l0pre).abs() < 5e-3, "post-attn-in std mismatch: rust={s_c_l0pre} py={m_p_l0pre}");
    for i in 0..8 { assert!((last_embed_v[i] - py.emb_last[i]).abs() < 5e-3, "embed elem[{i}] mismatch: {} vs {}", last_embed_v[i], py.emb_last[i]); }
    for i in 0..8 { assert!((xs1_last_v[i] - py.post_attn_in_last[i]).abs() < 2e-2, "post-attn-in elem[{i}] mismatch: {} vs {}", xs1_last_v[i], py.post_attn_in_last[i]); }

    // Python router parity: compute and compare indices/probabilities
    let script_router = {
        format!(
            r#"import json, torch
from transformers import AutoModelForCausalLM
snap = {snap}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=dtype, local_files_only=True)
model.to(device)
model.eval()
with torch.no_grad():
    # take captured post_attn_in_last from previous run is not directly available here;
    # recompute a quick forward to get it via a hook again
    tokens = {toks}
    cap = {{}}
    def pre_hook(module, inputs):
        cap['x'] = inputs[0][0, -1].detach().float().to(device)
        return None
    h = model.model.layers[0].post_attention_layernorm.register_forward_pre_hook(pre_hook)
    _ = model(input_ids=torch.tensor([tokens], dtype=torch.long, device=device))
    h.remove()
    x = cap['x']  # (H)
    router = model.model.layers[0].mlp.router
    x = x.to(router.weight.dtype)
    logits = torch.nn.functional.linear(x, router.weight, router.bias)  # (E)
    topv, topi = torch.topk(logits, k=4, dim=-1)
    probs = torch.softmax(topv, dim=-1)
    print(json.dumps({{"idx": topi.detach().cpu().tolist(), "probs": probs.detach().cpu().tolist()}}))
"#,
            snap = format!("{:?}", snapshot.display()),
            toks = serde_json::to_string(&input_ids).unwrap(),
        )
    };
    let out_router = Command::new("uv")
        .args(["run", "python", "-c", &script_router])
        .env("PYTHONUNBUFFERED", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .context("failed to run uv python for router parity")?;
    if !out_router.status.success() {
        let stderr = String::from_utf8_lossy(&out_router.stderr);
        anyhow::bail!("python router failed: {}", stderr);
    }
    let out_router_json = String::from_utf8_lossy(&out_router.stdout);
    #[derive(serde::Deserialize)]
    struct PyRouter { idx: Vec<i64>, probs: Vec<f32> }
    let pr: PyRouter = serde_json::from_str(&out_router_json).context("invalid python json for router")?;

    // Compare indices and probs (unordered compare ok; both are top-4 largest)
    for j in 0..4 {
        assert_eq!(idx_rust[j] as i64, pr.idx[j], "router top-idx[{j}] mismatch: rust={} py={}", idx_rust[j], pr.idx[j]);
        assert!((probs_rust[j] - pr.probs[j]).abs() < 2e-3, "router prob[{j}] mismatch: rust={} py={}", probs_rust[j], pr.probs[j]);
    }

    // Full layer0 parity
    let (m_c_l0, s_c_l0) = stats(&last_after_v);
    let (m_p_l0, s_p_l0) = stats(&py.layer0_last);
    assert!((m_c_l0 - m_p_l0).abs() < 5e-3, "layer0 mean mismatch: rust={m_c_l0} py={m_p_l0}");
    assert!((s_c_l0 - s_p_l0).abs() < 5e-3, "layer0 std mismatch: rust={s_c_l0} py={s_p_l0}");
    for i in 0..8 { assert!((last_after_v[i] - py.layer0_last[i]).abs() < 2e-2, "layer0 elem[{i}] mismatch: {} vs {}", last_after_v[i], py.layer0_last[i]); }

    Ok(())
}
