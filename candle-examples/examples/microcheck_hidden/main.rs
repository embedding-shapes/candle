use anyhow::{Context as _, Result};
use clap::Parser;

use candle::{DType, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;

use openai_harmony::chat::{Message, Role};
use gpt_oss_tokenizer::render_then_encode;

    #[derive(Parser, Debug)]
#[command(author, version, about = "Microcheck: final hidden (post-norm, pre-lm_head) parity")] 
    struct Args {
    /// The user prompt; default matches the reference script.
    #[arg(long, default_value = "Explain what MXFP4 quantization is")]
    prompt: String,

    /// If set, call the Python baseline to fetch reference vectors and compute metrics.
    #[arg(long)]
    compare_python: bool,

    /// If set, write (1+3L,H) last-token per-layer states to this npy file.
    #[arg(long)]
    dump_layers_npy: Option<std::path::PathBuf>,

    /// If set, read a Python npy (layers_step0_last.npy) and compare row-wise metrics.
    #[arg(long)]
    compare_layers_npy: Option<std::path::PathBuf>,

    /// Dump layer-0 attention Q/K/V last-token (pre/post-RoPE) stats.
    #[arg(long)]
    debug_l0: bool,

        /// Compare Rust layer-0 Q/K/V against Python baseline JSON.
        #[arg(long)]
        compare_python_l0: bool,

        /// Compare last-token attention logits/weights (head 0/1) between Rust and Python.
        #[arg(long)]
        compare_python_l0_attn: bool,
    }

const DEFAULT_SNAPSHOT_DIR: &str =
    "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";
const MODEL_INDEX_FILE: &str = "model.safetensors.index.json";

fn expand_tilde(p: &str) -> Result<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    if let Some(rest) = p.strip_prefix("~/") {
        let home = std::env::var("HOME").context("$HOME not set for ~ expansion")?;
        Ok(Path::new(&home).join(rest))
    } else {
        Ok(PathBuf::from(p))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let snapshot_dir = expand_tilde(DEFAULT_SNAPSHOT_DIR)?;
    let device = candle_examples::device(false /* cpu */)?;
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F16 };
    println!("Device: {:?}", device);
    println!("dtype: {}", match dtype { DType::BF16 => "bf16", DType::F16 => "f16", _ => "other" });
    println!("Snapshot: {}", snapshot_dir.display());

    // Tokenize via Harmony template to ensure parity.
    let user_msg = Message::from_role_and_content(Role::User, args.prompt.clone());
    let tokens: Vec<u32> = render_then_encode(&snapshot_dir, &[user_msg], true)
        .context("failed to render+encode with chat_template.jinja + tokenizer.json")?;
    println!("prompt len: {}", tokens.len());

    // Load config and weights
    let cfg_path = snapshot_dir.join("config.json");
    let cfg_bytes = std::fs::read(&cfg_path)
        .with_context(|| format!("failed to read config: {}", cfg_path.display()))?;
    let cfg: GptOssConfig = serde_json::from_slice(&cfg_bytes).context("invalid config.json")?;
    println!("hidden_size={} num_layers={}", cfg.hidden_size, cfg.num_hidden_layers);

    let model_files = candle_examples::hub_load_local_safetensors(&snapshot_dir, MODEL_INDEX_FILE)
        .with_context(|| format!("failed to read index {MODEL_INDEX_FILE} under {}", snapshot_dir.display()))?;
    if model_files.is_empty() {
        anyhow::bail!("no safetensors files found under {}", snapshot_dir.display());
    }
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device)? };
    let mut model = GptOssModel::load(vb, &cfg).context("failed to load GPT-OSS model weights")?;

    // Build input tensor and run one full forward to fetch last hidden (post-norm)
    let t = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &device)?;
    let last = model.forward_last_hidden_post_norm(&t, 0)?; // (hidden)
    let last_f32 = last.to_dtype(DType::F32)?;
    let v = last_f32.to_vec1::<f32>()?;
    let first8: Vec<f32> = v.iter().take(8).copied().collect();
    let mean = if v.is_empty() { 0.0 } else { v.iter().copied().sum::<f32>() / (v.len() as f32) };
    let std = if v.is_empty() { 0.0 } else { let var = v.iter().map(|x| { let d = *x - mean; d*d }).sum::<f32>() / (v.len() as f32); var.sqrt() };
    println!("last-hidden-post-norm: len={} first8={:?} mean={:.6} std={:.6}", v.len(), first8, mean, std);

    let rows_opt = if args.dump_layers_npy.is_some() || args.compare_layers_npy.is_some() {
        Some(model.debug_collect_layer_states_last_token(&t, 0)?)
    } else { None };
    if let Some(path) = args.dump_layers_npy.as_ref() {
        let rows = rows_opt.as_ref().unwrap();
        rows.write_npy(path).context("failed to write layers npy")?;
        println!("wrote per-layer states to {}", path.display());
    }
    if let Some(py_path) = args.compare_layers_npy.as_ref() {
        let rust_rows = rows_opt.as_ref().unwrap();
        let py_rows_raw = Tensor::read_npy(py_path).context("failed to read python npy")?;
        let (ra, ca) = rust_rows.dims2()?;
        let py_rows = py_rows_raw.reshape((ra, ca))?;
        let (rb, cb) = (ra, ca);
        if ra != rb || ca != cb { anyhow::bail!("shape mismatch rust={}x{} python={}x{}", ra, ca, rb, cb); }
        let rust = rust_rows.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let py = py_rows.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut row_names: Vec<String> = Vec::with_capacity(ra);
        row_names.push("h0".to_string());
        for i in 0..((ra - 1) / 3) {
            row_names.push(format!("l{}.pre_attn_norm", i));
            row_names.push(format!("l{}.post_attn_resid", i));
            row_names.push(format!("l{}.post_mlp_resid", i));
        }
        println!("Row-wise metrics vs Python (Linf, L2):");
        for r in 0..ra {
            let a = &rust[r];
            let b = &py[r];
            let mut linf = 0f32;
            let mut l2 = 0f64;
            for c in 0..ca {
                let d = a[c] - b[c];
                if d.abs() > linf { linf = d.abs(); }
                l2 += (d as f64) * (d as f64);
            }
            let l2 = l2.sqrt();
            println!("  {:>24}: Linf={:.6e} L2={:.6e}", row_names[r], linf, l2);
        }
    }

    if args.compare_python {
        // Ask the Python baseline to dump both post_mlp and post_norm vectors as JSON.
        let output = std::process::Command::new("uv")
            .arg("run")
            .arg("python")
            .arg("gpt_oss_transformers.py")
            .arg("--dump-final-hidden")
            .output()
            .context("failed to spawn Python baseline via `uv run`")?;
        if !output.status.success() {
            eprintln!("Python stderr:\n{}", String::from_utf8_lossy(&output.stderr));
            anyhow::bail!("python baseline failed");
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Expect the last line to be a JSON object.
        let json_line = stdout
            .lines()
            .rev()
            .find(|l| l.trim_start().starts_with('{'))
            .ok_or_else(|| anyhow::anyhow!("python baseline did not emit JSON"))?;
        let vjson: serde_json::Value = serde_json::from_str(json_line)?;
        let py_post_norm: Vec<f32> = vjson["post_norm_last"].as_array()
            .ok_or_else(|| anyhow::anyhow!("missing post_norm_last array"))?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
        if py_post_norm.len() != v.len() {
            anyhow::bail!("size mismatch: rust={} python={}", v.len(), py_post_norm.len());
        }
        // Compute metrics
        let mut l2 = 0f64;
        let mut linf = 0f32;
        for (a, b) in v.iter().zip(py_post_norm.iter()) {
            let d = *a - *b;
            l2 += (d as f64) * (d as f64);
            let ad = d.abs();
            if ad > linf { linf = ad; }
        }
        let l2 = (l2 as f64).sqrt();
        println!("compare-python: post_norm_last Linf={:.6e} L2={:.6e}", linf, l2);
    }

        if args.debug_l0 || args.compare_python_l0 || args.compare_python_l0_attn {
        // Force sinks parity by leaving sinks enabled (matches HF) and ensure YaRN
        // placement is on cos/sin to match Transformers (default unless overridden).
        // Users can set CANDLE_YARN_MODE to override, but we log what we use.
        let l0 = model.debug_l0_qkv_last_token(&t, 0)?;
        let (q_pre, k_pre, v_pre, q_rope, k_rope, softmax_scale, attn_mode) = l0;
        println!("l0.softmax_scale: {:.9}", softmax_scale);
        println!("l0.attn_mode: {:?}", attn_mode);
        println!("env CANDLE_YARN_MODE: {}", std::env::var("CANDLE_YARN_MODE").unwrap_or_else(|_| "<unset>".into()));

        // Shapes and dtype
        let (hq, dq) = q_pre.dims2()?; // (n_q, d)
        let (hkv, dkv) = k_pre.dims2()?;
        println!("q_pre: dtype=f32 shape=({}, {})", hq, dq);
        println!("k_pre: dtype=f32 shape=({}, {})", hkv, dkv);
        println!("v_pre: dtype=f32 shape=({}, {})", v_pre.dims2()?.0, v_pre.dims2()?.1);
        println!("q_rope: dtype=f32 shape=({}, {})", q_rope.dims2()?.0, q_rope.dims2()?.1);
        println!("k_rope: dtype=f32 shape=({}, {})", k_rope.dims2()?.0, k_rope.dims2()?.1);

        // Head-0 first-8 values and stats helper
        fn head_stats(tag: &str, m: &Tensor) -> anyhow::Result<()> {
            let (h, d) = m.dims2()?;
            let h0 = m.i((0, 0..d))?.to_vec1::<f32>()?;
            let first8: Vec<f32> = h0.iter().take(8).copied().collect();
            let flat = m.to_vec2::<f32>()?;
            let mut sum = 0f64;
            let mut cnt = 0usize;
            for r in &flat { for &x in r { sum += x as f64; cnt += 1; } }
            let mean = if cnt == 0 { 0.0 } else { (sum / cnt as f64) as f32 };
            let mut var = 0f64;
            for r in &flat { for &x in r { let d = x as f64 - mean as f64; var += d*d; } }
            let std = if cnt == 0 { 0.0 } else { (var / cnt as f64).sqrt() as f32 };
            println!("{}: heads={} dim={} head0.first8={:?} mean={:.6} std={:.6}", tag, h, d, first8, mean, std);
            Ok(())
        }

        head_stats("q_pre", &q_pre)?;
        head_stats("k_pre", &k_pre)?;
        head_stats("v_pre", &v_pre)?;
        head_stats("q_rope", &q_rope)?;
        head_stats("k_rope", &k_rope)?;

        // Also print l0.pre_attn_norm last-token first8 for both Rust and Python.
        let rows = model.debug_collect_layer_states_last_token(&t, 0)?; // (1+3L,H)
        let pre = rows.i((1, 0..8))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        println!("l0.pre_attn_norm.last.first8={:?}", pre);

        // Also sample q_proj weight first row first-8 for quick sanity.
        let (qw, _kw, _vw) = model.debug_l0_qkv_weights()?;
        let qw_samp = qw.i((0, 0..8))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        println!("q_proj.weight[0,0..8]={:?}", qw_samp);

        // Mask parity check at step-0: for sliding_window=128 and T<=75, windowed mask equals causal.
        // Rebuild masks and compare checksums.
        let qlen = t.dims2()?.1;
        let klen = qlen;
        let make_causal = || -> candle::Result<Tensor> {
            let mask: Vec<u8> = (0..qlen).flat_map(|i| (0..klen).map(move |j| u8::from(j > i))).collect();
            Tensor::from_slice(&mask, (qlen, klen), t.device())
        };
        let make_window = |left: usize, right: usize| -> candle::Result<Tensor> {
            let mask: Vec<u8> = (0..qlen)
                .flat_map(|i| {
                    (0..klen).map(move |j| {
                        let i = i as isize;
                        let j = j as isize;
                        let l = left as isize;
                        let r = right as isize;
                        let allow = (j > i - l) && (j <= i + r);
                        u8::from(!allow)
                    })
                })
                .collect();
            Tensor::from_slice(&mask, (qlen, klen), t.device())
        };
        if let candle_transformers::models::gpt_oss::AttnMode::Sliding { left, right } = attn_mode {
            let causal = make_causal()?;
            let window = make_window(left, right)?;
            let diff = (causal.to_dtype(DType::F32)? - &window.to_dtype(DType::F32)?)?.abs()?;
            let sum = diff.sum_all()?.to_scalar::<f32>()?;
            println!("mask parity (window vs causal): sum_abs_diff={:.0}", sum);
        } else {
            println!("mask parity: Full causal mode active");
        }

        // Optionally compare with Python JSON dump.
        if args.compare_python_l0 {
            let output = std::process::Command::new("uv")
                .arg("run")
                .arg("python")
                .arg("gpt_oss_transformers.py")
                .arg("--dump-l0-qkv")
                .output()
                .context("failed to spawn Python --dump-l0-qkv")?;
            if !output.status.success() {
                eprintln!("Python stderr:\n{}", String::from_utf8_lossy(&output.stderr));
                anyhow::bail!("python l0-qkv baseline failed");
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            let json_line = stdout
                .lines()
                .rev()
                .find(|l| l.trim_start().starts_with('{'))
                .ok_or_else(|| anyhow::anyhow!("python baseline (l0-qkv) did not emit JSON"))?;
            let vjson: serde_json::Value = serde_json::from_str(json_line)?;

            fn vec2_from_json(v: &serde_json::Value, rows: usize, cols: usize) -> anyhow::Result<Vec<Vec<f32>>> {
                let arr = v.as_array().ok_or_else(|| anyhow::anyhow!("expected array"))?;
                if arr.len() != rows * cols { anyhow::bail!("size mismatch: got {} wanted {}", arr.len(), rows*cols); }
                let mut out = vec![vec![0f32; cols]; rows];
                for i in 0..rows {
                    for j in 0..cols {
                        out[i][j] = arr[i*cols + j].as_f64().unwrap_or(0.0) as f32;
                    }
                }
                Ok(out)
            }

            let (hq, dq) = q_pre.dims2()?;
            let (hkv, dkv) = k_pre.dims2()?;
            let py_q_pre = vec2_from_json(&vjson["q_pre"], hq, dq)?;
            let py_k_pre = vec2_from_json(&vjson["k_pre"], hkv, dkv)?;
            let py_v_pre = vec2_from_json(&vjson["v_pre"], hkv, dkv)?;
            let py_q_rope = vec2_from_json(&vjson["q_rope"], hq, dq)?;
            let py_k_rope = vec2_from_json(&vjson["k_rope"], hkv, dkv)?;
            // Compare sinks vectors (first 8 values)
            if let Some(py_sinks) = vjson.get("sinks").and_then(|v| v.as_array()) {
                let py_s: Vec<f32> = py_sinks.iter().take(8).map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
                let rust_s = model.debug_l0_sinks()?.i(0..py_s.len())?.to_vec1::<f32>()?;
                println!("sinks head0..7 rust={:?} py={:?}", rust_s, py_s);
            }

            fn l2_linf(a: &Tensor, b_flat: &[Vec<f32>]) -> anyhow::Result<(f32, f64)> {
                let av = a.to_vec2::<f32>()?;
                let mut linf = 0f32; let mut l2 = 0f64;
                for i in 0..av.len() {
                    for j in 0..av[0].len() {
                        let d = av[i][j] - b_flat[i][j];
                        if d.abs() > linf { linf = d.abs(); }
                        l2 += (d as f64) * (d as f64);
                    }
                }
                Ok((linf, l2.sqrt()))
            }
            let (linf_qp, l2_qp) = l2_linf(&q_pre, &py_q_pre)?;
            let (linf_kp, l2_kp) = l2_linf(&k_pre, &py_k_pre)?;
            let (linf_vp, l2_vp) = l2_linf(&v_pre, &py_v_pre)?;
            let (linf_qr, l2_qr) = l2_linf(&q_rope, &py_q_rope)?;
            let (linf_kr, l2_kr) = l2_linf(&k_rope, &py_k_rope)?;
            println!("compare-python-l0: q_pre  Linf={:.6e} L2={:.6e}", linf_qp, l2_qp);
            println!("compare-python-l0: k_pre  Linf={:.6e} L2={:.6e}", linf_kp, l2_kp);
            println!("compare-python-l0: v_pre  Linf={:.6e} L2={:.6e}", linf_vp, l2_vp);
            println!("compare-python-l0: q_rope Linf={:.6e} L2={:.6e}", linf_qr, l2_qr);
            println!("compare-python-l0: k_rope Linf={:.6e} L2={:.6e}", linf_kr, l2_kr);
            // And show python l0.pre_attn_norm head0 first8 for context
            if let Some(pre_js) = vjson.get("l0_pre_norm_first8") {
                let arr = pre_js.as_array().unwrap();
                let py_pre: Vec<f32> = arr.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
                println!("py l0.pre_attn_norm.last.first8={:?}", py_pre);
            }

            // Compare attention output last token pre/post o_proj to pinpoint fault
            let (y_pre, y_post) = model.debug_l0_attn_last_token(&t, 0)?;
            let y_pre_r = y_pre.to_vec1::<f32>()?;
            let y_post_r = y_post.to_vec1::<f32>()?;

            let py_attn_pre = vjson["attn_pre_last"].as_array().unwrap()
                .iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<f32>>();
            let py_attn_post = vjson["attn_post_last"].as_array().unwrap()
                .iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<f32>>();
            let mut linf_a = 0f32; let mut l2_a = 0f64;
            for (a, b) in y_pre_r.iter().zip(py_attn_pre.iter()) { let d = a - b; if d.abs() > linf_a { linf_a = d.abs(); } l2_a += (d as f64)*(d as f64); }
            let mut linf_b = 0f32; let mut l2_b = 0f64;
            for (a, b) in y_post_r.iter().zip(py_attn_post.iter()) { let d = a - b; if d.abs() > linf_b { linf_b = d.abs(); } l2_b += (d as f64)*(d as f64); }
            println!("compare-python-l0: attn_pre  Linf={:.6e} L2={:.6e}", linf_a, l2_a.sqrt());
            println!("compare-python-l0: attn_post Linf={:.6e} L2={:.6e}", linf_b, l2_b.sqrt());

            // Optional: deep compare logits/weights rows for last token, head 0/1
            if args.compare_python_l0_attn {
                let (logits_last, weights_last) = model.debug_l0_attn_last_token_logits_weights(&t, 0, 2)?; // (h<=2, klen)
                let logits_r = logits_last.to_vec2::<f32>()?;
                let weights_r = weights_last.to_vec2::<f32>()?;

                fn vec_from_json(opt: Option<&serde_json::Value>) -> Option<Vec<f32>> {
                    opt.and_then(|v| v.as_array()).map(|arr| arr.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect::<Vec<f32>>())
                }

                if let Some(py_logits_h0) = vec_from_json(vjson.get("attn_logits_last_h0")) {
                    let mut linf = 0f32; let mut l2 = 0f64;
                    for (a, b) in logits_r[0].iter().zip(py_logits_h0.iter()) { let d = a - b; if d.abs() > linf { linf = d.abs(); } l2 += (d as f64)*(d as f64); }
                    println!("compare-python-l0: logits_last h0 Linf={:.6e} L2={:.6e}", linf, l2.sqrt());
                    let first8_r: Vec<f32> = logits_r[0].iter().take(8).copied().collect();
                    let first8_p: Vec<f32> = py_logits_h0.iter().take(8).copied().collect();
                    println!("  logits h0 first8 rust={:?} py={:?}", first8_r, first8_p);
                }
                if let Some(py_scores_h0) = vec_from_json(vjson.get("attn_scores_last_h0")) {
                    let mut linf = 0f32; let mut l2 = 0f64;
                    for (a, b) in weights_r[0].iter().zip(py_scores_h0.iter()) { let d = a - b; if d.abs() > linf { linf = d.abs(); } l2 += (d as f64)*(d as f64); }
                    println!("compare-python-l0: scores_last h0 Linf={:.6e} L2={:.6e}", linf, l2.sqrt());
                    let first8_r: Vec<f32> = weights_r[0].iter().take(8).copied().collect();
                    let first8_p: Vec<f32> = py_scores_h0.iter().take(8).copied().collect();
                    println!("  scores h0 first8 rust={:?} py={:?}", first8_r, first8_p);
                }
                if logits_r.len() > 1 {
                    if let Some(py_logits_h1) = vec_from_json(vjson.get("attn_logits_last_h1")) {
                        if let Some(py_logits_h1) = Some(py_logits_h1) {
                            let mut linf = 0f32; let mut l2 = 0f64;
                            for (a, b) in logits_r[1].iter().zip(py_logits_h1.iter()) { let d = a - b; if d.abs() > linf { linf = d.abs(); } l2 += (d as f64)*(d as f64); }
                            println!("compare-python-l0: logits_last h1 Linf={:.6e} L2={:.6e}", linf, l2.sqrt());
                        }
                    }
                    if let Some(py_scores_h1) = vec_from_json(vjson.get("attn_scores_last_h1")) {
                        if let Some(py_scores_h1) = Some(py_scores_h1) {
                            let mut linf = 0f32; let mut l2 = 0f64;
                            for (a, b) in weights_r[1].iter().zip(py_scores_h1.iter()) { let d = a - b; if d.abs() > linf { linf = d.abs(); } l2 += (d as f64)*(d as f64); }
                            println!("compare-python-l0: scores_last h1 Linf={:.6e} L2={:.6e}", linf, l2.sqrt());
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
