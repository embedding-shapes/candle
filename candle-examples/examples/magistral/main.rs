#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context as _, Error as E, Result};
use clap::Parser;

use candle::{DType, IndexOp, Tensor};
mod preproc;
use preproc::{compute_pixtral_resize_dims, load_image_pixtral};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral3::{
    config::Mistral3Config,
    model::{Model as Mistral3, Mistral3Cache},
};
use serde_json::json;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tekken::Tekkenizer;
use std::time::Instant;

// Constants/configurable values placed below imports
const DEFAULT_MODEL_ID: &str = "mistralai/magistral-small-2509";
const DEFAULT_REVISION: &str = "main";
const PIXTRAL_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const PIXTRAL_STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];
const IMAGE_RES: usize = 1540;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than GPU
    #[arg(long)]
    cpu: bool,

    /// Model id on the hub
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// Snapshot revision/branch
    #[arg(long, default_value = DEFAULT_REVISION)]
    revision: String,

    /// Path to image file
    #[arg(long)]
    image: String,

    /// Prompt text shown to the model
    #[arg(long, default_value = "Summarize this image comprehensively.")]
    prompt: String,

    /// Temperature for sampling
    #[arg(long)]
    temperature: Option<f64>,

    /// Top-p for nucleus sampling
    #[arg(long)]
    top_p: Option<f64>,

    /// RNG seed
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Max new tokens to generate
    #[arg(long, short = 'n', default_value_t = 512)]
    sample_len: usize,

    /// Print token ids for diagnostics
    #[arg(long)]
    print_token_ids: bool,
}

#[derive(Debug, Clone)]
struct SpecialIds {
    bos: u32,
    eos: u32,
    inst: u32,
    end_inst: u32,
    img: u32,
    img_end: u32,
    sys_start: u32,
    sys_end: u32,
    think: u32,
    end_think: u32,
}

fn load_special_ids(tekken_json: &PathBuf) -> Result<SpecialIds> {
    let data = fs::read(tekken_json)?;
    let v: serde_json::Value = serde_json::from_slice(&data)?;
    let arr = v
        .get("special_tokens")
        .and_then(|x| x.as_array())
        .context("missing special_tokens in tekken.json")?;
    let mut map: HashMap<String, u32> = HashMap::new();
    for tok in arr {
        let s = tok.get("token_str").and_then(|x| x.as_str());
        let r = tok.get("rank").and_then(|x| x.as_u64());
        if let (Some(s), Some(r)) = (s, r) {
            map.insert(s.to_string(), r as u32);
        }
    }
    let fetch = |k: &str| -> Result<u32> {
        map.get(k)
            .copied()
            .with_context(|| format!("missing {k} in tekken special_tokens"))
    };
    Ok(SpecialIds {
        bos: fetch("<s>")?,
        eos: fetch("</s>")?,
        inst: fetch("[INST]")?,
        end_inst: fetch("[/INST]")?,
        img: fetch("[IMG]")?,
        img_end: fetch("[IMG_END]")?,
        sys_start: fetch("[SYSTEM_PROMPT]")?,
        sys_end: fetch("[/SYSTEM_PROMPT]")?,
        think: fetch("[THINK]")?,
        end_think: fetch("[/THINK]")?,
    })
}

fn build_input_ids(
    tok: &Tekkenizer,
    sp: &SpecialIds,
    system_prompt: &str,
    user_prompt: &str,
    img_placeholders: usize,
) -> Result<Vec<u32>> {
    // Minimal instruct-style template using tekken special tokens.
    // <s>[SYSTEM_PROMPT]{pre}[THINK]{think}[/THINK]{post}[/SYSTEM_PROMPT][INST]{[IMG]*}[IMG_END]{user}[/INST]
    let mut ids: Vec<u32> = Vec::new();
    ids.push(sp.bos);

    // System prompt section
    ids.push(sp.sys_start);
    // Split the SYSTEM_PROMPT.txt into pre/think/post like the HF template
    let (pre, think, post) = {
        let start = system_prompt.find("[THINK]");
        let end = system_prompt.find("[/THINK]");
        match (start, end) {
            (Some(i0), Some(i1)) if i1 >= i0 => {
                let pre = &system_prompt[..i0];
                let think = &system_prompt[i0 + "[THINK]".len()..i1];
                let post = &system_prompt[i1 + "[/THINK]".len()..];
                (pre, Some(think), post)
            }
            _ => (system_prompt, None, ""),
        }
    };
    if !pre.is_empty() {
        let mut pre_ids = tok
            .encode(pre, false, false)
            .map_err(|e| E::msg(format!("tekken encode sys-pre: {e}")))?;
        ids.append(&mut pre_ids);
    }
    if let Some(t) = think {
        ids.push(sp.think);
        if !t.is_empty() {
            let mut t_ids = tok
                .encode(t, false, false)
                .map_err(|e| E::msg(format!("tekken encode sys-think: {e}")))?;
            ids.append(&mut t_ids);
        }
        ids.push(sp.end_think);
    }
    if !post.is_empty() {
        let mut post_ids = tok
            .encode(post, false, false)
            .map_err(|e| E::msg(format!("tekken encode sys-post: {e}")))?;
        ids.append(&mut post_ids);
    }
    ids.push(sp.sys_end);

    // User instruction with optional [IMG] placeholders
    ids.push(sp.inst);
    for _ in 0..img_placeholders {
        ids.push(sp.img);
    }
    // HF template closes the image block before the user text
    ids.push(sp.img_end);
    let mut user_ids = tok
        .encode(user_prompt, false, false)
        .map_err(|e| E::msg(format!("tekken encode user: {e}")))?;
    ids.append(&mut user_ids);
    ids.push(sp.end_inst);
    Ok(ids)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    // Print will be updated after defaults are applied below.

    // Resolve snapshot and required files
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));
    let start = std::time::Instant::now();
    let config_file = repo.get("config.json")?;
    let system_prompt_file = repo.get("SYSTEM_PROMPT.txt")?;
    let tekken_file = repo.get("tekken.json")?;
    let weight_files = candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;
    println!("retrieved metadata and shards in {:?}", start.elapsed());

    // Load tokenizer and special ids
    let t_tok = Instant::now();
    let tokenizer = Tekkenizer::from_file(&tekken_file).map_err(E::msg)?;
    let sp = load_special_ids(&tekken_file)?;
    println!("timing: tokenizer+special_ids: {:?}", t_tok.elapsed());

    // Load system prompt
    let t_sys = Instant::now();
    let system_prompt = fs::read_to_string(&system_prompt_file).context("read SYSTEM_PROMPT.txt")?;
    println!("timing: read SYSTEM_PROMPT.txt: {:?}", t_sys.elapsed());

    // Device and dtype
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F32 };
    // Prefer BF16 reduced-precision GEMM on CUDA for faster matmuls.
    #[cfg(feature = "cuda")]
    {
        candle::cuda::set_gemm_reduced_precision_bf16(true);
        eprintln!(
            "magistral: cuda gemm_bf16_fast now={}",
            candle::cuda::gemm_reduced_precision_bf16()
        );
    }

    // Config and model
    let t_cfg = Instant::now();
    let mut config: Mistral3Config =
        serde_json::from_slice(&fs::read(&config_file)?).context("parse config.json")?;
    println!("timing: parse config: {:?}", t_cfg.elapsed());
    // Prefer Flash-Attention when compiled with the feature.
    // This mirrors HF defaults and usually significantly speeds up attention on CUDA.
    let use_flash = cfg!(feature = "flash-attn");
    config.text_config.inner.use_flash_attn = use_flash;
    if use_flash {
        eprintln!("magistral: flash-attn enabled for text model");
    } else {
        eprintln!("magistral: flash-attn not compiled; using standard attention");
    }
    let t_vb = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
    println!("timing: mmap weights: {:?}", t_vb.elapsed());
    let t_model = Instant::now();
    let mut model = Mistral3::new(&config, vb)?;
    println!("timing: build model: {:?}", t_model.elapsed());

    // Image preprocessing (Pixtral-style):
    // - Resize preserving aspect ratio so the longest edge is 1540
    // - Round both H and W down to a multiple of the patch size (14)
    // - Normalize using Pixtral mean/std
    let patch = {
        // Prefer reading from config to avoid drift
        let p = config.vision_config.inner.patch_size as usize;
        if p == 0 { 14 } else { p }
    };
    let s = config.spatial_merge_size.max(1);
    // Resize rounding to multiples of the downsample ratio (patch * s),
    // matching HF/HuggingFace processors that operate on the merged-patch grid.
    let t_resize = Instant::now();
    let (new_h, new_w) = compute_pixtral_resize_dims(&args.image, IMAGE_RES, patch * s)?;
    println!("timing: compute_resize_dims: {:?}", t_resize.elapsed());
    let t_img = Instant::now();
    let img_loaded = load_image_pixtral(&args.image, new_h, new_w, &PIXTRAL_MEAN, &PIXTRAL_STD)?;
    println!("timing: load_image+bicubic+normalize (cpu f32): {:?}", t_img.elapsed());
    let t_cast = Instant::now();
    let img_cast = img_loaded.to_dtype(dtype)?;
    println!("timing: cast_to_model_dtype: {:?}", t_cast.elapsed());
    let t_h2d = Instant::now();
    let image = img_cast
        .to_device(&device)?
        .unsqueeze(0)?; // (1, C, H, W)
    println!("timing: host_to_device+unsqueeze: {:?}", t_h2d.elapsed());
    let (_b, _c, h, w) = image.dims4()?;
    let image_sizes: Vec<(u32, u32)> = vec![(h as u32, w as u32)];
    let grid_h = h / patch;
    let grid_w = w / patch;
    println!(
        "loaded image with shape {:?}, sizes {:?} (patch={} grid={}x{})",
        image.dims(), image_sizes, patch, grid_h, grid_w
    );

    // Compute the number of image placeholders needed after patch merging
    let num_image_tokens = (grid_h / s) * (grid_w / s);

    // Build input ids
    let input_ids_vec = build_input_ids(
        &tokenizer,
        &sp,
        &system_prompt,
        &args.prompt,
        num_image_tokens,
    )?;
    if args.print_token_ids {
        println!("input token ids ({}): {:?}", input_ids_vec.len(), input_ids_vec);
    }
    let input_ids = Tensor::new(input_ids_vec.as_slice(), &device)?.unsqueeze(0)?; // (1, S)

    // Unified debug JSON block to match Python output
    let (orig_w, orig_h) = image::image_dimensions(&args.image).map_err(candle::Error::wrap)?;
    let file_bytes = std::fs::metadata(&args.image)?.len() as u64;
    let s_u = s as usize;
    let eff_grid_h = grid_h / s_u;
    let eff_grid_w = grid_w / s_u;

    // Pixel stats on normalized image (float32 copy for stats)
    let img_cpu = image.to_dtype(DType::F32)?.to_device(&candle::Device::Cpu)?; // (1,3,H,W)
    let (_b1, _c1, hh, ww) = img_cpu.dims4()?;
    let mut per_channel = Vec::new();
    for cix in 0..3usize {
        let t = img_cpu.i((0, cix, .., ..))?;
        let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut sum = 0f64;
        let mut sumsq = 0f64;
        for &x in &v {
            if x < min_v { min_v = x; }
            if x > max_v { max_v = x; }
            let xd = x as f64;
            sum += xd;
            sumsq += xd * xd;
        }
        let n = v.len() as f64;
        let mean = if n > 0.0 { sum / n } else { 0.0 };
        let var = if n > 0.0 { (sumsq / n) - (mean * mean) } else { 0.0 };
        let std = if var > 0.0 { var.sqrt() } else { 0.0 };
        per_channel.push(json!({
            "min": min_v,
            "max": max_v,
            "mean": mean,
            "std": std,
            "sum": sum,
            "sum_sq": sumsq,
        }));
    }
    // Global stats
    let t_stats = Instant::now();
    let g: Vec<f32> = img_cpu.flatten_all()?.to_vec1()?;
    let mut gmin = f32::INFINITY;
    let mut gmax = f32::NEG_INFINITY;
    let mut gsum = 0f64;
    let mut gsum2 = 0f64;
    for &x in &g {
        if x < gmin { gmin = x; }
        if x > gmax { gmax = x; }
        let xd = x as f64;
        gsum += xd;
        gsum2 += xd * xd;
    }
    let gn = g.len() as f64;
    let gmean = if gn > 0.0 { gsum / gn } else { 0.0 };
    let gvar = if gn > 0.0 { (gsum2 / gn) - (gmean * gmean) } else { 0.0 };
    let gstd = if gvar > 0.0 { gvar.sqrt() } else { 0.0 };
    let first_vals_ch0_row0: Vec<f32> = img_cpu
        .i((0, 0, 0, 0..8))?
        .to_vec1::<f32>()
        .unwrap_or_else(|_| vec![]);
    println!("timing: pixel_stats_cpu: {:?}", t_stats.elapsed());

    // Derive dtype strings for debug
    let vision_dtype_str = match image.dtype() {
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::F16 => "f16",
        _ => "unknown",
    };
    let text_dtype_str = if dtype == DType::BF16 { "bf16" } else { "f32" };

    let debug = json!({
        "image": {
            "path": args.image,
            "orig_w": orig_w as i64,
            "orig_h": orig_h as i64,
            "file_bytes": file_bytes as i64,
        },
        "preproc": {
            "impl": "candle_preproc",
            "target_max_side": IMAGE_RES as i64,
            "resample": "bicubic",
            "patch_size": patch as i64,
            "spatial_merge_size": s as i64,
            "downsample_ratio": (patch * s) as i64,
            "resized_h": h as i64,
            "resized_w": w as i64,
            "grid_h": grid_h as i64,
            "grid_w": grid_w as i64,
            "eff_grid_h": eff_grid_h as i64,
            "eff_grid_w": eff_grid_w as i64,
            "placeholders": num_image_tokens as i64,
        },
        "pixels": {
            "dtype": vision_dtype_str,
            "shape": [1, 3, hh as i64, ww as i64],
            "per_channel": per_channel,
            "global": {"min": gmin, "max": gmax, "mean": gmean, "std": gstd, "sum": gsum, "sum_sq": gsum2},
            "first_values_ch0_row0": first_vals_ch0_row0,
        },
        "tokenizer": {
            "input_ids_len": input_ids_vec.len() as i64,
            "image_token_id": model.image_token_id as i64,
            "eos_token_id": sp.eos as i64,
            "pad_token_id": serde_json::Value::Null,
            "num_image_placeholders": num_image_tokens as i64,
        },
        "model": {
            "id": args.model_id,
            "device": if args.cpu { "cpu" } else { "cuda" },
            "vision_dtype": vision_dtype_str,
            "text_dtype": text_dtype_str,
        }
    });
    println!("=== magistral_debug ===\n{}", serde_json::to_string_pretty(&debug)?);

    // Generation setup
    let mut cache = Mistral3Cache::default();
    // Default to sampling like the Python reference unless explicitly overridden.
    let default_temp = Some(0.7);
    let default_top_p = Some(0.95);
    let use_temp = args.temperature.or(default_temp);
    let use_top_p = args.top_p.or(default_top_p);
    println!(
        "temp: {} top-p: {:?} seed: {}",
        use_temp.map(|v| format!("{v:.2}")).unwrap_or_else(|| "None".to_string()),
        use_top_p,
        args.seed
    );
    let mut logits_processor = LogitsProcessor::new(args.seed, use_temp, use_top_p);
    let mut tokens: Vec<u32> = input_ids_vec.clone();
    let initial_len = tokens.len();
    let mut generated = 0usize;
    let start_gen = std::time::Instant::now();
    println!("IMG placeholders: {} (merged grid {}x{})", num_image_tokens, grid_h / s, grid_w / s);

    let mut t_decode_total = std::time::Duration::ZERO;
    let mut t_prefill = std::time::Duration::ZERO;
    for step in 0..args.sample_len {
        let (inp, index_pos, pixel_opt, sizes_opt) = if step == 0 {
            (input_ids.clone(), 0usize, Some(&image), Some(image_sizes.as_slice()))
        } else {
            let last = *tokens.last().unwrap();
            let pos = initial_len + step - 1;
            (
                Tensor::new(&[last], &device)?.unsqueeze(0)?,
                pos,
                None,
                None,
            )
        };
        let t_fw = Instant::now();
        let logits = model.forward(&inp, pixel_opt, sizes_opt, &mut cache, index_pos, &config.vision_feature_layer)?;
        let dt_fw = t_fw.elapsed();
        if step == 0 { t_prefill = dt_fw; } else { t_decode_total += dt_fw; }
        // logits shape: [B, vocab] or [B, 1, vocab]
        let logits = if logits.dims().len() == 3 {
            logits.i((.., logits.dim(1)? - 1, ..))?
        } else {
            logits
        };
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated += 1;
        if next_token == sp.eos {
            break;
        }
    }

    // Decode only generated continuation
    let new_tokens = &tokens[initial_len..];
    let decoded = tokenizer
        .decode(new_tokens, tekken::SpecialTokenPolicy::Ignore)
        .map_err(|e| E::msg(format!("tekken decode: {e}")))?;
    let dt = start_gen.elapsed();
    println!("{decoded}");
    println!(
        "\n{} tokens generated ({:.2} token/s)",
        generated,
        generated as f64 / dt.as_secs_f64()
    );

    // Emit a final timing JSON summary to help pinpoint bottlenecks.
    #[allow(unused_mut)]
    let mut gemm_bf16_fast = None;
    #[cfg(feature = "cuda")]
    {
        gemm_bf16_fast = Some(candle::cuda::gemm_reduced_precision_bf16());
    }
    let timing_json = json!({
        "timing": {
            "tokenizer_ms": null, // printed above
            "system_prompt_ms": null, // printed above
            "parse_config_ms": null, // printed above
            "mmap_weights_ms": null, // printed above
            "build_model_ms": null, // printed above
            "preproc": {
                "compute_resize_ms": null,
                "load_image_ms": null,
                "cast_ms": null,
                "h2d_ms": null,
                "pixel_stats_ms": null
            },
            "prefill_ms": t_prefill.as_millis(),
            "decode_total_ms": t_decode_total.as_millis(),
            "decode_tokens": generated.saturating_sub(1)
        },
        "cuda": {
            "gemm_bf16_fast": gemm_bf16_fast,
        }
    });
    println!("=== magistral_timing ===\n{}", serde_json::to_string_pretty(&timing_json)?);

    Ok(())
}
