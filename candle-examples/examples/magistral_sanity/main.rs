#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context as ECtx, Error as E, Result};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::Deserialize;

use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral3::model::Model as Mistral3;
use candle_transformers::models::mistral3::{config::Mistral3Config, model::Mistral3Cache};
use tekken::Tekkenizer;

// Configurable constants at top
const DEFAULT_MODEL_ID: &str = "mistralai/magistral-small-2509";
const DEFAULT_REVISION: &str = "main";
const PIXTRAL_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const PIXTRAL_STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];
const IMAGE_RES: usize = 1540;

// Minimal, self-contained Pixtral preproc helpers to mirror the main example.
fn compute_resized_dims_from_wh(
    orig_w: u32,
    orig_h: u32,
    max_side: usize,
    divisor: usize,
) -> (usize, usize) {
    let divisor = divisor.max(1);
    let max_side = max_side.max(divisor);
    let (wf, hf) = (orig_w as f32, orig_h as f32);
    let (nw, nh) = if orig_w >= orig_h {
        let s = max_side as f32 / wf;
        (wf * s, hf * s)
    } else {
        let s = max_side as f32 / hf;
        (wf * s, hf * s)
    };
    let round_mul = |x: f32, d: usize| -> usize {
        let m = (x / d as f32).round();
        let m = if m < 1.0 { 1.0 } else { m };
        (m as usize) * d
    };
    (round_mul(nh, divisor), round_mul(nw, divisor))
}

fn load_image_pixtral(
    path: &str,
    h: usize,
    w: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Result<Tensor> {
    use candle::{Device, Tensor};
    if h == 0 || w == 0 {
        return Err(E::msg("invalid target size"));
    }
    let img = image::ImageReader::open(path)?
        .decode()
        .map_err(candle::Error::wrap)?
        .resize(w as u32, h as u32, image::imageops::FilterType::CatmullRom)
        .to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (h, w, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean = Tensor::new(mean, &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(std, &Device::Cpu)?.reshape((3, 1, 1))?;
    Ok(((data.to_dtype(DType::F32)? / 255.)?)
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?)
}

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
    #[arg(long, default_value = "./test-image.jpg")]
    image: String,

    /// Prompt text
    #[arg(long, default_value = "Summarize this slide image comprehensively.")]
    prompt: String,

    /// Temperature
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-p
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,

    /// RNG seed
    #[arg(long, default_value_t = 12345)]
    seed: u64,

    /// New tokens to generate for comparison
    #[arg(long, default_value_t = 16)]
    new_tokens: usize,
}

#[derive(Debug, Clone)]
struct SpecialIds {
    bos: u32,
    eos: u32,
    inst: u32,
    end_inst: u32,
    img: u32,
    sys_start: u32,
    sys_end: u32,
}

fn load_special_ids(tekken_json: &PathBuf) -> Result<SpecialIds> {
    let data = std::fs::read(tekken_json)?;
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
        sys_start: fetch("[SYSTEM_PROMPT]")?,
        sys_end: fetch("[/SYSTEM_PROMPT]")?,
    })
}

fn build_input_ids(
    tok: &Tekkenizer,
    sp: &SpecialIds,
    system_prompt: &str,
    user_prompt: &str,
    include_image: bool,
) -> Result<Vec<u32>> {
    let mut ids: Vec<u32> = Vec::new();
    ids.push(sp.bos);
    ids.push(sp.sys_start);
    let mut sys_ids = tok
        .encode(system_prompt, false, false)
        .map_err(|e| E::msg(format!("tekken encode system: {e}")))?;
    ids.append(&mut sys_ids);
    ids.push(sp.sys_end);

    ids.push(sp.inst);
    if include_image {
        ids.push(sp.img);
    }
    let mut user_ids = tok
        .encode(user_prompt, false, false)
        .map_err(|e| E::msg(format!("tekken encode user: {e}")))?;
    ids.append(&mut user_ids);
    ids.push(sp.end_inst);
    Ok(ids)
}

#[derive(Debug, Deserialize)]
struct PyMeta {
    input_ids_len: usize,
    image_h: usize,
    image_w: usize,
    patch_size: usize,
    grid_h: usize,
    grid_w: usize,
    text: String,
}

fn run_python_meta(args: &Args) -> Result<PyMeta> {
    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("infer_with_magistral.py")
        .arg("--image")
        .arg(&args.image)
        .arg("--model-id")
        .arg(&args.model_id)
        .arg("--max-new-tokens")
        .arg(format!("{}", args.new_tokens))
        .arg("--temperature")
        .arg(format!("{}", args.temperature))
        .arg("--top-p")
        .arg(format!("{}", args.top_p))
        .arg("--prompt")
        .arg(&args.prompt)
        .arg("--meta-json");
    let output = cmd.output().context("failed to spawn uv run")?;
    if !output.status.success() {
        let mut s = String::new();
        s.push_str("python stderr: \n");
        s.push_str(&String::from_utf8_lossy(&output.stderr));
        s.push_str("\npython stdout: \n");
        s.push_str(&String::from_utf8_lossy(&output.stdout));
        return Err(E::msg(format!(
            "python run failed: {}\n{}",
            output.status, s
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim();
    let meta: PyMeta = serde_json::from_str(line).context("invalid JSON from python meta")?;
    Ok(meta)
}

fn jaccard_words(a: &str, b: &str) -> f64 {
    let to_set = |s: &str| -> std::collections::BTreeSet<String> {
        s.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() >= 4)
            .map(|w| w.to_lowercase())
            .collect()
    };
    let sa = to_set(a);
    let sb = to_set(b);
    if sa.is_empty() && sb.is_empty() {
        return 1.0;
    }
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Resolve snapshot and artifacts
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        args.model_id.clone(),
        RepoType::Model,
        args.revision.clone(),
    ));
    let tekken_file = repo.get("tekken.json")?;
    let system_prompt_file = repo.get("SYSTEM_PROMPT.txt")?;
    let config_file = repo.get("config.json")?;
    let weight_files =
        candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?;

    // Tokenizer and special ids
    let tokenizer = Tekkenizer::from_file(&tekken_file).map_err(E::msg)?;
    let sp = load_special_ids(&tekken_file)?;
    let system_prompt =
        std::fs::read_to_string(&system_prompt_file).context("read SYSTEM_PROMPT.txt")?;

    // Build our input ids
    let input_ids_vec = build_input_ids(&tokenizer, &sp, &system_prompt, &args.prompt, true)?;
    println!("rust input_ids length: {}", input_ids_vec.len());
    let rust_img_count = input_ids_vec.iter().filter(|&&t| t == sp.img).count();
    println!("rust [IMG] count: {}", rust_img_count);

    // Python metadata
    let py = run_python_meta(&args)?;
    println!(
        "python input_ids length: {}, image: {}x{}, patch {} → grid {}x{}",
        py.input_ids_len, py.image_h, py.image_w, py.patch_size, py.grid_h, py.grid_w
    );

    // Compare tokenization length (allow minor deviation)
    let len_diff = (py.input_ids_len as isize - input_ids_vec.len() as isize).abs();
    if len_diff > 8 {
        println!(
            "WARNING: token length differs by {} (allowed small deviation)",
            len_diff
        );
    }
    assert!(
        rust_img_count == 1,
        "expected exactly one [IMG] token in rust"
    );

    // Device and model
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.supports_bf16() {
        DType::BF16
    } else {
        DType::F32
    };
    let config: Mistral3Config =
        serde_json::from_slice(&std::fs::read(&config_file)?).context("parse config.json")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
    let mut model = Mistral3::new(&config, vb)?;

    // Image preprocessing: use the same Pixtral pipeline as the main example
    let patch = {
        let p = config.vision_config.inner.patch_size as usize;
        if p == 0 {
            14
        } else {
            p
        }
    };
    let s = config.spatial_merge_size.max(1);
    let (ow, oh) = image::image_dimensions(&args.image).map_err(candle::Error::wrap)?;
    let (h_tgt, w_tgt) = compute_resized_dims_from_wh(ow, oh, IMAGE_RES, patch * s);
    let image = load_image_pixtral(&args.image, h_tgt, w_tgt, &PIXTRAL_MEAN, &PIXTRAL_STD)?
        .to_device(&device)?
        .unsqueeze(0)?; // (1, C, H, W)
    let (_b, _c, h, w) = image.dims4()?;
    let rust_grid_h = h / patch;
    let rust_grid_w = w / patch;
    println!(
        "rust image: {}x{}, patch {} → grid {}x{}",
        h, w, patch, rust_grid_h, rust_grid_w
    );

    assert_eq!(rust_grid_h, py.grid_h, "grid_h mismatch");
    assert_eq!(rust_grid_w, py.grid_w, "grid_w mismatch");

    // Prepare generation
    let input_ids = Tensor::new(input_ids_vec.as_slice(), &device)?.unsqueeze(0)?;
    let mut cache = Mistral3Cache::default();
    let mut logits_processor =
        LogitsProcessor::new(args.seed, Some(args.temperature), Some(args.top_p));
    let mut tokens: Vec<u32> = input_ids.to_vec2::<u32>()?.remove(0);
    let initial_len = tokens.len();

    let image_sizes: Vec<(u32, u32)> = vec![(h as u32, w as u32)];
    for step in 0..args.new_tokens {
        let (inp, index_pos, pixel_opt, sizes_opt) = if step == 0 {
            (
                input_ids.clone(),
                0usize,
                Some(&image),
                Some(image_sizes.as_slice()),
            )
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
        let logits = model.forward(
            &inp,
            pixel_opt,
            sizes_opt,
            &mut cache,
            index_pos,
            &config.vision_feature_layer,
        )?;
        let logits = if logits.dims().len() == 3 {
            logits.i((.., logits.dim(1)? - 1, ..))?
        } else {
            logits
        };
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        if next_token == sp.eos {
            break;
        }
    }

    let new_tokens = &tokens[initial_len..];
    let rust_text = tokenizer
        .decode(new_tokens, tekken::SpecialTokenPolicy::Ignore)
        .map_err(|e| E::msg(format!("tekken decode: {e}")))?;

    println!(
        "\npython text: {}\n---\nrust   text: {}\n",
        py.text, rust_text
    );
    let jac = jaccard_words(&py.text, &rust_text);
    println!("word Jaccard similarity (>=4 chars): {:.3}", jac);
    if jac < 0.10 {
        println!("WARNING: low similarity; sampling variance may be high");
    }

    Ok(())
}
