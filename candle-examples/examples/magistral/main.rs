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
use candle_examples::imagenet::load_image_with_std_mean;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral3::{
    config::Mistral3Config,
    model::{Model as Mistral3, Mistral3Cache},
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tekken::Tekkenizer;

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
    sys_start: u32,
    sys_end: u32,
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
    // Minimal instruct-style template using tekken special tokens.
    // <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{[IMG]?}{user}[/INST]
    let mut ids: Vec<u32> = Vec::new();
    ids.push(sp.bos);

    // System prompt section
    ids.push(sp.sys_start);
    let mut sys_ids = tok
        .encode(system_prompt, false, false)
        .map_err(|e| E::msg(format!("tekken encode system: {e}")))?;
    ids.append(&mut sys_ids);
    ids.push(sp.sys_end);

    // User instruction with optional [IMG]
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

fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} top-p: {:?} seed: {}",
        args.temperature.unwrap_or(0.0),
        args.top_p,
        args.seed
    );

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
    let tokenizer = Tekkenizer::from_file(&tekken_file).map_err(E::msg)?;
    let sp = load_special_ids(&tekken_file)?;

    // Load system prompt
    let system_prompt = fs::read_to_string(&system_prompt_file).context("read SYSTEM_PROMPT.txt")?;

    // Device and dtype
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.supports_bf16() { DType::BF16 } else { DType::F32 };

    // Config and model
    let config: Mistral3Config =
        serde_json::from_slice(&fs::read(&config_file)?).context("parse config.json")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
    let mut model = Mistral3::new(&config, vb)?;

    // Image preprocessing
    let image = load_image_with_std_mean(&args.image, IMAGE_RES, &PIXTRAL_MEAN, &PIXTRAL_STD)?
        .to_device(&device)?
        .unsqueeze(0)?; // (1, C, H, W)
    let (_b, _c, h, w) = image.dims4()?;
    let image_sizes: Vec<(u32, u32)> = vec![(h as u32, w as u32)];
    println!("loaded image with shape {:?}, sizes {:?}", image.dims(), image_sizes);

    // Build input ids
    let input_ids_vec = build_input_ids(&tokenizer, &sp, &system_prompt, &args.prompt, true)?;
    if args.print_token_ids {
        println!("input token ids ({}): {:?}", input_ids_vec.len(), input_ids_vec);
    }
    let input_ids = Tensor::new(input_ids_vec.as_slice(), &device)?.unsqueeze(0)?; // (1, S)

    // Generation setup
    let mut cache = Mistral3Cache::default();
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);
    let mut tokens: Vec<u32> = input_ids_vec.clone();
    let initial_len = tokens.len();
    let mut generated = 0usize;
    let start_gen = std::time::Instant::now();

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

        let logits = model.forward(&inp, pixel_opt, sizes_opt, &mut cache, index_pos, &config.vision_feature_layer)?;
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

    Ok(())
}
