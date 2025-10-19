use anyhow::Result;
use candle::{DType, Device, Tensor};
use candle_transformers::models::gpt_oss::load_expert_linear_mxfp4_grouped;

const SNAPSHOT_DIR: &str = "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee";

fn p(s: &str) -> std::path::PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        std::env::var("HOME").map(std::path::PathBuf::from).unwrap().join(rest)
    } else {
        std::path::PathBuf::from(s)
    }
}

#[test]
fn t40_mxfp4_decode_stats_match_python_fixture() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    // Map just the first shard files; helper resolves all from index
    let files = candle_examples::hub_load_local_safetensors(p(SNAPSHOT_DIR), "model.safetensors.index.json")?;
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, DType::BF16, &dev)? };

    // Layer 0 gate_up_proj (grouped MXFP4), expert 0
    let mlp_vb = vb.pp("model.layers.0.mlp");
    let scales = mlp_vb.to_dtype(DType::U8).get((32, 5760, 90), "experts.gate_up_proj_scales")?;
    let out_dim = scales.dim(1)?;
    assert_eq!(out_dim, 5760);
    // gate_up has 2*intermediate out; but grouped loader expects in/out dims per projection
    let inter = out_dim; // fused (gate_up) outputs out_dim, where inter == out_dim/2 per branch
    let hidden = 2880usize;
    let lin = load_expert_linear_mxfp4_grouped(
        hidden, // in_dim per projection branch
        inter,  // out_dim of fused gate_up (we'll slice to first half for stats)
        true,
        mlp_vb.clone(),
        "experts.gate_up_proj",
        0,
        32,
    )?;
    let w = lin.weight().to_dtype(DType::F32)?; // (out, in)
    // Compare stats for first 8 rows of the weight on first 2880 columns to Python fixture
    let full = w.narrow(1, 0, hidden)?; // (out, 2880)
    let mut stats = Vec::new();
    for r in 0..8usize {
        let row = full.narrow(0, r, 1)?.squeeze(0)?; // (2880)
        let mean = row.mean_all()?.to_scalar::<f32>()?;
        let mean_vec = Tensor::from_vec(vec![mean; hidden], (hidden,), &dev)?;
        let centered = (&row - &mean_vec)?;
        let var = centered.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let std = var.sqrt();
        stats.push((mean, std));
    }
    // Load Python fixture
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures/mxfp4_stats_layer0_expert0_gate_up_proj.json");
    let f = std::fs::File::open(fixture_path)?;
    let v: serde_json::Value = serde_json::from_reader(f)?;
    let rows = v.get("rows").unwrap().as_array().unwrap();
    for (i, (m, s)) in stats.iter().enumerate() {
        let mr = rows[i].get("mean").unwrap().as_f64().unwrap() as f32;
        let sr = rows[i].get("std").unwrap().as_f64().unwrap() as f32;
        assert!((m - mr).abs() < 1e-4, "row {i} mean mismatch: got {m}, ref {mr}");
        assert!((s - sr).abs() < 5e-4, "row {i} std mismatch: got {s}, ref {sr}");
    }
    Ok(())
}
