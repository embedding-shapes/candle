use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;
use std::collections::HashMap;

fn bf16(shape: (usize, usize), dev: &Device) -> Tensor {
    Tensor::zeros(shape, DType::BF16, dev).unwrap()
}

fn u8_zeros(shape: (usize, usize, usize), dev: &Device) -> Tensor {
    Tensor::zeros(shape, DType::U8, dev).unwrap()
}

#[test]
fn t20_missing_mxfp4_pair_reports_clear_error() -> Result<()> {
    let dev = Device::Cpu;
    // Small config; shapes not used beyond names.
    let cfg = GptOssConfig {
        vocab_size: 16,
        hidden_size: 16,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: Some(8),
        num_local_experts: 2,
        num_experts_per_tok: 1,
        intermediate_size: 8,
        max_position_embeddings: 1024,
        rope_theta: Some(15000.0),
        rope_scaling: None,
        layer_types: vec![candle_transformers::models::gpt_oss::GptOssLayerType::FullAttention],
        sliding_window: None,
        rms_norm_eps: Some(1e-5),
        swiglu_limit: Some(7.0),
    };

    let mut tmap: HashMap<String, Tensor> = HashMap::new();
    // Essentials
    tmap.insert(
        "model.embed_tokens.weight".into(),
        bf16((cfg.vocab_size, cfg.hidden_size), &dev),
    );
    tmap.insert(
        "model.norm.weight".into(),
        bf16((1, cfg.hidden_size), &dev).reshape(cfg.hidden_size)?,
    );
    let l0 = 0usize;
    let q_out = cfg.num_attention_heads * cfg.head_dim.unwrap();
    let kv_out = cfg.num_key_value_heads * cfg.head_dim.unwrap();
    tmap.insert(
        format!("model.layers.{l0}.self_attn.q_proj.weight"),
        bf16((q_out, cfg.hidden_size), &dev),
    );
    tmap.insert(
        format!("model.layers.{l0}.self_attn.k_proj.weight"),
        bf16((kv_out, cfg.hidden_size), &dev),
    );
    tmap.insert(
        format!("model.layers.{l0}.self_attn.v_proj.weight"),
        bf16((kv_out, cfg.hidden_size), &dev),
    );
    tmap.insert(
        format!("model.layers.{l0}.self_attn.o_proj.weight"),
        bf16((cfg.hidden_size, q_out), &dev),
    );
    tmap.insert(
        format!("model.layers.{l0}.self_attn.sinks"),
        bf16((1, cfg.num_attention_heads), &dev).reshape(cfg.num_attention_heads)?,
    );
    tmap.insert(
        format!("model.layers.{l0}.mlp.router.weight"),
        bf16((cfg.num_local_experts, cfg.hidden_size), &dev),
    );

    // Experts: provide only blocks, omit scales to trigger the graceful error.
    let nblocks_gate = cfg.hidden_size / 32; // 0 for small hidden; ensure >=1
    let nblocks_down = cfg.intermediate_size / 32;
    let nbg = nblocks_gate.max(1);
    let nbd = nblocks_down.max(1);
    for e in 0..cfg.num_local_experts {
        tmap.insert(
            format!("model.layers.{l0}.mlp.experts.{e}.gate_up_proj.blocks"),
            u8_zeros((2 * cfg.intermediate_size, nbg, 16), &dev),
        );
        // Intentionally do not insert gate_up_proj.scales
        tmap.insert(
            format!("model.layers.{l0}.mlp.experts.{e}.down_proj.blocks"),
            u8_zeros((cfg.hidden_size, nbd, 16), &dev),
        );
        // No down_proj.scales either
    }

    tmap.insert(
        "lm_head.weight".into(),
        bf16((cfg.vocab_size, cfg.hidden_size), &dev),
    );

    let vb = candle_nn::VarBuilder::from_tensors(tmap, DType::BF16, &dev);
    let err = GptOssModel::load(vb, &cfg).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("gate_up_proj") && msg.contains("missing") && msg.contains("scales"),
        "unexpected error message: {msg}"
    );
    Ok(())
}
