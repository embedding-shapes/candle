use candle::{DType, Device, Result, Tensor};
use candle_transformers::models::gpt_oss::config::GptOssConfig;
use candle_transformers::models::gpt_oss::model::GptOssModel;
use std::collections::HashMap;

fn bf16_rand(shape: (usize, usize), dev: &Device) -> Tensor {
    // Simple deterministic pattern rather than RNG for reproducibility
    let (r, c) = shape;
    let mut v = vec![0f32; r * c];
    for i in 0..v.len() {
        v[i] = ((i % 13) as f32) * 0.01 - 0.06;
    }
    Tensor::from_vec(v, shape, dev).unwrap().to_dtype(DType::BF16).unwrap()
}

fn u8_pattern(shape: (usize, usize, usize), dev: &Device) -> Tensor {
    let (a, b, c) = shape;
    let mut v = vec![0u8; a * b * c];
    for i in 0..v.len() {
        v[i] = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    Tensor::from_vec(v, shape, dev).unwrap()
}

#[test]
fn t19_minimal_forward_shape_dtype() -> Result<()> {
    let dev = Device::Cpu;
    // Tiny synthetic config
    let cfg = GptOssConfig {
        vocab_size: 32,
        hidden_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: Some(8),
        num_local_experts: 3,
        num_experts_per_tok: 2,
        intermediate_size: 32,
        max_position_embeddings: 1024,
        rope_theta: Some(15000.0),
        rope_scaling: None,
        layer_types: vec![candle_transformers::models::gpt_oss::GptOssLayerType::FullAttention],
        sliding_window: None,
        rms_norm_eps: Some(1e-5),
    };

    // Prepare weight tensors for the loader
    let mut tmap: HashMap<String, Tensor> = HashMap::new();

    // Embedding and norm
    tmap.insert("model.embed_tokens.weight".into(), bf16_rand((cfg.vocab_size, cfg.hidden_size), &dev));
    tmap.insert("model.norm.weight".into(), bf16_rand((1, cfg.hidden_size), &dev).reshape(cfg.hidden_size)?);

    // Layer 0 attention weights
    let l0 = 0usize;
    let q_out = cfg.num_attention_heads * cfg.head_dim.unwrap();
    let kv_out = cfg.num_key_value_heads * cfg.head_dim.unwrap();
    // Pre/post norms
    tmap.insert(
        format!("model.layers.{l0}.input_layernorm.weight"),
        bf16_rand((1, cfg.hidden_size), &dev).reshape(cfg.hidden_size)?,
    );
    tmap.insert(
        format!("model.layers.{l0}.post_attention_layernorm.weight"),
        bf16_rand((1, cfg.hidden_size), &dev).reshape(cfg.hidden_size)?,
    );
    tmap.insert(format!("model.layers.{l0}.self_attn.q_proj.weight"), bf16_rand((q_out, cfg.hidden_size), &dev));
    tmap.insert(format!("model.layers.{l0}.self_attn.k_proj.weight"), bf16_rand((kv_out, cfg.hidden_size), &dev));
    tmap.insert(format!("model.layers.{l0}.self_attn.v_proj.weight"), bf16_rand((kv_out, cfg.hidden_size), &dev));
    tmap.insert(format!("model.layers.{l0}.self_attn.o_proj.weight"), bf16_rand((cfg.hidden_size, q_out), &dev));
    tmap.insert(format!("model.layers.{l0}.self_attn.sinks"), bf16_rand((1, cfg.num_attention_heads), &dev).reshape(cfg.num_attention_heads)?);

    // Router
    tmap.insert(format!("model.layers.{l0}.mlp.router.weight"), bf16_rand((cfg.num_local_experts, cfg.hidden_size), &dev));

    // Experts: supply MXFP4 blocks+scales per expert with correct shapes
    let nblocks_gate = cfg.hidden_size / 32;
    let nblocks_down = cfg.intermediate_size / 32;
    for e in 0..cfg.num_local_experts {
        tmap.insert(
            format!("model.layers.{l0}.mlp.experts.{e}.gate_up_proj.blocks"),
            u8_pattern((2 * cfg.intermediate_size, nblocks_gate, 16), &dev),
        );
        tmap.insert(
            format!("model.layers.{l0}.mlp.experts.{e}.gate_up_proj.scales"),
            Tensor::zeros((2 * cfg.intermediate_size, nblocks_gate), DType::U8, &dev)?,
        );
        tmap.insert(
            format!("model.layers.{l0}.mlp.experts.{e}.down_proj.blocks"),
            u8_pattern((cfg.hidden_size, nblocks_down, 16), &dev),
        );
        tmap.insert(
            format!("model.layers.{l0}.mlp.experts.{e}.down_proj.scales"),
            Tensor::zeros((cfg.hidden_size, nblocks_down), DType::U8, &dev)?,
        );
    }

    // LM head
    tmap.insert("lm_head.weight".into(), bf16_rand((cfg.vocab_size, cfg.hidden_size), &dev));

    let vb = candle_nn::VarBuilder::from_tensors(tmap, DType::BF16, &dev);
    let model = GptOssModel::load(vb, &cfg)?;

    // Simple input
    let input_ids = Tensor::new(&[0u32, 1, 2, 3, 4, 5], &dev)?.reshape((2, 3))?;
    let logits = model.forward_logits_minimal(&input_ids)?;
    assert_eq!(logits.dims(), &[2, 3, cfg.vocab_size]);
    assert_eq!(logits.dtype(), DType::BF16);
    Ok(())
}
