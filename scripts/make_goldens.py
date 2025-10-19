import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoTokenizer
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssModel,
    GptOssAttention,
    GptOssRotaryEmbedding,
    apply_rotary_pos_emb,
)

SNAPSHOT = Path.home() / \
    ".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

def render_then_encode(tokenizer, messages, add_generation_prompt=True):
    text = tokenizer.apply_chat_template(
        messages,
        tools=None,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        return_tensors=None,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids

def main():
    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT))
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        (
            "chat_simple.json",
            [
                {"role": "user", "content": "Explain what MXFP4 quantization is"},
            ],
        ),
        (
            "chat_two_turns.json",
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "What is YARN RoPE?"},
            ],
        ),
        (
            "chat_system_dev.json",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "developer", "content": "Answer concisely."},
                {"role": "user", "content": "Who founded OpenAI?"},
            ],
        ),
    ]

    for name, messages in cases:
        ids = render_then_encode(tok, messages, add_generation_prompt=True)
        with open(fixtures_dir / name, "w") as f:
            json.dump({"input_ids": ids}, f)

    # Save special ids for sanity checks
    gen_cfg = json.load((SNAPSHOT / "generation_config.json").open())
    with open(fixtures_dir / "special_ids.json", "w") as f:
        json.dump({
            "bos": gen_cfg.get("bos_token_id"),
            "pad": gen_cfg.get("pad_token_id"),
            "eos": gen_cfg.get("eos_token_id"),
        }, f)

    # MXFP4 decode sanity: decode a single expert weight (layer 0 gate_up_proj, expert 0)
    idx_path = SNAPSHOT / "model.safetensors.index.json"
    with open(idx_path) as f:
        idx = json.load(f)["weight_map"]
    def load_tensor(key):
        path = SNAPSHOT / idx[key]
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(key).numpy()

    blocks = load_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks")
    scales = load_tensor("model.layers.0.mlp.experts.gate_up_proj_scales")
    # pick expert 0
    b0 = blocks[0]  # (out, nb, 16)
    s0 = scales[0]  # (out, nb)
    out, nb, _ = b0.shape
    cols = nb * 32

    def decode_fp4_e2m1(n):
        s = (n >> 3) & 0x1
        e = (n >> 1) & 0x3
        m = n & 0x1
        sign = -1.0 if s == 1 else 1.0
        if e == 0:
            frac = float(m) * 0.5
            exp = 0  # 2^(1-bias) with bias=1
            return sign * (2.0 ** exp) * frac
        else:
            frac = 1.0 + float(m) * 0.5
            exp = int(e) - 1
            return sign * (2.0 ** exp) * frac
    def pow2_e8m0(u):
        if u == 0xFF:
            return np.nan
        return float(2.0 ** (int(u) - 127))

    out_bf16 = np.zeros((out, cols), dtype=np.float32)
    for r in range(out):
        for b in range(nb):
            scale = pow2_e8m0(int(s0[r, b]))
            for j in range(16):
                byte = int(b0[r, b, j])
                lo = byte & 0x0F
                hi = (byte >> 4) & 0x0F
                c0 = b * 32 + 2 * j
                c1 = c0 + 1
                out_bf16[r, c0] = decode_fp4_e2m1(lo) * scale
                out_bf16[r, c1] = decode_fp4_e2m1(hi) * scale

    # Save stats for first 8 rows
    stats = []
    for r in range(min(8, out)):
        row = out_bf16[r]
        stats.append({
            "mean": float(np.nanmean(row)),
            "std": float(np.nanstd(row)),
        })
    with open(fixtures_dir / "mxfp4_stats_layer0_expert0_gate_up_proj.json", "w") as f:
        json.dump({"rows": stats, "shape": [int(out), int(cols)]}, f)

    # Also dump a small raw+decoded slice for row 0, block 0 (pre/post decode check)
    raw_bytes = [int(x) for x in b0[0, 0].tolist()]
    decoded32 = [float(x) for x in out_bf16[0, :32].tolist()]
    with open(fixtures_dir / "mxfp4_row0_block0_slice.json", "w") as f:
        json.dump({
            "raw_bytes": raw_bytes,  # 16 bytes (each carries 2 fp4 values)
            "scale_u8": int(s0[0, 0]),
            "decoded32": decoded32,
        }, f)

    # 2) Dump per-layer hidden states for the first 2 decode steps (tiny synthetic config for speed)
    torch.manual_seed(13)
    cfg = GptOssConfig(
        num_hidden_layers=3,
        num_local_experts=2,
        num_experts_per_tok=1,
        vocab_size=256,
        hidden_size=32,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": False,
            "original_max_position_embeddings": 16,
        },
        sliding_window=8,
        attention_dropout=0.0,
    )
    model = GptOssModel(cfg).eval()
    # Two decode steps with caching
    ids0 = torch.tensor([[42]], dtype=torch.long)
    out0 = model(ids0, use_cache=True, output_hidden_states=True)
    step1_hidden = [h.detach().cpu().numpy()[0, -1, :].tolist() for h in out0.hidden_states]
    # second token with cache
    ids1 = torch.tensor([[17]], dtype=torch.long)
    out1 = model(ids1, use_cache=True, past_key_values=out0.past_key_values, output_hidden_states=True)
    step2_hidden = [h.detach().cpu().numpy()[0, -1, :].tolist() for h in out1.hidden_states]
    with open(fixtures_dir / "hidden_states_step1.json", "w") as f:
        json.dump({
            "layers": len(step1_hidden),
            "hidden_size": len(step1_hidden[0]),
            "data": step1_hidden,
        }, f)
    with open(fixtures_dir / "hidden_states_step2.json", "w") as f:
        json.dump({
            "layers": len(step2_hidden),
            "hidden_size": len(step2_hidden[0]),
            "data": step2_hidden,
        }, f)

    # 3) Dump attention outputs for two short cases (use GptOssAttention directly, tiny config)
    def run_attn_case(seq_len: int, case_name: str):
        torch.manual_seed(123 + seq_len)
        attn_mod = GptOssAttention(cfg, layer_idx=0).eval()
        # hidden_states shape: (b, t, h)
        hs = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.float32)
        # Build rotary caches
        rope = GptOssRotaryEmbedding(config=cfg)
        pos_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        cos, sin = rope(hs, pos_ids)
        attn_out, attn_w = attn_mod(
            hidden_states=hs,
            attention_mask=None,
            position_ids=pos_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            position_embeddings=(cos, sin),
            output_attentions=True,
        )
        # Serialize minimal shapes for speed
        with open(fixtures_dir / case_name, "w") as f:
            json.dump({
                "output_shape": list(attn_out.shape),
                "weights_shape": list(attn_w.shape),
                "output_sample": attn_out.detach().cpu().numpy()[0, -1, :8].tolist(),
                "weights_sample": attn_w.detach().cpu().numpy()[0, 0].tolist(),
            }, f)

    run_attn_case(2, "attn_case1.json")
    run_attn_case(3, "attn_case2.json")

    # 4) Dump YARN-rotated Q,K slices at positions of interest using tiny head_dim
    def dump_yarn_qk(pos: int, fname: str):
        torch.manual_seed(999 + pos)
        rope = GptOssRotaryEmbedding(config=cfg)
        b, h, t, d = 1, 2, 1, cfg.head_dim
        q_in = torch.randn(b, h, t, d, dtype=torch.float32)
        k_in = torch.randn(b, 1, t, d, dtype=torch.float32)
        pos_ids = torch.tensor([[pos]])
        cos, sin = rope(q_in, pos_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q_in, k_in, cos, sin)
        with open(fixtures_dir / fname, "w") as f:
            json.dump({
                "head_dim": d,
                "pos": pos,
                "q_in": q_in.detach().cpu().numpy()[0, 0, 0].tolist(),
                "k_in": k_in.detach().cpu().numpy()[0, 0, 0].tolist(),
                "q_rot": q_rot.detach().cpu().numpy()[0, 0, 0].tolist(),
                "k_rot": k_rot.detach().cpu().numpy()[0, 0, 0].tolist(),
            }, f)

    dump_yarn_qk(0, "yarn_qk_pos0.json")
    dump_yarn_qk(17, "yarn_qk_pos17.json")

if __name__ == "__main__":
    main()
