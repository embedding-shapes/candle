import json
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
from safetensors import safe_open

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

if __name__ == "__main__":
    main()
