import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer as HFTokenizer

# Constants (paths and knobs)
SNAPSHOT_DIR = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    )
)
MODEL_INDEX_FILE = "model.safetensors.index.json"
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
ENV_DUMP_L1 = "CANDLE_DUMP_L1"  # truthy => dump first-layer debug vectors (parity with Rust)


def load_local_safetensors(snapshot: Path) -> List[Path]:
    idx_path = snapshot / MODEL_INDEX_FILE
    with open(idx_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Typical structure: {"weight_map": {"...": "model-00001-of-00003.safetensors", ...}}
    # We just need the set of shard filenames, in stable order.
    shard_set = []
    seen = set()
    for fname in data.get("weight_map", {}).values():
        if fname not in seen:
            seen.add(fname)
            shard_set.append(snapshot / fname)
    return shard_set


def load_stop_token_ids(snapshot: Path) -> List[int]:
    gen_cfg = snapshot / "generation_config.json"
    with open(gen_cfg, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    ids: List[int] = []
    eos = cfg.get("eos_token_id")
    if isinstance(eos, int):
        ids.append(eos)
    elif isinstance(eos, list):
        ids.extend([int(x) for x in eos])
    pad = cfg.get("pad_token_id")
    if isinstance(pad, int):
        ids.append(pad)
    # de-dup + sort for stable printing
    ids = sorted(set(ids))
    return ids


def lookup_special_ids(tokenizer_path: Path) -> Dict[str, int]:
    tk = HFTokenizer.from_file(str(tokenizer_path))
    def id_of(tok: str) -> int:
        tid = tk.token_to_id(tok)
        if tid is None:
            raise ValueError(f"missing token id for {tok}")
        return int(tid)
    return {
        "<|startoftext|>": id_of("<|startoftext|>"),
        "<|endoftext|>": id_of("<|endoftext|>"),
        "<|return|>": id_of("<|return|>"),
        "<|call|>": id_of("<|call|>"),
    }


def extract_final_assistant_text_from_decoded(decoded: str) -> str | None:
    ST_START = "<|start|>"
    ST_MESSAGE = "<|message|>"
    ST_END = "<|end|>"
    ST_CALL = "<|call|>"
    ST_RETURN = "<|return|>"
    ST_CHANNEL = "<|channel|>"

    preferred = f"{ST_START}assistant{ST_CHANNEL}final{ST_MESSAGE}"
    slice_after: str | None
    if preferred in decoded:
        pos = decoded.rfind(preferred)
        slice_after = decoded[pos + len(preferred) :]
    else:
        start_tag = f"{ST_START}assistant"
        pos = decoded.rfind(start_tag)
        if pos == -1:
            return None
        rest = decoded[pos + len(start_tag) :]
        if ST_CHANNEL in rest:
            rest2 = rest.split(ST_CHANNEL, 1)[1]
            if ST_MESSAGE not in rest2:
                return None
            slice_after = rest2.split(ST_MESSAGE, 1)[1]
        else:
            if ST_MESSAGE not in rest:
                return None
            slice_after = rest.split(ST_MESSAGE, 1)[1]

    end_idx = len(slice_after)
    for stop in (ST_RETURN, ST_CALL, ST_END, ST_START):
        p = slice_after.find(stop)
        if p != -1:
            end_idx = min(end_idx, p)
    return slice_after[:end_idx]


def main() -> None:
    # Messages (Harmony-style content). Keep identical to Rust example.
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": "Explain what MXFP4 quantization is"},
    ]

    # Device + dtype
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; this run expects a GPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    print(f"Device set to use {device}")
    print(f"dtype: {'bf16' if dtype == torch.bfloat16 else 'f16'}")

    # Print snapshot + shards summary
    print(f"Snapshot: {SNAPSHOT_DIR}")
    shards = load_local_safetensors(SNAPSHOT_DIR)
    print(f"Shards: count={len(shards)} first={[p.name for p in shards[:3]]}")

    # Load config summary
    cfg_path = SNAPSHOT_DIR / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    head_dim = cfg.get("head_dim") or (cfg["hidden_size"] // cfg["num_attention_heads"])
    rope_cfg = cfg.get("rope_scaling") or None
    if rope_cfg is None:
        rope_str = "null"
    else:
        rope_str = (
            "{'type': '" + str(rope_cfg.get("type", "?")) + "', "
            + "'factor': " + str(rope_cfg.get("factor", "null")) + ", "
            + "'beta_fast': " + str(rope_cfg.get("beta_fast", "null")) + ", "
            + "'beta_slow': " + str(rope_cfg.get("beta_slow", "null")) + ", "
            + "'original_max_position_embeddings': " + str(rope_cfg.get("original_max_position_embeddings", "null")) + "}"
        )
    print(
        "Config: hidden_size={hidden} layers={layers} heads={heads} kv_heads={kv} head_dim={hd} max_pos={maxp} rope={rope} sliding_window={sw}".format(
            hidden=cfg.get("hidden_size"),
            layers=cfg.get("num_hidden_layers"),
            heads=cfg.get("num_attention_heads"),
            kv=cfg.get("num_key_value_heads"),
            hd=head_dim,
            maxp=cfg.get("max_position_embeddings"),
            rope=rope_str,
            sw=cfg.get("sliding_window"),
        )
    )

    # Show the messages for parity
    print(f"messages: {[{'role': 'user', 'content': messages[0]['content']}]}")

    # Tokenizer parity: use HF fast tokenizer from snapshot
    tok = AutoTokenizer.from_pretrained(
        str(SNAPSHOT_DIR), trust_remote_code=True, local_files_only=True
    )
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids: List[int] = tok.encode(rendered, add_special_tokens=False)

    # Mirror Candle’s first-32 debug
    print(f"first 32 token ids: {input_ids[:32]}")
    tk_fast = HFTokenizer.from_file(str(SNAPSHOT_DIR / "tokenizer.json"))
    try:
        first32_dec = tk_fast.decode(input_ids[:32], skip_special_tokens=False)
    except Exception:
        first32_dec = "<decode-error>"
    print(f"first 32 decode: {first32_dec}")
    # Full prompt tokens and decode for strict parity with Rust
    print(f"prompt token ids: {input_ids}")
    try:
        full_dec = tk_fast.decode(input_ids, skip_special_tokens=False)
    except Exception:
        full_dec = "<decode-error>"
    tail_check = "<|start|>assistant<|channel|>final<|message|>"
    print(f"prompt tail contains '{tail_check}': {tail_check in full_dec}")
    print(f"prompt decode: {full_dec}")

    # Stop tokens
    stop_ids = load_stop_token_ids(SNAPSHOT_DIR)
    print(f"stop token ids: {stop_ids}")
    spec_ids = lookup_special_ids(SNAPSHOT_DIR / "tokenizer.json")
    print(f"special ids: {spec_ids}")

    # Load model on GPU
    model = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT_DIR), torch_dtype=dtype, device_map={"": 0}, local_files_only=True
    )
    model.eval()

    # Optional: synchronized first-layer debug hooks (post-norm and post-attn-residual)
    def _truthy_env(name: str) -> bool:
        v = os.environ.get(name)
        return v is not None and v.lower() in ("1", "true", "yes")

    if _truthy_env(ENV_DUMP_L1):
        printed = {"post_norm": False, "post_attn_resid": False}
        residual_holder: dict[str, torch.Tensor] = {}

        def _print_vec(tag: str, vec: torch.Tensor) -> None:
            v = vec.detach().float().cpu().numpy().tolist()
            first8 = v[:8]
            if len(v) == 0:
                mean = 0.0
                std = 0.0
            else:
                m = sum(v) / float(len(v))
                var = sum((x - m) * (x - m) for x in v) / float(len(v))
                mean, std = m, var ** 0.5
            print(
                f"[L1] {tag}: len={len(v)} first8={first8} mean={mean:.6f} std={std:.6f}"
            )

        # Hook: post-norm vector from the first layer's input_layernorm
        ln0 = model.model.layers[0].input_layernorm  # type: ignore[attr-defined]

        def _ln_hook(_m, _inp, out):
            if printed["post_norm"]:
                return
            try:
                y = out  # (b, t, h)
                last = y[0, -1, :]
                _print_vec("post-norm last-token", last)
            finally:
                printed["post_norm"] = True

        ln0.register_forward_hook(_ln_hook)

        # Pre-hook on the first decoder layer to capture residual before attention
        layer0 = model.model.layers[0]  # type: ignore[attr-defined]

        def _layer_pre(_m, inputs):
            # inputs: (hidden_states, ...)
            hs = inputs[0]
            residual_holder["resid"] = hs.detach()

        layer0.register_forward_pre_hook(_layer_pre)

        # Hook on the first layer's self-attention to capture attention output (after o_proj)
        attn0 = model.model.layers[0].self_attn  # type: ignore[attr-defined]

        def _attn_hook(_m, _inp, out):
            if printed["post_attn_resid"]:
                return
            try:
                attn_out = out[0] if isinstance(out, (tuple, list)) else out  # (b, t, h)
                resid = residual_holder.get("resid")
                if resid is None:
                    return
                vec = (resid + attn_out)[0, -1, :]
                _print_vec("post-attn-residual last-token", vec)
            finally:
                printed["post_attn_resid"] = True

        attn0.register_forward_hook(_attn_hook)

    # Top-10 next-token candidates at step 0
    with torch.no_grad():
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(input_ids=input_tensor).logits  # (1, T, V)
        last = logits[0, -1].float()
        probs = torch.nn.functional.softmax(last, dim=-1)
        topk = torch.topk(probs, k=10)
        inv_vocab = topk.indices.detach().cpu().tolist()
        inv_probs = topk.values.detach().cpu().tolist()
        print("top-10 next-token candidates:")
        for rank, (tok_id, p) in enumerate(zip(inv_vocab, inv_probs)):
            s = tk_fast.decode([tok_id], skip_special_tokens=False)
            print(f"  id={tok_id:6d} p={p:.4f} tok={s}")
        ch_id = tk_fast.token_to_id("<|channel|>")
        if ch_id is not None:
            ch_prob = probs[int(ch_id)].item()
            print(f"  special '<|channel|>' id={ch_id} p={ch_prob:.6f}")
        else:
            print("  special '<|channel|>' not present in tokenizer")

    # Generate
    gen = model.generate(
        torch.tensor([input_ids], dtype=torch.long, device=device),
        do_sample=True,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=stop_ids if stop_ids else None,
        use_cache=True,
    )
    out_ids = gen[0].detach().cpu().tolist()
    decoded_full = tk_fast.decode(out_ids, skip_special_tokens=False)
    print("\n" + decoded_full)
    final = extract_final_assistant_text_from_decoded(decoded_full)
    if final is not None:
        print(final.strip())
    else:
        # Fall back to skipping special tokens
        cleaned = tk_fast.decode(out_ids, skip_special_tokens=True)
        print(cleaned.strip())


if __name__ == "__main__":
    main()
