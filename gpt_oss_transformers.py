import json
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Any

import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
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


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-layer-states", action="store_true", help="Dump per-layer last-token states to layers_step0_last.npy and exit")
    parser.add_argument("--dump-final-hidden", action="store_true", help="Print JSON with last-token vectors: post_mlp_last and post_norm_last, then exit")
    parser.add_argument("--dump-l0-qkv", action="store_true", help="Print JSON with layer-0 last-token Q/K/V (pre and post-RoPE) plus softmax scale and sinks")
    program_start = time.perf_counter()
    args = parser.parse_args()
    # Messages (Harmony-style content). Keep identical to Rust example.
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": "What is the capital of France?"},
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
    cuda_sync()
    model_load_start = time.perf_counter()
    time_to_model_load_start = model_load_start - program_start
    model = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT_DIR), torch_dtype=dtype, device_map={"": 0}, local_files_only=True
    )
    model.eval()
    cuda_sync()
    model_load_duration = time.perf_counter() - model_load_start

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

    if args.dump_layer_states:
        # Dump h0 + (pre_attn_norm, post_attn_resid, post_mlp_resid) per layer for last prompt token.
        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            L = model.config.num_hidden_layers
            H = model.config.hidden_size

            pre_attn_norm: list[torch.Tensor] = []
            post_attn_resid: list[torch.Tensor] = []

            # Hook collectors per layer
            hooks = []
            for i in range(L):
                ln = model.model.layers[i].input_layernorm  # type: ignore[attr-defined]

                def _mk_ln_hook():
                    idx = len(pre_attn_norm)
                    def _ln_hook(_m, _inp, out):
                        y = out  # (b, t, h)
                        pre_attn_norm.append(y[0, -1, :].detach().float().cpu())
                    return _ln_hook
                hooks.append(ln.register_forward_hook(_mk_ln_hook()))

                post_ln = model.model.layers[i].post_attention_layernorm  # type: ignore[attr-defined]

                def _mk_pre_hook():
                    def _pre(_m, inputs):
                        hs = inputs[0]
                        post_attn_resid.append(hs[0, -1, :].detach().float().cpu())
                        return None
                    return _pre
                hooks.append(post_ln.register_forward_pre_hook(_mk_pre_hook()))

            out = model(input_ids=input_tensor, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states  # tuple(len = L+1)
            assert len(hs) == L + 1
            h0 = hs[0][0, -1, :].detach().float().cpu()
            post_mlp = [hs[i + 1][0, -1, :].detach().float().cpu() for i in range(L)]

            # Assemble rows: [h0] + triples per layer
            rows = [h0] + [x for i in range(L) for x in (pre_attn_norm[i], post_attn_resid[i], post_mlp[i])]
            arr = torch.stack(rows, dim=0).numpy()  # (1+3L, H) float32 on cpu
            np.save("layers_step0_last.npy", arr)
            for h in hooks:
                h.remove()
            print(json.dumps({"dump": "layers_step0_last.npy", "rows": int(arr.shape[0]), "hidden": int(arr.shape[1])}))
            return

    if args.dump_final_hidden:
        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            out = model(input_ids=input_tensor, output_hidden_states=True, return_dict=True)
            hs_last = out.hidden_states[-1]  # (1, T, H)
            post_mlp_last = hs_last[0, -1, :].detach().float().cpu().numpy().tolist()
            # Apply the final RMSNorm to match Candle's post-norm hidden pre-lm_head
            post_norm_last = model.model.norm(hs_last)[0, -1, :].detach().float().cpu().numpy().tolist()  # type: ignore[attr-defined]
            out_obj = {
                "prompt": messages[0]["content"],
                "dtype": "bf16" if dtype == torch.bfloat16 else "f16",
                "device": str(device),
                "post_mlp_last": post_mlp_last,
                "post_norm_last": post_norm_last,
            }
            print(json.dumps(out_obj))
        return

    if args.dump_l0_qkv:
        with torch.no_grad():
            # Tokenize and embed once to replicate Candle's taps.
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            hs0 = model.get_input_embeddings()(input_tensor)  # (1,T,H)
            L0 = model.model.layers[0]  # type: ignore[attr-defined]
            attn0 = L0.self_attn

            H = model.config.hidden_size
            n_q = model.config.num_attention_heads
            n_kv = model.config.num_key_value_heads
            hd = getattr(model.config, "head_dim", H // n_q)

            # position ids and cos/sin
            past_seen_tokens = 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + hs0.shape[1], device=hs0.device)
            position_ids = cache_position.unsqueeze(0)
            cos, sin = model.model.rotary_emb(hs0, position_ids)  # type: ignore[attr-defined]

            # Compute input layernorm then linear projections (matches forward path).
            x_norm = L0.input_layernorm(hs0)
            q_lin = torch.nn.functional.linear(x_norm, attn0.q_proj.weight, attn0.q_proj.bias)
            k_lin = torch.nn.functional.linear(x_norm, attn0.k_proj.weight, attn0.k_proj.bias)
            v_lin = torch.nn.functional.linear(x_norm, attn0.v_proj.weight, attn0.v_proj.bias)

            q = q_lin.view(1, -1, n_q, hd)  # (1,T,Hq,D)
            k = k_lin.view(1, -1, n_kv, hd)
            v = v_lin.view(1, -1, n_kv, hd)

            # Last-token slices pre-RoPE: (Hq,D), (Hkv,D)
            q_pre = q[0, -1].detach().float().cpu()
            k_pre = k[0, -1].detach().float().cpu()
            v_pre = v[0, -1].detach().float().cpu()

            # Post-RoPE: need (B,H,T,D) layout, apply_rotary_pos_emb, then take last token.
            q_bhtd = q.transpose(1, 2)
            k_bhtd = k.transpose(1, 2)
            v_bhtd = v.transpose(1, 2)
            from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

            q_rope_bhtd, k_rope_bhtd = apply_rotary_pos_emb(q_bhtd, k_bhtd, cos, sin)
            q_rope = q_rope_bhtd[0, :, -1, :].detach().float().cpu()
            k_rope = k_rope_bhtd[0, :, -1, :].detach().float().cpu()

            # Softmax scale and attn mode
            softmax_scale = float(hd ** -0.5)
            attn_type = model.config.layer_types[0] if hasattr(model.config, "layer_types") else "full_attention"

            # Compute attention output for the whole sequence and take last token, pre/post o_proj
            # Shapes to match eager_attention_forward semantics
            from transformers.models.gpt_oss.modeling_gpt_oss import repeat_kv
            key_states = repeat_kv(k_bhtd, attn0.num_key_value_groups)
            value_states = repeat_kv(v_bhtd, attn0.num_key_value_groups)
            attn_weights = torch.matmul(q_bhtd, key_states.transpose(2, 3)) * softmax_scale
            # Build attention mask from config utilities
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
            cache_position = torch.arange(0, hs0.shape[1], device=hs0.device)
            causal_mask_mapping = {
                "full_attention": create_causal_mask(config=model.config, input_embeds=hs0, attention_mask=None, cache_position=cache_position, past_key_values=None),
                "sliding_attention": create_sliding_window_causal_mask(config=model.config, input_embeds=hs0, attention_mask=None, cache_position=cache_position, past_key_values=None),
            }
            amask = causal_mask_mapping[attn_type][:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + amask
            sinks = attn0.sinks.reshape(1, -1, 1, 1).expand(q_bhtd.shape[0], -1, q_bhtd.shape[-2], -1)
            combined_logits = torch.cat([attn_weights, sinks], dim=-1)
            combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            probs = torch.nn.functional.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
            scores = probs[..., :-1]
            attn_output = torch.matmul(scores, value_states)  # (B, Hq, T, D)
            attn_output_bthd = attn_output.transpose(1, 2).contiguous() # (B, T, Hq, D)
            y_pre_full = attn_output_bthd.reshape(1, -1, n_q * hd)
            y_pre_last = y_pre_full[0, -1, :].detach().float().cpu().numpy().tolist()
            y_post_full = torch.nn.functional.linear(y_pre_full, attn0.o_proj.weight, attn0.o_proj.bias)
            y_post_last = y_post_full[0, -1, :].detach().float().cpu().numpy().tolist()

            # Weight sample for sanity (first row, first-8 cols)
            q_w_samp = attn0.q_proj.weight.detach().float().cpu()[0, :8].numpy().tolist()

            # Extract last-token attention logits/weights for head 0 (and head 1 if present)
            logits_last_h0 = attn_weights[0, 0, -1, :].detach().float().cpu().numpy().tolist()
            scores_last_h0 = scores[0, 0, -1, :].detach().float().cpu().numpy().tolist()
            logits_last_h1 = (
                attn_weights[0, 1, -1, :].detach().float().cpu().numpy().tolist() if n_q > 1 else None
            )
            scores_last_h1 = (
                scores[0, 1, -1, :].detach().float().cpu().numpy().tolist() if n_q > 1 else None
            )

            # For head-0 diagnostics: dump key/value sequences across time
            k_all_h0 = key_states[0, 0, :, :].detach().float().cpu().numpy().reshape(-1).tolist()
            v_all_h0 = value_states[0, 0, :, :].detach().float().cpu().numpy().reshape(-1).tolist()

            out_obj = {
                "q_pre": q_pre.reshape(-1).numpy().tolist(),
                "k_pre": k_pre.reshape(-1).numpy().tolist(),
                "v_pre": v_pre.reshape(-1).numpy().tolist(),
                "q_rope": q_rope.reshape(-1).numpy().tolist(),
                "k_rope": k_rope.reshape(-1).numpy().tolist(),
                "attn_pre_last": y_pre_last,
                "attn_post_last": y_post_last,
                "attn_logits_last_h0": logits_last_h0,
                "attn_scores_last_h0": scores_last_h0,
                "attn_logits_last_h1": logits_last_h1,
                "attn_scores_last_h1": scores_last_h1,
                "k_all_h0": k_all_h0,
                "v_all_h0": v_all_h0,
                "shapes": {
                    "q_pre": [int(q_pre.shape[0]), int(q_pre.shape[1])],
                    "k_pre": [int(k_pre.shape[0]), int(k_pre.shape[1])],
                    "v_pre": [int(v_pre.shape[0]), int(v_pre.shape[1])],
                },
                "softmax_scale": softmax_scale,
                "attn_type": attn_type,
                "sliding_window": int(model.config.sliding_window) if getattr(model.config, "sliding_window", None) is not None else None,
                "sinks": L0.self_attn.sinks.detach().float().cpu().numpy().tolist(),
                "q_w_samp": q_w_samp,
            }
            # Also return the pre-attn norm last-token first8 for context
            out_obj["l0_pre_norm_first8"] = x_norm[0, -1, :8].detach().float().cpu().numpy().tolist()
            print(json.dumps(out_obj))
        return

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Top-10 next-token candidates at step 0 with detailed timing breakdown
    with torch.no_grad():
        cuda_sync()
        t_forward_start = time.perf_counter()

        # Time embedding lookup
        t_embed_start = time.perf_counter()
        embeddings = model.get_input_embeddings()(input_tensor)
        cuda_sync()
        t_embed_dur = time.perf_counter() - t_embed_start

        # Time full forward pass (will include all layers + lm_head)
        t_full_start = time.perf_counter()
        logits = model(input_ids=input_tensor).logits  # (1, T, V)
        cuda_sync()
        t_full_dur = time.perf_counter() - t_full_start

        t_forward_total = time.perf_counter() - t_forward_start

        print(f"[PROFILE] First forward pass timing breakdown:")
        print(f"  - embedding lookup: {t_embed_dur*1000:.3f} ms")
        print(f"  - full forward (all layers + lm_head): {t_full_dur*1000:.3f} ms")
        print(f"  - total with syncs: {t_forward_total*1000:.3f} ms")

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

    # Set seed for deterministic sampling (matching Rust default)
    SEED = 299792458
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    generation_kwargs = {
        "input_ids": input_tensor,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_token_id": stop_ids if stop_ids else None,
        "streamer": streamer,
        "use_cache": True,
    }

    result_holder: Dict[str, torch.Tensor] = {}

    def _run_generate() -> None:
        cuda_sync()
        outputs = model.generate(**generation_kwargs)
        cuda_sync()
        result_holder["output_ids"] = outputs

    cuda_sync()
    generation_start = time.perf_counter()
    gen_thread = threading.Thread(target=_run_generate)
    gen_thread.start()

    streamed_chunks: List[str] = []
    response_stream_start: float | None = None
    response_stream_end: float | None = None
    prompt_processing_duration: float | None = None

    for chunk in streamer:
        cuda_sync()
        now = time.perf_counter()
        streamed_chunks.append(chunk)
        if response_stream_start is None:
            response_stream_start = now
            prompt_processing_duration = now - generation_start
        response_stream_end = now
        print(chunk, end="", flush=True)

    gen_thread.join()
    cuda_sync()
    generation_end = time.perf_counter()

    if response_stream_start is None:
        prompt_processing_duration = generation_end - generation_start
        response_stream_duration = 0.0
    else:
        response_stream_duration = (response_stream_end or generation_end) - response_stream_start

    out_tensor = result_holder.get("output_ids")
    if out_tensor is None:
        raise RuntimeError("generation did not produce output tokens")
    out_ids = out_tensor[0].detach().cpu().tolist()
    generated_tokens = max(len(out_ids) - len(input_ids), 0)

    prompt_processing_duration = prompt_processing_duration or 0.0
    total_response_time = response_stream_duration
    tokens_per_second = (generated_tokens / total_response_time) if total_response_time > 0.0 else 0.0

    decoded_full = tk_fast.decode(out_ids, skip_special_tokens=False)
    print()
    print("perf metrics:")
    print(f"- time_to_model_load_start: {time_to_model_load_start:.3f} s")
    print(f"- model_load_duration: {model_load_duration:.3f} s")
    print(f"- prompt_processing_duration: {prompt_processing_duration:.3f} s")
    print(f"- response_stream_duration: {response_stream_duration:.3f} s")
    print(f"- tokens_per_second: {tokens_per_second:.3f} tok/s over {generated_tokens} tokens")

    print(f"full harmony decode: {decoded_full}")
    final = extract_final_assistant_text_from_decoded(decoded_full)
    if final is not None and final.strip():
        print(f"assistant reply: {final.strip()}")
    else:
        cleaned = tk_fast.decode(out_ids, skip_special_tokens=True)
        print(f"assistant reply: {cleaned.strip()}")


if __name__ == "__main__":
    main()
