import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb, repeat_kv
from transformers.masking_utils import create_sliding_window_causal_mask


SNAPSHOT = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
    )
)


def main() -> None:
    # Deterministic text producing >128 tokens to hit the sliding boundary
    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
    messages = [{"role": "user", "content": ("abc ") * 80 + "please just pad"}]
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tok.encode(rendered, add_special_tokens=False)
    # Ensure we have at least 130 tokens
    if len(input_ids) < 130:
        # Append additional plain text until we reach the length
        extra = tok.encode((" filler") * 1000, add_special_tokens=False)
        need = 130 - len(input_ids)
        input_ids = input_ids + extra[:need]
    input_ids = input_ids[:130]

    # GPU, bfloat16 if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT), torch_dtype=dtype, device_map={"": 0} if device.type == "cuda" else None,
        trust_remote_code=True, local_files_only=True
    )
    model.eval()

    # Tensors
    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    embeds = model.model.embed_tokens(inp)

    # Prepare mask mapping and rotary caches similar to forward()
    cache_position = torch.arange(embeds.shape[1], device=device, dtype=torch.long)
    mask_map = {
        "sliding_attention": create_sliding_window_causal_mask(
            config=model.config, input_embeds=embeds, attention_mask=None,
            cache_position=cache_position, past_key_values=None
        ),
        "full_attention": None,
    }
    pos_ids = cache_position.unsqueeze(0)
    cos, sin = model.model.rotary_emb(embeds, pos_ids)

    # Take first layer
    layer0 = model.model.layers[0]
    x_in = embeds.detach()  # (1,T,H), pre-norm block input
    x_norm = layer0.input_layernorm(x_in)

    # Projections pre-RoPE
    H = model.config.hidden_size
    hd = getattr(model.config, "head_dim", H // model.config.num_attention_heads)
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads

    def view_heads(t, n):
        b, tlen, _ = t.shape
        return t.view(b, tlen, -1, hd).transpose(1, 2).contiguous()  # (b,n,t,d)

    q = view_heads(layer0.self_attn.q_proj(x_norm), n_q)
    k = view_heads(layer0.self_attn.k_proj(x_norm), n_kv)
    v = view_heads(layer0.self_attn.v_proj(x_norm), n_kv)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    # Eager attention forward (with sinks and sliding mask) in float32, then cast to dtype before o_proj
    with torch.no_grad():
        key_states = repeat_kv(k_rot, layer0.self_attn.num_key_value_groups)
        value_states = repeat_kv(v, layer0.self_attn.num_key_value_groups)
        attn_weights = torch.matmul(q_rot.float(), key_states.transpose(2, 3).float())
        attn_weights = attn_weights * (1.0 / (hd ** 0.5))
        if mask_map["sliding_attention"] is not None:
            raw_mask = mask_map["sliding_attention"][:, :, :, : key_states.shape[-2]]
            if raw_mask.dtype == torch.bool:
                minv = torch.finfo(attn_weights.dtype).min
                zeros = torch.zeros((), dtype=attn_weights.dtype, device=attn_weights.device)
                causal_mask = torch.where(raw_mask, zeros, torch.tensor(minv, dtype=attn_weights.dtype, device=attn_weights.device))
            else:
                causal_mask = raw_mask.to(dtype=attn_weights.dtype)
            attn_weights = attn_weights + causal_mask
        sinks_bhq1 = layer0.self_attn.sinks.reshape(1, -1, 1, 1).expand(q_rot.shape[0], -1, q_rot.shape[-2], -1).float()
        combined_logits = torch.cat([attn_weights, sinks_bhq1], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(combined_logits, dim=-1, dtype=torch.float32)
        scores = probs[..., :-1]
        attn_output = torch.matmul(scores, value_states.float())
        attn_output = attn_output.transpose(1, 2).contiguous()
        bsz, seqlen = attn_output.shape[0], attn_output.shape[1]
        attn_flat = attn_output.reshape(bsz, seqlen, -1)
        attn_projected = layer0.self_attn.o_proj(attn_flat.to(dtype))
    post_attn = x_in + attn_projected  # residual add, before MLP

    # Mask row counts (per q row) for head 0 using the same mask
    # In eager path, the mask is float: 0 for allowed, -inf for masked. Count where mask == 0.
    mm = mask_map["sliding_attention"]
    if mm is None:
        counts = [i + 1 for i in range(len(input_ids))]
    else:
        allowed = (mm[0, 0] == 0)  # (q,k) bool
        counts = allowed.sum(dim=-1).detach().cpu().tolist()

    # Sinks
    sinks = layer0.self_attn.sinks.detach().float().cpu().tolist()

    def to_f32_list(t: torch.Tensor):
        return t.detach().float().cpu().tolist()

    res: Dict[str, Any] = {
        "input_ids": input_ids,
        "x_in": to_f32_list(x_in[0]),  # (T,H)
        "q_pre": to_f32_list(q[0]),    # (n_q,T,d) via (b,n,t,d) -> take b=0
        "k_pre": to_f32_list(k[0]),    # (n_kv,T,d)
        "v_pre": to_f32_list(v[0]),
        "cos": to_f32_list(cos[0]),    # (T, d/2)
        "sin": to_f32_list(sin[0]),
        "mask_row_counts": counts,
        "sinks": sinks,
        "A0_last": to_f32_list(post_attn[0, -1]),  # last position only for compactness
        "hidden_size": H,
        "head_dim": hd,
        "n_q": n_q,
        "n_kv": n_kv,
    }

    print(json.dumps(res))


if __name__ == "__main__":
    main()
