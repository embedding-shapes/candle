Canonical spec (from the HF config/model card)

Decoder‑only MoE. 24 layers. hidden_size=2880, head_dim=64, num_attention_heads=64, num_key_value_heads=8 (GQA factor 8). RoPE with YARN: factor=32, beta_fast=32, beta_slow=1, original_max_position_embeddings=4096, rope_theta=150000. rms_norm_eps=1e-5. attention_bias=true. max_position_embeddings=131072. Sliding‑window size 128 with alternating sliding_attention and full_attention layers (see layer_types). MoE: num_local_experts=32, num_experts_per_tok=4 (top‑K=4). Untied LM head. HF chat requires Harmony formatting. 
Hugging Face
+1

Attention sinks (what it is and the constraint that affects Candle)

GPT‑OSS adds a per‑head sink logit to the attention scores. With flex/flash attention, you must renormalize after the kernel using the log‑sum‑exp (LSE) that the kernel computes; otherwise probabilities include the sink mass. Formula per (b,h,q): scale = 1 / (1 + exp(sink - lse)); multiply the output vector O by scale. So Candle’s flash‑attn wrapper must expose (O, LSE). 
Hugging Face
+1

Sliding‑window

Alternate full vs local exactly as in layer_types; local layers use left window 128 and right 0 in decode. Non‑FA path = build a causal+local mask; FA path = call windowed FA. 
Hugging Face

MoE details that matter for weights and math

Router: top‑K=4 per token, then softmax over the selected experts and weighted sum of expert outputs. Router logits can be optionally returned; default output_router_logits=false. Experts use fused gate_up and a down proj; activation silu. 
Hugging Face
+1

Weight files and names (HF → Candle VarBuilder)

Files: safetensors sharded; index lists canonical names. Map:

Attn: model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight,bias}

MoE router: model.layers.{i}.mlp.router.{weight,bias}

Experts: model.layers.{i}.mlp.experts.{gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias}

Embedding: model.embed_tokens.weight; head: lm_head.weight
These names appear in the shard index and discussions (e.g., self_attn.o_proj.bias, experts.gate_up_proj). Shapes match config; respect head_dim decoupled from hidden_size/num_heads. 
Hugging Face
+2
Hugging Face
+2

MXFP4: what it is here

GPT‑OSS uses MXFP4 post‑training quantization primarily on MoE weights. HF’s quantization_config shows modules excluded from conversion: self‑attn, router, embeddings, lm_head. I.e., experts are MXFP4, others stay BF16. HF loads Triton MXFP4 kernels automatically in Python; Candle has no MXFP4 runtime today. For Candle you must either:

Offline convert: dequantize MXFP4 expert tensors back to BF16 and write a new safetensors set; or

Add a tiny MXFP4 dequant path in Rust for experts (read u8 payload + per‑block scale, reconstruct FP weights to BF16 once on load).
HF docs also note MXFP4 requires CC≥7.5 in their stack; that’s a Python/Triton constraint, not Candle. 
Hugging Face
+1

Flash‑attention in Candle

Candle uses FA‑v2 kernels. Today the public API returns only O. For sinks, extend candle‑flash‑attn to return (O, LSE) for both standard and windowed variants, then apply the renorm above. Head‑dim constraints follow FA: multiples of 8 up to at least 128 are standard and 64 is fine. 
GitHub
+2
GitHub
+2

KV cache

One cache per layer. On sliding layers trim K/V to the configured window; on full layers cap by max_position_embeddings. Validate decode‑step vs. full‑pass equivalence on short sequences. (Matches general Candle patterns.) 
Hugging Face

CUDA/Blackwell notes

Build Candle with --features "cuda,flash-attn". If flash‑attn fails for a new SM (Blackwell), keep an eager attention fallback to validate correctness first; then fix kernel build flags. (Candle issues show FA build sensitivity; FA‑v2 itself supports head_dim=64.) 
GitHub
+1

Harmony prompt format

Model card states GPT‑OSS was trained on Harmony. Use your Rust Harmony crate to format messages before tokenization. Do not feed raw chat text.

[1]: https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json "config.json · openai/gpt-oss-20b at main"
[2]: https://huggingface.co/docs/transformers/main/model_doc/gpt_oss "GptOss"
[3]: https://huggingface.co/openai/gpt-oss-20b/blob/main/model.safetensors.index.json?utm_source=chatgpt.com "model.safetensors.index.json · openai/gpt-oss-20b at main"
[4]: https://github.com/huggingface/candle/issues/1683?utm_source=chatgpt.com "Slow generation compared to transformers + PyTorch #1683"
[5]: https://huggingface.github.io/candle/?utm_source=chatgpt.com "Introduction - Candle Documentation"
[6]: https://github.com/huggingface/candle/issues/1936?utm_source=chatgpt.com "Flash Attention not working on CUDA 12.1 #1936"
[7]: https://huggingface.co/openai/gpt-oss-20b "openai/gpt-oss-20b · Hugging Face"


## To keep in mind:

Keep these invariants and traps in mind.

**Flash‑attention + sinks**

* FA already computes `O = softmax(scores_non_sink)·V`. To match “softmax over [tokens + sink], then drop sink,” scale by
  `p_non_sink = 1 / (1 + exp(s_sink − LSE_non_sink))` per `(b,h,q)`, then `O *= p_non_sink`. This is exact.
  You must expose `softmax_lse` from `candle-flash-attn` to do this. The FFI already has `softmax_lse_ptr`; the public API does not return it yet. ([Docs.rs][1])
* Non‑FA path should **include** the sink in the softmax, then drop the sink column before `@V`. No extra renorm needed.
* Windowed FA: use the same entrypoint with `window_size_left/right`; return LSE as above. No new CUDA kernel required. ([Docs.rs][1])

**Sliding‑window layers**

* Alternate full vs local layers exactly as the HF reference describes for GPT‑OSS. Keep the left window equal to the configured `sliding_window` for causal decoding (right window = 0). ([Hugging Face][2])
* In non‑FA path, reuse the Mistral/Mixtral mask construction pattern to avoid off‑by‑one at layer boundaries. ([Docs.rs][3])

**MoE routing**

* Match Transformers’ contract: softmax‑after‑top‑K per token, then weighted sum of selected experts. Keep numerical clamps on router logits before softmax. ([Hugging Face][2])
* Candle’s Mixtral code shows a CPU fallback for top‑K selection and per‑expert partitioning; mirror that structure first, then optimize. ([GitHub][4])

**RoPE + YARN**

* Implement YARN scaling exactly (factor, betas, truncate, original max pos). Test rotary on tiny tensors vs. PyTorch to catch angle drift. Background: GPT‑OSS uses YARN for long context. ([cameronrwolfe.substack.com][5])

**Shapes and heads**

* Do not assume `head_dim = hidden_size/num_heads`. Project `hidden → (n_heads * head_dim)` and back.
* GQA/MQA: expand K/V heads to Q heads by view + repeat, not copy.
* Candle FA ships kernels for common head dims including 64. GPT‑OSS head_dim=64 is supported. ([GitHub][6])

**KV cache**

* One cache per layer. Trim to `sliding_window` on local layers. Cap by `max_position_embeddings` on full layers. Validate “decode one‑by‑one” equals “full pass” on short sequences.

**DTypes and numerics**

* Use BF16 for weights and activations where possible. Keep softmax, LSE, router softmax in FP32 to avoid denorm issues.
* RMSNorm ε and scaling must match HF defaults to hit parity.

**Weights and names**

* Map HF safetensors by exact names. Prefer a one‑time converter only if expert params are interleaved differently; otherwise load‑reshape at init.
* Keep LM head untied **if** config says so; do not hardcode tying.

**Blackwell (RTX Pro 6000)**

* Build kernels for your SM with `CUDA_COMPUTE_CAP=<cap>`. Ensure recent NVCC. If FA build fails, keep a clean non‑FA fallback path to validate correctness first. ([huggingface.github.io][7])

**Validation mindset**

* Equivalence tests first:

  1. Eager attention w/ sink vs. FA+LSE scaling on tiny shapes.
  2. Windowed FA vs. masked eager.
  3. Router+experts vs. Torch on random tiny nets.
  4. Rotary (base+YARN) vs. Torch on random Q/K.
* Only then run E2E tiny GPT‑OSS and compare logits for 2–3 prompts.

**Performance hygiene**

* Avoid reallocation in per‑token decode. Pre‑allocate attention scratch, router buffers, and per‑expert workspaces.
* Trim KV every step on local layers to bound memory.
* Expose a “no‑sink” toggle for A/B on throughput vs. parity.

**API integration**

* Add new `(o, lse)` FA wrappers instead of changing existing signatures to keep backward compatibility.
* Mirror existing `Model`/`Config` patterns in Candle to match LLaMA/Mistral ergonomics and generation API. ([Docs.rs][3])

**Spec references**

* Transformers’ GPT‑OSS docs flag special handling for sinks under flex/flash attention. Align behavior and tests to that spec. ([Hugging Face][8])

Follow these and your implementation will be simple, verifiable, and consistent with Candle’s design and GPT‑OSS’s reference behavior.

[1]: https://docs.rs/crate/candle-flash-attn/latest/source/src/ffi.rs "candle-flash-attn 0.9.1 - Docs.rs"
[2]: https://huggingface.co/blog/welcome-openai-gpt-oss?utm_source=chatgpt.com "GPT OSS, the new open-source model family from OpenAI!"
[3]: https://docs.rs/candle-transformers/latest/candle_transformers/all.html?utm_source=chatgpt.com "List of all items in this crate"
[4]: https://github.com/huggingface/candle/issues/1971?utm_source=chatgpt.com "How to use `topk`? · Issue #1971 · huggingface/candle"
[5]: https://cameronrwolfe.substack.com/p/gpt-oss?utm_source=chatgpt.com "GPT-oss from the Ground Up - by Cameron R. Wolfe, Ph.D."
[6]: https://github.com/huggingface/candle/issues/1936?utm_source=chatgpt.com "Flash Attention not working on CUDA 12.1 #1936"
[7]: https://huggingface.github.io/candle/guide/installation.html?utm_source=chatgpt.com "Installation - Candle Documentation"
[8]: https://huggingface.co/docs/transformers/main/model_doc/gpt_oss?utm_source=chatgpt.com "GptOss"

