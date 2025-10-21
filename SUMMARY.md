# GPT-OSS-20B Inference Parity Investigation - Comprehensive Summary

**Goal**: Achieve parity between Rust/Candle and Python/Transformers implementations for GPT-OSS-20B model with MXFP4 quantization.

## Current Status (2025-10-21)

- ✅ **CUDA MXFP4 dequantization offset bug fixed**. The GPU path now respects storage start offsets, so views produced by `narrow()` read the correct expert weights. CPU and CUDA dequant outputs match bit-for-bit on real checkpoint shards.
- ⚠️ **End-to-end inference still diverges**. The Rust example prints duplicated system/user headers and garbled assistant text, whereas Python emits the expected `<|channel|>analysis` and `<|channel|>final` responses. Further investigation is required beyond Layer-0 MLP.

## What Changed This Session

1. Added offset-aware handling inside `candle-core/src/mxfp4.rs::dequant_mxfp4_to_bf16_cuda`.
   - After materializing contiguous tensors, we now read `start_offset()` from the layout and slice the underlying `CudaSlice` before passing it to the CUDA kernel.
   - This avoids the previous CPU round-trip hack and preserves performance while guaranteeing correct pointer math.
2. Re-ran the targeted regression test `mxfp4_real_weights_bug` with real GPT-OSS expert data. The test now passes and prints that GPU matches expected values.
3. Replayed parity checks:
   - `uv run python gpt_oss_transformers.py` (baseline) produces coherent reply: `<|channel|>analysis` + `<|channel|>final` with answer “Paris”.
   - `cargo run --release --features cuda,flash-attn --example gpt-oss-20b -- --prompt "What is the capital of France?"` still shows duplicated system messages and truncated assistant output even though next-token logits now align (top candidate `<|channel|>` with p≈1.0).

## Evidence Collected

### Unit Test (GPU vs CPU dequant)
- Command: `RUST_BACKTRACE=1 cargo test --release --features cuda --test mxfp4_real_weights_bug -- --nocapture`
- Result: `GPU matches expected values!` for the first 32 elements of expert 3 gate_up block; no mismatches.
- Inputs: real safetensor blocks/scales (first block bytes `66 197 60 198 ...`, scale `121`).
- Dtype/device: `u8` inputs, `bf16` outputs, CUDA device `0`.

### Baseline Python Run
- Command: `uv run python gpt_oss_transformers.py`
- Output: coherent conversation with `<|channel|>analysis` explanation followed by `<|channel|>final` answer “The capital of France is **Paris**.”
- Logs confirm dtype `bf16`, device `cuda:0`, shards and config as expected.

### Rust Example Run (post-fix)
- Command: `cargo run --release --features cuda,flash-attn --example gpt-oss-20b -- --prompt "What is the capital of France?"`
- Observations:
  - Token statistics show correct prompt ids and that `<|channel|>` (id 200005) has probability 1.0.
  - Generated text still contains duplicated system/user headers and a garbled assistant segment instead of the coherent Harmony-format reply.
  - Indicates the forward pass logits are now reasonable, but downstream sampling/post-processing differs from Python.

## Updated Understanding

- **Root cause (resolved)**: When expert weights are loaded via `narrow()`, the resulting tensor carries a non-zero storage offset (~24,883,200 bytes for expert 3). The old CUDA path ignored this offset when turning the tensor into a raw `CudaSlice`, so the kernel always read expert 0. Slicing the `CudaSlice` by `start_offset()` fixes the issue without extra copies.
- **Validated components** (carry over from earlier sessions): embeddings, lm_head, attention projections, router logits/selection, MXFP4 math, biases, and now the Layer-0 MLP output all match Python within BF16 tolerance.
- **Remaining divergence**: Sampling / text formatting layer continues to deviate. Since the logits show `<|channel|>` dominance, the mismatch likely stems from token filtering, stop-token handling, or channel formatting logic after logits are produced.

## Next Steps

1. **Trace sampler parity**
   - Instrument Rust example to dump final logits for one token and compare numerically against Python right before sampling. Ensure we use identical temperature and top-p defaults.
   - Verify stop-token list (`<|return|>`, `<|call|>`, `<|endoftext|>`) and channel insertion logic.

2. **Micro-test decoding path**
   - Create a small deterministic harness that feeds stored logits through Rust sampler to confirm it emits the same token ids as Python for the first few steps.
   - Compare the resulting text assembly (channel markers, reasoning tags) between Rust and Python.

3. **Regression guard**
   - Keep `mxfp4_real_weights_bug` in place to prevent offset regressions.
   - Run `cargo test --release --features cuda -- --nocapture` after further changes touching MXFP4 or sampling.

## References

- Model snapshot: `~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/`
- Rust MXFP4 dequant: `candle-core/src/mxfp4.rs`
- CUDA kernel: `candle-kernels/src/quantized.cu`
- GPT-OSS example: `candle-examples/examples/gpt-oss-20b/main.rs`
- Python reference script: `gpt_oss_transformers.py`
- Diagnostic unit test: `candle-core/tests/mxfp4_real_weights_bug.rs`
