# GPT-OSS-20B Inference Parity Investigation - Comprehensive Summary

**Goal**: Achieve parity between Rust/Candle and Python/Transformers implementations for GPT-OSS-20B model with MXFP4 quantization.

## Current Status (CRITICAL BUG IDENTIFIED)

**Problem**: Rust implementation generates **incoherent output** (random tokens) while Python generates **coherent responses**.

**ROOT CAUSE FOUND**: CUDA MXFP4 dequantization reads from wrong memory offset after `narrow()` operations, causing GPU to read Expert 0's data instead of the requested expert's data.

### Observed Behavior

**Python output** (correct):
```
Top prediction: <|channel|> (id=200005) with p=1.0000
Response: <|channel|>analysis<|message|>User asks "What is the capital of France?" ...
Final answer: The capital of France is Paris.
```

**Rust output** (wrong):
```
Top prediction: , (id=11) with p=0.5859
<|channel|> (id=200005) has p=0.000000
Response: , ( else, [, j, [,Leftrich -bits won....
```

The outputs diverge completely - Python predicts `<|channel|>` with near certainty, Rust assigns it near-zero probability.

## Investigation Timeline & Key Findings

### ✅ Verified Correct Components

1. **Embeddings** - Match exactly
2. **lm_head weights** - Match exactly
3. **Layer-0 input_layernorm output** - Matches within BF16 precision
   - Python: `[-0.09375, 0.51953125, 0.345703125, ...]`
   - Rust: `[-0.09375, 0.51953125, 0.34570313, ...]`
4. **Layer-0 attention (Q/K/V/O projections)** - Match within BF16 precision
5. **Layer-0 post-attention-residual** - Matches
6. **Layer-0 post-attention-norm** (MLP input) - Matches within BF16 precision
7. **Router logits** - Match within BF16 precision
   - Python: `1.7734375`, Rust: `1.78125` (difference ~0.008)
   - Earlier confusion: Python's router returns softmax probabilities, not raw logits
8. **Router expert selection** - Correct: experts [3, 30, 11, 9]
   - Routing weights: `[0.43, 0.23, 0.18, 0.16]` (both Python and Rust)
9. **MXFP4 dequantization math** - Spec-compliant (F4E2M1 + E8M0)
10. **Gate/up interleaving** - Implemented correctly (even=gate, odd=up)
11. **Asymmetric clamping** - Correct (gate: max-only at 7.0, up: min-max at ±7.0)
12. **Bias loading** - Verified correct
    - Expert 3 gate_up_proj_bias: `[-0.486, -0.848, -0.539, ...]`
    - Expert 3 down_proj_bias: `[0.074, -0.221, -0.106, ...]`

### ❌ Divergence Point: MLP Output

Layer-0 MLP output for last token shows **significant mismatch**:
- Python: `[0.334, -0.316, 0.046, -0.289, -0.387, 2.469, -0.052, -0.135]`
- Rust: `[0.330, -0.336, -0.187, -0.213, -0.828, -5.5, -0.0005, -0.055]`

This cascades through all 24 layers, resulting in completely wrong final logits.

Full comparison at index positions:
```
[0] Python:  0.334, Rust:  0.330 (diff: 0.004) ✓
[1] Python: -0.316, Rust: -0.336 (diff: 0.020) ✓
[2] Python:  0.046, Rust: -0.187 (diff: 0.232) ✗
[3] Python: -0.289, Rust: -0.213 (diff: 0.076) ~
[4] Python: -0.387, Rust: -0.828 (diff: 0.441) ✗
[5] Python:  2.469, Rust: -5.500 (diff: 7.969) ✗✗✗
[6] Python: -0.052, Rust: -0.001 (diff: 0.051) ~
[7] Python: -0.135, Rust: -0.055 (diff: 0.079) ~
```

Individual Expert Outputs (Rust) - all produce negative values at index [5]:
```
Expert  3: -5.00 (before weight) → -2.16 (after weight)
Expert  9: -7.53 (before weight) → -1.20 (after weight)
Expert 11: -7.06 (before weight) → -1.28 (after weight)
Expert 30: -3.88 (before weight) → -0.89 (after weight)
Weighted sum: -5.5
```

But Python expects positive values that sum to **+2.47**.

## 🔍 MXFP4 Weight Layout Analysis

### Python Transformers Implementation
- **Storage**: safetensors has blocks `[E, out=5760, nb=90, 16]` and scales `[E, out, nb]`
- **Process**: Reshapes blocks to `[E, out, nb*16]`, then transposes to `[E, nb*16, out]`
- **Swizzle**: Applies hardware-specific layout transformations
- **Result**: Sets shape to `[E, in=2880, out=5760]`
- **Usage**: `input @ weight` (no transpose)
- **Python function**: `convert_moe_packed_tensors()` in `py-transformers/src/transformers/integrations/mxfp4.py`

### OpenAI Reference PyTorch Implementation
- **Storage**: Same MXFP4 format in safetensors
- **Process**: Dequantizes directly without transpose
- **Result**: Weights are `[E, out=5760, in=2880]`
- **Usage**: `torch.einsum("beck,bk->bec", weight, input)` which computes `weight[out,in] @ input[in]`
- **Location**: `./openai-gpt-oss/`

### Current Rust Implementation
- **Process**: Dequantizes directly without transpose
- **Result**: Weights are `[out=5760, in=2880]` per expert
- **Usage**: `candle::Linear` which does `input @ weight.T`
- **Equivalence**: `input[batch,in] @ weight.T` = `input[batch,in] @ [in,out]` = `[batch,out]` ✓
- **Location**: `candle-transformers/src/models/gpt_oss.rs:1620-1714`

### Analysis Conclusion
All three approaches are **mathematically equivalent**:
- Python transformers: `[batch,in] @ [in,out]`
- OpenAI reference: `[out,in] @ [in]` via einsum
- Rust candle: `[batch,in] @ [out,in].T` = `[batch,in] @ [in,out]`

**The weight layout is NOT the problem.**

## 🐛 ROOT CAUSE: CUDA MXFP4 Dequantization Bug (IDENTIFIED 2025-10-21)

### The Bug

The GPU CUDA kernel is **reading from the wrong memory offset** when tensors have been created via `narrow()` operations.

### Detailed Investigation

**Step 1: Layout Hypothesis Confirmed**
- Created test comparing Python's `convert_moe_packed_tensors()` output vs Rust simulation
- Test file: `/tmp/test_layout_hypothesis.py`
- Result: `torch.allclose(W_rs.T, W_py)` = **True** ✓
- Conclusion: Layout is mathematically correct when dequantization works properly

**Step 2: CPU vs CUDA Dequantization Comparison**
- Created standalone test: `/tmp/test_weights/src/main.rs`
- Loaded expert 3 gate_up blocks and scales from safetensors
- Raw data verified:
  - First block bytes: `[66, 197, 60, 198, 140, 69, 17, 237, 137, 198, 9, 1, 149, 4, 176, 37]`
  - First scale: `121` (E8M0 format → `2^(121-127) = 0.015625`)

**Expected values** (manually computed from F4E2M1 + E8M0 spec):
```
[0.015625, 0.03125, 0.046875, -0.03125, -0.03125, 0.0234375, 0.0625, -0.03125,
 -0.03125, -0.0, 0.046875, 0.03125, 0.0078125, 0.0078125, -0.046875, -0.0625,
 -0.0078125, -0.0, 0.0625, -0.03125, -0.0078125, 0.0, 0.0078125, 0.0,
 0.046875, -0.0078125, 0.03125, 0.0, 0.0, -0.0234375, 0.046875, 0.015625]
```

**CPU dequant output** (CORRECT):
```
[0.015625, 0.03125, 0.046875, -0.03125, -0.03125, 0.0234375, 0.0625, -0.03125, ...]
```

**GPU dequant output** (WRONG):
```
[0.0, 0.0, 0.0, -0.0625, 0.0, -0.0, -0.015625, -0.03125, 0.0, 0.015625, ...]
```

**Step 3: Identified GPU Reading Wrong Expert**
- Created verification test: `/tmp/verify_expert0_hypothesis.py`
- Loaded both Expert 0 and Expert 3 first block data
- **Expert 0** first block bytes: `[0, 192, 128, 169, 16, 27, 129, 34, 147, 228, 160, 178, 59, 25, 178, 49]`
- **Expert 0** dequantized: `[0.0, 0.0, 0.0, -0.0625, 0.0, -0.0, -0.015625, -0.03125, ...]`
- **Expert 3** dequantized: `[0.015625, 0.03125, 0.046875, -0.03125, ...]`

**CRITICAL FINDING**: GPU output **exactly matches Expert 0**, not Expert 3!

### Root Cause Analysis

The bug occurs in `candle-core/src/mxfp4.rs` function `dequant_mxfp4_to_bf16_cuda()`:

1. When loading expert weights, code does: `blocks_all.narrow(0, 3, 1)?.squeeze(0)?`
   - This creates a **view** with storage offset = `3 * 5760 * 90 * 16 = 24,883,200` bytes
2. `contiguous()` is called, but **doesn't copy** if tensor is already contiguous
   - The tensor remains a view with the offset
3. `to_device(&cuda)` transfers to GPU, maintaining the view/offset
4. **BUG**: `as_cuda_slice()` returns a CUDA pointer but doesn't account for the storage offset
5. CUDA kernel reads from index 0 of the pointer, which is Expert 0's data, not Expert 3's data

**Evidence**: Debug output shows `blocks offset: 24883200` but CUDA still reads wrong data.

### Unit Test Created

**Location**: `candle-core/tests/mxfp4_real_weights_bug.rs`

**Test demonstrates**:
- ✓ CPU dequant matches expected values (all 32 correct)
- ✗ GPU dequant has 30 mismatches out of 32 values
- GPU output exactly matches Expert 0 when it should match Expert 3

**Run test**:
```bash
RUST_BACKTRACE=1 cargo test --release --features cuda --test mxfp4_real_weights_bug -- --nocapture
```

**Current test status**: FAILING (as expected - demonstrates the bug)

## 🔧 Fix Attempted (In Progress)

**Location**: `candle-core/src/mxfp4.rs` lines 151-180

**Fix approach**: Force a real copy when storage offset is non-zero
```rust
let blocks_offset = {
    let (_blocks_storage, blocks_layout) = blocks_c.storage_and_layout();
    blocks_layout.start_offset()
};
let blocks_c = if blocks_offset > 0 {
    // Force a copy by converting to CPU and back to remove offset
    blocks_c.to_device(&Device::Cpu)?.to_device(blocks_c.device())?
} else {
    blocks_c
};
```

**Status**: Fix detects offset correctly (prints "blocks offset: 24883200, Forcing copy due to non-zero offset") but test still fails. The CPU-GPU-CPU roundtrip may not be properly removing the offset. Need to investigate alternative fix approaches.

## Files and Locations

### Model
- **Path**: `~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/`
- **Files**: `model-00000-of-00002.safetensors`, `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`
- **Config**: hidden_size=2880, layers=24, heads=64, kv_heads=8, head_dim=64

### Source Code
- **Rust MXFP4 dequant**: `candle-core/src/mxfp4.rs`
- **CUDA kernel**: `candle-kernels/src/quantized.cu:2551-2581`
- **GPT-OSS model**: `candle-transformers/src/models/gpt_oss.rs`
- **Expert loading**: `candle-transformers/src/models/gpt_oss.rs:1620-1714`
- **Expert forward**: `candle-transformers/src/models/gpt_oss.rs:111-167`

### Python References
- **Transformers MXFP4**: `py-transformers/src/transformers/integrations/mxfp4.py`
  - Key function: `convert_moe_packed_tensors()` lines 100-151
  - F4 lookup table: line 31-48
- **OpenAI reference**: `./openai-gpt-oss/`
- **Test implementation**: `gpt_oss_transformers.py`

### Test Files
- **Unit test (NEW)**: `candle-core/tests/mxfp4_real_weights_bug.rs` - Reproduces CUDA bug
- **Existing CUDA tests**: `candle-core/tests/mxfp4_cuda.rs` - Pass with synthetic data
- **Python helpers**: `/tmp/test_layout_hypothesis.py`, `/tmp/verify_expert0_hypothesis.py`
- Many other diagnostic scripts in `/tmp/*.py`

## Commands Reference

### Reproduce the Issue
```bash
# Python (correct output)
uv run python gpt_oss_transformers.py

# Rust (wrong output - before fix)
cargo run --release --features cuda,flash-attn --example gpt-oss-20b \
  --prompt "What is the capital of France?"

# Rust with debug output
CANDLE_DUMP_L1=1 cargo run --release --features cuda,flash-attn --example gpt-oss-20b \
  --prompt "What is the capital of France?" --sample-len 1
```

### Run Unit Test (Demonstrates Bug)
```bash
# This test currently FAILS - it demonstrates the bug
RUST_BACKTRACE=1 cargo test --release --features cuda --test mxfp4_real_weights_bug -- --nocapture
```

### Other Tests
```bash
# Existing CUDA tests (these PASS with synthetic data)
cargo test --release --features cuda --test mxfp4_cuda -- --nocapture

# F4 lookup table test
cargo test --release --features cuda --test t_mxfp4_fp4_lut -- --nocapture
```

## Where to Resume

### Immediate Next Steps

1. **Fix the CUDA pointer offset issue** (CURRENT PRIORITY)
   - The current fix attempt (CPU-GPU-CPU roundtrip) doesn't work
   - Need to investigate why `as_cuda_slice()` doesn't respect storage offset
   - Alternative approaches:
     a) Use `slice()` method on CudaSlice to offset the pointer
     b) Pass offset to CUDA kernel and adjust indexing
     c) Force actual memory copy via different method

2. **Verify fix with unit test**
   ```bash
   cargo test --release --features cuda --test mxfp4_real_weights_bug -- --nocapture
   ```
   - Test should show: "✓ GPU matches expected values!"

3. **Run full model inference**
   ```bash
   cargo run --release --features cuda,flash-attn --example gpt-oss-20b \
     --prompt "What is the capital of France?"
   ```
   - Should produce coherent output matching Python

4. **Run full test suite**
   ```bash
   cargo test --release --features cuda -- --nocapture
   ```

### Critical Observations

- The CUDA kernel itself is **correct** - it passes tests with synthetic data
- The CPU dequantization is **correct** - it produces expected values
- The bug is specifically in how **CUDA slice pointers are obtained from tensor views**
- This affects any MXFP4 tensor created via `narrow()`, which is how expert weights are loaded
- The offset is detected (24,883,200 bytes) but not properly handled

## Important Notes

- **DO NOT run `cargo clean`** - flash-attn takes 5+ minutes to rebuild
- **Always use `uv run python`** to respect .venv
- Test prompt: "What is the capital of France?"
- The bug affects ALL experts when loaded via grouped tensors with narrow()
- CPU workaround exists: `CANDLE_DEQUANT_ON_CPU=1` env var forces CPU dequant (slow but works)
