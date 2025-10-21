# GPT-OSS-20B Inference Parity Investigation Summary

**Goal**: Achieve parity between Rust/Candle and Python/Transformers implementations for GPT-OSS-20B model with MXFP4 quantization.

## Current Status

**Problem**: Rust implementation generates **incoherent output** (random tokens) while Python generates **coherent responses**.

### Observed Behavior

**Python output** (correct):
```
Top prediction: <|channel|> (id=200005) with p=1.0000
Response: <|channel|>analysis<|message|>User asks "What is the capital of France?" ...
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
4. **Layer-0 attention (Q/K/V/O projections)** - Match within BF16 precision
5. **Layer-0 post-attention-residual** - Matches
6. **Layer-0 post-attention-norm** (MLP input) - Matches within BF16 precision
   - Python: `[-0.09375, 0.51953125, 0.345703125, ...]`
   - Rust: `[-0.09375, 0.51953125, 0.34570313, ...]`
7. **Router logits** - Match within BF16 precision
   - Python: `1.7734375`, Rust: `1.78125` (difference ~0.008)
   - Earlier confusion: Python's router returns softmax probabilities, not raw logits
8. **Router expert selection** - Correct: experts [3, 30, 11, 9]
9. **MXFP4 dequantization math** - Spec-compliant (F4E2M1 + E8M0)
10. **Gate/up interleaving** - Implemented correctly (even=gate, odd=up)
11. **Asymmetric clamping** - Correct (gate: max-only at 7.0, up: min-max at ±7.0)

### ❌ Divergence Point: MLP Output

Layer-0 MLP output for last token shows **significant mismatch**:
- Python: `[0.334, -0.316, 0.046, -0.289, -0.387, 2.469, -0.052, -0.135]`
- Rust: `[0.330, -0.336, -0.187, -0.213, -0.828, -5.5, -0.0005, -0.055]`

This cascades through all 24 layers, resulting in completely wrong final logits.

### 🔍 MXFP4 Weight Layout Analysis (2025-10-21 Session)

#### Python Transformers Implementation
- **Storage**: safetensors has blocks `[E, out=5760, nb=90, 16]` and scales `[E, out, nb]`
- **Process**: Reshapes blocks to `[E, out, nb*16]`, then transposes to `[E, nb*16, out]`
- **Swizzle**: Applies hardware-specific layout transformations
- **Result**: Sets shape to `[E, in=2880, out=5760]`
- **Usage**: `input @ weight` (no transpose)

#### OpenAI Reference PyTorch Implementation
- **Storage**: Same MXFP4 format in safetensors
- **Process**: Dequantizes directly without transpose
- **Result**: Weights are `[E, out=5760, in=2880]`
- **Usage**: `torch.einsum("beck,bk->bec", weight, input)` which computes `weight[out,in] @ input[in]`

#### Current Rust Implementation
- **Process**: Dequantizes directly without transpose
- **Result**: Weights are `[out=5760, in=2880]` per expert
- **Usage**: `candle::Linear` which does `input @ weight.T`
- **Equivalence**: `input[batch,in] @ weight.T` = `input[batch,in] @ [in,out]` = `[batch,out]` ✓

#### Analysis Conclusion
All three approaches are **mathematically equivalent**:
- Python transformers: `[batch,in] @ [in,out]`
- OpenAI reference: `[out,in] @ [in]` via einsum
- Rust candle: `[batch,in] @ [out,in].T` = `[batch,in] @ [in,out]`

**The weight layout is NOT the problem.**

### 🔴 Critical Observation: Suspicious Behavior

When interleaved split was implemented (fixing from halves to even/odd indices), **the MLP output remained exactly the same**. This suggests:
1. The fix didn't actually take effect, OR
2. There's a more fundamental bug masking the fix, OR
3. Some other component is wrong

### 📊 Current Divergence Magnitude

Final hidden states show massive difference:
- Python gives `<|channel|>` probability = **1.0000**
- Rust gives `<|channel|>` probability = **0.000000**

This is not a minor numerical difference - it's a fundamental computation error.

## References

### OpenAI Official Repository
Located in `./openai-gpt-oss/` - contains reference PyTorch implementation:
- `gpt_oss/torch/model.py` - Model architecture
- `gpt_oss/torch/weights.py` - MXFP4 dequantization reference
- Confirms weight layout: `[num_experts, out, in]` with einsum usage

### Test Scripts
Located in `/tmp/`:
- `simple_mlp_test.py` - Captures MLP input/output
- `test_routing_detail.py` - Router analysis
- `test_mxfp4_dequant_layout.py` - Layout verification
- `test_actual_mxfp4_values.py` - Direct weight comparison
- Many others for component testing

## Next Steps

The bug is **NOT** in:
- MXFP4 weight layout/transpose
- Router computation
- Attention computation
- Embeddings or lm_head

The bug **IS** in the MLP forward computation somewhere. Need to create a **minimal focused parity test**:

1. Load identical input tensor in Python and Rust
2. Run through layer-0 MLP expert-3 ONLY
3. Compare gate_up output, GLU output, down output
4. Identify exact operation that diverges

## Commands

```bash
# Rust with debug
CANDLE_DUMP_L1=1 cargo run --release --features cuda,flash-attn --example gpt-oss-20b \
  --prompt "What is the capital of France?" --sample-len 1

# Python reference
uv run python gpt_oss_transformers.py

# Component tests
cargo test --release --features cuda --test <test_name> -- --nocapture
```

## Important Notes

- **DO NOT run `cargo clean`** - flash-attn takes 5+ minutes to rebuild
- **Always use `uv run python`** to respect .venv
- Model: `~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/`
- Test prompt: "What is the capital of France?"
