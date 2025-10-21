# GPT-OSS-20B Inference Parity Investigation Summary

**Goal**: Achieve parity between Rust/Candle and Python/Transformers implementations for GPT-OSS-20B model with MXFP4 quantization.

## Current Status

**Problem**: Rust implementation generates **incoherent output** (random tokens) while Python generates **coherent responses**.

### Observed Behavior

**Python output** (correct):
```
Top prediction: <|channel|> (id=200005) with p=1.0000
Response: <|channel|>analysis<|message|>The user asks: "What is the capital of France?"
This is a straightforward question. Provide answer: Paris.<|end|>...
```

**Rust output** (wrong):
```
Top prediction: , (id=11) with p=0.5859
<|channel|> (id=200005) has p=0.000000
Response: , ( else, [, j, [,Leftrich -bits won....g*,, do. SApp P, (α.:, split I...
```

The model predicts completely different next tokens, indicating **final logits are wrong**.

## Key Findings from Latest Investigation

### ✅ Router Logits Are CORRECT

**Discovery**: Router logits between Rust and Python **DO match**!
- Rust raw router logits for last token Expert 3: `1.78125`
- Python raw router logits for last token Expert 3: `1.7734375`
- Difference: ~0.008 (within BF16 precision)

**Confusion resolved**: Earlier tests showed Python router returning `0.427734375` for Expert 3, which was actually the **softmax probability after top-k selection**, NOT the raw logit. The Python `GptOssTopKRouter.forward()` returns a sparse tensor with softmax probabilities, not raw logits.

```python
# Python router behavior:
router_logits = F.linear(hidden_states, self.weight, self.bias)  # Raw logits
router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
router_top_value = torch.nn.functional.softmax(router_top_value, dim=1)  # Softmax on top-k
router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)  # Sparse
return router_scores, router_indices  # Returns softmax probs, not logits!
```

### ✅ MLP Input Matches

Layer-0 post-attention-norm output (MLP input) for last token:
- Python: `[-0.09375, 0.51953125, 0.345703125, -0.2275390625, -0.67578125, 0.41796875, -0.125, -0.625]`
- Rust: `[-0.09375, 0.51953125, 0.34570313, -0.23242188, -0.6796875, 0.41796875, -0.12695313, -0.625]`
- **Match within BF16 precision** ✅

### ❌ MLP Output Does NOT Match

Layer-0 MLP output for last token (from `/tmp/simple_mlp_test.py`):
- Python: `[0.334, -0.316, 0.046, -0.289, -0.387, 2.469, -0.052, -0.135]`
- Rust: `[0.330, -0.336, -0.187, -0.213, -0.828, -5.5, -0.0005, -0.055]`
- **Significant mismatch** ❌

This MLP output mismatch cascades through all remaining layers, causing wrong final logits and incoherent generation.

### ❌ Final Hidden State Does NOT Match

Rust debug output shows:
```
Hidden (after final norm) last token [:8]: [3.4375, -0.087402344, 10.0, 1.4765625, -6.59375, -3.609375, -1.453125, -10.375]
```

This needs to be compared with Python's final hidden state.

## Verified Correct Components

1. ✅ Embeddings match exactly
2. ✅ lm_head weights match exactly
3. ✅ Layer-0 input_layernorm output matches
4. ✅ Layer-0 Q/K/V/O projections match (within BF16 precision)
5. ✅ Layer-0 post-attention-residual matches
6. ✅ Layer-0 post-attention-norm matches
7. ✅ **Router logits match** (clarified in this session)
8. ✅ Router expert selection correct: experts [3, 30, 11, 9]
9. ✅ MXFP4 dequantization math is spec-compliant
10. ✅ Gate/up interleaving is implemented correctly (even=gate, odd=up)
11. ✅ Asymmetric clamping: gate max-only, up min-max (both 7.0)

## Known Issues / Uncertainties

### Issue: Interleaved Split Didn't Change Output

When switching from halves-based split to interleaved split (even=gate, odd=up), the MLP output **remained exactly the same**. This is suspicious and suggests:
1. Either the weights themselves happen to work with both layouts (unlikely)
2. Or there's another issue masking the interleaving fix
3. Or the change wasn't actually applied correctly despite debug output confirming it

### Next Steps: Ground-Up Testing Strategy

Create small, isolated Rust/Python test pairs to validate each component:

1. **Test embeddings**: Verify token embeddings match for exact prompt
2. **Test layer-0 attention**: Verify attention output matches
3. **Test layer-0 MLP**:
   - Test router forward pass
   - Test single expert forward (Expert 3) with known input
   - Test gate_up projection
   - Test gate/up split
   - Test GLU computation
   - Test down projection
   - Test expert aggregation
4. **Test final hidden state**: Verify output after all 24 layers + final norm
5. **Test lm_head**: Verify logits computation

## Testing Framework

### Existing Tests
- `gpt_oss_step2_mxfp4_parity.rs` - MXFP4 dequantization parity
- `gpt_oss_step3_matmul.rs` - Matrix multiplication test
- `gpt_oss_step6_cpu_vs_gpu.rs` - index_select CPU vs GPU
- Many component tests in `candle-transformers/tests/`

### Python Test Scripts (in /tmp/)
- `simple_mlp_test.py` - Basic MLP input/output capture
- `test_routing_detail.py` - Router logits analysis
- `test_router_manual.py` - Manual router computation

## Commands for Testing

```bash
# Run Rust with debug output (single token generation)
CANDLE_DUMP_L1=1 cargo run --release --features cuda,flash-attn --example gpt-oss-20b \
  --prompt "What is the capital of France?" --sample-len 1

# Run Python reference (full generation)
uv run python gpt_oss_transformers.py

# Run parity tests
cargo test --release --features cuda --test <test_name> -- --nocapture
```

## Important Notes

- **DO NOT run `cargo clean`** - flash-attn takes 5+ minutes to recompile
- **Always use `uv run python`** to respect .venv
- **Test prompt**: "What is the capital of France?"
- **Model path**: `~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/`
- **Python model** uses MXFP4GptOssExperts which fuses all 32 experts into single tensors

## Focus Area

The divergence starts at **Layer-0 MLP output**. Since router logits and MLP input match, the issue must be in:
- Expert MLP forward computation (gate_up projection, split, GLU, down projection)
- Expert output aggregation
- Or weight loading/layout for expert MLP weights

Need to create focused tests comparing Rust vs Python for single expert forward pass with known inputs.
