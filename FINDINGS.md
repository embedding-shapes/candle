# MLP Output Divergence Root Cause Analysis

## Summary
The Rust MLP implementation produces completely incorrect outputs compared to Python. The divergence is NOT in the router, attention, or embeddings - it's specifically in the expert forward computation.

## Evidence

### MLP Input (MATCHES ✓)
- Python: `[-0.09375, 0.51953125, 0.345703125, -0.2275390625, -0.67578125, 0.41796875, -0.125, -0.625]`
- Rust: `[-0.09375, 0.51953125, 0.34570313, -0.23242188, -0.6796875, 0.41796875, -0.12695313, -0.625]`
- **Status:** Within BF16 precision tolerance

### Router Output (MATCHES ✓)
- Experts selected: `[3, 30, 11, 9]` (both Python and Rust)
- Routing weights: `[0.43, 0.23, 0.18, 0.16]` (both Python and Rust)
- **Status:** Identical

### Bias Loading (VERIFIED ✓)
- Expert 3 gate_up_proj_bias: `[-0.486, -0.848, -0.539, ...]`
- Expert 3 down_proj_bias: `[0.074, -0.221, -0.106, ...]`
- **Status:** Rust loads correct values from safetensors

### MLP Output (BROKEN ✗)
Last token output at index [5]:
- Python: **+2.468750**
- Rust: **-5.500000**
- **Difference: 7.97** (opposite sign, completely wrong)

Full comparison:
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

### Individual Expert Outputs (Rust)
All experts produce negative values at index [5]:
```
Expert  3: -5.00 (before weight) → -2.16 (after weight)
Expert  9: -7.53 (before weight) → -1.20 (after weight)
Expert 11: -7.06 (before weight) → -1.28 (after weight)
Expert 30: -3.88 (before weight) → -0.89 (after weight)
Weighted sum: -5.5
```

But Python expects positive values that sum to **+2.47**.

## Hypothesis

The problem is in the **expert forward computation** (gate_up → SwiGLU → down). Possible causes:

1. **MXFP4 Dequantization Bug**
   - The dequantization produces wrong float values
   - Check: Compare dequantized weight values element-by-element with Python

2. **Weight Layout/Transpose Bug**
   - After dequantization, weights might be in wrong memory layout
   - Python: `[hidden, out]` = `[2880, 5760]`
   - Rust: `[out, in]` = `[5760, 2880]` (correct for candle::Linear)
   - But: Maybe the actual tensor data is wrong?

3. **Matrix Multiply Orientation Bug**
   - The matmul might be computing `weight @ input` instead of `input @ weight.T`

4. **SwiGLU Computation Bug**
   - Gate/up splitting is verified correct (interleaved)
   - But maybe the GLU formula is wrong?
   - Formula: `(up + 1) * (gate * sigmoid(gate * alpha))`
   - Alpha: 1.702, Limit: 7.0

## Next Debugging Steps

### Step 1: Verify MXFP4 Dequantization
Create test that:
1. Loads expert 3 gate_up MXFP4 weights in Python
2. Dequantizes using transformers library
3. Loads same weights in Rust
4. Dequantizes using candle
5. Compares first 100 float values element-by-element

**Expected:** If dequant is correct, values should match within FP precision
**If fails:** Fix MXFP4 dequant implementation in Rust

### Step 2: Test Matrix Multiply
If dequant is correct:
1. Use known input vector (e.g., all ones)
2. Multiply with expert 3 gate_up weights in both Python and Rust
3. Compare outputs

**Expected:** Should match
**If fails:** Check matmul orientation or tensor layout

### Step 3: Test Full Expert Forward
If matmul is correct:
1. Use the actual MLP input from last token
2. Run through expert 3 forward in both Python and Rust
3. Print intermediate values (gate, up, glu, down_input, down_output)
4. Compare at each step

**Expected:** Should match
**If fails:** Identify exact operation that diverges

## Commands for Next Session

```bash
# Python: Extract expert 3 dequantized weights
uv run python -c "
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained('~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee', torch_dtype=torch.bfloat16, device_map='cpu')
expert3_gate_up = model.model.layers[0].mlp.experts.gate_up_proj[3]
print('Shape:', expert3_gate_up.shape)
print('First 16 values:', expert3_gate_up[0, :16].tolist())
torch.save(expert3_gate_up, '/tmp/expert3_gate_up.pt')
"

# Rust: Print dequantized expert 3 weights
# Add debug output in load_expert_linear_mxfp4_grouped after line 1686
```

## Files Referenced
- `/tmp/test_mlp_final.py` - Shows MLP input/output mismatch
- `/tmp/test_expert3_weights.py` - Shows MXFP4 raw data
- `SUMMARY.md` - Previous investigation results
- `candle-transformers/src/models/gpt_oss.rs:1620-1714` - MXFP4 loading code
- `candle-transformers/src/models/gpt_oss.rs:111-167` - Expert forward code

## Conclusion

**The bug is in the expert forward computation, NOT in routing or infrastructure.**

The most likely cause is wrong MXFP4 dequantized values. Need to verify dequant correctness as the absolute next step.
