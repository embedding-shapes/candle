You need four things correct: on-disk layout, FP4 math, dequant/GEMM strategy, and integration points in Candle.

# 1) MXFP4 spec you must implement

* **Blockwise microscaling.** Values are stored as FP4 E2M1 elements in blocks of **k=32**, with one shared **E8M0** scale per block. Real value: `v_i = X * p_i`, where `X` is the block scale and `p_i` decodes the 4-bit FP4. The spec fixes `k=32`, element type FP4(E2M1), scale type E8M0. 
* **Decode rules.** For FP4(E2M1): if `E>0`, normal value is `(-1)^S * 2^(E-bias) * (1 + M * 2^-1)`. If `E=0`, subnormal is `(-1)^S * 2^(1-bias) * (M * 2^-1)`. Use the spec’s bias and rounding; handle NaN/Inf as defined. Scale E8M0 is a signed power-of-two exponent (no mantissa). 
* **Layout in GPT-OSS weights.** OpenAI stores each MXFP4 tensor as two arrays:

  * `<name>.blocks`: packed FP4, **two nibbles per byte**.
  * `<name>.scales`: block scales, blockwise over the **last dimension**.
    All non-MXFP4 tensors remain BF16. ([GitHub][1])

# 2) Where GPT-OSS uses MXFP4

* **Scope.** Only MoE expert weights use MXFP4. Attention, router, embeddings, and LM head stay BF16. This keeps accuracy while making 20B fit small GPUs; 120B fits 80 GB. ([Hugging Face][2])

# 3) CUDA strategy for Candle (phased)

**Phase A — minimal, correct, fast enough**

* **Dequant-on-load to BF16 (one-time).** Write a CUDA kernel that reads `blocks+scales`, decodes FP4(E2M1) to FP32, multiplies by `pow2(scale_e8m0)`, rounds to **BF16**, writes a contiguous weight tensor. Use shared-mem staging per 32-elem block; vectorize on bytes and use warp-wide bit ops. This avoids changing matmul paths and works on all SMs. Trade-off: 4× memory vs. MXFP4 in VRAM. ([Hugging Face][3])
* **CPU fallback path.** Provide a Rust decode for tests and CPU builds using lookup tables for FP4→FP32 and exponent→pow2 to avoid branches.

**Phase B — memory-efficient**

* **Tile dequant in GEMM.** Implement a custom GEMM for experts that, per tile:

  1. loads `scales` for the tile’s columns,
  2. loads packed FP4, unpacks to int4,
  3. table-decodes to FP32, multiplies by `pow2(scale)`, casts to FP16/BF16 **in registers**,
  4. feeds Tensor Cores via FP16/BF16 MMA.
     This keeps weights MXFP4 in VRAM. There is **no general Tensor Core MMA for MXFP4**, so you still do MMA in FP16/BF16 after dequant. NV’s **NVFP4** has hardware-aligned paths, but MXFP4 does not; plan accordingly. ([NVIDIA Developer][4])

# 4) Implementation details that bite

* **Packing/endianness.** Define: low nibble = column `2j`, high nibble = column `2j+1` along the **last dim**. Match HF reference; add a test that reconstructs a small known tensor from bytes. ([GitHub][1])
* **Block mapping.** Compute the block index as `col // 32` on the last dimension. Scales tensor shape must broadcast over the same last-dim blocking.
* **Zero and special values.** Implement FP4 subnormals. Preserve NaN propagation rules from the MX spec. Don’t silently clamp unless the spec allows. 
* **Rounding.** Use round-to-nearest-even when FP32→BF16 during dequant. The MX spec assumes IEEE754 semantics. 
* **Alignment/strides.** After dequant, persist weights in row-major contiguous layout expected by Candle’s Linear. If GPT-OSS experts are column-major in HF, transpose at load once.
* **Operator placement.** Only experts need MXFP4; keep router/attn/embeddings on existing Candle ops. That limits custom kernels to MoE matmuls.
* **Blackwell.** MXFP4 itself does not require a special SM capability. Your kernels should target SM90+ and SM100 with standard HMMA FP16/BF16. If you later add NVFP4, that is a different format and kernel path. ([NVIDIA Developer][4])

# 5) Candle code changes

* **safetensors loader:**

  * Detect paired `*.blocks` and `*.scales`.
  * Validate last-dim block alignment (size multiple of 32).
  * Expose an enum `QuantizedWeight::MXFP4 { blocks: TensorU8, scales: TensorU8, shape: [..] }`.
* **VarBuilder hook:** when a module parameter is MXFP4, either:

  * **A:** call `dequant_mxfp4_to_bf16_cuda()` and return BF16 Tensor; or
  * **B:** keep `QuantizedWeight` and route the module’s `Linear` to the custom MXFP4 GEMM.
* **CUDA kernels:**

  * `unpack_fp4_kernel(u8* blocks, fp32* tmp, int n)`: expand nibbles to table indices.
  * `scale_e8m0_to_fp32_kernel(u8* scales, fp32* out_pow2, int nblocks)`.
  * `dequant_tile_kernel(...)`: fused unpack+scale+cast to BF16 into output tile.
    Provide a single fused kernel for Phase A to reduce memory traffic.
* **Tests wiring:** keep kernels behind `feature = "cuda"`, with CPU mirror for CI.

# 6) Unit tests that prove correctness (no model load)

* **T1 FP4 tables.** Exhaustive decode of all 16 FP4 codes: compare to spec’s FP4(E2M1) values. Include subnormals and sign. 
* **T2 E8M0 scales.** Decode all 256 E8M0 codes to `pow2(int8_exp)`; verify overflow/NaN codes per spec. 
* **T3 Block reconstruct.** Create a 64×64 synthetic weight, quantize to MXFP4 in Python (reference), store `blocks+scales`, load in Rust and dequant; MSE/MAE thresholds vs. source.
* **T4 Layout invariants.** Given odd shapes and non-power-of-two last dims, ensure last-dim blocking and padding rules are consistent with HF’s “block over last dim”.
* **T5 GEMM parity (small).** BF16 GEMM of dequantized W vs. CPU FP32 matmul of original W on small K×N tiles; error within FP16/BF16 tolerance.
* **T6 Throughput sanity.** CUDA microbench: dequant-on-load time per GB and expert GEMM tokens/s versus BF16 baseline at the same hidden dims.

# 7) Performance notes

* **Bandwidth-bound.** MXFP4 reduces weight bandwidth by ~4×. If you dequant per-tile (Phase B), you approach MXFP4’s bandwidth benefit. If you dequant on load (Phase A), you lose that benefit but keep simplicity.
* **Kernel fusion.** Always fuse unpack→table-decode→scale-mul→cast. Avoid staging in global memory as FP32.
* **Vectorization.** Load `uint4`/`uint8` vectors, unpack nibbles with bitfield extract. Use warp shuffles to share scale exponents for a block.

# 8) What Transformers/vLLM do that you can mirror

* **Requirements.** Python stacks need Triton ≥3.4.0 for MXFP4 kernels; if missing, they **dequantize to BF16**. That validates Phase A as an accepted fallback. ([GitHub][5])
* **Docs.** vLLM explains MXFP4 as FP4(E2M1)+block scales over last dim; same mapping you’ll implement. ([Modal][6])

# 9) MXFP4 vs NVFP4

* **Do not mix.** NVFP4 uses different block sizing and scaling; some hardware paths exist for NVFP4 on NVIDIA. MXFP4 is OCP-standardized and what GPT-OSS uses; implement MXFP4 exactly. ([NVIDIA Developer][4])

# 10) Failure modes to guard against

* Wrong axis blocking (not last dim) → garbage outputs. Use shape-checked asserts at load. ([GitHub][1])
* Nibble order swapped → 50% values wrong. Fix endian and add byte-level tests.
* Ignoring subnormals → large relative error near zero. Implement E=0 path. 
* Casting to FP16 with round-toward-zero → bias. Use RNE to BF16.
* Transpose mismatch vs. Candle Linear weight layout. Measure with a 1-layer toy MLP parity test.

This is the minimal, correct path to bring **MXFP4** into Candle: implement OCP-accurate decode, honor GPT-OSS’s **blocks+scales over last dim**, start with **dequant-on-load** in CUDA for simplicity, and optionally add **tile-dequant GEMM** for memory/bandwidth wins. 

[1]: https://github.com/openai/gpt-oss?utm_source=chatgpt.com "openai/gpt-oss"
[2]: https://huggingface.co/openai/gpt-oss-120b "openai/gpt-oss-120b · Hugging Face"
[3]: https://huggingface.co/openai/gpt-oss-20b/tree/main?utm_source=chatgpt.com "openai/gpt-oss-20b at main"
[4]: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/?utm_source=chatgpt.com "Introducing NVFP4 for Efficient and Accurate Low- ..."
[5]: https://github.com/huggingface/transformers/issues/39945?utm_source=chatgpt.com "GPT-OSS mxfp4 with triton_kernel: ..."
[6]: https://modal.com/docs/examples/gpt_oss_inference?utm_source=chatgpt.com "Run OpenAI's gpt-oss model with vLLM"

