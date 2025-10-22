# Proposal: Modularize `candle-kernels/src/quantized.cu`

## Problem Statement

The file `candle-kernels/src/quantized.cu` is currently **5,368 lines** containing CUDA kernels for 11 different quantization formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, MXFP4). This monolithic structure creates several issues:

- **Slow compilation**: Any change requires recompiling all 5,368 lines
- **Poor navigation**: Finding specific kernel code requires scrolling through thousands of lines
- **Merge conflicts**: Multiple developers working on different quantization types will conflict
- **No parallelization**: Single compilation unit cannot utilize multiple CPU cores during build
- **Mental overhead**: Understanding one quantization format requires context from the entire file

## Proposed Solution: Split into 20 Files

Reorganize into **20 modular files**, averaging **268 lines each** (maximum 300 lines). Each file contains logically cohesive code for a specific purpose.

---

## File Structure

### **Core Headers (5 files)**

These headers contain shared infrastructure used across all quantization kernels.

#### **1. `quantized_common.cuh` (~280 lines)**

**Purpose**: Common definitions, constants, and data structures

**Contents**:
- Standard CUDA includes (`cuda_fp16.h`, `cuda_bf16.h`)
- Common macros: `GGML_UNUSED`, `GGML_CUDA_ASSUME`, `WARP_SIZE`
- Typedefs: `ggml_fp16_t`, `dfloat`, `dfloat2`, `dequantize_kernel_t`
- Compute capability constants:
  ```cuda
  #define CC_PASCAL     600
  #define CC_VOLTA      700
  #define CC_TURING     750
  #define CC_AMPERE     800
  #define MIN_CC_DP4A   610
  ```
- MMQ configuration for all GPU architectures (256+ lines of `#define` statements):
  ```cuda
  #define MMQ_X_Q4_0_AMPERE 64
  #define MMQ_Y_Q4_0_AMPERE 128
  #define NWARPS_Q4_0_AMPERE 4
  // ... repeated for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K through Q6_K
  // ... for RDNA2, RDNA1, AMPERE, PASCAL architectures
  ```
- Block structure definitions:
  ```cuda
  typedef struct {
      half d;
      uint8_t qs[QK4_0 / 2];
  } block_q4_0;

  typedef struct {
      half2 dm;
      uint8_t qs[QK4_1 / 2];
  } block_q4_1;
  // ... block_q5_0, block_q5_1, block_q8_0, block_q8_1
  // ... block_q2_K, block_q3_K, block_q4_K, block_q5_K, block_q6_K
  ```
- QK/QR/QI constants for each type (e.g., `#define QK4_0 32`)

---

#### **2. `quantized_device_helpers.cuh` (~120 lines)**

**Purpose**: General-purpose device utility functions

**Contents**:
- Warp-level reductions:
  ```cuda
  static __device__ __forceinline__ float warp_reduce_sum(float x);
  static __device__ __forceinline__ float warp_reduce_max(float x);
  ```
- Integer packing/unpacking helpers:
  ```cuda
  static __device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32);
  static __device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32);
  static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32);
  static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32);
  ```
- CUDA intrinsic wrappers:
  ```cuda
  static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c);
  ```
- MMA (Matrix Multiply-Accumulate) instruction wrapper:
  ```cuda
  static __device__ __forceinline__ void mma_m16n8k16_s32_s8(...);
  ```

---

#### **3. `quantized_mxfp4_common.cuh` (~100 lines)**

**Purpose**: MXFP4-specific constants and helper functions

**Contents**:
- MXFP4 constants:
  ```cuda
  #define QK_MXFP4  32  // 32 FP4 values per block
  #define QR_MXFP4  2   // Ratio
  #define QI_MXFP4  4   // 4-way interleaving
  ```
- FP4 → INT8 lookup table:
  ```cuda
  __device__ __constant__ int8_t kvalues_mxfp4[16] = {
      0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
  };
  ```
- Fast nibble extraction using `__byte_perm`:
  ```cuda
  static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32);
  static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table);
  ```
- E8M0 scale conversion to float32:
  ```cuda
  static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x);
  ```

---

#### **4. `quantized_dequant_vec_dot.cuh` (~300 lines)**

**Purpose**: Reusable dequantization and vector dot product templates

**Contents**:
- Generic dequantization function templates
- Dequantization implementations for basic types:
  ```cuda
  static __device__ __forceinline__ void dequantize_q4_0(...);
  static __device__ __forceinline__ void dequantize_q4_1(...);
  static __device__ __forceinline__ void dequantize_q5_0(...);
  static __device__ __forceinline__ void dequantize_q5_1(...);
  static __device__ __forceinline__ void dequantize_q8_0(...);
  ```
- Vector dot product template:
  ```cuda
  template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
  static __device__ void dequantize_mul_mat_vec(...);
  ```
- VDR (Vec Dot Ratio) constants:
  ```cuda
  #define VDR_Q4_0_Q8_1_MMVQ 2
  #define VDR_Q4_0_Q8_1_MMQ  4
  // ... for all quantization types
  ```

---

#### **5. `quantized_mmq_templates.cuh` (~280 lines)**

**Purpose**: MMQ (Multi-row Matrix Quantized) template infrastructure

**Contents**:
- Generic MMQ template:
  ```cuda
  template <int qk, int qr, int qi, bool need_sum, typename block_q_t,
            int mmq_x, int mmq_y, int nwarps,
            allocate_tiles_cuda_t allocate_tiles,
            load_tiles_cuda_t load_tiles,
            int vdr,
            vec_dot_q_mul_mat_cuda_t vec_dot>
  static __device__ __forceinline__ void mul_mat_q(...);
  ```
- Function pointer typedefs:
  ```cuda
  typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq,
                                     const block_q8_1 * __restrict__ bq8_1,
                                     const int & iqs);
  typedef void (*allocate_tiles_cuda_t)(int ** x_ql, half2 ** x_dm,
                                         int ** x_qh, int ** x_sc);
  typedef void (*load_tiles_cuda_t)(...);
  ```
- Shared memory tile iteration logic
- Y-tile loading helpers

---

### **Basic Quantization Types (5 files, ~220-250 lines each)**

Each file is **self-contained** for one quantization format. All follow the same structure:

#### **6. `q4_0.cu` (~230 lines)**

**Contents**:
1. **Dequantization kernels** (2 variants):
   ```cuda
   extern "C" __global__ void dequantize_block_q4_0_f32(const void * vx, float * y, const int k);
   extern "C" __global__ void dequantize_block_q4_0_f16(const void * vx, half * y, const int k);
   ```

2. **Vector dot product implementation**:
   ```cuda
   template <int vdr>
   static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(...);

   static __device__ __forceinline__ float vec_dot_q4_0_q8_1(...);
   ```

3. **Matrix-vector multiply**:
   ```cuda
   extern "C" __global__ void dequantize_mul_mat_vec_q4_0_cuda(...);
   ```

4. **Batch variants** (8 kernels for batch sizes 1-8):
   ```cuda
   extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda1(...);
   extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda2(...);
   // ... through cuda8
   ```

5. **MMQ tile operations**:
   ```cuda
   template<int mmq_y>
   static __device__ __forceinline__ void allocate_tiles_q4_0(...);

   template<int mmq_y, int nwarps, bool need_check>
   static __device__ __forceinline__ void load_tiles_q4_0(...);

   static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(...);
   ```

6. **Matrix multiply kernel**:
   ```cuda
   extern "C" __global__ void mul_mat_q4_0(...);
   ```

---

#### **7. `q4_1.cu` (~230 lines)**
Same structure as `q4_0.cu`, but for Q4_1 quantization format.

#### **8. `q5_0.cu` (~240 lines)**
Same structure as `q4_0.cu`, but for Q5_0 quantization format (includes 5th bit handling).

#### **9. `q5_1.cu` (~240 lines)**
Same structure as `q4_0.cu`, but for Q5_1 quantization format (includes 5th bit handling).

#### **10. `q8_0.cu` (~220 lines)**
Same structure as `q4_0.cu`, but for Q8_0 quantization format (8-bit quantization).

---

### **K-Type Quantization (5 files, ~290-300 lines each)**

K-types use more complex quantization with additional scale/min parameters. Each file follows this structure:

#### **11. `q2_k.cu` (~290 lines)**

**Contents**:
1. **Dequantization kernels**:
   ```cuda
   extern "C" __global__ void dequantize_block_q2_K_f32(const void * vx, float * y);
   extern "C" __global__ void dequantize_block_q2_K_f16(const void * vx, half * y);
   ```

2. **Vector dot product** (more complex due to K-type structure):
   ```cuda
   static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(...);
   static __device__ __forceinline__ float vec_dot_q2_K_q8_1(...);
   ```

3. **Matrix-vector multiply**:
   ```cuda
   extern "C" __global__ void dequantize_mul_mat_vec_q2_k(...);
   ```

4. **Batch variants** (8 kernels):
   ```cuda
   extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda1(...);
   // ... through cuda8
   ```

5. **MMQ implementation**:
   ```cuda
   template<int mmq_y>
   static __device__ __forceinline__ void allocate_tiles_q2_K(...);

   template<int mmq_y, int nwarps, bool need_check>
   static __device__ __forceinline__ void load_tiles_q2_K(...);

   static __device__ __forceinline__ float vec_dot_q2_K_q8_1_mul_mat(...);
   ```

6. **Matrix multiply kernel**:
   ```cuda
   extern "C" __global__ void mul_mat_q2_K(...);
   ```

---

#### **12. `q3_k.cu` (~295 lines)**
Same structure as `q2_k.cu`, for Q3_K quantization.

#### **13. `q4_k.cu` (~300 lines)**
Same structure as `q2_k.cu`, for Q4_K quantization.

#### **14. `q5_k.cu` (~300 lines)**
Same structure as `q2_k.cu`, for Q5_K quantization.

#### **15. `q6_k.cu` (~295 lines)**
Same structure as `q2_k.cu`, for Q6_K quantization.

---

### **MXFP4 Kernels (4 files)**

MXFP4 (FP4 E2M1 with E8M0 block-wise scale) is split into logical operation groups:

#### **16. `mxfp4_dequant.cu` (~150 lines)**

**Purpose**: Dequantization and format conversion utilities

**Contents**:
1. **FP4 E2M1 decoder**:
   ```cuda
   static __device__ __forceinline__ float decode_fp4_e2m1_device(uint8_t n);
   ```
   - Decodes sign (1 bit), exponent (2 bits), mantissa (1 bit)
   - Handles subnormals and normals with bias=1

2. **E8M0 scale converter**:
   ```cuda
   static __device__ __forceinline__ float pow2_e8m0_device(uint8_t bexp);
   ```
   - Converts biased 8-bit exponent to `2^(u8 - 127)`
   - Handles special case: 0xFF → NaN, 0x00 → smallest normal

3. **Lookup tables**:
   ```cuda
   __device__ __constant__ int8_t MXFP4_FP4_LUT[16] = {...};
   __device__ __constant__ int8_t MXFP4_INT8_LUT[16] = {...};
   ```

4. **Packing helper**:
   ```cuda
   __device__ __forceinline__ int pack_int8x4(int8_t x0, int8_t x1, int8_t x2, int8_t x3);
   ```

5. **Fused dequantization kernel**:
   ```cuda
   extern "C" __global__ void dequant_mxfp4_to_bf16(
       const uint8_t* blocks,  // [rows, nblocks, 16]
       const uint8_t* scales,  // [rows, nblocks]
       __nv_bfloat16* out,     // [rows, cols]
       const int rows, const int nblocks, const int cols);
   ```
   - Input: packed FP4 nibbles (2 per byte) + E8M0 scales
   - Output: dequantized BF16 values
   - Grid: `dim3(rows, nblocks)`, Block: `32 threads`

6. **Debug utility**:
   ```cuda
   extern "C" __global__ void mxfp4_unpack(
       const uint8_t* input,
       uint8_t* output_low,
       uint8_t* output_high,
       int num_bytes);
   ```
   - Splits packed bytes into separate nibbles for verification

---

#### **17. `mxfp4_load_tiles.cu` (~200 lines)**

**Purpose**: Tile loading for MMQ (shared memory cooperative loading)

**Contents**:
1. **Tile loading template**:
   ```cuda
   template<int mmq_y>
   static __device__ __forceinline__ void load_tiles_mxfp4(
       const uint8_t* __restrict__ blocks,  // [nblocks, 16]
       const uint8_t* __restrict__ scales,  // [nblocks]
       int* __restrict__ x_qs,              // Output: INT8 values in smem
       float* __restrict__ x_df,            // Output: scales in smem
       ...);
   ```
   - Cooperatively loads `mmq_y` rows of MXFP4 weights
   - Uses `kvalues_mxfp4` lookup for fast FP4→INT8 conversion
   - Stores dequantized INT8 values + float scales in shared memory
   - Each thread handles specific nibble positions

2. **Simple vector dot helper**:
   ```cuda
   static __device__ __forceinline__ float vec_dot_mxfp4_mmq_simple(
       const int* weight_int8,   // 32 INT8 values
       const int* activation,    // Q8_1 activation
       const float scale_mxfp4,  // E8M0 scale
       const float scale_q8_1);  // Q8_1 scale
   ```
   - Direct computation without shared memory complexity
   - Uses `__dp4a` for 4-way SIMD dot products

3. **Vector dot for MMQ**:
   ```cuda
   static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
       const int* __restrict__ x_qs,   // Pre-dequantized INT8 from smem
       const float* __restrict__ x_df, // Scales from smem
       const int* __restrict__ y_qs,   // Q8_1 activations
       const float* __restrict__ y_df, // Q8_1 scales
       const int i, const int j, const int k);
   ```
   - Core computation for one K-block (32 elements)
   - Works with pre-loaded shared memory tiles

---

#### **18. `mxfp4_matmul_mmq.cu` (~290 lines)**

**Purpose**: Optimized tiled matrix multiplication using MMQ approach

**Contents**:
1. **MMQ kernel**:
   ```cuda
   extern "C" __global__ void matmul_mxfp4_bf16_mmq(
       const uint8_t* blocks,         // [weight_rows, nblocks_per_row, 16]
       const uint8_t* scales,         // [weight_rows, nblocks_per_row]
       const __nv_bfloat16* act,      // [batch, act_cols]
       __nv_bfloat16* out,            // [batch, weight_rows]
       const int weight_rows,
       const int weight_cols,
       const int batch,
       const int act_cols);
   ```

**Characteristics**:
- **Tile-based approach**: Processes multiple rows and columns per block
- **Shared memory usage**:
  - Weight tiles: INT8 dequantized weights
  - Weight scales: float scales
  - Activation tiles: BF16 activations
- **Cooperative loading**: Threads work together to load tiles
- **Accumulation**: Each thread accumulates partial sums, then reduces across warp
- **Grid configuration**: Typically `dim3(batch, nblocks)` blocks
- **Block configuration**: `256-512 threads` (architecture-dependent)

**Performance characteristics**:
- Best for large matrices (>512 rows)
- High shared memory usage
- Maximum throughput via tile reuse

---

#### **19. `mxfp4_matmul_direct.cu` (~280 lines)**

**Purpose**: Simple direct matrix multiplication (no tiling overhead)

**Contents**:
1. **Direct BF16 matmul**:
   ```cuda
   extern "C" __global__ void matmul_mxfp4_bf16(
       const uint8_t* blocks,
       const uint8_t* scales,
       const __nv_bfloat16* act,
       __nv_bfloat16* out,
       const int weight_rows,
       const int weight_cols,
       const int batch,
       const int act_cols);
   ```
   - One thread per output element
   - Loads nibbles, dequantizes on-the-fly
   - No shared memory overhead
   - Good for small matrices or low batch sizes

2. **MXFP4 × Q8_1 matmul**:
   ```cuda
   extern "C" __global__ void matmul_mxfp4_q8_1(
       const uint8_t* blocks,
       const uint8_t* scales,
       const void* q8_1_data,        // Q8_1 quantized activations
       float* out,
       const int weight_rows,
       const int weight_cols,
       const int batch,
       const int act_cols);
   ```
   - Uses Q8_1 format activations instead of BF16
   - Leverages `__dp4a` for efficient INT8 × INT8 dot products
   - Output is FP32 for numerical stability

**Performance characteristics**:
- Best for small matrices (<512 rows) or low batch sizes
- Lower shared memory usage
- Simpler control flow

---

### **Special Kernels (1 file)**

#### **20. `special.cu` (~150 lines)**

**Purpose**: Specialized fusion kernels for specific model architectures

**Contents**:
1. **GPT-OSS expert activation fusion**:
   ```cuda
   extern "C" __global__ void fused_expert_activation_bf16(
       const __nv_bfloat16* gate_up,  // [batch, 2*expert_dim] (interleaved)
       __nv_bfloat16* output,          // [batch, expert_dim]
       const int batch,
       const int expert_dim,
       const float alpha,               // Sigmoid scaling factor
       const float limit);              // Asymmetric clamp limit
   ```

**Operation**:
- Input: `gate_up` tensor with interleaved gate and up vectors
  - Even indices: gate values
  - Odd indices: up values
- Computation: `output = (up+1) * gate * sigmoid(alpha*gate)`
- Steps:
  1. Split gate and up
  2. Asymmetric clamp on gate: `clamp(gate, -limit, +∞)`
  3. Compute sigmoid: `1 / (1 + exp(-alpha*gate))`
  4. Fuse multiplication: `(up+1) * gate * sigmoid`

**Motivation**:
- GPT-OSS models use this specific activation pattern
- Fusing avoids 3+ separate kernel launches
- Reduces memory bandwidth (no intermediate tensors)

**Future additions**:
- Other model-specific fusion kernels
- Experimental optimizations

---

## Build System Integration

Candle uses **Rust's `build.rs`** build script (not CMake). The file is located at:
```
candle-kernels/build.rs
```

### Current behavior
The `build.rs` likely compiles `quantized.cu` as a single monolithic file.

### Required changes

**Option A: Compile all .cu files individually**
```rust
// In build.rs
let cu_files = vec![
    "src/quantized_common.cuh",      // Header, no compilation
    "src/quantized_device_helpers.cuh", // Header
    "src/quantized_mxfp4_common.cuh",   // Header
    "src/quantized_dequant_vec_dot.cuh",// Header
    "src/quantized_mmq_templates.cuh",  // Header
    "src/q4_0.cu",
    "src/q4_1.cu",
    "src/q5_0.cu",
    "src/q5_1.cu",
    "src/q8_0.cu",
    "src/q2_k.cu",
    "src/q3_k.cu",
    "src/q4_k.cu",
    "src/q5_k.cu",
    "src/q6_k.cu",
    "src/mxfp4_dequant.cu",
    "src/mxfp4_load_tiles.cu",
    "src/mxfp4_matmul_mmq.cu",
    "src/mxfp4_matmul_direct.cu",
    "src/special.cu",
];

for file in cu_files.iter() {
    if file.ends_with(".cu") {
        // Compile .cu file
        // Link into final library
    }
}
```

**Option B: Use a single compilation unit that includes all files**

Create a new `quantized_all.cu`:
```cuda
#include "quantized_common.cuh"
#include "quantized_device_helpers.cuh"
#include "quantized_mxfp4_common.cuh"
#include "quantized_dequant_vec_dot.cuh"
#include "quantized_mmq_templates.cuh"
#include "q4_0.cu"
#include "q4_1.cu"
// ... etc
```

**Option A is preferred** for parallel compilation and incremental build benefits.

---

## Benefits Summary

### **Developer Experience**
- **Easy navigation**: Need Q4_0 code? → `q4_0.cu`. Need MXFP4 matmul? → `mxfp4_matmul_*.cu`
- **Focused editing**: Work on MXFP4 without seeing 4,000+ irrelevant lines
- **Clear structure**: Headers → implementations, utilities → kernels
- **IDE performance**: Syntax highlighting, autocomplete, and go-to-definition work better on 200-line files than 5,000-line files

### **Build Performance**
- **Parallel compilation**: 15 `.cu` files compile concurrently (up to CPU core limit)
- **Incremental builds**: Changing `mxfp4_matmul_mmq.cu` doesn't recompile `q4_0.cu`
- **Faster iteration**: Fix a bug in Q4_0 → only recompile `q4_0.cu` (~230 lines) instead of entire `quantized.cu` (5,368 lines)

### **Code Quality**
- **Easier code review**: Small, focused diffs instead of changes buried in a 5,000-line file
- **Reduced merge conflicts**: Different developers can work on different quantization types without conflicts
- **Better testing**: Can test individual quantization formats in isolation
- **Self-documenting**: File structure makes the code organization obvious

### **Maintainability**
- **Future-proof**: Adding a new quantization format = add 1 new file following established pattern
- **Reduced cognitive load**: Understand one file at a time, not the entire system
- **Easier debugging**: Smaller compilation units mean faster debug builds
- **Industry standard**: Matches modern CUDA code organization (e.g., llama.cpp uses similar modular structure)

---

## Migration Strategy

### Phase 1: Preparation
1. Create header files (no code changes, just extract definitions)
2. Verify headers compile independently
3. Add `#include` guards

### Phase 2: Split implementation files
1. Start with one type (e.g., `q4_0.cu`)
2. Extract relevant code
3. Add includes to headers
4. Verify it compiles and links
5. Run tests to confirm no functional changes
6. Repeat for remaining types

### Phase 3: Update build system
1. Modify `build.rs` to compile new file structure
2. Verify parallel compilation works
3. Measure build time improvements

### Phase 4: Validation
1. Run full test suite
2. Benchmark kernel performance (ensure no regression)
3. Verify binary size unchanged (no code duplication)

---

## Risks and Mitigations

### Risk: Build system complexity
**Mitigation**: Start with Option B (single compilation unit) if `build.rs` modifications are complex. Still get code organization benefits without build changes.

### Risk: Circular dependencies
**Mitigation**: Clear header hierarchy prevents this:
- `quantized_common.cuh` depends on nothing
- Other headers depend only on `quantized_common.cuh`
- `.cu` files depend on headers only

### Risk: Code duplication
**Mitigation**: Use headers for shared code. Each `.cu` file should only contain type-specific implementations.

### Risk: Performance regression
**Mitigation**: CUDA inlining (`__forceinline__`) ensures no performance loss. Verify with benchmarks before/after split.

---

## Conclusion

This reorganization transforms a 5,368-line monolithic file into a clean, modular structure with **20 files averaging 268 lines each**. The split respects logical boundaries (headers vs implementations, basic types vs K-types vs MXFP4), enables parallel compilation, and significantly improves developer experience—all without changing any kernel logic or affecting runtime performance.
