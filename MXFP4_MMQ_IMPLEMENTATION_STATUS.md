# MXFP4 MMQ Implementation Status

**Date**: 2025-10-22
**Status**: ⚠️ **NOT YET APPLIED**

## Summary

The MMQ tile-based implementation infrastructure exists but has **NOT** been applied to the GPT-OSS example yet. The current implementation uses a simple per-element kernel without tiling or shared memory.

## Current Implementation

### CUDA Kernel: `matmul_mxfp4_bf16_mmq`

**Location**: `candle-kernels/src/quantized.cu:2961-3047`

**Current Approach**:
- ✗ No tiling (mmq_x, mmq_y not used)
- ✗ No shared memory
- ✗ No cooperative loading via `load_tiles_mxfp4<>`
- ✗ Simple 1-thread-per-output element
- ✓ Uses `get_int_from_table_16` for efficient FP4→INT8 lookup
- ✓ Processes 8 blocks at a time for ILP

**Grid/Block Configuration**:
```rust
grid_dim: (rows, (out_dim + 256 - 1) / 256, 1)
block_dim: (256, 1, 1)
shared_mem_bytes: 0  // ← No shared memory!
```

Each thread computes one output element independently.

### Rust Wrapper: `matmul_mxfp4_bf16_mmq_cuda`

**Location**: `candle-core/src/mxfp4.rs:400-500`

**Current Configuration**:
```rust
const THREADS_PER_BLOCK: usize = 256;
let grid_x = rows;
let grid_y = (out_dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
```

No tile size parameters used.

## What Exists But Is NOT Used

### Template Functions (Available But Unused)

1. **`load_tiles_mxfp4<int mmq_y>`**
   - Location: `candle-kernels/src/quantized.cu:2743-2787`
   - Status: ✓ Implemented, ✗ Not called
   - Functionality: Cooperative loading into shared memory

2. **`load_tiles_mxfp4_fast<int mmq_x, int mmq_y>`**
   - Location: `candle-kernels/src/quantized.cu:2849-2910`
   - Status: ✓ Implemented, ✗ Not called
   - Functionality: Optimized tile loading for MMQ kernels

3. **`vec_dot_mxfp4_bf16_mmq<int mmq_x, int mmq_y>`**
   - Location: `candle-kernels/src/quantized.cu:2914-2957`
   - Status: ✓ Implemented, ✗ Not called
   - Functionality: Vectorized dot product using shared memory

## What Needs to Be Done

To apply the MMQ tile-based implementation, the following changes are needed:

### 1. Create New Templated MMQ Kernel

Replace or supplement `matmul_mxfp4_bf16_mmq` with a templated version:

```cuda
template <int mmq_x, int mmq_y>
__global__ void matmul_mxfp4_bf16_mmq_tiled(
    const __nv_bfloat16* __restrict__ act,
    const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    const int rows,
    const int out_dim,
    const int nblocks,
    const int in_dim,
    const int act_row_stride,
    const int out_row_stride
) {
    // Allocate shared memory for tiles
    __shared__ int weight_qs_shared[mmq_y * 65];
    __shared__ float weight_scales_shared[mmq_y * 32];

    // Use load_tiles_mxfp4<mmq_y>() to load weights
    // Use vec_dot_mxfp4_bf16_mmq<mmq_x, mmq_y>() for computation
    // ...
}

// Instantiate for specific tile sizes
extern "C" __global__ void matmul_mxfp4_bf16_mmq_64_2(/* ... */) {
    matmul_mxfp4_bf16_mmq_tiled<64, 2>(/* ... */);
}

extern "C" __global__ void matmul_mxfp4_bf16_mmq_64_128(/* ... */) {
    matmul_mxfp4_bf16_mmq_tiled<64, 128>(/* ... */);
}
```

### 2. Update Rust Launch Code

Modify `candle-core/src/mxfp4.rs` to use the new tiled kernel:

```rust
pub fn matmul_mxfp4_bf16_mmq_cuda(/* ... */) -> Result<Tensor> {
    // Choose tile configuration
    const MMQ_X: usize = 64;
    const MMQ_Y: usize = 2;  // Start conservative
    const NWARPS: usize = 8;

    // Calculate grid dimensions
    let nty = (rows + MMQ_Y - 1) / MMQ_Y;
    let ntx = (out_dim + MMQ_X - 1) / MMQ_X;

    // Calculate shared memory needed
    let shared_mem_bytes = calculate_shared_memory(MMQ_X, MMQ_Y);

    let cfg = LaunchConfig {
        grid_dim: (nty as u32, ntx as u32, 1),
        block_dim: (32, NWARPS as u32, 1),
        shared_mem_bytes,
    };

    // Load tiled kernel
    let kernel_name = format!("matmul_mxfp4_bf16_mmq_{}_{}", MMQ_X, MMQ_Y);
    let func = dev.get_or_load_func(&kernel_name, &candle_kernels::QUANTIZED)?;

    // ...
}
```

### 3. Validate with Tests

Before switching to tiled implementation:

1. ✓ Unit test `load_tiles_mxfp4<2>` (completed in Step 4)
2. ✓ Unit test `get_int_from_table_16` (completed in Step 3)
3. ⚠️ Create end-to-end test comparing tiled vs. simple kernel
4. ⚠️ Verify numeric parity
5. ⚠️ Benchmark performance improvement

### 4. Gradual Rollout

**Phase 1** (Steps 6-8): Conservative testing
- Implement `matmul_mxfp4_bf16_mmq_64_2`
- Validate correctness
- Measure baseline performance

**Phase 2** (Steps 9-11): Scale up
- Add `matmul_mxfp4_bf16_mmq_64_64`
- Profile and optimize
- Compare with Step 1 baseline

**Phase 3** (Production): Optimize
- Add `matmul_mxfp4_bf16_mmq_64_128`
- Match llama.cpp performance
- Deploy to GPT-OSS example

## Recommended Tile Sizes

See `MXFP4_MMQ_TILE_SIZES.md` for detailed analysis.

**Quick reference**:
- Start: `mmq_x=64, mmq_y=2` (9.8 KB shared mem, 62.5% occupancy)
- Target: `mmq_x=64, mmq_y=128` (45.9 KB shared mem, 62.5% occupancy)

## Files to Modify

1. **CUDA Kernel**:
   - `candle-kernels/src/quantized.cu`
   - Add templated kernel instances
   - Wire up `load_tiles_mxfp4<>` and `vec_dot_mxfp4_bf16_mmq<>`

2. **Rust Wrapper**:
   - `candle-core/src/mxfp4.rs`
   - Update `matmul_mxfp4_bf16_mmq_cuda()`
   - Add tile size configuration
   - Calculate shared memory
   - Adjust grid/block dimensions

3. **Tests** (create if not exists):
   - `candle-core/tests/mxfp4_mmq_correctness.rs`
   - `candle-core/benches/mxfp4_mmq_performance.rs`

## Current vs. Target Performance

**Current Implementation** (simple per-element):
- Memory access pattern: Each thread independently loads weights
- Shared memory usage: 0 KB
- Occupancy: Limited by register pressure
- Expected: ~30-40% of peak memory bandwidth

**Target Implementation** (MMQ tiled):
- Memory access pattern: Cooperative loading, shared within thread block
- Shared memory usage: 9.8-45.9 KB (depending on mmq_y)
- Occupancy: 50-62.5% (calculated)
- Expected: ~60-80% of peak memory bandwidth (llama.cpp experience)

**Expected Speedup**: **2-3x** for typical workloads

## Verification Checklist

Before claiming MMQ implementation is complete:

- [ ] Templated kernel implemented with `<mmq_x, mmq_y>` parameters
- [ ] `load_tiles_mxfp4<mmq_y>()` called from kernel
- [ ] Shared memory allocated and configured
- [ ] Grid dimensions: `(nty, ntx, 1)` where `nty = rows/mmq_y`, `ntx = out_dim/mmq_x`
- [ ] Block dimensions: `(32, nwarps, 1)` where `nwarps = 8`
- [ ] Numeric parity test passes vs. simple kernel
- [ ] Performance improvement measured (target: 2-3x)
- [ ] GPT-OSS example runs successfully with tiled kernel

## References

- Tile size analysis: `MXFP4_MMQ_TILE_SIZES.md`
- Step 3 report: `/tmp/step3_verification_report.md`
- Step 4 report: `/tmp/step4_verification_report.md`
- Step 5 report: `/tmp/step5_tile_size_determination.md`

## Next Steps

Proceed to **Step 6**: Implement the first tiled MMQ kernel instance with `mmq_x=64, mmq_y=2`.
