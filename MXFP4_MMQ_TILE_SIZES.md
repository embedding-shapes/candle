# MXFP4 MMQ Tile Size Configuration for Blackwell Architecture

**GPU**: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
**Compute Capability**: 12.0
**Date**: 2025-10-22
**Based on**: llama.cpp tile sizing logic and occupancy analysis

## Quick Reference

### Recommended Tile Sizes

| Phase | mmq_x | mmq_y | Shared Mem | Occupancy | Use Case |
|-------|-------|-------|------------|-----------|----------|
| **Initial/Testing** | 64 | 2 | 9.8 KB | 62.5% | Correctness verification, debugging |
| **Development** | 64 | 64 | 27.6 KB | 62.5% | Performance tuning, profiling |
| **Production** | 64 | 128 | 45.9 KB | 62.5% | llama.cpp default, maximum performance |
| Alternative | 128 | 128 | 55.1 KB | 50.0% | Large output dimensions |

### Current Implementation Status

Check these files to see which configuration is currently active:
- `candle-kernels/src/quantized.cu` - CUDA kernel implementation
- `candle-core/src/mxfp4.rs` - Rust wrapper
- `candle-examples/examples/gpt-oss-20b/main.rs` - GPT-OSS example

## Architecture Specifications

**Blackwell (SM 12.0)**:
- Max threads per SM: 2048
- Max blocks per SM: 32
- Warp size: 32
- Warps per SM: 64
- Registers per SM: 65,536
- Shared memory per SM: 232 KB (configurable)
- Shared memory per block (max): 232 KB

## llama.cpp Default Configuration

For compute capability 12.0 (Blackwell >= Volta 7.0):

```cpp
mmq_y = 128          // get_mmq_y_host() for cc >= Volta
mmq_x_max = 64       // MMQ_DP4A_MAX_BATCH_SIZE (default)
nwarps = 8           // 256 threads per block
threads_per_block = 256
```

**MXFP4 Tile Layout** (uses `MMQ_DP4A_TXS_Q8_1`):
```cpp
qs = mmq_y * 65 ints          // Quantized INT8 values
dm = mmq_y * 8 + mmq_y/4      // Scales (half2 format)
sc = 0                         // No additional scales
```

## Shared Memory Usage

**Formula**:
```
total = nbs_ids + nbs_x + nbs_y

where:
  nbs_ids = mmq_x * 4 bytes
  nbs_x = qs*4 + dm*4 + sc*4 bytes
  nbs_y = GGML_PAD(mmq_x * 144, 1024) bytes
```

**Calculated Values**:

| mmq_x | mmq_y | nbs_ids | nbs_x | nbs_y | Total | % of 232 KB |
|-------|-------|---------|-------|-------|-------|-------------|
| 64 | 2 | 256 B | 592 B | 9 KB | 9.8 KB | 4.2% |
| 64 | 64 | 256 B | 18.7 KB | 9 KB | 27.6 KB | 11.9% |
| 64 | 128 | 256 B | 37.5 KB | 9 KB | 45.9 KB | 19.8% |
| 128 | 128 | 512 B | 37.5 KB | 18 KB | 55.1 KB | 23.8% |

✓ All configurations well below the 232 KB limit

## Register Pressure

**Estimated**: ~44 registers per thread
- Base registers: ~25 (indices, addresses, temporaries)
- Accumulator registers: ~20 (partial sums for dot products)
- Work registers: ~15 (loop variables, intermediate calculations)

**Register Limit**:
```
Registers per block = 256 threads × 44 regs = 11,264
Blocks per SM (reg-limited) = 65,536 / 11,264 ≈ 5.8 → 5 blocks
```

## Theoretical Occupancy

**Methodology**:
```
blocks_per_sm = min(
    2048 / 256,                    # Thread limit = 8
    32,                            # Max blocks = 32
    232KB / shared_mem,            # Shared memory limit
    65536 / 11264                  # Register limit ≈ 5
)

warp_occupancy = (blocks_per_sm × 8 warps) / 64 warps × 100%
```

**Results**:

| mmq_x | mmq_y | Blocks/SM | Active Warps | Warp Occ. | Limiting Factor |
|-------|-------|-----------|--------------|-----------|-----------------|
| 64 | 2   | 5 | 40 | **62.5%** | registers |
| 64 | 64  | 5 | 40 | **62.5%** | registers |
| 64 | 128 | 5 | 40 | **62.5%** | shared_mem |
| 128 | 128 | 4 | 32 | **50.0%** | shared_mem |

## Trade-offs Analysis

### High mmq_y (128) - Production Configuration

**Advantages**:
- ✓ Maximizes work per thread block
- ✓ Better amortization of shared memory loading overhead
- ✓ Reduces grid size (fewer blocks needed)
- ✓ Better for memory-bound workloads
- ✓ Proven configuration in llama.cpp

**Disadvantages**:
- ✗ Higher shared memory usage (45.9 KB)
- ✗ Becomes shared memory limited
- ✗ More complex debugging

**Best for**: Production deployment, memory-bound workloads

### Low mmq_y (2) - Testing Configuration

**Advantages**:
- ✓ Minimal shared memory (9.8 KB - 95% headroom)
- ✓ Register limited, not memory limited
- ✓ Maximum flexibility for experimentation
- ✓ Easier to debug with small tiles
- ✓ Good occupancy (62.5%)

**Disadvantages**:
- ✗ More blocks needed (larger grid)
- ✗ Less work amortization per block
- ✗ More kernel launch overhead

**Best for**: Initial implementation, correctness verification, debugging

### Balanced mmq_y (64) - Development Configuration

**Advantages**:
- ✓ Moderate shared memory (27.6 KB)
- ✓ Good balance of work per block
- ✓ Still register limited
- ✓ Used in llama.cpp for pre-Volta GPUs

**Disadvantages**:
- ✗ Not optimal for either extreme

**Best for**: Development, performance tuning, profiling

## Implementation Phases

### Phase 1: Initial Implementation (Steps 6-8)
```rust
// Conservative starting values
const MMQ_X: usize = 64;
const MMQ_Y: usize = 2;
```

**Goal**: Verify correctness, establish baseline
**Priority**: Correctness over performance

### Phase 2: Development (Steps 9-11)
```rust
// Balanced configuration
const MMQ_X: usize = 64;
const MMQ_Y: usize = 64;
```

**Goal**: Performance tuning, identify bottlenecks
**Priority**: Balance performance and debuggability

### Phase 3: Production Optimization
```rust
// llama.cpp proven configuration
const MMQ_X: usize = 64;
const MMQ_Y: usize = 128;
```

**Goal**: Maximum performance for deployment
**Priority**: Match llama.cpp performance

### Optional: Runtime Selection

For maximum flexibility, implement dynamic selection based on output dimension:

```rust
fn get_mmq_x_y(cc: i32, output_cols: usize) -> (usize, usize) {
    let mmq_y = if cc >= 700 { 128 } else { 64 };
    let mmq_x_max = if cc >= 700 { 64 } else { 64 };

    // Select mmq_x to minimize tiles needed
    let mmq_x = (8..=mmq_x_max)
        .step_by(8)
        .min_by_key(|&x| (output_cols + x - 1) / x)
        .unwrap_or(mmq_x_max);

    (mmq_x, mmq_y)
}
```

## Verification Checklist

When changing tile sizes, verify:

- [ ] Shared memory usage < 232 KB
- [ ] Grid dimensions correctly calculated
- [ ] Block dimensions = (32, nwarps) = (32, 8)
- [ ] Shared memory arrays sized correctly:
  - `weight_qs_shared[mmq_y * 65]`
  - `weight_scales_shared[mmq_y * 32]`
- [ ] Kernel correctness with micro-tests
- [ ] Performance measured vs. baseline
- [ ] Occupancy measured with nsys/ncu

## Performance Expectations

**Based on llama.cpp experience**:

- MMQ kernels typically achieve **60-80%** of peak memory bandwidth
- Occupancy of **50-62.5%** is normal and acceptable
- MXFP4 matmul should be **2-4x faster** than FP16/BF16 baseline
- Memory bandwidth is the primary bottleneck, not compute

**Profiling recommendations**:
```bash
# Measure occupancy
nsys profile --stats=true ./example

# Detailed kernel metrics
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second \
    ./example
```

## References

### Source Code Analysis
- llama.cpp tile sizing: `llama.cpp/ggml/src/ggml-cuda/mmq.cuh:94-160`
- Shared memory calculation: `llama.cpp/ggml/src/ggml-cuda/mmq.cuh:3537-3570`
- MXFP4 tile configuration: `llama.cpp/ggml/src/ggml-cuda/mmq.cuh:168-189`

### Analysis Artifacts
- `/tmp/blackwell_specs.py` - Basic tile calculations
- `/tmp/blackwell_occupancy_analysis.py` - Full occupancy analysis
- `/tmp/step5_tile_size_determination.md` - Detailed analysis

### NVIDIA Documentation
- [Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/blackwell/)
- [CUDA Programming Guide - Occupancy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy-calculator)
- [Compute Capability 12.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-12-x)

## History

- **2025-10-22**: Initial analysis for Blackwell (SM 12.0)
  - Determined conservative starting values: mmq_x=64, mmq_y=2
  - Validated llama.cpp default: mmq_x=64, mmq_y=128
  - Calculated theoretical occupancy: 50-62.5%
