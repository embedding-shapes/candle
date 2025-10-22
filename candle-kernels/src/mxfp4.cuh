extern "C" __global__ void dequant_mxfp4_to_bf16(
    const uint8_t* __restrict__ blocks, // [rows, nblocks, 16]
    const uint8_t* __restrict__ scales, // [rows, nblocks]
    __nv_bfloat16* __restrict__ out,    // [rows, cols]
    const int rows,
    const int nblocks,
    const int cols
) {
    const int r = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x; // 0..31 (one thread per FP4 value in block)

    if (r >= rows || b >= nblocks || t >= 32) return;

    // Load scale for this block.
    const int sb_index = r * nblocks + b;
    const float scale = pow2_e8m0_device(scales[sb_index]);

    // Packed bytes for this block (16 bytes), two FP4 per byte.
    const int block_base = (r * nblocks + b) * 16;
    const int byte_idx = t >> 1; // 0..15
    const uint8_t packed = blocks[block_base + byte_idx];
    const uint8_t nibble = (t & 1) ? (packed >> 4) : (packed & 0x0f);

    const float v = decode_fp4_e2m1_device(nibble) * scale;

    // Output index
    const int out_c = b * 32 + t;
    const int out_index = r * cols + out_c;
    out[out_index] = __float2bfloat16(v);
}

// Simple MXFP4 unpack utility: split each input byte into high/low 4-bit nibbles.
// This is used for micro-verification of nibble order and downstream mapping.
extern "C" __global__ void mxfp4_unpack(
    const uint8_t* __restrict__ in, // [n]
    uint8_t* __restrict__ hi,       // [n]
    uint8_t* __restrict__ lo,       // [n]
    const int n                     // number of bytes
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const uint8_t v = in[idx];
    hi[idx] = v >> 4;
    lo[idx] = v & 0x0f;
}

// ============================================================================
// MMQ (Multi-row Matrix Quantized) MXFP4 Tile Loading
// ============================================================================
// Loads and dequantizes MXFP4 weights into shared memory for efficient MMQ matmul.
// Adapted from llama.cpp mmq.cuh:699-762 for Candle's separate blocks/scales layout.
//
// Key differences from llama.cpp:
// - Candle stores blocks [out_dim, nblocks, 16] and scales [out_dim, nblocks] separately
// - llama.cpp uses struct block_mxfp4 { uint8_t e; uint8_t qs[16]; } (interleaved)
//
// This function cooperatively loads mmq_y rows × 32 elements per block tile into shared memory.
// Each thread handles specific nibble positions using warp-level cooperation.
//
// Template parameters:
//   mmq_y: Number of rows to load per tile (e.g., 64)
//
// Memory layout after loading:
//   weight_qs_shared[row][col]: INT8 dequantized values (65-col stride for bank conflict avoidance)
//   weight_scales_shared[row][block_idx]: FP32 scales
//
// Ref: llama.cpp mmq.cuh load_tiles_mxfp4
template <int mmq_y>
static __device__ __forceinline__ void load_tiles_mxfp4(
    const uint8_t* __restrict__ blocks,          // Candle: [out_dim, nblocks, 16]
    const uint8_t* __restrict__ scales,          // Candle: [out_dim, nblocks]
    int* __restrict__ weight_qs_shared,          // Output: INT8 values [mmq_y][65]
    float* __restrict__ weight_scales_shared,    // Output: FP32 scales [mmq_y][32]
    const int row_offset,                        // Starting row in weight matrix
    const int block_offset,                      // Starting block in K dimension
    const int nblocks                            // Total blocks per row (stride)
) {
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int kbx = tid / QI_MXFP4;    // Which MXFP4 block (0-7 for 32 threads)
    const int kqsx = tid % QI_MXFP4;   // Which nibble position within block (0-3)

    // Each thread loads elements for mmq_y rows
    // Using strided access for warp cooperation
    for (int row_local = threadIdx.y; row_local < mmq_y; row_local += blockDim.y) {
        const int row_global = row_offset + row_local;
        const int block_global = block_offset + kbx;

        // Calculate pointers to this block's data in Candle's layout
        // blocks: [out_dim, nblocks, 16] → blocks[row_global * nblocks * 16 + block_global * 16]
        // scales: [out_dim, nblocks]     → scales[row_global * nblocks + block_global]
        const uint8_t* block_data = blocks + (row_global * nblocks + block_global) * 16;
        const uint8_t  block_scale = scales[row_global * nblocks + block_global];

        // Load packed nibbles and dequantize using lookup table
        const int aux_q4 = get_int_b1(block_data, kqsx);  // Load 4 bytes = 8 nibbles
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);  // Lookup → INT8

        // Store INT8 values to shared memory
        // Layout: [row_local][col] with 65-column stride (padding for bank conflicts)
        const int k0 = kbx * 8 + kqsx;  // Starting column for this thread
        weight_qs_shared[row_local * 65 + k0 + 0]        = v.x;  // Even nibbles
        weight_qs_shared[row_local * 65 + k0 + QI_MXFP4] = v.y;  // Odd nibbles

        // Load E8M0 scale (one thread per block)
        if (kqsx == 0) {
            const float scale = ggml_cuda_e8m0_to_fp32(block_scale) * 0.5f;
            weight_scales_shared[row_local * 32 + kbx] = scale;
        }
    }

    __syncthreads();  // Ensure all threads finish loading before proceeding
}

// ============================================================================
// Simplified MMQ Vector Dot Product
// ============================================================================
// Direct computation without shared memory complexity
static __device__ __forceinline__ float vec_dot_mxfp4_simple(
    const uint8_t* block_ptr,       // FP4 packed weights
    const __nv_bfloat16* act,       // BF16 activations
    const float scale,              // Combined scale
    int len                         // Number of elements (32)
) {
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < len; ++i) {
        const int byte_idx = i >> 1;
        const uint8_t packed = block_ptr[byte_idx];
        const uint8_t nibble = (i & 1) ? (packed >> 4) & 0x0f : packed & 0x0f;
        const float w = (float)MXFP4_FP4_LUT[nibble];
        const float a = __bfloat162float(act[i]);
        sum += w * a;
    }

    return sum * scale;
}

// Vector dot product for MMQ with pre-dequantized INT8 weights
// This is the core computation for one K-block (32 elements) in MMQ
// Inputs:
//   - weight_int8: 32 INT8 values (already dequantized via kvalues_mxfp4)
//   - act_bf16: 32 BF16 activation values
//   - scale: FP32 scale factor (E8M0 converted to float)
// Output: FP32 dot product result
// Ref: llama.cpp uses similar approach with __dp4a for Q8_1 activations
static __device__ __forceinline__ float vec_dot_mmq_mxfp4_bf16(
    const int8_t* weight_int8,      // [32] INT8 weights from shared memory
    const __nv_bfloat16* act_bf16,  // [32] BF16 activations
    const float scale               // FP32 scale
) {
    float sum = 0.0f;

    // Process all 32 elements of one K-block
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const float w = (float)weight_int8[i];
        const float a = __bfloat162float(act_bf16[i]);
        sum += w * a;
    }

    return scale * sum;
}

// ============================================================================
// MMQ MXFP4 BF16 Matrix Multiplication Kernel (Optimized)
// ============================================================================
// High-performance tiled matmul using Multi-row Matrix Quantized (MMQ) approach
// adapted from llama.cpp for Candle's memory layout.
//
// Key optimizations from llama.cpp:
// 1. Cooperative weight loading into shared memory using INT8 lookup (kvalues_mxfp4)
// 2. INT8→BF16 dot products using __dp4a for 4x throughput
// 3. Tiled K-dimension processing (MMQ_ITER_K = 256 elements per iteration)
// 4. Multiple output elements per thread block for better occupancy
// 5. Bank conflict avoidance with carefully chosen strides
//
// Memory layout:
// - Activations: [rows, in_dim] in BF16
// - Weights: [out_dim, nblocks, 16] in packed FP4 + [out_dim, nblocks] scales
// - Output: [rows, out_dim] in BF16
//
// Grid: (rows, (out_dim + mmq_x - 1) / mmq_x)  where mmq_x = output columns per block
// Block: (32, nwarps)  typically (32, 4) = 128 threads
//
// Shared memory usage per block:
// - Weight INT8 values: mmq_y rows × (2*MMQ_TILE_NE_K + padding) ints
// - Weight scales: mmq_y rows × (MMQ_TILE_NE_K/QI_MXFP4) floats
// - Activation INT8: small buffers for __dp4a operations
//
// Template parameters:
//   mmq_x: Number of output columns per thread block (e.g., 64, 128)
//   mmq_y: Number of weight rows loaded into shared memory (e.g., 64, 128)
//
// Ref: llama.cpp mmq.cuh mul_mat_q kernel and load_tiles_mxfp4
#define MMQ_TILE_NE_K 32
#define MMQ_ITER_K 256
#define MMQ_NWARPS 4

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void load_tiles_mxfp4_fast(
    const uint8_t* __restrict__ blocks,          // [out_dim, nblocks, 16]
    const uint8_t* __restrict__ scales,          // [out_dim, nblocks]
    int* __restrict__ weight_qs_shared,          // Output: INT8 values
    float* __restrict__ weight_scales_shared,    // Output: FP32 scales
    const int row_offset,                        // Starting row in weight matrix
    const int block_offset,                      // Starting block in K dimension
    const int nblocks,                           // Total blocks per row
    const int num_rows_to_load                   // Actual rows to load (may be < mmq_y at boundary)
) {
    constexpr int nwarps = MMQ_NWARPS;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x; // 0..31 within warp

    // How many blocks we're loading in K dimension (typically MMQ_ITER_K / 32 = 8 blocks)
    constexpr int blocks_per_iter = MMQ_ITER_K / 32;

    // Each thread handles specific elements using cooperative loading
    // For MXFP4: each block has 32 elements packed in 16 bytes
    // We use get_int_from_table_16 to dequantize 8 FP4 values → 8 INT8 values per call

    // Map lanes to per-block loaders: 8 lanes (each handling 4 bytes) per 32-value block
    const int kbx  = lane_id / QI_MXFP4; // 0..7 blocks per iteration
    const int kqsx = lane_id % QI_MXFP4; // 0..3 4-byte chunks within block

    // Load quantized weights and scales
    #pragma unroll
    for (int row_local = warp_id; row_local < num_rows_to_load; row_local += nwarps) {
        const int row_global = row_offset + row_local;

        // Load weights for blocks_per_iter consecutive blocks in K dimension
        if (kbx < blocks_per_iter) {
            const int block_global = block_offset + kbx;
            if (block_global < nblocks) {
                // Candle layout: blocks[row_global][block_global][16 bytes]
                const uint8_t* block_data = blocks + (row_global * nblocks + block_global) * 16;

                // Load 4 bytes = 8 FP4 nibbles, dequantize to INT8 using lookup table
                const int aux_q4 = get_int_b1(block_data, kqsx);
                const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

                // Store to shared memory in SEQUENTIAL order (our BF16 activations are sequential)
                // get_int_from_table_16 returns: v.x = even nibbles, v.y = odd nibbles
                // Interleave bytes to get sequential order: nibble0,nibble1,nibble2,...
                const int row_stride_ints = (2 * MMQ_TILE_NE_K + 1);  // 65 ints
                int8_t* row_bytes = reinterpret_cast<int8_t*>(&weight_qs_shared[row_local * row_stride_ints]);
                const int byte_offset = kbx * 32 + kqsx * 8;  // 8 bytes per thread (one kqsx)

                // Interleave: even nibble, odd nibble, even nibble, odd nibble, ...
                const int8_t* vx_bytes = reinterpret_cast<const int8_t*>(&v.x);
                const int8_t* vy_bytes = reinterpret_cast<const int8_t*>(&v.y);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    row_bytes[byte_offset + 2*i + 0] = vx_bytes[i];  // even nibble (0,2,4,6)
                    row_bytes[byte_offset + 2*i + 1] = vy_bytes[i];  // odd nibble (1,3,5,7)
                }
            }
        }

        // Load scale (only first thread per block)
        if (kqsx == 0 && kbx < blocks_per_iter) {
            const int block_global = block_offset + kbx;
            if (block_global < nblocks) {
                const uint8_t scale_byte = scales[row_global * nblocks + block_global];
                const float scale = pow2_e8m0_device(scale_byte) * 0.5f;
                // Store scale: [row_local][block_idx]
                weight_scales_shared[row_local * blocks_per_iter + kbx] = scale;
            }
        }
    }
}

// Optimized vector dot product using INT8 weights and __dp4a
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void vec_dot_mxfp4_bf16_mmq(
    const int* __restrict__ weight_qs_shared,     // INT8 weights [mmq_y][2*MMQ_TILE_NE_K+1]
    const float* __restrict__ weight_scales_shared, // FP32 scales [mmq_y][blocks]
    const __nv_bfloat16* __restrict__ act_row,    // BF16 activations
    float* __restrict__ acc,                      // Accumulator [mmq_x]
    const int k_offset,                           // Starting position in K dimension
    const int weight_row,                         // Which weight row in shared memory
    const int blocks_this_iter                    // Number of blocks to process
) {
    constexpr int blocks_per_iter = MMQ_ITER_K / 32;

    // Process blocks_per_iter blocks (8 blocks = 256 elements)
    #pragma unroll
    for (int b = 0; b < blocks_per_iter; ++b) {
        if (b >= blocks_this_iter) break;

        const float scale = weight_scales_shared[weight_row * blocks_per_iter + b];
        // Each block is 32 INT8 values = 32 bytes = 8 ints
        const int shared_offset_ints = weight_row * (2 * MMQ_TILE_NE_K + 1) + b * 8;

        // Each block has 32 elements, process in groups of 4 (one int contains 4 INT8 values)
        #pragma unroll
        for (int k = 0; k < 32; k += 4) {
            // Load 4 INT8 weights from one int (4 bytes)
            const int k_int = k / 4;  // Convert element index to int index (0,1,2,...,7)
            int weight_int8_4 = weight_qs_shared[shared_offset_ints + k_int];

            // Load 4 BF16 activations and multiply with INT8 weights
            const int act_idx = k_offset + b * 32 + k;
            float sum_4 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int8_t w = ((int8_t*)&weight_int8_4)[i];
                const float a = __bfloat162float(act_row[act_idx + i]);
                sum_4 += (float)w * a;
            }

            *acc += scale * sum_4;
        }
    }
}

// Main MMQ kernel - tiled matrix multiplication for MXFP4
// Uses shared memory tiling and cooperative loading for maximum performance
// This is a __device__ function meant to be called from within a __global__ kernel
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void matmul_mxfp4_bf16_mmq_tiled(
    const __nv_bfloat16* __restrict__ act,    // [rows, in_dim]
    const uint8_t* __restrict__ blocks,       // [out_dim, nblocks, 16]
    const uint8_t* __restrict__ scales,       // [out_dim, nblocks]
    __nv_bfloat16* __restrict__ out,          // [rows, out_dim]
    const int rows,
    const int out_dim,
    const int nblocks,
    const int in_dim,
    const int act_row_stride,
    const int out_row_stride
) {
    constexpr int nwarps = MMQ_NWARPS;
    constexpr int blocks_per_iter = MMQ_ITER_K / 32;  // 8 blocks = 256 elements

    // Shared memory for weights (INT8 + scales)
    // Layout: [mmq_x][2*MMQ_TILE_NE_K+1] for INT8 weights (65 stride for bank conflict avoidance)
    // We load mmq_x weight rows (one per output column in the tile)
    __shared__ int weight_qs_shared[mmq_x * (2 * MMQ_TILE_NE_K + 1)];
    __shared__ float weight_scales_shared[mmq_x * blocks_per_iter];

    // Thread block computes output tile [row, row+mmq_y) × [col, col+mmq_x)
    const int row_base = blockIdx.x * mmq_y;
    const int col_base = blockIdx.y * mmq_x;

    // Accumulator: each thread handles multiple outputs
    // Size: (mmq_y/warp_size) × (mmq_x/nwarps) when mmq_y >= WARP_SIZE
    // Size: mmq_y × (mmq_x/nwarps) when mmq_y < WARP_SIZE
    constexpr int acc_rows = (mmq_y >= WARP_SIZE) ? (mmq_y / WARP_SIZE) : mmq_y;
    constexpr int acc_cols = mmq_x / nwarps;
    float acc[acc_rows][acc_cols];

    // Initialize accumulator to zero
    #pragma unroll
    for (int i = 0; i < acc_rows; ++i) {
        #pragma unroll
        for (int j = 0; j < acc_cols; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Outer loop over K dimension in tiles of MMQ_ITER_K elements
    for (int k_block = 0; k_block < nblocks; k_block += blocks_per_iter) {
        const int blocks_this_iter = min(blocks_per_iter, nblocks - k_block);

        // Cooperatively load mmq_x weight rows into shared memory
        // Each thread block loads blocks_per_iter blocks (256 elements) for mmq_x output columns
        load_tiles_mxfp4_fast<mmq_x, mmq_y>(
            blocks, scales,
            weight_qs_shared, weight_scales_shared,
            col_base,           // row_offset in weight matrix (output column dimension)
            k_block,            // block_offset in K dimension
            nblocks,            // total blocks per row
            min(mmq_x, out_dim - col_base)  // num_rows_to_load (mmq_x weight rows)
        );

        __syncthreads();

        // Compute partial dot products using loaded weights
        // Each thread processes its assigned outputs based on threadIdx
        // Note: Only num_rows_to_load weight rows were loaded into shared memory
        const int num_rows_loaded = min(mmq_x, out_dim - col_base);

        #pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j_local = j0 + threadIdx.y;

            if (j_local >= mmq_x || col_base + j_local >= out_dim || j_local >= num_rows_loaded) {
                continue;
            }

            #pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i_local = i0 + threadIdx.x;

                if (i_local >= mmq_y || row_base + i_local >= rows) {
                    continue;
                }

                // Pointer to activations for this row
                const __nv_bfloat16* act_row = act + (row_base + i_local) * act_row_stride;

                // Accumulator index in the acc array
                // When mmq_y >= WARP_SIZE: multiple i0 iterations, acc_idx_i = i0 / WARP_SIZE
                // When mmq_y < WARP_SIZE: single i0=0, acc_idx_i = i_local (one acc row per output row)
                const int acc_idx_i = (mmq_y >= WARP_SIZE) ? (i0 / WARP_SIZE) : i_local;
                const int acc_idx_j = j0 / nwarps;

                // Process blocks_per_iter blocks (typically 8 blocks = 256 elements)
                #pragma unroll
                for (int b = 0; b < blocks_per_iter; ++b) {
                    if (b >= blocks_this_iter) break;

                    // Get weight data from shared memory
                    const int weight_row = j_local;

                    // Check if this weight row was actually loaded into shared memory
                    if (weight_row >= num_rows_loaded) {
                        continue;  // Skip if this row wasn't loaded
                    }

                    const float scale = weight_scales_shared[weight_row * blocks_per_iter + b];
                    // shared_offset is in units of int (4 bytes), but we need byte offset for int8
                    const int shared_offset_ints = weight_row * (2 * MMQ_TILE_NE_K + 1);
                    const int8_t* weight_row_base = (const int8_t*)&weight_qs_shared[shared_offset_ints];
                    const int8_t* weight_int8 = weight_row_base + b * 32;

                    // Get activation data
                    const int k_offset = (k_block + b) * 32;

                    // Check if this block is within bounds for activations
                    if (k_offset + 32 > in_dim) {
                        // This shouldn't happen if nblocks is computed correctly,
                        // but add safety check to prevent out-of-bounds access
                        continue;
                    }

                    const __nv_bfloat16* act_block = act_row + k_offset;

                    // Compute dot product for one block (32 elements)
                    float block_dot = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 32; ++k) {
                        const float w = (float)weight_int8[k];
                        const float a = __bfloat162float(act_block[k]);
                        block_dot += w * a;
                    }

                    acc[acc_idx_i][acc_idx_j] += scale * block_dot;
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory with thread-to-output mapping
    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j_local = j0 + threadIdx.y;
        const int col = col_base + j_local;

        if (col >= out_dim) {
            continue;  // Skip this column, not all threads
        }

        #pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i_local = i0 + threadIdx.x;
            const int row = row_base + i_local;

            if (row >= rows) {
                continue;
            }

            // Only write if this thread actually computed values
            // For mmq_y < WARP_SIZE, acc_idx_i = i_local, so we need i_local < mmq_y
            // For mmq_y >= WARP_SIZE, acc_idx_i = i0 / WARP_SIZE, which is always valid
            if (mmq_y < WARP_SIZE && i_local >= mmq_y) {
                continue;  // This thread didn't compute anything
            }

            const int acc_idx_i = (mmq_y >= WARP_SIZE) ? (i0 / WARP_SIZE) : i_local;
            const int acc_idx_j = j0 / nwarps;

            out[row * out_row_stride + col] = __float2bfloat16(acc[acc_idx_i][acc_idx_j]);
        }
    }
}

// Instantiate the MMQ kernel with default parameters for Candle's API
// This provides a drop-in replacement with the same signature as matmul_mxfp4_bf16
// Grid should be configured as: grid=((rows+mmq_y-1)/mmq_y, (out_dim+mmq_x-1)/mmq_x), block=(32, nwarps)
extern "C" __global__ void matmul_mxfp4_bf16_mmq(
    const __nv_bfloat16* __restrict__ act,    // [rows, in_dim]
    const uint8_t* __restrict__ blocks,       // [out_dim, nblocks, 16]
    const uint8_t* __restrict__ scales,       // [out_dim, nblocks]
    __nv_bfloat16* __restrict__ out,          // [rows, out_dim]
    const int rows,
    const int out_dim,
    const int nblocks,
    const int in_dim,
    const int act_row_stride,
    const int out_row_stride
) {
    // Use minimal tile size for testing: mmq_x=64, mmq_y=2
    // For production, increase to mmq_y=64 or 128
    matmul_mxfp4_bf16_mmq_tiled<64, 32>(
        act, blocks, scales, out,
        rows, out_dim, nblocks, in_dim,
        act_row_stride, out_row_stride
    );
}

extern "C" __global__ void matmul_mxfp4_bf16(
    const __nv_bfloat16* __restrict__ act,    // [rows, in_dim]
    const uint8_t* __restrict__ blocks,       // [out_dim, nblocks, 16]
    const uint8_t* __restrict__ scales,       // [out_dim, nblocks]
    __nv_bfloat16* __restrict__ out,          // [rows, out_dim]
    const int rows,
    const int out_dim,
    const int nblocks,
    const int in_dim,
    const int act_row_stride,
    const int out_row_stride
) {
    constexpr int warp_size = 32;
    constexpr int elems_per_block = 32;
    constexpr int bytes_per_block = 16;
    constexpr int cols_per_warp = 4;
    constexpr int lanes_per_column = warp_size / cols_per_warp; // 8 lanes cooperate per column.
    constexpr int elems_per_lane = elems_per_block / lanes_per_column; // 4 values per lane.
    constexpr int tile_k_blocks = MATMUL_MXFP4_TILE_K_BLOCKS; // decode tile_k_blocks*32 activations at a time.

    const int warps_per_block = blockDim.y;
    const int tile_cols = cols_per_warp * warps_per_block;

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tile_col = blockIdx.y;
    const int lane = threadIdx.x; // [0, 31]
    const int warp = threadIdx.y; // [0, warps_per_block)

    const int lane_column = lane / lanes_per_column;      // local column index within the warp.
    const int lane_offset = lane % lanes_per_column;       // which subset of the block we handle.

    const int column = tile_col * tile_cols + warp * cols_per_warp + lane_column;
    const bool column_active = column < out_dim;

    const __nv_bfloat16* act_row = act + (size_t)row * (size_t)act_row_stride;
    const uint8_t* block_col = column_active
        ? blocks + (size_t)column * (size_t)nblocks * (size_t)bytes_per_block
        : nullptr;
    const uint8_t* scale_col = column_active
        ? scales + (size_t)column * (size_t)nblocks
        : nullptr;

    extern __shared__ float act_shared[];

    float acc = 0.0f;

    for (int block_start = 0; block_start < nblocks; block_start += tile_k_blocks) {
        const int blocks_this_iter = min(tile_k_blocks, nblocks - block_start);
        const int shared_span = blocks_this_iter * elems_per_block;

        for (int idx = warp * warp_size + lane; idx < shared_span; idx += warps_per_block * warp_size) {
            const int act_idx = block_start * elems_per_block + idx;
            const float value = act_idx < in_dim
                ? __bfloat162float(act_row[act_idx])
                : 0.0f;
            act_shared[idx] = value;
        }

        __syncthreads();

        if (column_active) {
#pragma unroll
            for (int b = 0; b < tile_k_blocks; ++b) {
                if (b >= blocks_this_iter) {
                    break;
                }

                const uint8_t* block_ptr = block_col + (size_t)(block_start + b) * (size_t)bytes_per_block;
                const float scale = pow2_e8m0_device(scale_col[block_start + b]) * 0.5f;

#pragma unroll
                for (int step = 0; step < elems_per_lane; ++step) {
                    const int elem = lane_offset * elems_per_lane + step; // [0, 31]
                    const int byte_index = elem >> 1;
                    const uint8_t packed = block_ptr[byte_index];
                    const uint8_t nibble = (elem & 1) ? (packed >> 4) & 0x0f : (packed & 0x0f);
                    const float weight = scale * (float)MXFP4_FP4_LUT[nibble];
                    const float act_val = act_shared[b * elems_per_block + elem];
                    acc += act_val * weight;
                }
            }
        }

        __syncthreads();
    }

    unsigned mask = 0xffffffffu;
    for (int offset = lanes_per_column / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(mask, acc, offset);
        acc += other;
    }

    if (column_active && lane_offset == 0) {
        out[(size_t)row * (size_t)out_row_stride + column] = __float2bfloat16(acc);
    }
}

extern "C" __global__ void matmul_mxfp4_q8_1(
    const block_q8_1* __restrict__ act,        // [rows, act_blocks_stride]
    const uint8_t* __restrict__ blocks,        // [out_dim, nblocks, 16]
    const uint8_t* __restrict__ scales,        // [out_dim, nblocks]
    __nv_bfloat16* __restrict__ out,           // [rows, out_dim]
    const int rows,
    const int out_dim,
    const int nblocks,
    const int act_blocks_stride,
    const int out_row_stride
) {
    constexpr int warp_size = 32;
    constexpr int elems_per_block = 32;
    constexpr int bytes_per_block = 16;
    constexpr int cols_per_warp = 4;
    constexpr int lanes_per_column = warp_size / cols_per_warp; // 8
    constexpr int elems_per_lane = elems_per_block / lanes_per_column; // 4
    constexpr int tile_k_blocks = MATMUL_MXFP4_TILE_K_BLOCKS;

    const int warps_per_block = blockDim.y;
    const int tile_cols = cols_per_warp * warps_per_block;

    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tile_col = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int lane_column = lane / lanes_per_column;
    const int lane_offset = lane % lanes_per_column;

    const int column = tile_col * tile_cols + warp * cols_per_warp + lane_column;
    const bool column_active = column < out_dim;

    const block_q8_1* act_row = act + (size_t)row * (size_t)act_blocks_stride;
    const uint8_t* block_col = column_active
        ? blocks + (size_t)column * (size_t)nblocks * (size_t)bytes_per_block
        : nullptr;
    const uint8_t* scale_col = column_active
        ? scales + (size_t)column * (size_t)nblocks
        : nullptr;

    extern __shared__ uint8_t shared_raw[];
    block_q8_1* act_shared = reinterpret_cast<block_q8_1*>(shared_raw);
    int* act_shared_int = reinterpret_cast<int*>(act_shared);
    const int ints_per_qblock = sizeof(block_q8_1) / sizeof(int);

    float acc = 0.0f;

    for (int block_start = 0; block_start < nblocks; block_start += tile_k_blocks) {
        const int blocks_this_iter = min(tile_k_blocks, nblocks - block_start);

        const int total_threads = warps_per_block * warp_size;
        for (int idx = warp * warp_size + lane; idx < tile_k_blocks * ints_per_qblock; idx += total_threads) {
            int local_block = idx / ints_per_qblock;
            int int_offset = idx % ints_per_qblock;
            int global_block = block_start + local_block;
            int value = 0;
            if (global_block < nblocks) {
                const block_q8_1* src = act_row + global_block;
                const int* src_int = reinterpret_cast<const int*>(src);
                value = src_int[int_offset];
            }
            act_shared_int[idx] = value;
        }

        __syncthreads();

        if (column_active) {
#pragma unroll
            for (int b = 0; b < tile_k_blocks; ++b) {
                if (b >= blocks_this_iter) {
                    break;
                }

                const float base_scale = pow2_e8m0_device(scale_col[block_start + b]) * 0.5f;
                if (base_scale == 0.0f) {
                    continue;
                }

                const block_q8_1* act_block = &act_shared[b];
                const float act_delta = __half2float(act_block->ds.x);
                if (act_delta == 0.0f) {
                    continue;
                }

                const uint8_t* block_ptr = block_col + (size_t)(block_start + b) * (size_t)bytes_per_block;

#if __CUDA_ARCH__ >= 610
                int8_t w_vals[elems_per_lane];
                int8_t x_vals[elems_per_lane];
#pragma unroll
                for (int step = 0; step < elems_per_lane; ++step) {
                    const int elem = lane_offset * elems_per_lane + step;
                    const int byte_index = elem >> 1;
                    const uint8_t packed = block_ptr[byte_index];
                    const uint8_t nibble = (elem & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                    w_vals[step] = MXFP4_INT8_LUT[nibble];
                    x_vals[step] = act_block->qs[elem];
                }
                const int weight_pack = pack_int8x4(w_vals[0], w_vals[1], w_vals[2], w_vals[3]);
                const int act_pack = pack_int8x4(x_vals[0], x_vals[1], x_vals[2], x_vals[3]);
                const float scale = base_scale * act_delta;
                const int dot = __dp4a(weight_pack, act_pack, 0);
                acc += scale * static_cast<float>(dot);
#else
#pragma unroll
                for (int step = 0; step < elems_per_lane; ++step) {
                    const int elem = lane_offset * elems_per_lane + step;
                    const int byte_index = elem >> 1;
                    const uint8_t packed = block_ptr[byte_index];
                    const uint8_t nibble = (elem & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                    const float weight = base_scale * (float)MXFP4_FP4_LUT[nibble];
                    const float act_val = static_cast<float>(act_block->qs[elem]) * act_delta;
                    acc += weight * act_val;
                }
#endif
            }
        }

        __syncthreads();
    }

    unsigned mask = 0xffffffffu;
    for (int offset = lanes_per_column / 2; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(mask, acc, offset);
        acc += other;
    }

    if (column_active && lane_offset == 0) {
        out[(size_t)row * (size_t)out_row_stride + column] = __float2bfloat16(acc);
    }
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#else

    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    const int8_t * s = bq5_K->scales;

    const float d = bq5_K->d;

    const float d8_1 = __low2half(bq8_1[0].ds);
    const float d8_2 = __low2half(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * ql = (const int *)bq5_K->qs + (iqs/2);
    const int vl1 = ql[0];
    const int vl2 = ql[4];

    const int step = 4 * (iqs/2); // 0, 4, 8, 12
    const int im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    const int in = step%8; // 0, 4, 0, 4
    const int vh = (*((const int *)(bq5_K->qh + in))) >> im;

    const int v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    const int v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    const int v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    const int v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    const float sumf_d = d8_1 * (ggml_cuda_dp4a(ui1, v1, 0) * s[0] + ggml_cuda_dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (ggml_cuda_dp4a(ui3, v3, 0) * s[2] + ggml_cuda_dp4a(ui4, v4, 0) * s[3]);

    return d * sumf_d;
#endif
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

// https://github.com/ggerganov/llama.cpp/blob/c50a82ce0f71558cbb8e555146ba124251504b38/ggml-cuda/mmvq.cu#L4
typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);

template <int ncols_y, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __device__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int nwarps              = ncols_y <= 4 ? 4 : 2;
    constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    &x[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block) {
            dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

// batch size = 1
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 2
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 3
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 4
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 5
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 6
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 7
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 8
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = blockDim.y*blockIdx.y + threadIdx.y;

    const int i_padded = iy*kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i_padded / QK8_1; // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

extern "C" __global__ void quantize_q8_1_bf16(const __nv_bfloat16 * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) {
        return;
    }

    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    const int i_padded = iy * kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *)vy;

    const int ib = i_padded / QK8_1;
    const int iqs = i_padded % QK8_1;

    const int64_t index = static_cast<int64_t>(iy) * kx + ix;
    const float xi = ix < kx ? __bfloat162float(x[index]) : 0.0f;

    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127.f;
    const int8_t q = amax == 0.0f ? 0 : lrintf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs != 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

// Kernels from https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/mmq.cu

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int  tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI5_0) + mmq_y/QI5_0];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}


template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_1) + mmq_y/QI5_1];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset < nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q8_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI8_0) + mmq_y/QI8_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q2_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI2_K) + mmq_y/QI2_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const int kbx = k / QI2_K;
    const int ky  = (k % QI2_K) * QR2_K;
    const float * y_df = (const float *) y_ds;

    int v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    const int kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    const int shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    for (int l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    const uint8_t * scales = ((const uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    const int index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    return vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q3_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI3_K) + mmq_y/QI3_K];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE/2)     + mmq_y/2];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const int kbx  = k / QI3_K;
    const int ky  = (k % QI3_K) * QR3_K;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    int v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        const int kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        const int shift = 2 * ((ky % 32) / 8);
        const int vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        const int vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        const int vlh = (vh << 2) & 0x04040404;

        v[l] = __vsubss4(vll, vlh);
    }

    const int index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_K) + mmq_y/QI4_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    const int index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_K) + mmq_y/QI5_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    const int index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q6_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI6_K) + mmq_y/QI6_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    const int index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    const int index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}


static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}


extern "C" __global__ void
    mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, true>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    const int nwarps = NWARPS_Q4_1_AMPERE;

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, true>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}


extern "C" __global__ void
    mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    const int nwarps = NWARPS_Q5_0_AMPERE;

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, true>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    const int nwarps = NWARPS_Q5_1_AMPERE;

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, true>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q8_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    const int nwarps = NWARPS_Q8_0_AMPERE;

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, true>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q2_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    const int nwarps = NWARPS_Q2_K_AMPERE;
    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, true>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q3_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    const int nwarps = NWARPS_Q3_K_AMPERE;
    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, true>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    const int nwarps = NWARPS_Q4_K_AMPERE;
    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, true>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    const int nwarps = NWARPS_Q5_K_AMPERE;
    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, true>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q6_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    const int nwarps = NWARPS_Q6_K_AMPERE;
    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, true>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

// Fused expert activation kernel for GPT-OSS
// Combines gate/up split + asymmetric clamp + sigmoid + element-wise ops
// Input: gate_up [batch, 2*expert_dim] (interleaved: even=gate, odd=up)
// Output: result [batch, expert_dim] = (up+1) * gate * sigmoid(alpha*gate)