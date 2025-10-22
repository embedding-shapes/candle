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
