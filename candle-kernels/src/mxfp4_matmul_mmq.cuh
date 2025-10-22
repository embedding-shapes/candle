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

