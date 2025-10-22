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
