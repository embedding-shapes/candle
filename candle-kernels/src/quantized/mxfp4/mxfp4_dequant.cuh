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
