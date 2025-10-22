__device__ __constant__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// Extract 32-bit int from byte array at 4-byte aligned position
// Used for reading packed FP4 nibbles from MXFP4 blocks
// Ref: llama.cpp vecdotq.cuh:7
static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;
    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;
    return x32;
}

// Ultra-efficient 4-bit nibble lookup using __byte_perm instruction
// Takes 8 nibbles (4-bit indices) packed in q4, returns int2 with:
//   .x = bytes at even indices (0,2,4,6)
//   .y = bytes at odd indices (1,3,5,7)
// This is CRITICAL for MMQ performance - enables fast MXFP4 dequantization
// Ref: llama.cpp vecdotq.cuh:34-94
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
#if defined(__HIP_PLATFORM_AMD__)
    // AMD ROCm implementation using __builtin_amdgcn_perm
    const uint32_t *values = (const uint32_t *)table;
    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // Lookup in lower half (indices 0-7)
    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);

    // Lookup in upper half (indices 8-15)
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    // Select based on MSB of each index nibble
    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);

    return make_int2(res_x, res_y);
#else
    // NVIDIA CUDA implementation using __byte_perm
    // __byte_perm selects bytes using 3-bit indices (lower 16 bits of 3rd arg)
    // To handle the 4th bit, we do two __byte_perm calls for low/high halves,
    // then select between them based on the 4th bit
    const uint32_t * table32 = (const uint32_t *) table;

    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));

#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;

        // Select from low 64 bits of table using low 3 bits
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        // Select from high 64 bits of table using low 3 bits
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        // Select between low and high based on 4th bit
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }

    // tmp contains bytes in same order as nibbles in q4
    // Now reorder to put even/odd indices into separate ints
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}

// Convert E8M0 scale (8-bit exponent, 0 mantissa bits) to float32
// E8M0 stores just the exponent byte of IEEE 754, used as block-wise scale in MXFP4
// Special case: x=0 maps to 2^-127 (smallest normal float32)
// Otherwise: directly use x as exponent bits (shift left 23 to FP32 position)
// Ref: llama.cpp common.cuh:604
static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
    // Manual conversion: E8M0 byte becomes exponent bits of float32
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;  // 2^-127 (smallest positive normal float)
    } else {
        bits = (uint32_t) x << 23;  // Move exponent to bits [30:23]
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// MXFP4 constants (ref: llama.cpp ggml-common.h)
#define QK_MXFP4  32  // 32 FP4 values per block
#define QR_MXFP4  2   // Ratio (related to expansion when dequantizing)
#define QI_MXFP4  4   // 4-way interleaving in quantized representation (QK_MXFP4/(4*QR_MXFP4))

