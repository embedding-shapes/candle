static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}


#define WARP_SIZE 32
#define CUDART_HMAX     11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)

#define CC_PASCAL     600
#define MIN_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define CC_VOLTA      700
#define CC_OFFSET_AMD 1000000
#define CC_RDNA1      (CC_OFFSET_AMD + 1010)
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)
#define CC_RDNA3      (CC_OFFSET_AMD + 1100)

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= MIN_CC_DP4A
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}

// MXFP4 lookup table for FP4 -> INT8 conversion (e2m1 values doubled)
// Used with __dp4a for efficient int8 dot products
// Ref: llama.cpp ggml-common.h:1094