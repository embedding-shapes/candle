// Kernels adapted from llama.cpp ggml-cuda.cu
// https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda.cu
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>
#include<math.h>

#ifndef MATMUL_MXFP4_TILE_K_BLOCKS
#define MATMUL_MXFP4_TILE_K_BLOCKS 32
#endif

#define GGML_UNUSED(x) (void)(x)
#define GGML_CUDA_ASSUME(x)

#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

#undef GGML_CUDA_F16
#define GGML_CUDA_DMMV_X 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define K_QUANTS_PER_ITERATION 2

typedef uint16_t ggml_fp16_t;
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);

#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_0_AMPERE 4
#define  MMQ_Y_Q4_0_AMPERE 32
#define NWARPS_Q4_0_AMPERE 4
#else
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_1_AMPERE 4
#define  MMQ_Y_Q4_1_AMPERE 32
#define NWARPS_Q4_1_AMPERE 4
#else
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_0_AMPERE 4
#define  MMQ_Y_Q5_0_AMPERE 32
#define NWARPS_Q5_0_AMPERE 4
#else
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_1_AMPERE 4
#define  MMQ_Y_Q5_1_AMPERE 32
#define NWARPS_Q5_1_AMPERE 4
#else
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q8_0_AMPERE 4
#define  MMQ_Y_Q8_0_AMPERE 32
#define NWARPS_Q8_0_AMPERE 4
#else
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q2_K_AMPERE 4
#define  MMQ_Y_Q2_K_AMPERE 32
#define NWARPS_Q2_K_AMPERE 4
#else
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q3_K_AMPERE 4
#define  MMQ_Y_Q3_K_AMPERE 32
#define NWARPS_Q3_K_AMPERE 4
#else
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_K_AMPERE 4
#define  MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#else
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_K_AMPERE 4
#define  MMQ_Y_Q5_K_AMPERE 32
#define NWARPS_Q5_K_AMPERE 4
#else
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q6_K_AMPERE 4
#define  MMQ_Y_Q6_K_AMPERE 32
#define NWARPS_Q6_K_AMPERE 4
#else
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8


// QK = number of values after dequantization
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
    half2   ds;             // ds.x = delta, ds.y = sum
    int8_t  qs[QK8_0];      // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0, "wrong q8_1 block size/padding");

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);
typedef void (*allocate_tiles_cuda_t)(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc);
typedef void (*load_tiles_cuda_t)(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row);
typedef float (*vec_dot_q_mul_mat_cuda_t)(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ms, const int & i, const int & j, const int & k);

#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
#ifdef GGML_QKK_64
    uint8_t scales[2]; // scales, quantized with 8 bits
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    half d;             // super-block scale
} block_q3_K;
//static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + K_SCALE_SIZE, "wrong q3_K block size/padding");

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
#ifdef GGML_QKK_64
typedef struct {
    half    dm[2];             // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == sizeof(half2) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
    half2 dm;                  // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2, "wrong q4_K block size/padding");
#endif

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
#ifdef GGML_QKK_64
typedef struct {
    half d;                  // super-block scale
    int8_t scales[QK_K/16];  // block scales
    uint8_t qh[QK_K/8];      // quants, high bit
    uint8_t qs[QK_K/2];      // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K/2 + QK_K/8 + QK_K/16, "wrong q5_K block size/padding");
#else
typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
#endif

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;         // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + 13*QK_K/16, "wrong q6_K block size/padding");

// In llama.cpp this is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

