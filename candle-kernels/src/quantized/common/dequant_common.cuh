extern "C" __global__ void dequantize_block_##QNAME##_f32(const void * __restrict__ vx, float * __restrict__ y) { \
  dequantize_block_##QNAME(vx, y); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_f16(const void * __restrict__ vx, half * __restrict__ y) { \
  dequantize_block_##QNAME(vx, y); \
} \

#define DEQUANTIZE(QNAME) \
extern "C" __global__ void dequantize_block_##QNAME##_f32(const void * __restrict__ vx, float * __restrict__ y, const int k) { \
  dequantize_block_##QNAME(vx, y, k); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_f16(const void * __restrict__ vx, half * __restrict__ y, const int k) { \
  dequantize_block_##QNAME(vx, y, k); \
} \

DEQUANTIZE_K(q2_K)
DEQUANTIZE_K(q3_K)
DEQUANTIZE_K(q4_K)
DEQUANTIZE_K(q5_K)
DEQUANTIZE_K(q6_K)
DEQUANTIZE_K(q8_K)
DEQUANTIZE(q4_0)
DEQUANTIZE(q4_1)
DEQUANTIZE(q5_0)
DEQUANTIZE(q5_1)
DEQUANTIZE(q8_0)

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q4_0_Q8_1_MMVQ 2
