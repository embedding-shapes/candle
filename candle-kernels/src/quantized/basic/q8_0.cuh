#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * sumi;
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

