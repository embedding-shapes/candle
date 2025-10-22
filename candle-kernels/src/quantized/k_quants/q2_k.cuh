#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float & d8) {

    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = ggml_cuda_dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m    = ggml_cuda_dp4a(m,    u[i], sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const float2 dm2f = __half22float2(dm2);

    return d8 * (dm2f.x*sumi_d - dm2f.y*sumi_m);
}

