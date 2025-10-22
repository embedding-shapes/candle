// batch size = 4
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 5
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 6
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

