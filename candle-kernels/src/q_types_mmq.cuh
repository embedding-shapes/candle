// Q-type MMQ implementations - split into logical modules
// This file includes all the MMQ kernel implementations for different Q-types

// vec_dot helper functions for Q5_K and Q6_K (MMVQ variants)
#include "q_mmq_vec_dot_helpers.cuh"

// mul_mat_vec template and batch size variants (MMVQ kernels)
#include "q_mmq_mul_mat_vec_1_3.cuh"
#include "q_mmq_mul_mat_vec_4_6.cuh"
#include "q_mmq_mul_mat_vec_7_8.cuh"

// MMQ implementations for individual Q-types
#include "q_mmq_q5_0.cuh"
#include "q_mmq_q5_1.cuh"
#include "q_mmq_q8_0.cuh"
#include "q_mmq_q2_k.cuh"
#include "q_mmq_q3_k.cuh"
#include "q_mmq_q4_k.cuh"
#include "q_mmq_q5_k.cuh"
#include "q_mmq_q6_k.cuh"
#include "q_mmq_q4_0_q4_1.cuh"

// Kernel launch functions
#include "q_mmq_kernels.cuh"
