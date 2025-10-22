//================================== k-quants
// Split into per-Q-type files for better maintainability

// Q2_K dequantization
#include "dequant_q2_k.cuh"

// Q3_K dequantization
#include "dequant_q3_k.cuh"

// Q4_K dequantization (includes get_scale_min_k4 helper used by Q5_K)
#include "dequant_q4_k.cuh"

// Q5_K dequantization (uses get_scale_min_k4 from Q4_K)
#include "dequant_q5_k.cuh"

// Q6_K dequantization
#include "dequant_q6_k.cuh"

// Misc Q8 and Q5 dequantization functions
#include "dequant_q8_misc.cuh"
