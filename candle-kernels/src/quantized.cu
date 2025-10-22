// Kernels adapted from llama.cpp ggml-cuda.cu
// https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda.cu
//
// Refactored into modular files for better maintainability
// Structure follows REFACTOR_PLAN.md

// Core definitions and structures
#include "quantized_common.cuh"

// Device utility functions
#include "quantized_device_helpers.cuh"

// MXFP4-specific constants and helpers
#include "quantized_mxfp4_common.cuh"

// MMQ template infrastructure
#include "quantized_mmq_templates.cuh"

// Dequantization and vector dot product templates
#include "quantized_dequant_vec_dot.cuh"

// Q-type implementations (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K through Q6_K)
#include "q_types.cuh"

// MXFP4 kernel implementations
#include "mxfp4.cuh"

// Special kernels (fused expert activation)
#include "special.cuh"
