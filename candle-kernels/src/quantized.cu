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

// Individual Q-type vec_dot implementations
#include "q4_0.cuh"
#include "q4_1.cuh"
#include "q5_0.cuh"
#include "q5_1.cuh"
#include "q8_0.cuh"
#include "q2_k.cuh"
#include "q3_k.cuh"
#include "q4_k.cuh"
#include "q5_k.cuh"
#include "q6_k.cuh"

// MXFP4 kernel implementations
#include "mxfp4_dequant.cuh"
#include "mxfp4_load_tiles.cuh"
#include "mxfp4_matmul_mmq.cuh"
#include "mxfp4_matmul_direct.cuh"

// Q-type MMQ implementations (allocate/load tiles, vec_dot, mul_mat)
#include "q_types_mmq.cuh"

// Special kernels (fused expert activation)
#include "special.cuh"
