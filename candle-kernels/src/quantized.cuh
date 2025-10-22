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

// Individual Q-type implementations
#include "q4_0.cu"
#include "q4_1.cu"
#include "q5_0.cu"
#include "q5_1.cu"
#include "q8_0.cu"
#include "q2_k.cu"
#include "q3_k.cu"
#include "q4_k.cu"
#include "q5_k.cu"
#include "q6_k.cu"

// MXFP4 kernel implementations
#include "mxfp4.cuh"

// Special kernels (fused expert activation)
#include "special.cuh"
