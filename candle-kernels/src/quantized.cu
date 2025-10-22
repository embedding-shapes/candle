// Kernels adapted from llama.cpp ggml-cuda.cu
// https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda.cu
//
// Refactored into modular files for better maintainability
// Structure follows REFACTOR_PLAN.md

// Core definitions and structures
#include "quantized/common/quantized_common.cuh"

// Device utility functions
#include "quantized/common/quantized_device_helpers.cuh"

// MXFP4-specific constants and helpers
#include "quantized/mxfp4/quantized_mxfp4_common.cuh"

// MMQ template infrastructure
#include "quantized/common/quantized_mmq_templates.cuh"

// Dequantization and vector dot product templates
#include "quantized/common/quantized_dequant_vec_dot.cuh"

// Individual Q-type vec_dot implementations
#include "quantized/basic/q4_0.cuh"
#include "quantized/basic/q4_1.cuh"
#include "quantized/basic/q5_0.cuh"
#include "quantized/basic/q5_1.cuh"
#include "quantized/basic/q8_0.cuh"
#include "quantized/k_quants/q2_k.cuh"
#include "quantized/k_quants/q3_k.cuh"
#include "quantized/k_quants/q4_k.cuh"
#include "quantized/k_quants/q5_k.cuh"
#include "quantized/k_quants/q6_k.cuh"

// MXFP4 kernel implementations
#include "quantized/mxfp4/mxfp4_dequant.cuh"
#include "quantized/mxfp4/mxfp4_load_tiles.cuh"
#include "quantized/mxfp4/mxfp4_matmul_mmq.cuh"
#include "quantized/mxfp4/mxfp4_matmul_direct.cuh"

// Q-type MMQ implementations (allocate/load tiles, vec_dot, mul_mat)
#include "quantized/mmq/q_types_mmq.cuh"

// Special kernels (fused expert activation)
#include "quantized/special/special.cuh"
