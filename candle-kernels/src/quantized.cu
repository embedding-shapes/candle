// Kernels adapted from llama.cpp ggml-cuda.cu
// https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda.cu
//
// Refactored into modular files for better maintainability

#include "quantized_common.cuh"
#include "quantized_mmq_templates.cuh"
#include "quantized_dequant_vec_dot.cuh"
#include "quantized_impl.cuh"
