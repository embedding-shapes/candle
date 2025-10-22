extern "C" __global__ void fused_expert_activation_bf16(
    const __nv_bfloat16* __restrict__ gate_up,  // [batch, 2*expert_dim]
    __nv_bfloat16* __restrict__ output,          // [batch, expert_dim]
    const int batch,
    const int expert_dim,
    const float alpha,
    const float limit
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elems = batch * expert_dim;

    if (idx >= total_elems) {
        return;
    }

    const int b = idx / expert_dim;
    const int e = idx % expert_dim;

    // Read gate (even index) and up (odd index)
    const int gate_idx = b * (2 * expert_dim) + e * 2;
    const int up_idx = gate_idx + 1;

    float gate = __bfloat162float(gate_up[gate_idx]);
    float up = __bfloat162float(gate_up[up_idx]);

    // Asymmetric clamp: gate has only max, up has both min and max
    gate = fminf(gate, limit);
    up = fmaxf(fminf(up, limit), -limit);

    // Compute activation: (up + 1.0) * gate * sigmoid(alpha * gate)
    const float gate_alpha = alpha * gate;
    const float sig = 1.0f / (1.0f + expf(-gate_alpha));
    const float glu = gate * sig;
    const float up_plus = up + 1.0f;
    const float result = up_plus * glu;

    output[idx] = __float2bfloat16(result);
}
