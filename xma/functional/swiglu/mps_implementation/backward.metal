// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

struct GradPair {
    float dg;
    float du;
};

static inline GradPair swiglu_backward(float g, float u, float dy) {
    float g_sigmoid = sigmoid(g);
    float g_silu = g * g_sigmoid;
    return {dy * u * (g_sigmoid + g_silu * (1.0f - g_sigmoid)), dy * g_silu};
}

kernel void swiglu_backward_fp32(device const float *g [[buffer(0)]],
                                 device const float *u [[buffer(1)]],
                                 device const float *dy [[buffer(2)]],
                                 device float *dg [[buffer(3)]],
                                 device float *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    auto grads = swiglu_backward(g[id], u[id], dy[id]);
    dg[id] = grads.dg;
    du[id] = grads.du;
}

kernel void swiglu_backward_fp16(device const half *g [[buffer(0)]],
                                 device const half *u [[buffer(1)]],
                                 device const half *dy [[buffer(2)]],
                                 device half *dg [[buffer(3)]],
                                 device half *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    auto grads = swiglu_backward(float(g[id]), float(u[id]), float(dy[id]));
    dg[id] = half(grads.dg);
    du[id] = half(grads.du);
}

kernel void swiglu_backward_bf16(device const bfloat *g [[buffer(0)]],
                                 device const bfloat *u [[buffer(1)]],
                                 device const bfloat *dy [[buffer(2)]],
                                 device bfloat *dg [[buffer(3)]],
                                 device bfloat *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    auto grads = swiglu_backward(float(g[id]), float(u[id]), float(dy[id]));
    dg[id] = bfloat(grads.dg);
    du[id] = bfloat(grads.du);
}
