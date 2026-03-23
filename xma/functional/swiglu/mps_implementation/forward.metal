// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <metal_stdlib>

using namespace metal;

static inline float sigmoid(float g) {
    return 1.0f / (1.0f + exp(-g));
}

kernel void swiglu_forward_fp32(device const float *g [[buffer(0)]],
                                device const float *u [[buffer(1)]],
                                device float *y [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    float _g = g[id];
    y[id] = u[id] * _g * sigmoid(_g);
}

kernel void swiglu_forward_fp16(device const half *g [[buffer(0)]],
                                device const half *u [[buffer(1)]],
                                device half *y [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    float _g = float(g[id]);
    y[id] = half(float(u[id]) * _g * sigmoid(_g));
}

kernel void swiglu_forward_bf16(device const bfloat *g [[buffer(0)]],
                                device const bfloat *u [[buffer(1)]],
                                device bfloat *y [[buffer(2)]],
                                uint id [[thread_position_in_grid]]) {
    float _g = float(g[id]);
    y[id] = bfloat(float(u[id]) * _g * sigmoid(_g));
}
