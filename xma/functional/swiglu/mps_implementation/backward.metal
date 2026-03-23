// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************
//
// NOTE: This file is concatenated after forward.metal at compile time.
// The #include <metal_stdlib>, using namespace metal, and sigmoid() helper
// are defined there — do not redeclare them here.
//
// Backward pass for SwiGLU: y = u * g * sigmoid(g)
// Given dy, g, u, computes:
//   dg = dy * u * (sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g)))
//   du = dy * g * sigmoid(g)

kernel void swiglu_backward_fp32(device const float *g [[buffer(0)]],
                                 device const float *u [[buffer(1)]],
                                 device const float *dy [[buffer(2)]],
                                 device float *dg [[buffer(3)]],
                                 device float *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    float _g = g[id];
    float g_sigmoid = sigmoid(_g);
    float g_silu = _g * g_sigmoid;
    dg[id] = dy[id] * u[id] * (g_sigmoid + g_silu * (1.0f - g_sigmoid));
    du[id] = dy[id] * g_silu;
}

kernel void swiglu_backward_fp16(device const half *g [[buffer(0)]],
                                 device const half *u [[buffer(1)]],
                                 device const half *dy [[buffer(2)]],
                                 device half *dg [[buffer(3)]],
                                 device half *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    float _g = float(g[id]);
    float _u = float(u[id]);
    float _dy = float(dy[id]);
    float g_sigmoid = sigmoid(_g);
    float g_silu = _g * g_sigmoid;
    dg[id] = half(_dy * _u * (g_sigmoid + g_silu * (1.0f - g_sigmoid)));
    du[id] = half(_dy * g_silu);
}

kernel void swiglu_backward_bf16(device const bfloat *g [[buffer(0)]],
                                 device const bfloat *u [[buffer(1)]],
                                 device const bfloat *dy [[buffer(2)]],
                                 device bfloat *dg [[buffer(3)]],
                                 device bfloat *du [[buffer(4)]],
                                 uint id [[thread_position_in_grid]]) {
    float _g = float(g[id]);
    float _u = float(u[id]);
    float _dy = float(dy[id]);
    float g_sigmoid = sigmoid(_g);
    float g_silu = _g * g_sigmoid;
    dg[id] = bfloat(_dy * _u * (g_sigmoid + g_silu * (1.0f - g_sigmoid)));
    du[id] = bfloat(_dy * g_silu);
}
