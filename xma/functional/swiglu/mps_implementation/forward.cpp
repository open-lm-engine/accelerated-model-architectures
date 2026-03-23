// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void swiglu_forward_mps_kernel(device T *g [[buffer(0)]],
                                      device T *u [[buffer(1)]],
                                      device T *y [[buffer(2)]],
                                      uint id [[thread_position_in_grid]]) {
    _g = g[id];
    y[id] = u[id] * _g * sigmoid(_g);
}
