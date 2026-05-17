// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace flash_model_architectures {
    // define dtype aliases
    using fp64 = double;
    using fp64_2 = double2;
    using fp64_4 = double4;

    using fp32 = float;
    using fp32_2 = float2;
    using fp32_4 = float4;

    using fp16 = half;
    using fp16_2 = half2;

    using bf16 = __nv_bfloat16;
    using bf16_2 = __nv_bfloat162;

    using int64 = long;
    using uint64 = ulong;
    using uint64_2 = ulong2;

    using uint8 = uint8_t;

    using int32 = int;
    using int32_4 = int4;

    using uint32 = uint;
    using uint32_2 = uint2;
    using uint32_4 = uint4;

    using int16 = short;
    using uint16 = ushort;
}  // namespace flash_model_architectures
