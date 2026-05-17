// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "alias.h"
#include "common.h"
#include "cutlass/half.h"

namespace flash_model_architectures {
    template <>
    struct DType<c10::Half> {
        using c10_dtype = c10::Half;
        using nv_dtype = fp16;
        using nv_dtype2 = fp16_2;
        using cutlass_dtype = cutlass::half_t;

        inline __device__ static fp32 upcast(const c10_dtype &value) { return upcast(static_cast<nv_dtype>(value)); }
        inline __device__ static fp32 upcast(const nv_dtype &value) { return __half2float(value); }
        inline __device__ static fp32_2 upcast(const nv_dtype2 &value) { return __half22float2(value); }

        inline __device__ static nv_dtype downcast(const fp32 &value) { return __float2half(value); }
        inline __device__ static nv_dtype2 downcast(const fp32_2 &value) { return __float22half2_rn(value); }

        inline __device__ static nv_dtype2 make2(const nv_dtype &value) { return __half2half2(value); }
        inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_half2(x, y); }
    };

    template <>
    struct DType<fp16> : public DType<c10::Half> {};
}  // namespace flash_model_architectures
