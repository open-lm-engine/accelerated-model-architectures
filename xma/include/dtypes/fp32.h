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

namespace flash_model_architectures {
    template <>
    struct DType<fp32> {
        using c10_dtype = fp32;
        using nv_dtype = fp32;
        using nv_dtype2 = fp32_2;
        using nv_dtype4 = fp32_4;
        using cutlass_dtype = fp32;

        inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_float2(x, y); }
        inline __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_float2(array[0], array[1]); }

        inline __device__ static nv_dtype4 make4(const nv_dtype &x,
                                                 const nv_dtype &y,
                                                 const nv_dtype &z,
                                                 const nv_dtype &t) {
            return make_float4(x, y, z, t);
        }
        inline __device__ static nv_dtype4 make4(const nv_dtype *array) {
            return make_float4(array[0], array[1], array[2], array[3]);
        }

        inline __device__ static nv_dtype upcast(const nv_dtype &value) { return value; }
        inline __device__ static nv_dtype2 upcast(const nv_dtype2 &value) { return value; }
        inline __device__ static nv_dtype4 upcast(const nv_dtype4 &value) { return value; }

        inline __device__ static nv_dtype downcast(const nv_dtype &value) { return value; }
        inline __device__ static nv_dtype2 downcast(const nv_dtype2 &value) { return value; }
        inline __device__ static nv_dtype4 downcast(const nv_dtype4 &value) { return value; }
    };
}  // namespace flash_model_architectures
