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
#include "cutlass/bfloat16.h"

namespace flash_model_architectures {
    template <>
    struct DType<c10::BFloat16> {
        using c10_dtype = c10::BFloat16;
        using nv_dtype = bf16;
        using nv_dtype2 = bf16_2;
        using cutlass_dtype = cutlass::bfloat16_t;

        inline __device__ static fp32 upcast(const c10_dtype &value) { return upcast(static_cast<nv_dtype>(value)); }
        inline __device__ static fp32 upcast(const nv_dtype &value) { return __bfloat162float(value); }
        inline __device__ static fp32_2 upcast(const nv_dtype2 &value) { return __bfloat1622float2(value); }

        inline __device__ static nv_dtype downcast(const fp32 &value) { return __float2bfloat16(value); }
        inline __device__ static nv_dtype2 downcast(const fp32_2 &value) { return __float22bfloat162_rn(value); }

        inline __device__ static nv_dtype2 make2(const c10_dtype &value) {
            return make2(static_cast<nv_dtype>(value));
        }
        inline __device__ static nv_dtype2 make2(const c10_dtype &x, const c10_dtype &y) {
            return make2(static_cast<nv_dtype>(x), static_cast<nv_dtype>(y));
        }

        inline __device__ static nv_dtype2 make2(const nv_dtype &value) { return __bfloat162bfloat162(value); }
        inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_bfloat162(x, y); }
    };

    template <>
    struct DType<bf16> : public DType<c10::BFloat16> {};
}  // namespace flash_model_architectures
