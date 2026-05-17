// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "alias.h"
#include "common.h"

namespace flash_model_architectures {
    template <>
    struct DType<uint32> {
        using nv_dtype = uint32;
        using nv_dtype2 = uint32_2;
        using nv_dtype4 = uint32_4;

        inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_uint2(x, y); }
        inline __device__ static nv_dtype4 make4(const nv_dtype &x,
                                                 const nv_dtype &y,
                                                 const nv_dtype &z,
                                                 const nv_dtype &t) {
            return make_uint4(x, y, z, t);
        }
    };

    inline int32 safe_cast_uint32_to_int32(const uint32 value) {
        if (value > std::numeric_limits<int32>::max()) {
            throw std::runtime_error("value exceeds int32's max value");
        }

        return static_cast<int32>(value);
    }
}  // namespace flash_model_architectures
