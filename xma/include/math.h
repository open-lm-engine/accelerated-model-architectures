// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include <cuda.h>

namespace flash_model_architectures {
    template <typename T>
    inline __host__ __device__ T ceil_divide(const T &x, const T &d) {
        return (x / d) + (x % d != 0);
    }
}  // namespace flash_model_architectures
