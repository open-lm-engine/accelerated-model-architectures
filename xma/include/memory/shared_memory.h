// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace flash_model_architectures::memory {
    template <typename scalar_t>
    inline __device__ scalar_t *get_dynamic_shared_memory() {
        extern __shared__ char shared_memory_raw[];
        alignas(scalar_t) scalar_t *shared_memory = reinterpret_cast<scalar_t *>(shared_memory_raw);
        return shared_memory;
    }
}  // namespace flash_model_architectures::memory
