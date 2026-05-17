// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include "alias.h"

namespace flash_model_architectures {
    // base struct for converting torch ScalarType to NVIDIA's dtype
    template <typename scalar_t>
    struct DType {
        using c10_dtype = scalar_t;
    };
}  // namespace flash_model_architectures
