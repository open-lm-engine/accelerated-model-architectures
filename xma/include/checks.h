// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include "dtypes.h"

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS_TENSOR(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CONTIGUOUS_CUDA_TENSOR(x) \
    CHECK_CUDA_DEVICE(x);               \
    CHECK_CONTIGUOUS(x)

#define CHECK_WITHIN_UINT32(x) TORCH_CHECK(x <= uint64(std::numeric_limits<uint32>::max()))
