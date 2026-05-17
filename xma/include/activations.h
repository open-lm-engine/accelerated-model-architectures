// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include "dtypes.h"

#define MAX_EXP_FP32 88.3762626647949
#define MIN_EXP_FP32 -88.3762626647949

namespace flash_model_architectures {
    template <typename input_T, typename output_T>
    __device__ output_T sigmoid(const input_T &x) {
        fp32 x_fp32 = DType<input_T>::upcast(x);
        x_fp32 = fmaxf(MIN_EXP_FP32, x_fp32);
        x_fp32 = fminf(MAX_EXP_FP32, x_fp32);

        x_fp32 = 1 / (1 + expf(-x_fp32));

        return DType<output_T>::downcast(x_fp32);
    }
}  // namespace flash_model_architectures
