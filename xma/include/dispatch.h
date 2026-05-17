// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#include <torch/extension.h>

#define DISPATCH_CASE(ENUM_TYPE, SCALAR_NAME, ...) AT_PRIVATE_CASE_TYPE_USING_HINT(ENUM_TYPE, SCALAR_NAME, __VA_ARGS__)

#define DISPATCH_FLOAT_KERNEL(TYPE, NAME, SCALAR_NAME, ...)                                  \
    AT_DISPATCH_SWITCH(TYPE,                                                                 \
                       NAME,                                                                 \
                       DISPATCH_CASE(at::ScalarType::Half, SCALAR_NAME, __VA_ARGS__)         \
                           DISPATCH_CASE(at::ScalarType::BFloat16, SCALAR_NAME, __VA_ARGS__) \
                               DISPATCH_CASE(at::ScalarType::Float, SCALAR_NAME, __VA_ARGS__))

#define DISPATCH_INT_KERNEL(TYPE, NAME, SCALAR_NAME, ...)                                  \
    AT_DISPATCH_SWITCH(TYPE,                                                               \
                       NAME,                                                               \
                       DISPATCH_CASE(at::ScalarType::Int, SCALAR_NAME, __VA_ARGS__)        \
                           DISPATCH_CASE(at::ScalarType::UInt32, SCALAR_NAME, __VA_ARGS__) \
                               DISPATCH_CASE(at::ScalarType::Long, SCALAR_NAME, __VA_ARGS__))
