# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32, Numeric, const_expr, range_constexpr
from cutlass.cute import TensorSSA
from cutlass.cutlass_dsl import dsl_user_op


F32x2 = tuple[Float32, Float32]


@dsl_user_op
def _tanh(x: Float32 | F32x2, *, loc=None, ip=None) -> Float32 | F32x2:
    if const_expr(isinstance(x, tuple)):
        x = (
            cute.math.tanh(x[0], fastmath=True, loc=loc, ip=ip),
            cute.math.tanh(x[1], fastmath=True, loc=loc, ip=ip),
        )
    else:
        x = cute.math.tanh(Float32(x), fastmath=True, loc=loc, ip=ip)

    return x


@dsl_user_op
def _sigmoid(x: Float32 | F32x2, *, loc=None, ip=None) -> Float32 | F32x2:
    if const_expr(isinstance(x, tuple)):
        x = cute.arch.mul_packed_f32x2((0.5, 0.5), x)
        x = _tanh(x, loc=loc, ip=ip)
        x = cute.arch.fma_packed_f32x2(x, (0.5, 0.5), (0.5, 0.5))
    else:
        x = 0.5 * _tanh(0.5 * Float32(x), loc=loc, ip=ip) + 0.5

    return x


@cute.jit
def tanh(x: Numeric | TensorSSA, output_dtype: Numeric | None = None) -> Numeric | TensorSSA:
    if const_expr(output_dtype is None):
        output_dtype = x.dtype

    if const_expr(isinstance(x, TensorSSA)):
        y = cute.make_rmem_tensor(x.shape, Float32)
        y.store(x.to(Float32))

        n = cute.size(y.shape)
        for i in range_constexpr(n >> 1):
            y[2 * i], y[2 * i + 1] = _tanh((y[2 * i], y[2 * i + 1]))

        if const_expr(n % 2 == 1):
            y[n - 1] = _tanh(y[n - 1])

        y = y.load()
    else:
        y = _tanh(x.to(Float32))

    return y.to(output_dtype)


@cute.jit
def sigmoid(x: Numeric | TensorSSA, output_dtype: Numeric | None = None) -> Numeric | TensorSSA:
    if const_expr(output_dtype is None):
        output_dtype = x.dtype

    if const_expr(isinstance(x, TensorSSA)):
        y = cute.make_rmem_tensor(x.shape, Float32)
        y.store(x.to(Float32))

        n = cute.size(y.shape)
        for i in range_constexpr(n >> 1):
            y[2 * i], y[2 * i + 1] = _sigmoid((y[2 * i], y[2 * i + 1]))

        if const_expr(n % 2 == 1):
            y[n - 1] = _sigmoid(y[n - 1])

        y = y.load()
    else:
        y = _sigmoid(x.to(Float32))

    return y.to(output_dtype)
