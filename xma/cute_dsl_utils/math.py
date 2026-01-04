# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32, Numeric, const_expr, range_constexpr
from cutlass._mlir.dialects import llvm
from cutlass.cute import TensorSSA
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _tanh(x: Float32 | float, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            res=T.f32(),
            operands_=[Float32(x).ir_value(loc=loc, ip=ip)],
            asm_string="tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def tanh(x: Numeric | TensorSSA, output_dtype: Numeric | None = None) -> Numeric | TensorSSA:
    if const_expr(output_dtype is None):
        output_dtype = x.dtype

    if const_expr(isinstance(x, TensorSSA)):
        y = cute.make_fragment(x.shape, Float32)
        y.store(x.to(Float32))

        for i in range_constexpr(cute.size(y.shape)):
            y[i] = _tanh(y[i])

        y = y.load()
    else:
        y = _tanh(x.to(Float32))
        y = y.to(output_dtype)

    return y


def sigmoid(x: Numeric | TensorSSA, output_dtype: Numeric | None = None) -> Numeric | TensorSSA:
    if const_expr(output_dtype is None):
        output_dtype = x.dtype

    x = x.to(Float32)
    x = 0.5 * tanh(0.5 * x, output_dtype=Float32) + 0.5
    x = x.to(output_dtype)

    return x
