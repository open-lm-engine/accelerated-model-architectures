# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import cutlass.cute as cute


@cute.jit
def tanh(x, output_dtype: tl.constexpr = None):
    if output_dtype is None:
        output_dtype = x.dtype

    x = x.to(tl.float32)
    x = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1)
    x = x.to(output_dtype)

    return x
