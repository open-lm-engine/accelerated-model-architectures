# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def compute_p_norm(x, P, P_inv, eps):
    if P_inv is None:
        P_inv = 1 / P

    x = tl.abs(x)
    m = tl.max(x, axis=1)
    x = x.to(tl.float32)
    x /= m
    x += eps
    x = tl.log2(x)
    x *= P
    x = tl.exp2(x)
    x = tl.sum(x, axis=1) + eps
    x = tl.log2(x)
    x *= P_inv
    x = tl.exp2(x)
    x *= m

    return x
