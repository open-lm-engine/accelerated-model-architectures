# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl


@triton.jit
def store_or_atomic_add(x_ptrs, x, atomic_add=False, mask=None, sem="relaxed"):
    if atomic_add:
        tl.store(x_ptrs, x, mask=mask)
    else:
        tl.atomic_add(x_ptrs, x, mask=mask, sem=sem)
