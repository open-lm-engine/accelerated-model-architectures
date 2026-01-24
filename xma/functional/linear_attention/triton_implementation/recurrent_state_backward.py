# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import triton
import triton.language as tl

from ....math import get_powers_of_2
from ....triton_utils import matmul


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for num_warps in get_powers_of_2(4, 8):
        for num_stages in range(1, 5):
            for BLOCK_SIZE_S in [1] + get_powers_of_2(16, 64):
                for BLOCK_SIZE_K in get_powers_of_2(32, 64):
                    for BLOCK_SIZE_V in get_powers_of_2(32, 64):
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_SIZE_S": BLOCK_SIZE_S,
                                    "BLOCK_SIZE_K": BLOCK_SIZE_K,
                                    "BLOCK_SIZE_V": BLOCK_SIZE_V,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=[])
@triton.jit
def recurrent_state_backward_triton_kernel(
    q_ptr,
    q_stride,
    dy_ptr,
    dy_stride,
    dht_ptr,
    dht_stride,
    dh_ptr,
    dh_stride,
    dh0_ptr,
    dh0_stride,
    attention_multiplier,
    cu_seqlens_ptr,
    cu_seqlens_stride,
    S,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    Gq: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE % BLOCK_SIZE_S == 0)

    BLOCK_ID_BN = tl.program_id(0)
    BLOCK_ID_K = tl.program_id(1)
    BLOCK_ID_V = tl.program_id(2)

    BLOCK_ID_B = BLOCK_ID_BN // N
    BLOCK_ID_N = BLOCK_ID_BN % N

    BLOCK_ID_Nq = BLOCK_ID_N // Gq

    NUM_BLOCKS_S = tl.cdiv(S, BLOCK_SIZE_S)

    BLOCK_S = NUM_BLOCKS_S * tl.arange(0, BLOCK_SIZE_S)
    BLOCK_K = BLOCK_ID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    BLOCK_V = BLOCK_ID_V * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)

    MASK_K = BLOCK_K < K
    MASK_V = BLOCK_V < V

    MASK_KV = MASK_K[:, None] & MASK_V[None, :]

    if dht_ptr is None:
        dh = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_V), dtype=tl.float32)
    else:
        dh = tl.load(
            dht_ptr
            + BLOCK_ID_B * dht_stride[0]
            + BLOCK_ID_N * dht_stride[1]
            + BLOCK_K[:, None] * dht_stride[2]
            + BLOCK_V[None, :] * dht_stride[3],
            mask=MASK_KV,
        ).to(tl.float32)

    IS_VARLEN: tl.constexpr = cu_seqlens_ptr is not None
    S_DIM: tl.constexpr = 1 - IS_VARLEN
    N_DIM: tl.constexpr = 2 - IS_VARLEN
    K_DIM: tl.constexpr = 3 - IS_VARLEN

    if IS_VARLEN:
        cu_seqlens_ptrs = cu_seqlens_ptr + BLOCK_ID_B * cu_seqlens_stride[0]
        start = tl.load(cu_seqlens_ptrs)
        end = tl.load(cu_seqlens_ptrs + cu_seqlens_stride[0])

        S = end - start
        BLOCK = start + BLOCK_S

    _B = BLOCK[:, None] if IS_VARLEN else BLOCK_ID_B
    _S = 0 if IS_VARLEN else BLOCK_S[:, None]

    q_ptrs = (
        q_ptr
        + _B * q_stride[0]
        + _S * q_stride[S_DIM]
        + BLOCK_ID_Nq * q_stride[N_DIM]
        + BLOCK_K[None, :] * q_stride[K_DIM]
    )

    dy_ptrs = (
        dy_ptr
        + _B * dy_stride[0]
        + _S * dy_stride[S_DIM]
        + BLOCK_ID_N * dy_stride[N_DIM]
        + BLOCK_V[None, :] * dy_stride[K_DIM]
    )

    dh_ptrs = (
        dh_ptr
        + _B * dh_stride[0]
        + _S * dh_stride[1]
        + BLOCK_ID_N * dh_stride[2]
        + BLOCK_K[:, None] * dh_stride[3]
        + BLOCK_V[None, :] * dh_stride[4]
    )

    for s in range(NUM_BLOCKS_S - 1, -1, -1):
        MASK_S = BLOCK_S < S

        MASK_SK = MASK_S[:, None] & MASK_K[None, :]
        MASK_SV = MASK_S[:, None] & MASK_V[None, :]

        q = tl.load(q_ptrs, mask=MASK_SK)
        q_ptrs -= BLOCK_SIZE_S * q_stride[S_DIM]

        dy = tl.load(dy_ptrs, mask=MASK_SV)
        dy_ptrs += BLOCK_SIZE_S * dy_stride[S_DIM]

        q *= attention_multiplier
        dh = matmul(A=q.T, B=dy, C=dh, output_dtype=dh.dtype)

        if dh_ptr is not None and (s * BLOCK_SIZE_S) % CHUNK_SIZE == 0 and s != 0:
            tl.store(dh_ptrs, dh, mask=MASK_KV)
            dh_ptrs += dh_stride[S_DIM]

        BLOCK_S -= BLOCK_SIZE_S

    if dh0_ptr is not None:
        tl.store(
            dh0_ptr
            + BLOCK_ID_B * dh0_stride[0]
            + BLOCK_ID_N * dh0_stride[1]
            + BLOCK_K[:, None] * dh0_stride[2]
            + BLOCK_V[None, :] * dh0_stride[3],
            dh,
            mask=MASK_KV,
        )
