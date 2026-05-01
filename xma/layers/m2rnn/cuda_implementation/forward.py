# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr, range_constexpr

from ....constants import WARP_SIZE
from ....custom_op import xma_op
from ....cute_dsl_utils import get_fake_cute_tensor, tanh
from ....math import ceil_divide, get_next_power_of_2


class _M2RNNForwardCUDAKernel:
    def __init__(self, K: int, V: int, Gq: int, Gk: int, Gv: int, Gw: int, Gxf: int) -> None:
        self.K = K
        self.V = V
        self.Gq = Gq
        self.Gk = Gk
        self.Gv = Gv
        self.Gw = Gw
        self.Gxf = Gxf

        # One thread owns one row of the recurrent state. A power-of-two CTA
        # keeps the warp shuffle reduction simple and gives enough threads to
        # cooperatively load W into shared memory.
        self.threads_per_cta = max(WARP_SIZE, min(128, get_next_power_of_2(max(K, V))))
        assert self.threads_per_cta % WARP_SIZE == 0

        self.num_warps = self.threads_per_cta // WARP_SIZE
        self.w_loads_per_thread = ceil_divide(V * V, self.threads_per_cta)
        self.v_loads_per_thread = ceil_divide(V, self.threads_per_cta)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mW: cute.Tensor,
        mXf: cute.Tensor,
        mH0: cute.Tensor,
        mHt: cute.Tensor,
        mY: cute.Tensor,
        mCuSeqlens: cute.Tensor | None,
    ) -> None:
        BLOCK_ID_B, BLOCK_ID_N, _ = cute.arch.block_idx()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        if const_expr(mCuSeqlens is None):
            S = cute.size(mQ, mode=[1])
        else:
            start = mCuSeqlens[BLOCK_ID_B]
            end = mCuSeqlens[BLOCK_ID_B + 1]
            S = end - start

        BLOCK_ID_Nq = BLOCK_ID_N // self.Gq
        BLOCK_ID_Nk = BLOCK_ID_N // self.Gk
        BLOCK_ID_Nv = BLOCK_ID_N // self.Gv
        BLOCK_ID_Nw = BLOCK_ID_N // self.Gw
        BLOCK_ID_Nxf = BLOCK_ID_N // self.Gxf

        LANE_ID = THREAD_ID % WARP_SIZE
        WARP_ID = THREAD_ID // WARP_SIZE

        io_dtype = mQ.element_type
        acc_dtype = Float32

        smem = cutlass.utils.SmemAllocator()
        sW = smem.allocate_tensor(
            element_type=acc_dtype,
            layout=cute.make_layout((V, V), stride=(V, 1)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(element_type=acc_dtype, layout=cute.make_layout(V), byte_alignment=16)
        sF = smem.allocate_tensor(element_type=acc_dtype, layout=cute.make_layout(1), byte_alignment=4)
        sY = smem.allocate_tensor(
            element_type=acc_dtype, layout=cute.make_layout((NUM_WARPS, V), stride=(V, 1)), byte_alignment=16
        )

        # Cooperative load of the VxV state transition matrix.
        for i in range_constexpr(self.w_loads_per_thread):
            idx = tid + i * self.threads_per_cta
            if idx < V * V:
                row = idx // V
                col = idx % V
                sW[row, col] = mW[bid_nw, row, col].to(acc_dtype)

        cute.arch.sync_threads()

        # Each CTA only needs the query vector for the active timestep.
        # We first carve out the per-threadblock row, then tile it across threads.
        q_thread_tiler = cute.make_layout(self.threads_per_cta)

        # One row of the state matrix lives in each thread.
        h_row = cute.make_rmem_tensor(cute.make_layout(V), acc_dtype)
        h_next = cute.make_rmem_tensor(cute.make_layout(V), acc_dtype)

        if tid < K:
            for v in range_constexpr(V):
                h_row[v] = mH0[bid_b, bid_n, tid, v].to(acc_dtype)
        else:
            for v in range_constexpr(V):
                h_row[v] = acc_dtype(0.0)

        start = bid_b * S
        seqlen = S
        if const_expr(is_varlen):
            start = mCuSeqlens[bid_b]
            seqlen = mCuSeqlens[bid_b + 1] - start

        for s in cutlass.range(S, unroll=1):
            if (not const_expr(is_varlen)) or (s < seqlen):
                if const_expr(is_varlen):
                    time_idx = start + s
                else:
                    time_idx = s

                # Load value and forget tensors for this timestep.
                for i in range_constexpr(self.v_loads_per_thread):
                    idx = tid + i * self.threads_per_cta
                    if idx < V:
                        if const_expr(is_varlen):
                            sV[idx] = mV[time_idx, bid_nv, idx].to(acc_dtype)
                        else:
                            sV[idx] = mV[bid_b, s, bid_nv, idx].to(acc_dtype)

                if tid == 0:
                    if const_expr(is_varlen):
                        sF[0] = mXf[time_idx, bid_nxf].to(acc_dtype)
                    else:
                        sF[0] = mXf[bid_b, s, bid_nxf].to(acc_dtype)

                cute.arch.sync_threads()

                f_val = sF[0]

                if tid < K:
                    if const_expr(is_varlen):
                        k_val = mK[time_idx, bid_nk, tid].to(acc_dtype)
                    else:
                        k_val = mK[bid_b, s, bid_nk, tid].to(acc_dtype)

                    for vp in range_constexpr(V):
                        acc = k_val * sV[vp]
                        for vv in range_constexpr(V):
                            acc += h_row[vv] * sW[vv, vp]

                        z_vp = tanh(acc, output_dtype=acc_dtype)
                        h_next[vp] = f_val * h_row[vp] + (acc_dtype(1.0) - f_val) * z_vp

                    if const_expr(is_varlen):
                        q_row = cute.tiled_divide(mQ[time_idx, bid_nq], q_thread_tiler)
                    else:
                        q_row = cute.tiled_divide(mQ[bid_b, s, bid_nq], q_thread_tiler)

                    q_val = q_row[tid, 0].to(acc_dtype)
                else:
                    q_val = acc_dtype(0.0)

                for vp in range_constexpr(V):
                    partial = q_val * (h_next[vp] if tid < K else acc_dtype(0.0))
                    for off in range_constexpr(5):
                        partial += cute.arch.shuffle_sync_bfly(
                            partial,
                            offset_or_lane=Int32(1 << off),
                            mask_and_clamp=Int32(0x1F),
                        )

                    if lane == 0:
                        sY[warp, vp] = partial

                cute.arch.sync_threads()

                if tid < V:
                    y_val = acc_dtype(0.0)
                    for w in range_constexpr(NUM_WARPS):
                        y_val += sY[w, tid]

                    if const_expr(is_varlen):
                        mY[time_idx, bid_n, tid] = y_val.to(io_dtype)
                    else:
                        mY[bid_b, s, bid_n, tid] = y_val.to(io_dtype)

                cute.arch.sync_threads()

                if tid < K:
                    for vp in range_constexpr(V):
                        h_row[vp] = h_next[vp]

        if tid < K:
            for vp in range_constexpr(V):
                mHt[bid_b, bid_n, tid, vp] = h_row[vp].to(io_dtype)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mW: cute.Tensor,
        mXf: cute.Tensor,
        mH0: cute.Tensor | None,
        mHt: cute.Tensor,
        mY: cute.Tensor,
        mCuSeqlens: cute.Tensor | None,
        stream: cuda.CUstream,
    ) -> None:
        if const_expr(mCuSeqlens is None):
            B = cute.size(mQ, mode=[0])
        else:
            B = cute.size(mCuSeqlens, mode=[0]) - 1

        N = cute.size(mW, mode=[0]) * self.Gw

        self.kernel(
            mQ=mQ,
            mK=mK,
            mV=mV,
            mW=mW,
            mXf=mXf,
            mH0=mH0,
            mHt=mHt,
            mY=mY,
            mCuSeqlens=mCuSeqlens,
        ).launch(
            grid=(B, N, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            smem=(self.V * self.V + self.V + 1 + self.num_warps * self.V) * 4,
        )


_CACHE: dict = {}


@xma_op(mutates_args={"ht", "y"})
def _m2rnn_forward_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    h0: torch.Tensor | None,
    ht: torch.Tensor,
    y: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    Nq: int,
    Nk: int,
    Nv: int,
    Nw: int,
    Nxf: int,
    N: int,
) -> None:
    is_varlen = cu_seqlens is not None

    K = q.size(-1)
    V = v.size(-1)

    div_K = math.gcd(16 // q.dtype.itemsize, K)
    div_V = math.gcd(16 // v.dtype.itemsize, V)
    div_Nxf = math.gcd(16 // xf.dtype.itemsize, Nxf)

    key = (q.dtype, div_K, div_V, div_Nxf, K, V, Nq, Nk, Nv, Nw, Nxf, N, is_varlen)
    function = _CACHE.get(key)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if function is None:
        if is_varlen:
            _q = get_fake_cute_tensor(q.dtype, (cute.sym_int(), Nq, K), divisibility=div_K)
            _k = get_fake_cute_tensor(k.dtype, (cute.sym_int(), Nk, K), divisibility=div_K)
            _v = get_fake_cute_tensor(v.dtype, (cute.sym_int(), Nv, V), divisibility=div_V)
            _xf = get_fake_cute_tensor(xf.dtype, (cute.sym_int(), Nxf), divisibility=div_Nxf)
            _y = get_fake_cute_tensor(y.dtype, (cute.sym_int(), N, V), divisibility=div_V)
        else:
            _q = get_fake_cute_tensor(q.dtype, (cute.sym_int(), cute.sym_int(), Nq, K), divisibility=div_K)
            _k = get_fake_cute_tensor(k.dtype, (cute.sym_int(), cute.sym_int(), Nk, K), divisibility=div_K)
            _v = get_fake_cute_tensor(v.dtype, (cute.sym_int(), cute.sym_int(), Nv, V), divisibility=div_V)
            _xf = get_fake_cute_tensor(xf.dtype, (cute.sym_int(), cute.sym_int(), Nxf), divisibility=div_Nxf)
            _y = get_fake_cute_tensor(y.dtype, (cute.sym_int(), cute.sym_int(), N, V), divisibility=div_V)

        _W = get_fake_cute_tensor(W.dtype, (Nw, V, V), divisibility=div_V)
        _h0 = None if h0 is None else get_fake_cute_tensor(h0.dtype, (cute.sym_int(), N, K, V), divisibility=div_V)
        _ht = get_fake_cute_tensor(ht.dtype, (cute.sym_int(), N, K, V), divisibility=div_V)
        _cu_seqlens = (
            None if cu_seqlens is None else get_fake_cute_tensor(cu_seqlens.dtype, (cute.sym_int(),), divisibility=1)
        )

        function = _M2RNNForwardCUDAKernel(K=K, V=V, Gq=N // Nq, Gk=N // Nk, Gv=N // Nv, Gw=N // Nw, Gxf=N // Nxf)
        function = cute.compile(
            function, _q, _k, _v, _W, _xf, _h0, _ht, _y, _cu_seqlens, stream, options="--enable-tvm-ffi"
        )

        _CACHE[key] = function

    function(q, k, v, W, xf, h0, ht, y, cu_seqlens, stream)
