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


# =====================================================================
# Hopper (SM90) M2RNN forward CUTE-DSL kernel.
#
# Recurrence (per timestep s):
#     x = k ⊗ v                      # (K, V)
#     z = tanh(h_prev @ W + x)       # (K, V)
#     h = f * h_prev + (1 - f) * z   # (K, V)
#     y = q @ h                      # (V,)
#
# Tile / thread decomposition (specialized for K=64, V=16):
#   * Grid:  (B, N) — one CTA per (batch, head).
#   * Block: K threads (= 64 = 2 warps) — thread `tid` owns row `tid`
#     of the (K, V) state in registers (V floats per thread).
#   * Smem:
#       - sW  : (V, V) — weight tile, loaded once, reused for S steps.
#       - sV  : (V,)   — current step's `v` vector, refreshed each step.
#       - sF  : scalar — broadcast forget gate.
#       - sY  : (num_warps, V) — cross-warp reduction buffer for `q @ h`.
#
# All math runs in fp32; loads / stores honor the user's IO dtype.
# Recurrence is sequential, so the kernel is latency-bound — keeping the
# state in registers and W in smem minimizes per-step memory traffic.
# Tensor cores are *not* used: V=16 makes the inner GEMM too small for
# WGMMA to amortize launch overhead inside a recurrent loop.
# =====================================================================


class M2RNNForwardCUTEKernel:
    def __init__(self, K: int, V: int, Gq: int, Gk: int, Gv: int, Gw: int, Gxf: int) -> None:
        # specialized for K = 64, V = 16
        assert K == 64, f"this implementation requires K == 64, got {K}"
        assert V == 16, f"this implementation requires V == 16, got {V}"

        self.K = K
        self.V = V
        self.Gq = Gq
        self.Gk = Gk
        self.Gv = Gv
        self.Gw = Gw
        self.Gxf = Gxf

        self.threads_per_cta = K
        self.num_warps = self.threads_per_cta // WARP_SIZE
        assert self.threads_per_cta % WARP_SIZE == 0
        assert (V * V) % self.threads_per_cta == 0

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
        S: Int32,
        HAS_H0: cutlass.Constexpr,
        HAS_HT: cutlass.Constexpr,
        HAS_Y: cutlass.Constexpr,
    ) -> None:
        K: cutlass.Constexpr = self.K
        V: cutlass.Constexpr = self.V
        Gq: cutlass.Constexpr = self.Gq
        Gk: cutlass.Constexpr = self.Gk
        Gv: cutlass.Constexpr = self.Gv
        Gw: cutlass.Constexpr = self.Gw
        Gxf: cutlass.Constexpr = self.Gxf
        NUM_WARPS: cutlass.Constexpr = self.num_warps
        W_ELEMS_PER_THREAD: cutlass.Constexpr = (V * V) // self.threads_per_cta

        bid_b, bid_n, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        bid_nq = bid_n // Gq
        bid_nk = bid_n // Gk
        bid_nv = bid_n // Gv
        bid_nw = bid_n // Gw
        bid_nxf = bid_n // Gxf

        io_dtype = mK.element_type

        lane = tid % WARP_SIZE
        warp = tid // WARP_SIZE

        # -----------------------------------------------------------------
        # Smem: W (V*V), sV (V), sF (1), sY (num_warps, V) — used by the
        # output reduction; declared here so all paths see the same layout.
        # -----------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()

        sW = smem.allocate_tensor(
            element_type=io_dtype,
            layout=cute.make_layout((V, V), stride=(V, 1)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(
            element_type=io_dtype,
            layout=cute.make_layout(V),
            byte_alignment=16,
        )
        sF = smem.allocate_tensor(
            element_type=io_dtype,
            layout=cute.make_layout(1),
            byte_alignment=4,
        )
        sY = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((NUM_WARPS, V), stride=(V, 1)),
            byte_alignment=16,
        )

        # -----------------------------------------------------------------
        # Cooperative load of W (V x V) into smem (one-time).
        # 64 threads * 4 elems = 256 covers the full 16x16 weight tile
        # in row-major order; consecutive threads hit consecutive elements
        # so loads are coalesced.
        # -----------------------------------------------------------------
        for i in range_constexpr(W_ELEMS_PER_THREAD):
            idx = tid * W_ELEMS_PER_THREAD + i
            row = idx // V
            col = idx % V
            sW[row, col] = mW[bid_nw, row, col]

        # -----------------------------------------------------------------
        # Initialize register-resident row of h: h_row = h0[bid_b, bid_n, tid, :]
        # -----------------------------------------------------------------
        h_row = cute.make_rmem_tensor(cute.make_layout(V), Float32)

        if const_expr(HAS_H0):
            for v in range_constexpr(V):
                h_row[v] = mH0[bid_b, bid_n, tid, v].to(Float32)
        else:
            for v in range_constexpr(V):
                h_row[v] = Float32(0.0)

        cute.arch.sync_threads()

        # -----------------------------------------------------------------
        # Main recurrent loop.
        # -----------------------------------------------------------------
        for s in cutlass.range(S, unroll=1):
            # --- load k[tid] from gmem (one element per thread) -----------
            k_val = mK[bid_b, s, bid_nk, tid].to(Float32)

            # --- load v[V], f scalar into smem cooperatively --------------
            if tid < V:
                sV[tid] = mV[bid_b, s, bid_nv, tid]
            if tid == 0:
                sF[0] = mXf[bid_b, s, bid_nxf]

            cute.arch.sync_threads()

            f_val = sF[0].to(Float32)

            # --- z[v'] = sum_v h_row[v] * W[v,v'] + k_val * v[v'] ----------
            # --- z = tanh(z); h_row = f*h_row + (1-f)*z --------------------
            for vp in range_constexpr(V):
                acc = k_val * sV[vp].to(Float32)
                for v in range_constexpr(V):
                    acc += h_row[v] * sW[v, vp].to(Float32)
                z_vp = tanh(acc, output_dtype=Float32)
                h_row[vp] = f_val * h_row[vp] + (Float32(1.0) - f_val) * z_vp

            # --- optional output: y = q @ h -------------------------------
            if const_expr(HAS_Y):
                q_val = mQ[bid_b, s, bid_nq, tid].to(Float32)

                # warp-level butterfly reduction across the 32 lanes for
                # each of V output columns; lane 0 of each warp ends up
                # with the warp's contribution.
                for vp in range_constexpr(V):
                    partial = q_val * h_row[vp]
                    for off in range_constexpr(5):  # log2(WARP_SIZE)
                        partial += cute.arch.shuffle_sync_bfly(
                            partial,
                            offset_or_lane=Int32(1 << off),
                            mask_and_clamp=Int32(0x1F),
                        )
                    if lane == 0:
                        sY[warp, vp] = partial

                cute.arch.sync_threads()

                # cross-warp reduction + final write: V threads each
                # accumulate the per-warp partials and store one output.
                if tid < V:
                    final = Float32(0.0)
                    for w in range_constexpr(NUM_WARPS):
                        final += sY[w, tid]
                    mY[bid_b, s, bid_n, tid] = final.to(io_dtype)

            # --- end-of-step barrier: protects sV/sF before next iter -----
            cute.arch.sync_threads()

        # -----------------------------------------------------------------
        # Store final state ht.
        # -----------------------------------------------------------------
        if const_expr(HAS_HT):
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
        mH0: cute.Tensor,
        mHt: cute.Tensor,
        mY: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        B = cute.size(mK, mode=[0])
        S = cute.size(mK, mode=[1])
        N = cute.size(mW, mode=[0]) * self.Gw  # Nw * Gw == N

        # smem layout footprint (matches kernel allocator order)
        io_bytes = mK.element_type.width // 8
        smem_bytes = (
            self.V * self.V * io_bytes  # sW
            + self.V * io_bytes  # sV
            + max(4, io_bytes)  # sF (one element, padded)
            + self.num_warps * self.V * 4  # sY (fp32)
            + 64  # alignment slack
        )

        self.kernel(
            mQ,
            mK,
            mV,
            mW,
            mXf,
            mH0,
            mHt,
            mY,
            S,
        ).launch(
            grid=(B, N, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            smem=smem_bytes,
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
    h: torch.Tensor | None,
    ht: torch.Tensor | None,
    y: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    Nq: int,
    Nk: int,
    Nv: int,
    Nw: int,
    Nxf: int,
    N: int,
) -> None:
    assert cu_seqlens is None
    assert h is None

    if cu_seqlens is None:
        B, S, _, K = k.size()
    else:
        B = cu_seqlens.size(0) - 1
        S = None
        K = k.size(-1)

    V = v.size(-1)

    assert K == 64 and V == 16, f"cute kernel requires K=64, V=16; got K={K}, V={V}"

    has_h0 = h0 is not None
    has_ht = ht is not None
    has_y = y is not None

    Gq = N // Nq
    Gk = N // Nk
    Gv = N // Nv
    Gw = N // Nw
    Gxf = N // Nxf

    div_K = math.gcd(16 // k.dtype.itemsize, K)
    div_V = math.gcd(16 // v.dtype.itemsize, V)
    div_Nxf = math.gcd(16 // xf.dtype.itemsize, Nxf)
    div_N = math.gcd(16 // y.dtype.itemsize, N)

    key = (k.dtype, div_K, div_V, div_Nxf, div_N, K, V, Gq, Gk, Gv, Gw, Gxf, has_h0, has_ht, has_y)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    function = _CACHE.get(key)

    if function is None:
        _q = get_fake_cute_tensor(q.dtype, (cute.sym_int(), cute.sym_int(), Nq, K), divisibility=div_K)
        _k = get_fake_cute_tensor(k.dtype, (cute.sym_int(), cute.sym_int(), Nk, K), divisibility=div_K)
        _v = get_fake_cute_tensor(v.dtype, (cute.sym_int(), cute.sym_int(), Nv, V), divisibility=div_V)
        _W = get_fake_cute_tensor(W.dtype, (Nw, V, V), divisibility=div_V)
        _xf = get_fake_cute_tensor(xf.dtype, (cute.sym_int(), cute.sym_int(), Nxf), divisibility=div_Nxf)
        _y = get_fake_cute_tensor(y.dtype, (cute.sym_int(), cute.sym_int(), N, V), divisibility=div_V)

        _h0 = None if h0 is None else get_fake_cute_tensor(h0.dtype, (cute.sym_int(), N, K, V), divisibility=div_V)
        _ht = None if ht is None else get_fake_cute_tensor(ht.dtype, (cute.sym_int(), N, K, V), divisibility=div_V)

        function = M2RNNForwardCUTEKernel(K=K, V=V, Gq=Gq, Gk=Gk, Gv=Gv, Gw=Gw, Gxf=Gxf)

        function = cute.compile(
            function,
            _q,
            _k,
            _v,
            _W,
            _xf,
            _h0,
            _ht,
            _y,
            stream,
            options="--enable-tvm-ffi",
        )

        _CACHE[key] = function

    function(q, k, v, W, xf, h0, ht, y, stream)
