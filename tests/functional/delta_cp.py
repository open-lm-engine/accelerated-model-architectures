# **************************************************
# Copyright (c) 2026, Mayank Mishra, Han Guo
# **************************************************

import os


os.environ["TRITON_F32_DEFAULT"] = "ieee"

import torch
from einops import rearrange, reduce, repeat
from fla.ops.cp import FLACPContext, build_cp_context

from xma.functional.delta_rule.chunk import chunk_delta_rule_bwd, chunk_delta_rule_fwd


def prepare_data(
    B: int,
    T: int,
    DK: int,
    DV: int,
    H: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Seed identically on every rank so all ranks see the same inputs
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, DK, generator=g, dtype=dtype, device=device), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, DK, generator=g, dtype=dtype, device=device), dim=-1)
    v = torch.randn(B, T, H, DV, generator=g, dtype=dtype, device=device)
    b = torch.randn(B, T, H, generator=g, dtype=dtype, device=device).sigmoid()
    S = torch.randn(1, H, DK, DV, generator=g, dtype=dtype, device=device)
    do = torch.randn(B, T, H, DV, generator=g, dtype=dtype, device=device)
    return q, k, v, b, S, do


def fwd_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    S: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context: FLACPContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    o, A, _, h0_cp = chunk_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        beta=b,
        scale=scale,
        initial_state=S,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
    )
    dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
        q=q,
        k=k,
        v=v,
        beta=b,
        A=A,
        scale=scale,
        initial_state=h0_cp,
        do=do,
        dht=None,
        cu_seqlens=cu_seqlens,
        cp_context=cp_context,
    )
    return o, dq, dk, dv, db, dh0


rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl")
group = torch.distributed.group.WORLD
device = f"cuda:{local_rank}"

B = 1
T = 512
H = 21
DK = 128
DV = 1024
scale = DK**-0.5

q, k, v, b, S, do = prepare_data(
    B=B,
    T=T,
    DK=DK,
    DV=DV,
    H=H,
    dtype=torch.float32,
    device=device,
)

S_ = repeat(S, "1 ... -> b ...", b=B).contiguous()
o0, dq0, dk0, dv0, db0, dS0 = fwd_bwd(
    q=q,
    k=k,
    v=v,
    b=b,
    S=S_,
    do=do,
    scale=scale,
)

flat = lambda x: rearrange(x, "b t ... -> 1 (b t) ...")
tokens_per_rank = (B * T) // world_size
t0 = rank * tokens_per_rank
t1 = t0 + tokens_per_rank
cu_seqlens = torch.tensor(
    [i * T for i in range(B + 1)],
    dtype=torch.long,
    device=device,
)
cp_context = build_cp_context(
    cu_seqlens=cu_seqlens,
    group=group,
    conv1d_kernel_size=None,
)
o1, dq1, dk1, dv1, db1, dS1 = fwd_bwd(
    q=flat(q)[:, t0:t1].contiguous(),
    k=flat(k)[:, t0:t1].contiguous(),
    v=flat(v)[:, t0:t1].contiguous(),
    b=flat(b)[:, t0:t1].contiguous(),
    S=S,
    do=flat(do)[:, t0:t1].contiguous(),
    scale=scale,
    cu_seqlens=cp_context.cu_seqlens,
    cp_context=cp_context,
)


# Per-gradient max-abs-diff across all ranks, for diagnostic visibility.
# Every rank must call this (it does an all_reduce internally); only rank 0 prints.
def max_abs_diff_across_ranks(a: torch.Tensor, b: torch.Tensor) -> float:
    amax = (a - b).abs().max()
    torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=group)
    return amax.item()


diffs = {
    name: max_abs_diff_across_ranks(a, b)
    for name, a, b in [
        ("o", o1, flat(o0)[:, t0:t1]),
        ("dq", dq1, flat(dq0)[:, t0:t1]),
        ("dk", dk1, flat(dk0)[:, t0:t1]),
        ("dv", dv1, flat(dv0)[:, t0:t1]),
        ("db", db1, flat(db0)[:, t0:t1]),
    ]
}
if rank == 0:
    print("  max abs diff: " + " ".join(f"{n}={v:.2e}" for n, v in diffs.items()))

# Per-token outputs: per-rank CP value matches the reference slice.
torch.testing.assert_close(o1, flat(o0)[:, t0:t1])
torch.testing.assert_close(dq1, flat(dq0)[:, t0:t1], rtol=1.3e-6, atol=2e-5)
torch.testing.assert_close(dk1, flat(dk0)[:, t0:t1], rtol=1.3e-6, atol=2e-5)
torch.testing.assert_close(dv1, flat(dv0)[:, t0:t1])
torch.testing.assert_close(db1, flat(db0)[:, t0:t1], rtol=1.3e-6, atol=2e-5)

# State grad: sum across ranks (CP zeros non-rank-0); sum across sequences
# for the reference (one shared h0 → ∂L/∂h0 = Σ_seq ∂L/∂h0_seq).
torch.distributed.all_reduce(dS1, op=torch.distributed.ReduceOp.SUM, group=group)
dS0 = reduce(dS0, "b ... -> 1 ...", "sum")
if rank == 0:
    print(f"  max abs diff: dS={(dS1 - dS0).abs().max().item():.2e}")
torch.testing.assert_close(dS1, dS0)

if rank == 0:
    print(f"PASS: world={world_size}")

torch.distributed.destroy_process_group()
