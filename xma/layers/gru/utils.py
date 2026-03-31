# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


def _get_num_heads(
    x: torch.Tensor,
    W: torch.Tensor,
    xf: torch.Tensor,
    Wf: torch.Tensor,
    xr: torch.Tensor,
    Wr: torch.Tensor,
    run_check: bool,
) -> tuple[int, int, int, int, int, int, int]:
    Nx = x.size(-2)
    Nxf = xf.size(-2)
    Nxr = xr.size(-2)

    Nw = W.size(0)
    Nwf = Wf.size(0)
    Nwr = Wr.size(0)

    N = max(Nx, Nxf, Nxr, Nw, Nwf, Nwr)

    if run_check:
        assert N % Nx == 0
        assert N % Nxf == 0
        assert N % Nxr == 0

        assert N % Nw == 0
        assert N % Nwf == 0
        assert N % Nwr == 0

    return Nx, Nxf, Nxr, Nw, Nwf, Nwr, N
