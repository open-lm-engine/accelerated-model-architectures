# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest


torch = pytest.importorskip("torch")

from torch.testing import assert_close

from xma import Accelerator, KernelBackend
from xma.layers import DepthwiseCausalConvolution
from xma.utils import is_causal_conv1d_available

from ..utils import skip_if_incompatible_kernel_backend


_HIDDEN_SIZE = 8
_BATCH = 2
_PREFILL_LEN = 6


def _make_conv(
    hidden_size: int = _HIDDEN_SIZE, kernel_size: int = 4, activation: str | None = "silu", add_bias: bool = True
) -> DepthwiseCausalConvolution:
    return DepthwiseCausalConvolution(
        hidden_size=hidden_size, kernel_size=kernel_size, activation_function=activation, add_bias=add_bias
    )


@pytest.mark.parametrize("kernel_size", [4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
@pytest.mark.parametrize("output_state", [False, True])
@pytest.mark.parametrize("short_seq", [False, True])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.torch])
def test_prefill_shapes(
    kernel_size: int,
    add_bias: bool,
    activation: str | None,
    output_state: bool,
    short_seq: bool,
    kernel_backend: KernelBackend,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    if kernel_backend == KernelBackend.cuda and not is_causal_conv1d_available():
        pytest.skip("causal_conv1d unavailable")

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    seq_len = max(1, kernel_size - 1) if short_seq else _PREFILL_LEN
    x = torch.randn(_BATCH, seq_len, _HIDDEN_SIZE, device=device)

    out, state = conv(
        x, input_state=None, attention_mask=None, output_state=output_state, kernel_backend=kernel_backend
    )

    assert out.size() == x.size()

    if output_state:
        assert state is not None
        assert state.size() == (_BATCH, _HIDDEN_SIZE, kernel_size - 1)
    else:
        assert state is None


@pytest.mark.parametrize("kernel_size", [4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
@pytest.mark.parametrize("output_state", [False, True])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.torch])
def test_generation_shapes(
    kernel_size: int, add_bias: bool, activation: str | None, output_state: bool, kernel_backend: KernelBackend
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    if kernel_backend == KernelBackend.cuda and not is_causal_conv1d_available():
        pytest.skip("causal_conv1d unavailable")

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    x_prefill = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    _, state = conv(x_prefill, input_state=None, attention_mask=None, output_state=True, kernel_backend=kernel_backend)

    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)
    out, state_out = conv(
        x_gen, input_state=state, attention_mask=None, output_state=output_state, kernel_backend=kernel_backend
    )

    assert out.size() == (_BATCH, 1, _HIDDEN_SIZE)

    if output_state:
        assert state_out is not None
        assert state_out.size() == state.size()
    else:
        assert state_out is None


@pytest.mark.parametrize("kernel_size", [4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
@pytest.mark.parametrize("seq_len", [1, 2, 4])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.torch])
def test_zero_state_matches_fresh_prefill(
    kernel_size: int, add_bias: bool, activation: str | None, seq_len: int, kernel_backend: KernelBackend
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    if kernel_backend == KernelBackend.cuda and not is_causal_conv1d_available():
        pytest.skip("causal_conv1d unavailable")

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    x = torch.randn(_BATCH, seq_len, _HIDDEN_SIZE, device=device)
    zero_state = torch.zeros(_BATCH, _HIDDEN_SIZE, kernel_size - 1, device=device)

    out_fresh, state_fresh = conv(
        x, input_state=None, attention_mask=None, output_state=True, kernel_backend=kernel_backend
    )

    out_zero, state_zero = conv(
        x, input_state=zero_state, attention_mask=None, output_state=True, kernel_backend=kernel_backend
    )

    assert_close(out_zero, out_fresh, rtol=1e-5, atol=1e-5)
    assert_close(state_zero, state_fresh, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kernel_size", [4])
@pytest.mark.parametrize("add_bias", [False, True])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
@pytest.mark.parametrize("continuation_len", [1, 2, 4])
@pytest.mark.parametrize("n_gen_steps", [1, 2])
@pytest.mark.parametrize("short_prefill", [False, True])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.torch])
def test_consistency(
    kernel_size: int,
    add_bias: bool,
    activation: str | None,
    continuation_len: int,
    n_gen_steps: int,
    short_prefill: bool,
    kernel_backend: KernelBackend,
) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    if kernel_backend == KernelBackend.cuda and not is_causal_conv1d_available():
        pytest.skip("causal_conv1d unavailable")

    rtol, atol = (2e-2, 2e-3) if Accelerator.get_accelerator() == Accelerator.tpu else (1e-5, 1e-5)

    with torch.device(device):
        conv = _make_conv(kernel_size=kernel_size, add_bias=add_bias, activation=activation)

    conv.eval()

    prefill_len = max(1, kernel_size - 1) if short_prefill else _PREFILL_LEN
    total_gen_steps = continuation_len + n_gen_steps
    x_full = torch.randn(_BATCH, prefill_len + total_gen_steps, _HIDDEN_SIZE, device=device)

    out_full, state_full = conv(
        x_full, input_state=None, attention_mask=None, output_state=True, kernel_backend=kernel_backend
    )

    _, state = conv(
        x_full[:, :prefill_len],
        input_state=None,
        attention_mask=None,
        output_state=True,
        kernel_backend=kernel_backend,
    )

    out_continue, state = conv(
        x_full[:, prefill_len : prefill_len + continuation_len],
        input_state=state,
        attention_mask=None,
        output_state=True,
        kernel_backend=kernel_backend,
    )

    assert_close(
        out_continue,
        out_full[:, prefill_len : prefill_len + continuation_len],
        rtol=rtol,
        atol=atol,
    )

    for step in range(n_gen_steps):
        start = prefill_len + continuation_len + step
        x_step = x_full[:, start : start + 1]
        is_last = step == n_gen_steps - 1

        out_step, state = conv(
            x_step,
            input_state=state,
            attention_mask=None,
            output_state=not is_last,
            kernel_backend=kernel_backend,
        )

        assert_close(out_step, out_full[:, start : start + 1], rtol=rtol, atol=atol)

    assert state is None
    _, state = conv(
        x_full[:, :prefill_len],
        input_state=None,
        attention_mask=None,
        output_state=True,
        kernel_backend=kernel_backend,
    )

    _, state = conv(
        x_full[:, prefill_len:],
        input_state=state,
        attention_mask=None,
        output_state=True,
        kernel_backend=kernel_backend,
    )
    assert_close(state, state_full, rtol=rtol, atol=atol)


@pytest.mark.parametrize("kernel_size", [4])
@pytest.mark.parametrize("kernel_backend", [KernelBackend.cuda, KernelBackend.torch])
def test_attention_mask(kernel_size: int, kernel_backend: KernelBackend) -> None:
    skip_if_incompatible_kernel_backend(kernel_backend)
    device = kernel_backend.get_compatible_accelerator().get_current_device()

    if kernel_backend == KernelBackend.cuda and not is_causal_conv1d_available():
        pytest.skip("causal_conv1d unavailable")

    rtol, atol = (2e-2, 2e-3) if Accelerator.get_accelerator() == Accelerator.tpu else (1e-5, 1e-5)

    conv = _make_conv(kernel_size=kernel_size, activation=None).to(device)
    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)

    # all-ones mask is a no-op
    mask_ones = torch.ones(_BATCH, _PREFILL_LEN, dtype=x.dtype, device=device)
    out_no_mask, _ = conv(x, input_state=None, attention_mask=None, output_state=False, kernel_backend=kernel_backend)

    out_ones, _ = conv(
        x, input_state=None, attention_mask=mask_ones, output_state=False, kernel_backend=kernel_backend
    )
    assert_close(out_no_mask, out_ones, rtol=rtol, atol=atol)

    # padding positions in the output must be exactly zero
    mask = mask_ones.clone()
    mask[1, :3] = 0
    out_masked, _ = conv(x, input_state=None, attention_mask=mask, output_state=False, kernel_backend=kernel_backend)
    assert (out_masked[1, :3] == 0).all()

    # non-padding positions must match a manually zeroed input (no mask)
    x_zeroed = x.clone()
    x_zeroed[1, :3] = 0
    out_zeroed, _ = conv(
        x_zeroed, input_state=None, attention_mask=None, output_state=False, kernel_backend=kernel_backend
    )
    assert_close(out_masked[0], out_zeroed[0], rtol=rtol, atol=atol)
    assert_close(out_masked[1, 3:], out_zeroed[1, 3:], rtol=rtol, atol=atol)


@pytest.mark.parametrize("kernel_size", [4])
@pytest.mark.parametrize("activation", [None, "silu", "gelu"])
def test_kernel_vs_fallback(kernel_size: int, activation: str | None) -> None:
    skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    device = Accelerator.get_current_device()

    if not is_causal_conv1d_available():
        pytest.skip("causal_conv1d unavailable")

    conv = _make_conv(kernel_size=kernel_size, activation=activation).to(device)
    conv.eval()

    x = torch.randn(_BATCH, _PREFILL_LEN, _HIDDEN_SIZE, device=device)
    x_gen = torch.randn(_BATCH, 1, _HIDDEN_SIZE, device=device)

    out_k, state_k = conv(
        x, input_state=None, attention_mask=None, output_state=True, kernel_backend=KernelBackend.cuda
    )
    out_gen_k, _ = conv(
        x_gen, input_state=state_k, attention_mask=None, output_state=False, kernel_backend=KernelBackend.cuda
    )

    out_f, state_f = conv(x, input_state=None, attention_mask=None, output_state=True)
    out_gen_f, _ = conv(x_gen, input_state=state_f, attention_mask=None, output_state=False)

    assert_close(out_k, out_f, rtol=1e-5, atol=1e-5)
    assert_close(state_k, state_f, rtol=1e-5, atol=1e-5)
    assert_close(out_gen_k, out_gen_f, rtol=1e-5, atol=1e-5)
