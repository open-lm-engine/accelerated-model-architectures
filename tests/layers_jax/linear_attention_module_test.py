# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest


haliax = pytest.importorskip("haliax")

import jax

from xma import KernelBackend, LinearAttentionJAX


@pytest.mark.parametrize("has_input_state", [False, True])
def test_linear_attention_module(has_input_state: bool) -> None:
    Embed = haliax.Axis("embed", 32)
    Output = haliax.Axis("output", 24)
    Batch = haliax.Axis("batch", 2)
    Pos = haliax.Axis("position", 16)

    key_init, key_input, key_state = jax.random.split(jax.random.PRNGKey(0), 3)

    module = LinearAttentionJAX.init(
        Embed,
        Output,
        key_head_dim=8,
        value_head_dim=8,
        num_query_heads=4,
        num_key_heads=2,
        num_value_heads=1,
        add_bias=True,
        key=key_init,
    )

    input = haliax.random.normal(key_input, (Batch, Pos, Embed))
    input_state = haliax.random.normal(key_state, (Batch, module.Heads, module.StateSize)) if has_input_state else None

    # this is a smoke test: it only checks that the module runs end to end and returns the expected shapes, not
    # that the output is numerically correct (that's covered at the op level by linear_attention_jax_test.py)
    output, output_state = module(input, input_state, kernel_backend=KernelBackend.jax)

    assert output.axes == (Batch, Pos, Output)
    assert output_state.axes == (Batch, module.Heads, module.StateSize)
