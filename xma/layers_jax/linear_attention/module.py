# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ...accelerator import Accelerator, KernelBackend
from ...math import divide_if_divisible
from ..depthwise_causal_convolution import DepthwiseCausalConvolutionJAX
from .op import linear_attention_packed_jax


class _Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array | None

    @staticmethod
    def init(in_size: int, out_size: int, use_bias: bool, *, key: PRNGKeyArray) -> _Linear:
        weight_key, bias_key = jax.random.split(key, 2)
        bound = in_size**-0.5

        weight = jax.random.uniform(weight_key, (in_size, out_size), minval=-bound, maxval=bound)
        bias = jax.random.uniform(bias_key, (out_size,), minval=-bound, maxval=bound) if use_bias else None

        return _Linear(weight=weight, bias=bias)

    def __call__(self, input: jax.Array) -> jax.Array:
        output = input @ self.weight
        if self.bias is not None:
            output = output + self.bias

        return output


class LinearAttentionJAX(eqx.Module):
    input_projection: _Linear
    output_projection: _Linear
    forget_gate_projection: _Linear | None
    output_gate_projection: _Linear | None
    output_norm_weight: jax.Array | None
    conv1d: DepthwiseCausalConvolutionJAX | None

    input_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    num_query_heads: int = eqx.field(static=True)
    num_key_heads: int = eqx.field(static=True)
    num_value_heads: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    key_head_dim: int = eqx.field(static=True)
    value_head_dim: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    forget_gate: str | None = eqx.field(static=True)
    norm_eps: float = eqx.field(static=True)

    attention_multiplier: float | None = eqx.field(static=True)
    BLOCK_SIZE_S: int = eqx.field(static=True)
    BLOCK_SIZE_V: int = eqx.field(static=True)

    @staticmethod
    def init(
        input_size: int,
        output_size: int,
        key_head_dim: int,
        value_head_dim: int,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        add_bias: bool,
        *,
        attention_multiplier: float | None = None,
        BLOCK_SIZE_S: int = 256,
        BLOCK_SIZE_V: int = 128,
        kernel_size: int | None = None,
        conv_activation_function: str | Callable[[jax.Array], jax.Array] | None = None,
        forget_gate: str | None = None,
        output_gate: bool = False,
        norm_eps: float = 1e-5,
        key: PRNGKeyArray,
    ) -> LinearAttentionJAX:
        # forget_gate: None (Linear Attention), "scalar" (one decay per head, Mamba-2) or "diagonal" (one decay per
        # head and key channel, GLA); log f = log_sigmoid of a projection of the layer input
        # output_gate: normalize each head's output with RMSNorm (a learned scale shared by the heads) and multiply it by
        # swish of a projection of the layer input before the output projection, as FLA's GLA layer does
        assert forget_gate in (None, "scalar", "diagonal")

        num_heads = max(num_query_heads, num_key_heads, num_value_heads)

        divide_if_divisible(num_heads, num_query_heads)
        divide_if_divisible(num_heads, num_key_heads)
        divide_if_divisible(num_heads, num_value_heads)

        lane_count = Accelerator.get_lane_count()

        if BLOCK_SIZE_V <= 0 or BLOCK_SIZE_V % lane_count != 0:
            raise ValueError(
                f"BLOCK_SIZE_V ({BLOCK_SIZE_V}) must be a positive multiple of {lane_count} "
                "(pallas-kernel envelope; enforced at init so a later kernel_backend='pallas' call cannot fail mid-run)"
            )

        query_size = num_query_heads * key_head_dim
        key_size = num_key_heads * key_head_dim
        value_size = num_value_heads * value_head_dim
        qkv_size = query_size + key_size + value_size
        heads_value_size = num_heads * value_head_dim

        key_input_projection, key_output_projection, key_conv, key_forget_gate, key_output_gate = jax.random.split(
            key, 5
        )

        input_projection = _Linear.init(input_size, qkv_size, use_bias=add_bias, key=key_input_projection)
        output_projection = _Linear.init(heads_value_size, output_size, use_bias=add_bias, key=key_output_projection)

        forget_gate_projection = None
        if forget_gate is not None:
            key_forget_gate, key_retention = jax.random.split(key_forget_gate)
            forget_gate_projection = _Linear.init(
                input_size,
                num_heads * (1 if forget_gate == "scalar" else key_head_dim),
                use_bias=True,
                key=key_forget_gate,
            )
            # the gate always has a bias, set so every channel starts with a per-token retention sigmoid(b) spread over
            # [0.9, 0.999] (as Mamba-2's dt_bias does): slow decays at init, and a chunk's summed log decay stays small
            retention = jax.random.uniform(key_retention, forget_gate_projection.bias.shape, minval=0.9, maxval=0.999)
            forget_gate_projection = eqx.tree_at(
                lambda m: m.bias, forget_gate_projection, jnp.log(retention) - jnp.log1p(-retention)
            )

        output_gate_projection = None
        output_norm_weight = None
        if output_gate:
            output_gate_projection = _Linear.init(input_size, heads_value_size, use_bias=add_bias, key=key_output_gate)
            output_norm_weight = jnp.ones((value_head_dim,))

        conv1d = (
            None
            if kernel_size is None
            else DepthwiseCausalConvolutionJAX.init(
                qkv_size,
                kernel_size=kernel_size,
                activation_function=conv_activation_function,
                add_bias=add_bias,
                key=key_conv,
            )
        )

        return LinearAttentionJAX(
            input_projection=input_projection,
            output_projection=output_projection,
            forget_gate_projection=forget_gate_projection,
            output_gate_projection=output_gate_projection,
            output_norm_weight=output_norm_weight,
            conv1d=conv1d,
            input_size=input_size,
            output_size=output_size,
            num_query_heads=num_query_heads,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            num_heads=num_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            state_size=key_head_dim * value_head_dim,
            forget_gate=forget_gate,
            norm_eps=norm_eps,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_V=BLOCK_SIZE_V,
        )

    def __call__(
        self,
        input: jax.Array,
        input_state: jax.Array | None = None,
        conv_state: jax.Array | None = None,
        output_conv_state: bool = False,
        output_state: bool = True,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
        log_forget = None
        if self.forget_gate_projection is not None:
            log_forget = jax.nn.log_sigmoid(self.forget_gate_projection(input).astype(jnp.float32))
            if self.forget_gate == "diagonal":
                # GLA's gate_logit_normalizer: keeps a chunk's summed log decay inside the fp32 exp range
                log_forget = log_forget / 16
                log_forget = log_forget.reshape(*log_forget.shape[:-1], self.num_heads, self.key_head_dim)

        output_gate = None if self.output_gate_projection is None else self.output_gate_projection(input)

        projected = self.input_projection(input)

        if self.conv1d is None:
            assert conv_state is None
        else:
            projected, conv_state = self.conv1d(
                projected, input_state=conv_state, output_state=output_conv_state, kernel_backend=kernel_backend
            )

        input, input_state = linear_attention_packed_jax(
            projected,
            num_query_heads=self.num_query_heads,
            num_key_heads=self.num_key_heads,
            num_value_heads=self.num_value_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            log_forget=log_forget,
            input_state=input_state,
            attention_multiplier=self.attention_multiplier,
            output_state=output_state,
            BLOCK_SIZE_S=self.BLOCK_SIZE_S,
            BLOCK_SIZE_V=self.BLOCK_SIZE_V,
            kernel_backend=kernel_backend,
        )

        if output_gate is not None:
            input = self._gated_rms_norm(input, output_gate)

        input = self.output_projection(input)

        return input, input_state, conv_state

    def _gated_rms_norm(self, input: jax.Array, output_gate: jax.Array) -> jax.Array:
        # per-head RMSNorm of (..., N * V), times swish of the gate. Splitting the lane dimension into (N, V) makes XLA
        # relayout fp32 copies of the whole activation on TPU, so each head's mean square is taken with a (N * V, N)
        # block-indicator matmul instead, and broadcast back with its transpose. Both are single bf16 MXU passes: the
        # squares are all positive, so their rounding averages out, and the bf16 scale is no coarser than the output
        dtype = input.dtype
        x = input.astype(jnp.float32)
        heads = jnp.arange(self.num_heads * self.value_head_dim) // self.value_head_dim
        indicator = (heads[:, None] == jnp.arange(self.num_heads)[None, :]).astype(jnp.bfloat16)

        mean_square = jnp.dot((x * x).astype(jnp.bfloat16), indicator, preferred_element_type=jnp.float32)
        rms = jax.lax.rsqrt(mean_square / self.value_head_dim + self.norm_eps).astype(jnp.bfloat16)
        scale = jnp.dot(rms, indicator.T, preferred_element_type=jnp.bfloat16).astype(jnp.float32)

        weight = jnp.tile(self.output_norm_weight.astype(jnp.float32), self.num_heads)
        return (x * scale * weight * jax.nn.silu(output_gate.astype(jnp.float32))).astype(dtype)
