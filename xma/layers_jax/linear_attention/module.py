# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import equinox as eqx
import haliax as hax
import jax
from haliax import Axis, NamedArray
from jaxtyping import PRNGKeyArray

from ...accelerator import KernelBackend
from ...math import divide_if_divisible
from .op import linear_attention_jax


class LinearAttentionJAX(eqx.Module):
    input_projection: hax.nn.Linear
    output_projection: hax.nn.Linear

    Embed: Axis = eqx.field(static=True)
    Output: Axis = eqx.field(static=True)
    QHeads: Axis = eqx.field(static=True)
    KHeads: Axis = eqx.field(static=True)
    VHeads: Axis = eqx.field(static=True)
    Heads: Axis = eqx.field(static=True)
    KeyHeadDim: Axis = eqx.field(static=True)
    ValueHeadDim: Axis = eqx.field(static=True)
    StateSize: Axis = eqx.field(static=True)

    attention_multiplier: float | None = eqx.field(static=True)
    BLOCK_SIZE_S: int = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: Axis,
        Output: Axis,
        key_head_dim: int,
        value_head_dim: int,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        add_bias: bool,
        *,
        attention_multiplier: float | None = None,
        BLOCK_SIZE_S: int = 64,
        key: PRNGKeyArray,
    ) -> LinearAttentionJAX:
        num_heads = max(num_query_heads, num_key_heads, num_value_heads)

        divide_if_divisible(num_heads, num_query_heads)
        divide_if_divisible(num_heads, num_key_heads)
        divide_if_divisible(num_heads, num_value_heads)

        QHeads = Axis("query_heads", num_query_heads)
        KHeads = Axis("key_heads", num_key_heads)
        VHeads = Axis("value_heads", num_value_heads)
        Heads = Axis("heads", num_heads)
        KeyHeadDim = Axis("key_head_dim", key_head_dim)
        ValueHeadDim = Axis("value_head_dim", value_head_dim)
        StateSize = Axis("state_size", key_head_dim * value_head_dim)

        QuerySize = Axis("query_size", num_query_heads * key_head_dim)
        KeySize = Axis("key_size", num_key_heads * key_head_dim)
        ValueSize = Axis("value_size", num_value_heads * value_head_dim)
        QKV = Axis("qkv", QuerySize.size + KeySize.size + ValueSize.size)
        HeadsValueSize = Axis("heads_value_size", num_heads * value_head_dim)

        key_input_projection, key_output_projection = jax.random.split(key, 2)

        input_projection = hax.nn.Linear.init(Embed, QKV, key=key_input_projection, use_bias=add_bias)
        output_projection = hax.nn.Linear.init(HeadsValueSize, Output, key=key_output_projection, use_bias=add_bias)

        return LinearAttentionJAX(
            input_projection=input_projection,
            output_projection=output_projection,
            Embed=Embed,
            Output=Output,
            QHeads=QHeads,
            KHeads=KHeads,
            VHeads=VHeads,
            Heads=Heads,
            KeyHeadDim=KeyHeadDim,
            ValueHeadDim=ValueHeadDim,
            StateSize=StateSize,
            attention_multiplier=attention_multiplier,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
        )

    def __call__(
        self,
        input: NamedArray,
        input_state: NamedArray | None = None,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[NamedArray, NamedArray]:
        # input: (Batch, Pos, Embed); the underlying pallas kernel only supports a single leading batch axis,
        # so Batch and Pos are whatever's left over once Embed is accounted for, in their existing order
        Batch, Pos = [axis for axis in input.axes if axis != self.Embed]

        QuerySize = Axis("query_size", self.QHeads.size * self.KeyHeadDim.size)
        KeySize = Axis("key_size", self.KHeads.size * self.KeyHeadDim.size)
        ValueSize = Axis("value_size", self.VHeads.size * self.ValueHeadDim.size)

        projected = self.input_projection(input)
        query, key, value = projected.split("qkv", [QuerySize, KeySize, ValueSize])

        query = query.unflatten_axis(QuerySize, (self.QHeads, self.KeyHeadDim))
        key = key.unflatten_axis(KeySize, (self.KHeads, self.KeyHeadDim))
        value = value.unflatten_axis(ValueSize, (self.VHeads, self.ValueHeadDim))

        if input_state is not None:
            input_state = input_state.unflatten_axis(self.StateSize, (self.KeyHeadDim, self.ValueHeadDim))
            input_state = input_state.array

        output, output_state = linear_attention_jax(
            query=query.array,
            key=key.array,
            value=value.array,
            input_state=input_state,
            attention_multiplier=self.attention_multiplier,
            BLOCK_SIZE_S=self.BLOCK_SIZE_S,
            kernel_backend=kernel_backend,
        )

        HeadsValueSize = Axis("heads_value_size", self.Heads.size * self.ValueHeadDim.size)

        output = hax.named(output, (Batch, Pos, self.Heads, self.ValueHeadDim))
        output = output.flatten_axes((self.Heads, self.ValueHeadDim), HeadsValueSize)
        output = self.output_projection(output)

        output_state = hax.named(output_state, (Batch, self.Heads, self.KeyHeadDim, self.ValueHeadDim))
        output_state = output_state.flatten_axes((self.KeyHeadDim, self.ValueHeadDim), self.StateSize)

        return output, output_state
