# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn

from ..accelerator import KernelBackend
from ..functional import linear_attention
from ..math import divide_if_divisible


class LinearAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        key_head_dim: int,
        value_head_dim: int,
        output_size: int,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        add_bias: bool,
    ) -> None:
        super().__init__()

        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim

        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.num_heads = max(num_query_heads, num_key_heads, num_value_heads)

        divide_if_divisible(self.num_heads, self.num_query_heads)
        divide_if_divisible(self.num_heads, self.num_key_heads)
        divide_if_divisible(self.num_heads, self.num_value_heads)

        self.query_size = self.num_query_heads * self.key_head_dim
        self.key_size = self.num_key_heads * self.key_head_dim
        self.value_size = self.num_value_heads * self.value_head_dim
        self.state_size = self.num_heads * self.key_head_dim * self.value_head_dim

        self.input_projection = nn.Linear(input_size, self.query_size + self.key_size + self.value_size, bias=add_bias)
        self.output_projection = nn.Linear(self.num_heads * self.value_head_dim, output_size, bias=add_bias)

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        *,
        kernel_backend: KernelBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = self.input_projection(input)

        query, key, value = input.split((self.query_size, self.key_size, self.value_size), dim=-1)

        query = query.view(*query.size()[:-1], self.num_query_heads, self.key_head_dim)
        key = key.view(*key.size()[:-1], self.num_key_heads, self.key_head_dim)
        value = value.view(*value.size()[:-1], self.num_value_heads, self.value_head_dim)

        if input_state is not None:
            input_state = input_state.view(-1, self.num_heads, self.key_head_dim, self.value_head_dim)

        input, input_state = linear_attention(
            query=query,
            key=key,
            value=value,
            input_state=input_state,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kernel_backend=kernel_backend,
        )

        input = input.flatten(-2, -1)
        input_state = input_state.flatten(-2, -1)

        input = self.output_projection(input)

        return input, input_state
