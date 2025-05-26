# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch


def pack_sequence_torch(
    inputs: torch.Tensor | list[torch.Tensor], cu_seqlens: torch.Tensor, padding_side: str = "left"
) -> torch.Tensor | list[torch.Tensor]:
    is_list = isinstance(inputs, (list, tuple))
    if not is_list:
        inputs = [inputs]

    outputs = []

    for x in inputs:
        B, S = x.size()[:2]
        other_dims = x.shape[2:]

        sum_seqlens = cu_seqlens[-1]
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

        if padding_side == "left":
            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])
        elif padding_side == "right":
            seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])
        else:
            raise ValueError(f"unexpected padding_side ({padding_side})")

        unpadded = torch.zeros(sum_seqlens, *other_dims, dtype=x.dtype, device=x.device)
        unpadded = x[batch_indices, seq_indices]

        outputs.append(unpadded)

    if not is_list:
        outputs = outputs[0]

    return outputs


def unpack_sequence_torch(
    inputs: torch.Tensor | list[torch.Tensor],
    cu_seqlens: torch.Tensor,
    desired_shape: tuple[int],
    padding_side: str = "left",
) -> torch.Tensor | list[torch.Tensor]:

    is_list = isinstance(inputs, (list, tuple))
    if not is_list:
        inputs = [inputs]

    outputs = []

    for x in inputs:
        B, S = desired_shape[:2]
        assert cu_seqlens.size(0) - 1 == B
        assert desired_shape[2:] == x.size()[1:]

        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

        if padding_side == "left":
            pad_tokens = S - seqlens
            seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])
        elif padding_side == "right":
            seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])
        else:
            raise ValueError(f"unexpected padding_side ({padding_side})")

        padded = torch.zeros(desired_shape, dtype=x.dtype, device=x.device)
        padded[batch_indices, seq_indices] = x

        outputs.append(padded)

    if not is_list:
        outputs = outputs[0]

    return outputs
