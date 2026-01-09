import torch

import cutlass.cute as cute
from cutlass import Boolean, Float32, range_constexpr

from ....constants import LOG_WARP_SIZE, WARP_SIZE
from ....custom_op import xma_op
from ....cute_dsl_utils import torch_tensor_to_cute_tensor


@xma_op(mutates_args={"y"})
def softmax_forward_cuda(x: torch.Tensor, y: torch.Tensor, logits_multiplier: float | None) -> None:
    x = torch_tensor_to_cute_tensor(x, leading_dim=-1)
    y = torch_tensor_to_cute_tensor(y, leading_dim=-1)
