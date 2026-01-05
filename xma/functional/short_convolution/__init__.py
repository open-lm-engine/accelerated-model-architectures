import torch
import torch.nn.functional as F

from ...custom_op import CustomOp


class _ShortConvolution1D(CustomOp):
    @staticmethod
    def forward_backward_torch(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: int, padding: int, groups: int
    ) -> torch.Tensor:
        x = F.conv1d(
            input=x,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            groups=groups,
        )

        return x


def short_convolution_1D(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: int, padding: int, groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    x = _ShortConvolution1D.run(x=x, weight=weight, bias=bias, stride=stride, padding=padding, groups=groups)

    return x
