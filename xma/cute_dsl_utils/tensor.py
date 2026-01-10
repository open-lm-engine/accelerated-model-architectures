import cutlass.cute as cute
from cutlass import Int32


def get_predication_tensor(cX: cute.Tensor, boundary: Int32) -> cute.Tensor:
    pX = cute.make_rmem_tensor_like(cX)
