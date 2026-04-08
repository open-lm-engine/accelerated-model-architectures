// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void _pack_unpack_sequence_cuda(const torch::Tensor &x,
                                torch::Tensor &output,
                                const torch::Tensor &cu_seqlens,
                                const std::string &padding_side,
                                const bool &pack,
                                const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_pack_unpack_sequence_cuda", &_pack_unpack_sequence_cuda, "pack unpack sequence (CUDA)");
}
