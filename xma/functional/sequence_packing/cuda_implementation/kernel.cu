// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/xma.h"

namespace ck = flash_model_architectures;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

enum class PaddingSide { left, right };

template <typename scalar_t, bool is_packing>
inline __device__ void _copy_array(const scalar_t *source,
                                   scalar_t *destination,
                                   const uint32 &b,
                                   const uint32 &s,
                                   const uint32 &t,
                                   const uint32 &S,
                                   const uint32 &N) {
    constexpr uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
    const uint32 N_vec = N / N_per_thread;

    uint32 unpacked_offset = (b * S + s) * N_vec;
    uint32 packed_offset = t * N_vec;

    for (uint32 i = threadIdx.x; i < N_vec; i += blockDim.x) {
        if (is_packing) {
            const scalar_t *source_vec = ck_mem::load_128_bits<scalar_t>(source, unpacked_offset + i);
            ck_mem::store_128_bits<scalar_t>(source_vec, destination, packed_offset + i);
        } else {
            const scalar_t *source_vec = ck_mem::load_128_bits<scalar_t>(source, packed_offset + i);
            ck_mem::store_128_bits<scalar_t>(source_vec, destination, unpacked_offset + i);
        }
    }
}

template <typename scalar_t, typename integer_t, PaddingSide padding_side, bool is_packing>
__global__ void pack_unpack_sequence_cuda_kernel(
    const scalar_t *x, scalar_t *output, const integer_t *cu_seqlens, const uint32 S, const uint32 N) {
    const uint32 s = blockIdx.x;
    const uint32 b = blockIdx.y;

    const integer_t start = cu_seqlens[b];
    const integer_t end = cu_seqlens[b + 1];
    const uint32 seqlens = end - start;

    if (padding_side == PaddingSide::left) {
        const uint32 pad_tokens = S - seqlens;
        if (s >= pad_tokens) {
            _copy_array<scalar_t, is_packing>(x, output, b, s, start + s - pad_tokens, S, N);
        }
    } else {
        if (s < seqlens) {
            _copy_array<scalar_t, is_packing>(x, output, b, s, start + s, S, N);
        }
    }
}

void _pack_unpack_sequence_cuda(const torch::Tensor &x,
                                torch::Tensor &output,
                                const torch::Tensor &cu_seqlens,
                                const std::string &padding_side,
                                const bool &pack,
                                const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);
    CHECK_CUDA_TENSOR(cu_seqlens);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    TORCH_CHECK(padding_side == "left" || padding_side == "right");

    uint32 B, S, N;
    if (pack) {
        B = x.size(0);
        S = x.size(1);
        N = x.numel() / (B * S);
    } else {
        B = output.size(0);
        S = output.size(1);
        N = output.numel() / (B * S);
    }

    const dim3 NUM_BLOCKS = dim3(S, B);
    const uint32 shared_memory_size = B * sizeof(uint32);

    DISPATCH_FLOAT_KERNEL(
        x.scalar_type(), "pack_unpack_sequence_cuda_kernel_float", scalar_t, ([&] {
            constexpr uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
            TORCH_CHECK(N % N_per_thread == 0);

            DISPATCH_INT_KERNEL(
                cu_seqlens.scalar_type(), "pack_unpack_sequence_cuda_kernel_int", integer_t, ([&] {
                    if (pack) {
                        if (padding_side == "left") {
                            pack_unpack_sequence_cuda_kernel<scalar_t, integer_t, PaddingSide::left, true>
                                <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(x.data_ptr<scalar_t>(),
                                                                                 output.data_ptr<scalar_t>(),
                                                                                 cu_seqlens.data_ptr<integer_t>(),
                                                                                 S,
                                                                                 N);
                        } else {
                            pack_unpack_sequence_cuda_kernel<scalar_t, integer_t, PaddingSide::right, true>
                                <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(x.data_ptr<scalar_t>(),
                                                                                 output.data_ptr<scalar_t>(),
                                                                                 cu_seqlens.data_ptr<integer_t>(),
                                                                                 S,
                                                                                 N);
                        }
                    } else {
                        if (padding_side == "left") {
                            pack_unpack_sequence_cuda_kernel<scalar_t, integer_t, PaddingSide::left, false>
                                <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(x.data_ptr<scalar_t>(),
                                                                                 output.data_ptr<scalar_t>(),
                                                                                 cu_seqlens.data_ptr<integer_t>(),
                                                                                 S,
                                                                                 N);
                        } else {
                            pack_unpack_sequence_cuda_kernel<scalar_t, integer_t, PaddingSide::right, false>
                                <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(x.data_ptr<scalar_t>(),
                                                                                 output.data_ptr<scalar_t>(),
                                                                                 cu_seqlens.data_ptr<integer_t>(),
                                                                                 S,
                                                                                 N);
                        }
                    }
                }));
        }));
}
