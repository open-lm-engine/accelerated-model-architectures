// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#pragma once

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

#define CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE) TORCH_CHECK(BLOCK_SIZE % WARP_SIZE == 0)

#include <cuda.h>
#include <cuda_runtime.h>

#include "dtypes.h"
#include "math.h"

namespace flash_model_architectures {
    inline __device__ uint32 get_num_blocks() { return gridDim.x * gridDim.y * gridDim.z; }

    inline __host__ int get_max_thread_blocks(const uint32 &sm_count, const uint32 &thread_block_cluster_size) {
        int max_num_blocks = sm_count;
        if (max_num_blocks % thread_block_cluster_size != 0) {
            max_num_blocks = thread_block_cluster_size * (max_num_blocks / thread_block_cluster_size);
        }

        return max_num_blocks;
    }

    inline __host__ std::tuple<uint32, uint32> get_num_blocks(const uint64 &num_elements,
                                                              const uint32 &BLOCK_SIZE,
                                                              const uint32 &sm_count = 0,
                                                              const uint32 &max_thread_block_cluster_size = 1) {
        uint32 NUM_BLOCKS = ceil_divide<uint64>(num_elements, BLOCK_SIZE);
        if (sm_count != 0 && NUM_BLOCKS > sm_count) {
            NUM_BLOCKS = sm_count;
        }

        int thread_block_cluster_size = max_thread_block_cluster_size;
        if (thread_block_cluster_size > NUM_BLOCKS) {
            thread_block_cluster_size = NUM_BLOCKS;
        }

        NUM_BLOCKS = thread_block_cluster_size * (NUM_BLOCKS / thread_block_cluster_size);

        return std::make_tuple(NUM_BLOCKS, thread_block_cluster_size);
    }
}  // namespace flash_model_architectures
