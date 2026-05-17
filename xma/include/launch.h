// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include "dtypes.h"

namespace flash_model_architectures {
    template <typename T>
    struct ChunkedArray {
        T *array;
        uint64 num_elements;
    };

    template <typename T>
    std::vector<ChunkedArray<T> > chunk_array(T *x, const uint64 &num_elements) {
        const uint64 num_elements_per_chunk = (uint64)std::numeric_limits<uint>::max() + 1;
        uint64 start = 0;

        std::vector<ChunkedArray<T> > result;

        // We will "yield" chunks of the array, each of size 'chunk_size' or less
        while (start < num_elements) {
            uint64 remaining = num_elements - start;
            uint64 current_chunk_size = std::min(remaining, num_elements_per_chunk);

            ChunkedArray<T> chunked_x;
            chunked_x.array = (T *)&x[start];
            chunked_x.num_elements = current_chunk_size;

            result.push_back(chunked_x);
            start += current_chunk_size;
        }

        return result;
    }
}  // namespace flash_model_architectures
