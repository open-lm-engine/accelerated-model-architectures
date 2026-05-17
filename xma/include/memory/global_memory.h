// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#include "../dtypes.h"

namespace flash_model_architectures::memory {
    template <typename T>
    inline __device__ T *load_128_bits(const T *array, const uint64 &index) {
        const int32_4 *vector_array = reinterpret_cast<const int32_4 *>(array);
        int32_4 vector_element = vector_array[index];
        T *output = reinterpret_cast<T *>(&vector_element);
        return output;
    }

    template <typename T>
    inline __device__ void store_128_bits(const T *source, T *destination, const uint64 &index) {
        int32_4 *destination_vector_array = reinterpret_cast<int32_4 *>(destination);
        const int32_4 source_vector = reinterpret_cast<const int32_4 *>(&source[0])[0];
        destination_vector_array[index] = source_vector;
    }

    template <typename T>
    constexpr inline __device__ uint32 get_num_elements_for_vector_load_stores() {
        return sizeof(int32_4) / sizeof(T);
    }
}  // namespace flash_model_architectures::memory
