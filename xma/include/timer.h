// **************************************************
// Copyright (c) 2026, Mayank Mishra
// **************************************************

#include "dtypes.h"

namespace flash_model_architectures {
    struct GpuTimer {
        cudaStream_t _stream_id;
        cudaEvent_t _start;
        cudaEvent_t _stop;

        /// Constructor
        GpuTimer() : _stream_id(0) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
        }

        /// Destructor
        ~GpuTimer() {
            cudaEventDestroy(_start);
            cudaEventDestroy(_stop);
        }

        /// Start the timer for a given stream (defaults to the default stream)
        void start(cudaStream_t stream_id = 0) {
            _stream_id = stream_id;
            cudaEventRecord(_start, _stream_id);
        }

        /// Stop the timer
        void stop() { cudaEventRecord(_stop, _stream_id); }

        /// Return the elapsed time (in milliseconds)
        fp32 elapsed_millis() {
            fp32 elapsed = 0.0;
            cudaEventSynchronize(_stop);
            cudaEventElapsedTime(&elapsed, _start, _stop);
            return elapsed;
        }
    };
}  // namespace flash_model_architectures
