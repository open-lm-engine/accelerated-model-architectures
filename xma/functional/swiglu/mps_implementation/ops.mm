// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

// PyTorch C++ extension header — gives us torch::Tensor, TORCH_CHECK, pybind11 bindings, etc.
#include <torch/extension.h>

// Apple's Metal GPU framework — the Obj-C API for talking to the GPU on macOS/iOS
// (similar role to the CUDA runtime API on NVIDIA)
#import <Metal/Metal.h>

// PyTorch's internal MPS stream — lets us submit GPU work on the same command queue
// that PyTorch uses, so our kernel executes in order with other PyTorch MPS operations
#include <ATen/mps/MPSStream.h>
// PyTorch's MPS utilities — gives us getMTLBufferStorage() to extract the underlying
// Metal buffer (GPU memory) from a PyTorch MPS tensor
#include <ATen/native/mps/OperationUtils.h>

// ---- cached Metal objects (created once, reused across calls) ----

// MTLLibrary = a compiled collection of GPU functions (like a .so/.dylib but for the GPU).
// We compile forward.metal once and cache the resulting library here.
static id<MTLLibrary> mtl_library = nil;
// MTLComputePipelineState = a ready-to-dispatch GPU kernel for a specific function.
// Think of it as the GPU equivalent of a compiled + linked function pointer.
// We cache one per dtype since Metal doesn't support templates — we have separate
// kernel functions for float32 and float16.
static id<MTLComputePipelineState> float_pipeline = nil;
static id<MTLComputePipelineState> half_pipeline = nil;

// get_pipeline: compiles the Metal shader (if needed) and returns a cached pipeline
// for the requested dtype. This is the Metal equivalent of cuModuleLoad + cuModuleGetFunction.
static id<MTLComputePipelineState> get_pipeline(id<MTLDevice> device, torch::ScalarType dtype) {
    // Compile the Metal source file into a library (only on first call)
    if (!mtl_library) {
        // @autoreleasepool = Obj-C automatic memory management scope.
        // Any temporary Obj-C objects created inside are freed when the block exits.
        // (Similar to a destructor scope in C++ RAII, but for Obj-C reference counting.)
        @autoreleasepool {
            // __FILE__ is the path to this .mm file at compile time.
            // We strip the filename to get the directory, then append "forward.metal"
            // to locate the shader source relative to this file.
            NSString *dir = [@__FILE__ stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"forward.metal"];

            // Read the .metal source file into a string
            NSError *error = nil;
            NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
            // TORCH_CHECK = assert that throws a C++ exception with a nice error message
            TORCH_CHECK(!error, "failed to read Metal source: ", [[error localizedDescription] UTF8String]);

            // Compile the Metal source code into a MTLLibrary (collection of GPU functions).
            // This is the runtime equivalent of running `xcrun metal` + `xcrun metallib`.
            // The Metal compiler runs here and produces GPU machine code.
            mtl_library = [device newLibraryWithSource:source options:nil error:&error];
            TORCH_CHECK(!error, "failed to compile Metal shader: ", [[error localizedDescription] UTF8String]);
        }
    }

    // Pick the right kernel function name and cache slot based on dtype
    id<MTLComputePipelineState> *cached;  // pointer to the cache slot we'll use
    NSString *name;                       // name of the kernel function in forward.metal

    if (dtype == torch::kFloat) {
        cached = &float_pipeline;
        name = @"swiglu_forward_float";  // matches `kernel void swiglu_forward_float(...)` in .metal
    } else if (dtype == torch::kHalf) {
        cached = &half_pipeline;
        name = @"swiglu_forward_half";  // matches `kernel void swiglu_forward_half(...)` in .metal
    } else {
        TORCH_CHECK(false, "unsupported dtype for MPS swiglu");
    }

    // Create the compute pipeline for this kernel (only on first call per dtype)
    if (!*cached) {
        @autoreleasepool {
            // Look up the kernel function by name from the compiled library
            // (like dlsym() for GPU code)
            id<MTLFunction> fn = [mtl_library newFunctionWithName:name];
            TORCH_CHECK(fn, "Metal function not found: ", [name UTF8String]);

            // Create a pipeline state object from the function. This finalizes the GPU
            // kernel for dispatch — Metal validates the function signature, allocates
            // registers, and determines max threads per threadgroup.
            NSError *error = nil;
            *cached = [device newComputePipelineStateWithFunction:fn error:&error];
            TORCH_CHECK(!error, "failed to create compute pipeline: ", [[error localizedDescription] UTF8String]);
        }
    }

    return *cached;
}

// swiglu_forward_mps: the main entry point called from Python.
// Dispatches the Metal kernel to compute y = u * g * sigmoid(g) element-wise.
void swiglu_forward_mps(const torch::Tensor &g, const torch::Tensor &u, torch::Tensor &y) {
    // Validate that all tensors live on the MPS device (Apple GPU)
    TORCH_CHECK(g.is_mps() && u.is_mps() && y.is_mps(), "all tensors must be on MPS device");
    // Metal kernels read memory linearly by thread ID, so tensors must be contiguous
    TORCH_CHECK(g.is_contiguous() && u.is_contiguous() && y.is_contiguous(), "all tensors must be contiguous");

    // Get a handle to the Metal GPU (the Apple Silicon GPU)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    // Get (or compile + cache) the right pipeline for this tensor's dtype
    auto pipeline = get_pipeline(device, g.scalar_type());

    // Total number of elements to process — we launch one GPU thread per element
    uint32_t num_elements = g.numel();
    // Threads per threadgroup — capped at what the GPU hardware supports for this kernel.
    // A threadgroup is like a CUDA block: threads that execute together and can share memory.
    // maxTotalThreadsPerThreadgroup is hardware-dependent (typically 1024 on Apple Silicon).
    NSUInteger thread_group_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_elements);

    @autoreleasepool {
        // Get PyTorch's MPS stream — this is the command queue PyTorch uses for all MPS
        // operations. Submitting here ensures correct ordering with other PyTorch ops.
        auto stream = at::mps::getCurrentMPSStream();
        // Get a command buffer from the stream — a batch of GPU commands that will be
        // submitted together (like recording commands into a CUDA stream)
        auto command_buffer = stream->commandBuffer();
        // Create a compute command encoder — records GPU compute commands into the
        // command buffer (this is where we configure and launch the kernel)
        auto encoder = [command_buffer computeCommandEncoder];

        // Tell the encoder which compiled kernel to run
        [encoder setComputePipelineState:pipeline];
        // Bind tensor g's GPU memory to buffer index 0 (matches [[buffer(0)]] in .metal).
        // getMTLBufferStorage() extracts the underlying id<MTLBuffer> from a PyTorch MPS tensor.
        // storage_offset * element_size converts the tensor's element offset to a byte offset.
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(g)
                    offset:g.storage_offset() * g.element_size()
                   atIndex:0];
        // Bind tensor u's GPU memory to buffer index 1 (matches [[buffer(1)]] in .metal)
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(u)
                    offset:u.storage_offset() * u.element_size()
                   atIndex:1];
        // Bind output tensor y's GPU memory to buffer index 2 (matches [[buffer(2)]] in .metal)
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(y)
                    offset:y.storage_offset() * y.element_size()
                   atIndex:2];

        // Launch the kernel: num_elements total threads, grouped into threadgroups.
        // Metal automatically divides the 1D grid into threadgroups of thread_group_size.
        // This is analogous to cudaLaunchKernel(grid=num_elements, block=thread_group_size).
        [encoder dispatchThreads:MTLSizeMake(num_elements, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
        // Finalize the encoder — no more commands can be recorded after this
        [encoder endEncoding];

        // Submit the command buffer to the GPU and flush immediately (true = flush).
        // The kernel is now queued for execution on the GPU.
        stream->commit(true);
    }
}

// pybind11 module definition — this is what torch.utils.cpp_extension.load() looks for.
// It creates a Python module and exposes our C++ function as a callable Python function.
// TORCH_EXTENSION_NAME is a macro set by the build system to the module name we specified.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("swiglu_forward_mps", &swiglu_forward_mps, "SwiGLU forward (MPS)"); }
