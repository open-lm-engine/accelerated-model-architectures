// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

#import <Metal/Metal.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

static id<MTLLibrary> mtl_library = nil;
static id<MTLComputePipelineState> float_pipeline = nil;
static id<MTLComputePipelineState> half_pipeline = nil;

static id<MTLComputePipelineState> get_pipeline(id<MTLDevice> device, torch::ScalarType dtype) {
    if (!mtl_library) {
        @autoreleasepool {
            NSString *dir = [@__FILE__ stringByDeletingLastPathComponent];
            NSString *path = [dir stringByAppendingPathComponent:@"forward.metal"];

            NSError *error = nil;
            NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
            TORCH_CHECK(!error, "failed to read Metal source: ", [[error localizedDescription] UTF8String]);

            mtl_library = [device newLibraryWithSource:source options:nil error:&error];
            TORCH_CHECK(!error, "failed to compile Metal shader: ", [[error localizedDescription] UTF8String]);
        }
    }

    id<MTLComputePipelineState> *cached;
    NSString *name;

    if (dtype == torch::kFloat) {
        cached = &float_pipeline;
        name = @"swiglu_forward_float";
    } else if (dtype == torch::kHalf) {
        cached = &half_pipeline;
        name = @"swiglu_forward_half";
    } else {
        TORCH_CHECK(false, "unsupported dtype for MPS swiglu");
    }

    if (!*cached) {
        @autoreleasepool {
            id<MTLFunction> fn = [mtl_library newFunctionWithName:name];
            TORCH_CHECK(fn, "Metal function not found: ", [name UTF8String]);

            NSError *error = nil;
            *cached = [device newComputePipelineStateWithFunction:fn error:&error];
            TORCH_CHECK(!error, "failed to create compute pipeline: ", [[error localizedDescription] UTF8String]);
        }
    }

    return *cached;
}

void swiglu_forward_mps(const torch::Tensor &g, const torch::Tensor &u, torch::Tensor &y) {
    TORCH_CHECK(g.is_mps() && u.is_mps() && y.is_mps(), "all tensors must be on MPS device");
    TORCH_CHECK(g.is_contiguous() && u.is_contiguous() && y.is_contiguous(), "all tensors must be contiguous");

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    auto pipeline = get_pipeline(device, g.scalar_type());

    uint32_t num_elements = g.numel();
    NSUInteger thread_group_size = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)num_elements);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto command_buffer = stream->commandBuffer();
        auto encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(g)
                    offset:g.storage_offset() * g.element_size()
                   atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(u)
                    offset:u.storage_offset() * u.element_size()
                   atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(y)
                    offset:y.storage_offset() * y.element_size()
                   atIndex:2];

        [encoder dispatchThreads:MTLSizeMake(num_elements, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(thread_group_size, 1, 1)];
        [encoder endEncoding];

        stream->commit(true);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("swiglu_forward_mps", &swiglu_forward_mps, "SwiGLU forward (MPS)"); }
