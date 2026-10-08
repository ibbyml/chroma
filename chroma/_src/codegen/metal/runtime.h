#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "core/tensor.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace chroma::metal {
struct Binding {
    size_t index, buffer, offset;
};
struct Kernel {
    const char* name;
    std::vector<Binding> bindings;
    MTLSize groups, threads;
    std::vector<uint8_t> params;
    id<MTLComputePipelineState> pipeline = nil;
};

inline std::string description(NSError* error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "Unknown Metal error";
}

class Runner {
public:
    Runner(const char* weight_file, size_t weight_bytes, size_t arena_bytes, std::vector<size_t> input_bytes,
           std::vector<size_t> output_bytes, const char* source, std::vector<Kernel> kernels)
        : input_bytes_(std::move(input_bytes)), output_bytes_(std::move(output_bytes)), kernels_(std::move(kernels)) {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            throw std::runtime_error("No Metal device is available");
        }
        queue_ = [device_ newCommandQueue];
        if (!queue_) {
            throw std::runtime_error("Cannot create Metal command queue");
        }
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_1;
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        library_ = [device_ newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
        if (!library_) {
            throw std::runtime_error("Metal shader compilation failed: " + description(error));
        }
        std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
        for (auto& kernel : kernels_) {
            auto& pipeline = pipelines[kernel.name];
            if (!pipeline) {
                id<MTLFunction> function = [library_ newFunctionWithName:[NSString stringWithUTF8String:kernel.name]];
                if (!function) {
                    throw std::runtime_error(std::string("Missing Metal kernel: ") + kernel.name);
                }
                pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline) {
                    throw std::runtime_error(description(error));
                }
            }
            kernel.pipeline = pipeline;
            const size_t threads = kernel.threads.width * kernel.threads.height * kernel.threads.depth;
            if (threads > kernel.pipeline.maxTotalThreadsPerThreadgroup) {
                throw std::runtime_error(std::string("Unsupported threadgroup size for ") + kernel.name);
            }
        }
        allocate(weight_bytes);
        allocate(arena_bytes);
        allocate(sizeof(uint32_t));
        for (size_t size : input_bytes_) {
            allocate(size);
        }
        for (size_t size : output_bytes_) {
            allocate(size);
        }
        read_weights(weight_file, buffers_[0].contents, weight_bytes);
    }

    void run(const void* const* inputs, void* const* outputs) {
        for (size_t i = 0; i < input_bytes_.size(); ++i) {
            std::memcpy(buffers_[3 + i].contents, inputs[i], input_bytes_[i]);
        }
        *static_cast<uint32_t*>(buffers_[2].contents) = 0;
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        if (!command) {
            throw std::runtime_error("Cannot create Metal command buffer");
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            throw std::runtime_error("Cannot create Metal compute encoder");
        }
        for (const auto& kernel : kernels_) {
            [encoder setComputePipelineState:kernel.pipeline];
            if (!kernel.params.empty()) {
                [encoder setBytes:kernel.params.data() length:kernel.params.size() atIndex:0];
            }
            for (const auto& binding : kernel.bindings) {
                [encoder setBuffer:buffers_[binding.buffer] offset:binding.offset atIndex:binding.index];
            }
            [encoder dispatchThreadgroups:kernel.groups threadsPerThreadgroup:kernel.threads];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) {
            throw std::runtime_error(description(command.error));
        }
        if (*static_cast<uint32_t*>(buffers_[2].contents)) {
            throw std::out_of_range("Tensor index out of range");
        }
        for (size_t i = 0; i < output_bytes_.size(); ++i) {
            std::memcpy(outputs[i], buffers_[3 + input_bytes_.size() + i].contents, output_bytes_[i]);
        }
    }
    const char* device_name() const { return device_.name.UTF8String; }

private:
    void allocate(size_t bytes) {
        if (bytes > device_.maxBufferLength) {
            throw std::runtime_error("Tensor exceeds Metal's maximum buffer length");
        }
        id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max(bytes, size_t{4})
                                                    options:MTLResourceStorageModeShared];
        if (!buffer) {
            throw std::bad_alloc();
        }
        buffers_.push_back(buffer);
    }
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_;
    std::vector<id<MTLBuffer>> buffers_;
    std::vector<size_t> input_bytes_, output_bytes_;
    std::vector<Kernel> kernels_;
};
}
