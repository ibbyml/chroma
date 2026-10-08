#pragma once

#include "core/tensor.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace chroma::cuda {
static_assert(sizeof(__nv_bfloat16) == 2);

__device__ inline size_t index(int64_t value, size_t count, bool negative, unsigned int* error) {
    if (value >= 0) {
        const size_t res = static_cast<size_t>(value);
        if (res < count) {
            return res;
        }
    } else if (negative) {
        const size_t distance = static_cast<size_t>(-(value + 1)) + 1;
        if (distance <= count) {
            return count - distance;
        }
    }
    atomicExch(error, 1u);
    return 0;
}

__device__ inline float warp_sum(float value) {
    for (int offset = 16; offset; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ inline float warp_max(float value) {
    for (int offset = 16; offset; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ inline float warp_min(float value) {
    for (int offset = 16; offset; offset /= 2) {
        value = fminf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ inline unsigned int warp_min(unsigned int value) {
    for (int offset = 16; offset; offset /= 2) {
        unsigned int other = __shfl_xor_sync(0xffffffffu, value, offset);
        value = value < other ? value : other;
    }
    return value;
}

inline void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
inline void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class DeviceGuard {
public:
    explicit DeviceGuard(int device) {
        check(cudaGetDevice(&previous_), "cudaGetDevice");
        check(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceGuard() { cudaSetDevice(previous_); }
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_ = 0;
};

class Runner {
public:
    Runner() = default;
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    ~Runner() {
        if (device_ < 0) {
            return;
        }
        int previous = 0;
        cudaGetDevice(&previous);
        cudaSetDevice(device_);
        if (stream_) {
            cudaStreamSynchronize(stream_);
        }
        if (blas_) {
            cublasDestroy(blas_);
        }
        for (void* buffer : buffers_) {
            cudaFree(buffer);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        cudaSetDevice(previous);
    }

    void initialize(const char* filename, size_t weight_bytes, size_t workspace_bytes, std::vector<size_t> input_bytes,
                    std::vector<size_t> output_bytes, bool blas) {
        check(cudaGetDevice(&device_), "cudaGetDevice");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device_), "cudaGetDeviceProperties");
        if (properties.major * 10 + properties.minor < 75) {
            throw std::runtime_error("compute capability < 7.5");
        }
        name_ = properties.name;
        input_bytes_ = std::move(input_bytes);
        output_bytes_ = std::move(output_bytes);
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        if (blas) {
            check(cublasCreate(&blas_), "cublasCreate");
            check(cublasSetStream(blas_, stream_), "cublasSetStream");
            check(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
        }
        allocate(weight_bytes);
        allocate(workspace_bytes);
        allocate(sizeof(unsigned int));
        for (size_t bytes : input_bytes_) {
            allocate(bytes);
        }
        for (size_t bytes : output_bytes_) {
            allocate(bytes);
        }
        std::vector<char> data(weight_bytes);
        read_weights(filename, data.data(), weight_bytes);
        check(cudaMemcpy(weights(), data.data(), weight_bytes, cudaMemcpyHostToDevice), "Upload weights");
    }

    template <class Execute> void run(const void* const* inputs, void* const* outputs, Execute execute) {
        DeviceGuard guard(device_);
        unsigned int index_error = 0;
        try {
            check(cudaMemsetAsync(error(), 0, sizeof(unsigned int), stream_), "Reset index error");
            for (size_t i = 0; i < input_bytes_.size(); ++i) {
                check(cudaMemcpyAsync(input(i), inputs[i], input_bytes_[i], cudaMemcpyHostToDevice, stream_),
                      "Upload input");
            }
            execute();
            check(cudaMemcpyAsync(&index_error, error(), sizeof(index_error), cudaMemcpyDeviceToHost, stream_),
                  "Read index error");
            check(cudaStreamSynchronize(stream_), "CUDA execution");
            if (index_error) {
                throw std::out_of_range("Tensor index out of range");
            }
            for (size_t i = 0; i < output_bytes_.size(); ++i) {
                check(cudaMemcpyAsync(outputs[i], output(i), output_bytes_[i], cudaMemcpyDeviceToHost, stream_),
                      "Download output");
            }
            check(cudaStreamSynchronize(stream_), "Download output");
        } catch (...) {
            // Host arrays must stay alive until all queued transfers have stopped.
            cudaStreamSynchronize(stream_);
            throw;
        }
    }

    void linear(float* out, const float* x, const float* weight, int rows, int columns, int inner) {
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major Y = X W^T becomes column-major Y^T = W X^T.
        check(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, inner, &alpha, weight, inner, x, inner, &beta,
                          out, columns),
              "cublasSgemm");
    }
    const char* device_name() const { return name_.c_str(); }
    cudaStream_t stream() const { return stream_; }
    std::byte* weights() { return buffers_[0]; }
    std::byte* workspace() { return buffers_[1]; }
    unsigned int* error() { return reinterpret_cast<unsigned int*>(buffers_[2]); }
    std::byte* input(size_t index) { return buffers_[3 + index]; }
    std::byte* output(size_t index) { return buffers_[3 + input_bytes_.size() + index]; }

private:
    void allocate(size_t bytes) {
        buffers_.push_back(nullptr);
        check(cudaMalloc(reinterpret_cast<void**>(&buffers_.back()), std::max(bytes, size_t{1})), "cudaMalloc");
    }
    int device_ = -1;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    std::string name_;
    std::vector<std::byte*> buffers_;
    std::vector<size_t> input_bytes_, output_bytes_;
};
}
