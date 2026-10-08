# Backends

All three backends compile the same [operators](runtime.md#supported-operators) in FP32 or BF16 with int64 indices. Import, fusion, and memory planning are shared, and CUDA and Metal also share one GPU kernel generator.

| | CPU | CUDA | Metal |
| --- | --- | --- | --- |
| Host | Linux, or macOS 15+ | Linux, including WSL2 and Colab | macOS 15+ |
| Matrix products | CBLAS or Accelerate | cuBLAS (FP32), scalar kernel (BF16) | Tiled and BF16-weight kernels |
| Wide reductions | Loops | One warp per row | One SIMD group per row |
| Tested on | Apple M3, Colab x86 | NVIDIA A100 | Apple M3 |

## CPU

- `CXX` picks the compiler (default `c++`). Linux needs CBLAS headers and libraries, such as `libopenblas-dev`, and macOS uses Accelerate.
- Fused expressions become C++20 expression templates with shapes and strides as template parameters, and each one evaluates in a single loop.
- Linear layers and compatible contractions call BLAS with static strides, and selected-expert contractions read expert weights in place. Other einsums are loops specialized to the shape.

## CUDA

You need a driver, CUDA Toolkit 12+ (`nvcc`, runtime, cuBLAS, headers), and a host compiler the toolkit supports. `NVCC` selects the compiler and can carry arguments, for example `NVCC="nvcc -ccbin /usr/bin/g++"`. Generating source needs no GPU.

- Builds `sm_75` and `sm_80` cubins plus `compute_80` PTX, so it loads on Turing and newer. Only the A100 has been tested.
- Loading needs the CUDA runtime and cuBLAS, not Torch or `nvcc`. If they aren't found, add the toolkit's `lib64` to `LD_LIBRARY_PATH`. Choose a GPU with `CUDA_VISIBLE_DEVICES`.
- FP32 linear layers call `cublasSgemm` in pedantic mode, so TF32 is never used. BF16 layers and `blas=False` use a scalar kernel that accumulates in FP32. Tensor cores are not used.
- Softmax, RMSNorm, and sums use one warp per row once a row has 32 elements, and top-k always uses one warp per row. FMA contraction and fast math are off.
- Each `Program` owns its device buffers, a stream, and a cuBLAS handle. Weights upload once. A call copies inputs to the device, runs, copies outputs back, and synchronizes.

```sh
CHROMA_TEST_CUDA=1 uv run --extra dev pytest tests/test_cuda.py -v
```

Without `CHROMA_TEST_CUDA=1`, the hardware tests skip.

## Metal

Shaders compile at load time in safe math mode, with precise functions and contraction disabled. Every operation runs on the GPU, with no CPU fallback.

- **BF16-weight matmul and RMSNorm**, adapted from gpt-oss. These run when a constant weight is exactly representable in BF16, and weights are never rounded to qualify. Matmul needs an inner dimension of at least 128, and RMSNorm a row of at least 4096, both divisible by 4.
- **Tiled FP32 linear and einsum.** 8×32 output tiles are staged through threadgroup memory, for all but the smallest shapes.
- **Reductions and top-k.** Reductions use one 32-lane SIMD group per row once a row has 32 elements, and top-k always uses one per row.

Weights, the arena, staging buffers, and an index-error flag are shared Metal buffers allocated once. Each call encodes one command buffer and waits for it. For small models the CPU can be faster, because dispatch and copies dominate.

```sh
uv run --extra dev pytest tests/test_metal.py -v
```
