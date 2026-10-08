# Benchmarks

All numbers use the dev model: a 12-layer GPT-OSS-style mixture of experts with 59,079,408 parameters, seeded random FP32 weights, and a 201,088-token vocabulary. The raw reports are in [`benchmarks/`](benchmarks/), and each one records its environment, weight digest, source digest, and every sample.

| Reports | Hardware | Software |
| --- | --- | --- |
| `benchmark*.json`, `decode-benchmark*.json` | Apple M3, macOS 27.0 | Python 3.14.0, Torch 2.14.0, Apple clang 21 |
| `cuda-*.json` | NVIDIA A100-SXM4-40GB (Colab) | Python 3.14.7, Torch 2.14.0+cu130, CUDA 13.0, GCC 13.3 |

Each benchmark ran twice, and the tables pool both runs. CPU runs use one thread for both Torch and BLAS.

## Forward pass

Each run times 100 eight-token calls after 10 warmups, with all paths interleaved in random order. Torch runs eagerly with the same weights. On a GPU, Torch's weights stay resident and TF32 is off. For both Torch and Chroma, inputs and outputs start and end on the CPU. "Reuse" calls pass `out=`.

| Median of 200 calls | Chroma | Chroma, reuse | Torch eager |
| --- | ---: | ---: | ---: |
| Apple M3 CPU | 5.533 ms | 5.509 ms | 10.447 ms |
| Apple M3 Metal (Torch: MPS) | 4.138 ms | 4.150 ms | 16.243 ms |
| NVIDIA A100 | 4.255 ms | 4.167 ms | 24.865 ms |

Compiling took about 6 s on CPU, 5 s on Metal, and 43 s with nvcc. The planned arena is 75,776 bytes, against 1,989,120 bytes if no temporary were reused.

## Cached decoding

Each round fills a 256-token cache with a 32-token prompt, then runs 32 decode steps. There are 3 timed rounds. Each cached step passes one token through the decode graph, including host-side mask and cache work. It is compared against Chroma's own padded, uncached 256-token graph.

| Median | Cached step | Uncached step | Prefill (32 tokens) |
| --- | ---: | ---: | ---: |
| Apple M3 CPU | 2.336 ms | 53.186 ms | 78.152 ms |
| Apple M3 Metal | 3.194 ms | 37.076 ms | 45.877 ms |
| NVIDIA A100 | 3.267 ms | 7.389 ms | 9.880 ms |

Compiling all three graphs took about 17 s on CPU and on Metal, and 137 s with nvcc.

## Reproducing

```sh
uv run --extra dev python scripts/export.py
uv run --extra dev python scripts/benchmark.py --tokens 8 --runs 100 --warmup 10 --threads 1 --torch-gpu \
  --weights build/graph/dev/weights.safetensors
uv run --extra dev python scripts/benchmark_decode.py --context 256 --prompt 32 --steps 32 --rounds 3 --threads 1 \
  --weights build/graph/dev/weights.safetensors
```

Both scripts run CPU plus Metal on macOS and CPU elsewhere, and `--backends` overrides that. Passing the same `--weights` file everywhere keeps machines comparable. Reports go to `build/benchmarks/`. On a CUDA machine, pass `--backends cuda`; that is how the A100 reports were produced, on Colab.
