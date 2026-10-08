# Benchmarks

All numbers use the dev model: a 12-layer GPT-OSS-style mixture of experts with 59,079,408 parameters, seeded random FP32 weights, and a 201,088-token vocabulary. The raw reports are in [`benchmarks/`](benchmarks/), and each one records its environment, weight digest, source digest, and every sample.

<!-- benchmarks:environments -->
| Reports | Hardware | Software |
| --- | --- | --- |
| `benchmark*.json`, `decode-benchmark*.json` | Apple M3, macOS 27.0.1 | Python 3.14.0, Torch 2.14.0, Apple clang 21 |
| `cuda-*.json` | NVIDIA A100-SXM4-40GB | Python 3.14.7, Torch 2.14.0+cu130, CUDA 13.0, GCC 13.3 |
<!-- /benchmarks:environments -->

Each benchmark ran twice, and the tables pool both runs. CPU runs use one thread for both Torch and BLAS.

## Forward pass

Each run times 100 eight-token calls after 10 warmups, with all paths interleaved in random order. Torch runs eagerly with the same weights. On a GPU, Torch's weights stay resident and TF32 is off. For both Torch and Chroma, inputs and outputs start and end on the CPU. "Reuse" calls pass `out=`.

<!-- benchmarks:forward -->
| Median of 200 calls | Chroma | Chroma, reuse | Torch eager |
| --- | ---: | ---: | ---: |
| Apple M3 CPU | 5.31 ms | 5.28 ms | 10.63 ms |
| Apple M3 Metal (Torch: MPS) | 4.15 ms | 4.14 ms | 16.78 ms |
| NVIDIA A100 | 4.25 ms | 4.17 ms | 24.86 ms |

Compiling took about 7 s on CPU, 5 s on Metal, and 43 s with nvcc. The planned arena is 75,776 bytes, against 1,989,120 bytes if no temporary were reused.
<!-- /benchmarks:forward -->

## Cached decoding

Each round fills a 256-token cache with a 32-token prompt, then runs 32 decode steps. There are 3 timed rounds. Each cached step passes one token through the decode graph, including host-side mask and cache work. It is compared against Chroma's own padded, uncached 256-token graph.

<!-- benchmarks:decode -->
| Median | Cached step | Uncached step | Prefill (32 tokens) |
| --- | ---: | ---: | ---: |
| Apple M3 CPU | 2.18 ms | 37.58 ms | 54.08 ms |
| Apple M3 Metal | 3.21 ms | 37.17 ms | 44.96 ms |
| NVIDIA A100 | 3.27 ms | 7.39 ms | 9.88 ms |

Compiling all three graphs took about 17 s on CPU, 16 s on Metal, and 137 s with nvcc.
<!-- /benchmarks:decode -->

## Reproducing

```sh
uv run --extra dev python scripts/export.py
uv run --extra dev python scripts/benchmark.py --tokens 8 --runs 100 --warmup 10 --threads 1 --torch-gpu \
  --weights build/graph/dev/weights.safetensors
uv run --extra dev python scripts/benchmark_decode.py --context 256 --prompt 32 --steps 32 --rounds 3 --threads 1 \
  --weights build/graph/dev/weights.safetensors
```

Both scripts run CPU plus Metal on macOS and CPU elsewhere, and `--backends` overrides that. Passing the same `--weights` file everywhere keeps machines comparable. Reports go to `build/benchmarks/`. On a CUDA machine, pass `--backends cuda`; that is how the A100 reports were produced, on Colab.

Copy new reports into `docs/benchmarks/`, then run `uv run scripts/summarize_benchmarks.py`. It pools each pair of runs into `summary.json` and refreshes the numbers above; the README, figures, and website read the same file.
