# Chroma

Chroma compiles a fixed-shape PyTorch export into a native inference program for CPU, CUDA, or Metal. Because every shape and dtype is known ahead of time, reshapes stay as indexing, chains of arithmetic fuse into a single loop, and temporaries share memory once their last reader has run. The compiled program takes NumPy arrays and runs without Torch.

## Getting started

Chroma uses Python 3.14, [uv](https://docs.astral.sh/uv/), and a C++20 compiler.

```sh
# Clone repo and sync environment
git clone https://github.com/ibbyml/chroma
cd chroma
uv sync --extra dev

# Chat with the dev model
uv run scripts/chat.py --backend cpu
```

The first run exports and compiles a GPT-OSS-shaped dev model with random weights, so the replies are gibberish. It exists to exercise the runtime end to end. `--backend` is `cpu`, `cuda`, or `metal`.

## How it works

1. **Import.** Load a `torch.export` graph, reject anything outside the [supported operators](docs/runtime.md#supported-operators), recognize RMSNorm, and evaluate work that depends only on constants at compile time.
2. **Lower.** Views become layout metadata and elementwise ops become deferred expressions, so chains fuse. An expression is materialized only when several ops read it or it grows too deep.
3. **Generate.** The CPU backend emits C++20 expression templates and BLAS calls. CUDA and Metal share one GPU generator that writes each kernel body once with per-target types and intrinsics.
4. **Plan memory.** Each temporary gets a lifetime from its producer to its last reader, and the lifetimes are packed into one aligned arena.
5. **Build.** The source and weights are compiled into a Python extension with a manifest. Loading needs NumPy, not Torch.

## Backends

| Target | Execution | Requires |
| --- | --- | --- |
| CPU, Linux | C++20 and CBLAS | A computer |
| CPU, macOS | C++20 and Accelerate | macOS 15+ |
| CUDA | Generated kernels and cuBLAS | Linux, CUDA Toolkit 12+, compute capability 7.5+ |
| Metal | Generated shaders and an Objective-C++ runtime | macOS 15+ |

Setup, dispatch details, and validation for each target are in [docs/backends.md](docs/backends.md).

## API

A model can be exported via:

```python
chroma.compile(export, directory, backend=...) 
```

 This emits an `ExportedProgram` or a `.pt2` path into `directory` and returns a `Program`. 

 `Program(directory)` loads it again later, without Torch. To compile a one-layer example:

```sh
uv run scripts/export.py --example
uv run scripts/run.py build/graph/example/example.pt2 --output build/models/example
```

And now this model can be invoked programmatically:

```python
import numpy as np
from chroma import Program

with Program("build/models/example") as model:
    x = np.arange(8, dtype=np.float32)
    output = model(x)
    model(x, out=output)
```

The first call allocates outputs; `out=` writes into contiguous arrays you own. Inputs must match the exported shapes and dtypes. Multiple outputs come back as a flat tuple.

`chroma.jit` wraps a function or a Torch module and compiles it once per input shape and dtype:

```python
import torch
from chroma import jit

@jit(backend="cpu")
def activation(x: torch.Tensor) -> torch.Tensor:
    return x.sin() + x


res = activation(torch.ones(8))
```

`chroma.Decoder` runs KV-cached generation from a pair of prefill and decode programs. See [docs/runtime.md](docs/runtime.md#kv-cached-decoding).

## Benchmarks

The dev model is a 12-layer GPT-OSS-style mixture-of-experts network with 59,079,408 parameters and seeded random weights. It exercises expert routing, attention, and a 201,088-token vocabulary projection; it says nothing about language quality.

Eight-token FP32 forward passes with identical weights, pooled from two runs of 100 calls after 10 warmups:

| Target | Chroma | Torch eager |
| --- | ---: | ---: |
| Apple M3 CPU, macOS 27.0 | 5.533 ms | 10.447 ms |
| Apple M3 Metal / MPS | 4.138 ms | 16.243 ms |
| NVIDIA A100 | 4.255 ms | 24.865 ms |

Calls return fresh NumPy arrays, and GPU times include host transfers and synchronization. On this graph, 1.90 MiB of distinct temporaries fit in a 74 KiB arena.

With a 256-token KV cache, a decode step takes 2.336 ms on CPU, 3.194 ms on Metal, and 3.267 ms on the A100, against 53.186, 37.076, and 7.389 ms for Chroma's own padded, uncached graph. Method, compile times, and the raw reports are in [docs/benchmarks.md](docs/benchmarks.md).

## Development

```sh
uv run ruff check . && uv run ruff format --check . && uv run ty check .
uv run --extra dev pytest  # a few minutes; CUDA hardware tests need CHROMA_TEST_CUDA=1

# Native runtime tests, plus build/cpp/compile_commands.json for clangd
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

On Linux without CBLAS, configure with `-DCHROMA_BLAS=OFF`. Everything generated goes under `build/`, which is safe to delete.

## Documentation

- [Backends](docs/backends.md): setup, execution, and validation for CPU, CUDA, and Metal
- [Runtime](docs/runtime.md): API, supported operators, numerics, memory, and cached decoding
- [Benchmarks](docs/benchmarks.md): method, results, and how to reproduce them

## Citation

If you find Chroma useful in your research, please cite:

```bibtex
@misc{Chroma,
  author = {Ibrahim Khan},
  title = {Chroma: An AOT fixed-shape compiler and runtime for PyTorch graph exports},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ibbyml/chroma}
}
```

## License

[MIT](LICENSE)