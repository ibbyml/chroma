# Runtime

## API

| Call | Purpose |
| --- | --- |
| `compile(export, directory, *, backend="cpu", blas=True)` | Compile an `ExportedProgram` or `.pt2` path into `directory` and return a loaded `Program` |
| `Program(directory)` | Load a compiled program |
| `jit(fn, *, backend="cpu", blas=True)` | Compile a function or module once per input shape and dtype |
| `Decoder(prefill, decode, *, sliding_window=0)` | Run KV-cached generation from a prefill and decode pair |

Compiling needs Torch and pybind11. Loading and running need only NumPy and `ml-dtypes`. `blas=False` swaps BLAS for portable scalar matrix kernels on CPU and CUDA.

A `Program` takes NumPy arrays or CPU Torch tensors and returns NumPy arrays:

- Inputs must match the exported shapes and dtypes. Non-contiguous inputs are copied.
- `out=` takes writable, contiguous, non-overlapping arrays. An error can leave them partly written, but the program stays usable.
- Multiple outputs come back as a flat tuple, and a value returned twice comes back as two arrays.
- Calls on one instance are serialized and release the GIL. Separate instances have separate arenas.
- `model.stats` reports kernels, fused expressions, folded constants, and arena, output, and weight bytes. It excludes BLAS, driver, and allocator memory.

## Supported operators

Computation is FP32 or BF16 and indices are int64. Every dimension must be static and nonzero, though rank-0 tensors are fine.

| Family | Forms |
| --- | --- |
| Arithmetic | add, subtract, reverse subtract, multiply, divide, scalar powers, reciprocal, rsqrt, sigmoid, sin, cos, exp, neg, abs, square, scalar clamp |
| Layout | view, reshape, unsqueeze, expand, positive-step slice, split, permute, transpose, clone, contiguous, same-dtype conversion |
| Selection | embedding; first-axis indexing by one int64 tensor; top-k along an axis of at most 4096 |
| Aggregation | mean and sum over fixed axes, softmax, concatenation, triu and tril |
| Linear algebra | linear; two-operand einsum with an explicit output and no ellipses |
| Constants | arange, full, new_full, and expressions over constants |
| Casts | FP32 to and from BF16 |

Anything else fails at compile time: other operators or dtypes, dynamic shapes, non-tensor inputs or outputs, and mutation of inputs or buffers. In-place ops on intermediates are functionalized. A bad runtime index raises `IndexError`.

## Numerics

- Generated code disables fast math and floating-point contraction. BLAS and reduction order can still differ from Torch in the last bits.
- BF16 uses round-to-nearest-even. Matmuls, reductions, and softmax accumulate in FP32, and fused arithmetic still rounds at every exported BF16 op. Compare BF16 results with dtype-sized tolerances.
- Tied top-k values have no guaranteed order. With `sorted=False`, CPU may return the selection unsorted, while the GPU backends always sort.

## Memory and artifacts

- Each `Program` allocates its weights and one activation arena. Native code allocates nothing during a call.
- Each temporary lives from its producing kernel to its last reader. The planner packs these lifetimes best fit into one aligned arena, and a temporary that is also an output writes straight into the output buffer.
- Contiguous views alias their base. Constants are deduplicated in a 64-byte-aligned weight file.
- A compiled directory holds `model.json`, the extension `_chroma_<id>.<abi>.so`, `weights-<hash>.bin`, and the generated source. Compiling into it again replaces the old library and weights.
- Loading checks the manifest version, the Python ABI, and that the manifest's tensors match the library. Recompile after changing weights, Python, or platform.

## KV-cached decoding

`chroma.models.cached.export_cached(model, context, steps)` exports one step of a GPT-OSS model with the cache as explicit inputs. Use `steps=context` for prefill and `steps=1` for decode. Each graph takes token IDs, positions, K and V caches shaped `[layers, context, kv_heads, head_dim]`, two attention masks, and the position whose logits to return. It returns those logits and the new K/V rows.

`Decoder` owns both programs and a rolling cache in host memory:

- `append(tokens)` takes only new tokens and returns a `[1, vocab]` row of logits. A single token runs the decode graph, and anything longer runs prefill in chunks.
- When the cache is full, the oldest rows are overwritten and positions keep advancing. That gives bounded causal attention rather than a fresh pass over a cropped prompt.
- K/V rows commit only after the native call returns, so an interrupted `append` leaves the cache at the last committed `position`.
- `sliding_window` must match the export, and 0 means the whole cache. `reset()` clears the cache and `close()` releases both programs.

## Source layout

| Path | Responsibility |
| --- | --- |
| `chroma/_src/graph.py` | Load and validate exports, recognize RMSNorm |
| `chroma/_src/compiler.py` | Bind inputs and weights, fold constants, emit nodes in order |
| `chroma/_src/lowering.py`, `value.py` | Normalize ATen ops; views, expressions, and contraction geometry |
| `chroma/_src/memory.py` | Lifetime-based arena planning |
| `chroma/_src/codegen/` | Weight packing and storage (`base.py`), CPU templates, GPU kernel bodies (`gpu.py`), CUDA and Metal runtimes |
| `chroma/_src/build.py` | Compile, load, and publish an artifact |
| `chroma/_src/program.py`, `decoder.py`, `jit.py` | Public runtime |
| `chroma/_src/core/` | Native tensor types, BF16, and Python bindings |
| `chroma/models/` | GPT-OSS reference model and cached-decoding exports |
