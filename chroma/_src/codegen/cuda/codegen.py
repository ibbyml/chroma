import math
from collections.abc import Callable, Sequence
from typing import ClassVar

import torch

from chroma._src.codegen.base import GeneratedCode, nbytes
from chroma._src.codegen.gpu import GPUCodegen
from chroma._src.program import Stats, TensorSpec
from chroma._src.value import Shape, Value, View, strides


def generation_template(
    stats: Stats,
    body: str,
    declarations: str,
    input_bytes: Sequence[int],
    output_bytes: Sequence[int],
    blas: bool,
) -> str:
    return f"""#include "codegen/cuda/runtime.h"
#include "core/bindings.h"
#include <cmath>
#include <cstdint>
{declarations}
struct ModelProgram {{
    chroma::cuda::Runner runtime;
    const char* device_name() const {{ return runtime.device_name(); }}
    void run(const void* const* inputs, void* const* outputs) {{
        runtime.run(inputs, outputs, [&] {{
{body}
        }});
    }}
}};
auto create_model(const char* filename) {{
    auto model = std::make_unique<ModelProgram>();
    model->runtime.initialize(filename, {stats["weight_bytes"]}, {stats["workspace_bytes"]},
        {{{", ".join(map(str, input_bytes))}}}, {{{", ".join(map(str, output_bytes))}}}, {str(blas).lower()});
    return model;
}}
"""


class CUDACodegen(GPUCodegen):
    backend = "CUDA"
    ctypes: ClassVar = {torch.float32: "float", torch.bfloat16: "__nv_bfloat16", torch.int64: "int64_t"}
    functions: ClassVar = {"exp": "expf", "sqrt": "sqrtf", "sin": "sinf", "cos": "cosf", "abs": "fabsf", "pow": "powf", "fmax": "fmaxf"}
    lanes = "chroma::cuda::warp_"
    any_lane = "__any_sync(0xffffffffu, {})"
    all_lanes = "__all_sync(0xffffffffu, {})"
    shuffle = "__shfl_sync(0xffffffffu, {}, {})"
    index_function = "chroma::cuda::index"
    helper_prefix = "__device__ inline "
    pointer_prefix = ""
    error_parameter = "unsigned int* error"

    def __init__(self, *, blas: bool = True) -> None:
        super().__init__()
        self.blas = blas
        self.lines: list[str | Value] = []

    def encode(self, dtype: torch.dtype, expression: str) -> str:
        return f"__float2bfloat16_rn({expression})" if dtype == torch.bfloat16 else expression

    def decode(self, expression: str) -> str:
        return f"__bfloat162float({expression})"

    def extern(self, tensor: torch.Tensor, kind: str, offset: int) -> Value:
        dtype = self.ctype(tensor.dtype)
        name = f"v{next(self.counter)}"
        shape = tuple(tensor.shape)
        pointer = f"runtime.input({offset})" if kind == "input" else f"runtime.weights() + {offset}"
        self.lines.append(f"auto {name} = reinterpret_cast<const {dtype}*>({pointer});")
        self.sources[name] = {name: tensor.dtype}
        return Value(name, shape, tensor.dtype, stored=True, view=View(None, strides(shape)))

    def allocate(self, shape: Shape, dtype: torch.dtype) -> Value:
        value = super().allocate(shape, dtype)
        self.lines.append(value)
        return value

    def record(self, code: str, inputs: Sequence[Value]) -> None:
        self.record_usage(inputs)
        self.lines.append(code)
        self.kernels += 1

    def launch(
        self, outputs: Sequence[Value], inputs: Sequence[Value], count: int, body: Callable[[], str], *, parallel: bool = False
    ) -> None:
        threads = 32 if parallel else 256
        groups = count if parallel else (count + threads - 1) // threads
        if groups > 2**31 - 1:
            raise ValueError("CUDA dispatch exceeds the x grid limit")
        sources = {name: dtype for value in inputs for name, dtype in self.sources[value.name].items()}
        output_names = {out.name for out in outputs}
        sources.update({out.name: out.dtype for out in outputs})
        parameters = [f"{'' if name in output_names else 'const '}{self.ctype(dtype)}* {name}" for name, dtype in sources.items()]
        parameters.append("unsigned int* error")
        name = f"chroma_kernel_{len(self.shaders)}"
        index = "blockIdx.x" if parallel else "static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x"
        prefix = "const unsigned int tid = threadIdx.x;\n" if parallel else ""
        self.shaders.append(
            f"__global__ void {name}({', '.join(parameters)}) {{\n"
            f"const size_t gid = {index};\n"
            f"if (gid >= {count}ull) return;\n{prefix}{body()}\n}}"
        )
        self.record(
            f"{name}<<<{groups}, {threads}, 0, runtime.stream()>>>({', '.join([*sources, 'runtime.error()'])});\n"
            'chroma::cuda::check(cudaGetLastError(), "CUDA kernel launch");',
            inputs,
        )

    def alias(self, meta: torch.Tensor, root: Value, offset: int) -> Value:
        name = f"v{next(self.counter)}"
        shape = tuple(meta.shape)
        self.lines.append(f"auto {name} = {root.name} + {offset};")
        self.sources[name] = {name: meta.dtype}
        return Value(name, shape, meta.dtype, root.deps, stored=True, view=View(root, strides(shape), offset))

    def output(self, index: int, value: Value) -> None:
        if self.promote_output(index, value):
            return
        out = Value(f"output{index}", value.shape, value.dtype, stored=True)
        self.lines.append(f"auto {out.name} = reinterpret_cast<{self.ctype(out.dtype)}*>(runtime.output({index}));")
        self.copy(out, value)

    def linear(self, meta: torch.Tensor, x: Value, weight: Value, bias: Value | None) -> Value:
        rows, columns, inner = math.prod(x.shape[:-1]), weight.shape[0], weight.shape[1]
        if max(rows, columns, inner) > 2**31 - 1:
            raise ValueError("Matrix dimensions exceed the BLAS integer range")
        if not (self.blas and x.dtype == weight.dtype == meta.dtype == torch.float32):
            out = self.allocate(tuple(meta.shape), meta.dtype)
            self.scalar_linear(out, x, weight, bias)
            return out
        x, weight = self.materialize(x), self.materialize(weight)
        out = self.allocate(tuple(meta.shape), meta.dtype)
        self.record(f"runtime.linear({out.name}, {x.name}, {weight.name}, {rows}, {columns}, {inner});", [x, weight])
        if bias is not None:
            self.launch([out], [out, bias], rows * columns, lambda: f"{out.name}[gid] += {self.access(bias, f'gid % {columns}ull')};")
        return out

    def render_line(self, line: str | Value) -> str:
        if isinstance(line, str):
            return line
        if line.name in self.output_buffers:
            pointer = f"runtime.output({self.output_buffers[line.name]})"
        else:
            pointer = f"runtime.workspace() + {self.allocations[line.name].offset}"
        return f"auto {line.name} = reinterpret_cast<{self.ctype(line.dtype)}*>({pointer});"

    def generate(self, inputs: list[TensorSpec], outputs: list[TensorSpec], stats: Stats) -> GeneratedCode:
        input_bytes, output_bytes = list(map(nbytes, inputs)), list(map(nbytes, outputs))
        stats = stats | {
            "backend": "cuda",
            "input_bytes": sum(input_bytes),
            "cuda_buffer_bytes": stats["weight_bytes"] + stats["workspace_bytes"] + sum(input_bytes) + sum(output_bytes) + 4,
            "parallel_reductions": self.parallel_reductions,
        }
        body = "\n".join(self.render_line(line) for line in self.lines)
        declarations = "\n".join(self.helpers) + "\n" + "\n".join(self.shaders)
        source = generation_template(stats, body, declarations, input_bytes, output_bytes, self.blas)
        flags = [
            "--generate-code=arch=compute_75,code=sm_75",
            "--generate-code=arch=compute_80,code=[sm_80,compute_80]",
            "--fmad=false",
            "-lcublas",
        ]
        return GeneratedCode({"model.cu": source}, "model.cu", flags, stats, inputs, outputs, bytes(self.weights))
