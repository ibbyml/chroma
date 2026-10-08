import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import torch

from chroma._src.codegen.base import GeneratedCode, nbytes
from chroma._src.codegen.gpu import GPUCodegen, coordinate, flat_index
from chroma._src.program import Stats, TensorSpec
from chroma._src.value import DTYPES, Shape, Value, View, matrix_axes, strides

_KERNELS = Path(__file__).resolve().parent / "kernels.metal"

type Grid = tuple[int, int, int]


@dataclass(frozen=True)
class Binding:
    kind: str  # weight, arena, input, output, or error
    index: int | str = 0  # Input or output position, or the arena allocation name.
    offset: int = 0


@dataclass(frozen=True)
class Launch:
    name: str
    bindings: list[tuple[int, Binding]]
    groups: Grid
    threads: Grid
    params: bytes


def generation_template(
    shader: str,
    stats: Stats,
    input_bytes: Sequence[int],
    output_bytes: Sequence[int],
    kernels: Sequence[str],
) -> str:
    return f"""#include "codegen/metal/runtime.h"
#include "core/bindings.h"
using ModelProgram = chroma::metal::Runner;
static const char* shader = R"CHROMA_SHADER({shader})CHROMA_SHADER";
auto create_model(const char* filename) {{
@autoreleasepool {{
    return std::make_unique<ModelProgram>(filename, {stats["weight_bytes"]}, {stats["workspace_bytes"]},
        std::vector<size_t>{{{", ".join(map(str, input_bytes))}}}, std::vector<size_t>{{{", ".join(map(str, output_bytes))}}},
        shader, std::vector<chroma::metal::Kernel>{{{", ".join(kernels)}}});
}}
}}
"""


class MetalCodegen(GPUCodegen):
    backend = "Metal"
    ctypes: ClassVar = {torch.float32: "float", torch.bfloat16: "bfloat", torch.int64: "int64_t"}
    functions: ClassVar = {"exp": "exp", "sqrt": "sqrt", "sin": "sin", "cos": "cos", "abs": "abs", "pow": "pow", "fmax": "fmax"}
    lanes = "simd_"
    any_lane = "simd_any({})"
    all_lanes = "simd_all({})"
    shuffle = "simd_shuffle({}, {})"
    index_function = "chroma_index"
    helper_prefix = "inline "
    pointer_prefix = "device "
    error_parameter = "device atomic_uint* error"

    def __init__(self) -> None:
        super().__init__()
        self.bindings: dict[str, Binding] = {}
        self.constant_arrays: dict[str, np.ndarray] = {}
        self.packed_weights: dict[str, Binding] = {}
        self.shader_cache: dict[tuple[tuple[str, ...], str], str] = {}
        self.launches: list[Launch] = []
        self.arguments: dict[tuple[str, torch.dtype], int] = {}
        self.bf16_kernels = 0
        self.tiled_matmuls = 0

    def encode(self, dtype: torch.dtype, expression: str) -> str:
        return f"bfloat({expression})" if dtype == torch.bfloat16 else expression

    def decode(self, expression: str) -> str:
        return f"float({expression})"

    def extern(self, tensor: torch.Tensor, kind: str, offset: int) -> Value:
        name = f"v{next(self.counter)}"
        shape = tuple(tensor.shape)
        self.bindings[name] = Binding("input", offset) if kind == "input" else Binding("weight", offset=offset)
        self.sources[name] = {name: tensor.dtype}
        return Value(name, shape, tensor.dtype, stored=True, view=View(None, strides(shape)))

    def weight(self, tensor: torch.Tensor) -> Value:
        value = super().weight(tensor)
        if tensor.dtype == torch.float32:
            self.constant_arrays[value.name] = tensor.detach().numpy()
        return value

    def allocate(self, shape: Shape, dtype: torch.dtype) -> Value:
        value = super().allocate(shape, dtype)
        self.bindings[value.name] = Binding("arena", value.name)
        return value

    def pointer(self, name: str, dtype: torch.dtype) -> str:
        # Fused helpers take raw pointers; kernels bind each distinct buffer to a numbered slot.
        if self.local_reads is not None:
            return name
        key = name, dtype
        if key not in self.arguments:
            self.arguments[key] = len(self.arguments)
        return f"b{self.arguments[key]}"

    def launch(
        self,
        outputs: Sequence[Value],
        inputs: Sequence[Value],
        count: int,
        body: Callable[[], str],
        *,
        parallel: bool = False,
        grid: Grid | None = None,
    ) -> None:
        if count > 2**32 - 1:
            raise ValueError("Metal dispatch exceeds the 32-bit grid limit")
        self.arguments = {}
        code = body()
        if len(self.arguments) > 30:
            raise ValueError("Fused Metal kernel exceeds the buffer binding limit")
        parameters: list[str] = []
        bindings: list[tuple[int, Binding]] = []
        output_names = {value.name for value in outputs}
        for (variable, dtype), index in self.arguments.items():
            const = "" if variable in output_names else "const "
            parameters.append(f"device {const}{self.ctype(dtype)}* b{index} [[buffer({index})]]")
            bindings.append((index, self.bindings[variable]))
        parameters.append("device atomic_uint* error [[buffer(30)]]")
        threads = 256
        if parallel:
            grid, threads = (count, 1, 1), 32
            code = "size_t gid = group.x;\n" + code
        if grid is None:
            parameters.append("uint gid [[thread_position_in_grid]]")
            code = f"if (gid >= {count}u) return;\n" + code
        else:
            parameters += ["uint3 group [[threadgroup_position_in_grid]]", "uint tid [[thread_index_in_threadgroup]]"]
        key = tuple(parameters), code
        if key not in self.shader_cache:
            name = f"chroma_kernel_{len(self.shaders)}"
            self.shaders.append(f"kernel void {name}({', '.join(parameters)}) {{\n{code}\n}}")
            self.shader_cache[key] = name
        bindings.append((30, Binding("error")))
        self.record(self.shader_cache[key], bindings, grid or (math.ceil(count / threads), 1, 1), (threads, 1, 1), inputs)

    def record(
        self,
        name: str,
        bindings: list[tuple[int, Binding]],
        groups: Grid,
        threads: Grid,
        inputs: Sequence[Value],
        params: bytes = b"",
    ) -> None:
        for _, binding in bindings:
            if binding.kind == "arena":
                allocation = self.allocations[cast(str, binding.index)]
                allocation.end = max(allocation.end, self.kernels)
        self.record_usage(inputs)
        self.launches.append(Launch(name, bindings, groups, threads, params))
        self.kernels += 1

    def packable(self, value: Value) -> bool:
        # FP32 constants qualify for BF16 storage only when packing loses nothing.
        array = self.constant_arrays.get(value.name)
        return array is not None and not np.any(array.view(np.uint32) & 0xFFFF)

    def pack(self, value: Value) -> Binding:
        if value.name not in self.packed_weights:
            self.packed_weights[value.name] = Binding("weight", offset=len(self.weights))
            self.weights.extend((self.constant_arrays[value.name].view(np.uint32) >> 16).astype(np.uint16).tobytes())
            self.weights.extend(bytes(-len(self.weights) % 64))
        return self.packed_weights[value.name]

    def alias(self, meta: torch.Tensor, root: Value, offset: int) -> Value:
        shape = tuple(meta.shape)
        name = f"v{next(self.counter)}"
        binding = self.bindings[root.name]
        itemsize = np.dtype(DTYPES[meta.dtype][1]).itemsize
        self.bindings[name] = replace(binding, offset=binding.offset + offset * itemsize)
        self.sources[name] = {name: meta.dtype}
        return Value(name, shape, meta.dtype, root.deps, stored=True, view=View(root, strides(shape), offset))

    def output(self, index: int, value: Value) -> None:
        if self.promote_output(index, value):
            return
        destination = Value(f"output{index}", value.shape, value.dtype, stored=True)
        self.bindings[destination.name] = Binding("output", index)
        self.copy(destination, value)

    def linear(self, meta: torch.Tensor, x: Value, weight: Value, bias: Value | None) -> Value:
        rows, columns, inner = math.prod(x.shape[:-1]), weight.shape[0], weight.shape[1]
        floats = all(v.dtype == torch.float32 for v in (meta, x, weight, *([bias] if bias else [])))
        packed = floats and inner >= 128 and inner % 4 == 0 and self.packable(weight) and (bias is None or self.packable(bias))
        if not x.stored and columns >= 8:
            x = self.materialize(x)
        out = self.allocate(tuple(meta.shape), meta.dtype)
        if packed:
            x = self.materialize(x)
            packed_weight = self.pack(weight)
            self.record(
                "chroma_f32_bf16w_matmul",
                [
                    (1, self.bindings[x.name]),
                    (2, packed_weight),
                    (3, self.pack(bias) if bias else packed_weight),
                    (4, self.bindings[out.name]),
                    (5, Binding("error")),
                ],
                (columns, rows, 1),
                (32, 1, 1),
                [x, weight] + ([bias] if bias else []),
                struct.pack("<III", inner // 4, columns, int(bias is not None)),
            )
            self.bf16_kernels += 1
        elif floats and rows >= 4 and columns >= 64 and inner >= 16:
            x, weight = self.materialize(x), self.materialize(weight)
            bias = self.materialize(bias) if bias else None
            self.record(
                "chroma_f32_linear_tiled",
                [
                    (1, self.bindings[x.name]),
                    (2, self.bindings[weight.name]),
                    (3, self.bindings[bias.name if bias else weight.name]),
                    (4, self.bindings[out.name]),
                ],
                ((columns + 31) // 32, (rows + 7) // 8, 1),
                (256, 1, 1),
                [x, weight] + ([bias] if bias else []),
                struct.pack("<IIII", rows, columns, inner, int(bias is not None)),
            )
            self.tiled_matmuls += 1
        else:
            self.scalar_linear(out, x, weight, bias)
        return out

    def norm(self, meta: torch.Tensor, x: Value, weight: Value, epsilon: float) -> Value:
        columns = x.shape[-1]
        floats = all(v.dtype == torch.float32 for v in (meta, x, weight))
        if not (floats and columns >= 4096 and columns % 4 == 0 and self.packable(weight)):
            return super().norm(meta, x, weight, epsilon)
        x = self.materialize(x)
        out = self.allocate(tuple(meta.shape), meta.dtype)
        self.record(
            "chroma_f32_bf16w_rmsnorm",
            [(1, self.bindings[x.name]), (2, self.pack(weight)), (3, self.bindings[out.name]), (4, Binding("error"))],
            (math.prod(x.shape[:-1]), 1, 1),
            (1024, 1, 1),
            [x, weight],
            struct.pack("<Iff", columns // 4, columns, epsilon),
        )
        self.bf16_kernels += 1
        return out

    def matrix(
        self,
        meta: torch.Tensor,
        labels: Sequence[str],
        output: str,
        sizes: dict[str, int],
        reduction: Sequence[str],
        values: Sequence[Value],
    ) -> Value | None:
        axes = matrix_axes(labels, output, reduction)
        if axes is None:
            return None
        batch_axes, row_axis, column_axis, contract_axis = axes
        rows, columns, inner = sizes[row_axis], sizes[column_axis], sizes[contract_axis]
        if rows < 8 or columns < 32 or inner < 16 or rows * columns * inner < 32768:
            return None
        out = self.allocate(tuple(meta.shape), meta.dtype)
        batch_shape = tuple(sizes[dim] for dim in batch_axes)

        # 8x32 output tiles, staged through threadgroup memory 16 reduction elements at a time.
        def body() -> str:
            coordinates = {dim: coordinate("group.z", batch_shape, i) for i, dim in enumerate(batch_axes)}
            coordinates.update({row_axis: "row", column_axis: "column", contract_axis: "k"})
            factors = []
            for label, value in zip(labels, values, strict=True):
                indices = [coordinates[dim] if size != 1 else "0ull" for dim, size in zip(label, value.shape, strict=True)]
                factors.append(self.access(value, flat_index(indices, value.shape)))
            destination = self.address(out, flat_index([coordinates[dim] for dim in output], out.shape))
            return f"""threadgroup float a[8 * 16];
threadgroup float b[32 * 16];
float sum = 0.0f;
for (size_t offset = 0; offset < {inner}ull; offset += 16) {{
for (uint i = tid; i < 8 * 16; i += 256) {{
size_t row = group.y * 8 + i / 16, k = offset + i % 16;
a[i] = row < {rows}ull && k < {inner}ull ? {factors[0]} : 0.0f;
}}
for (uint i = tid; i < 32 * 16; i += 256) {{
size_t column = group.x * 32 + i / 16, k = offset + i % 16;
b[i] = column < {columns}ull && k < {inner}ull ? {factors[1]} : 0.0f;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = 0; j < 16; ++j) sum += a[tid / 32 * 16 + j] * b[tid % 32 * 16 + j];
threadgroup_barrier(mem_flags::mem_threadgroup);
}}
size_t row = group.y * 8 + tid / 32, column = group.x * 32 + tid % 32;
if (row < {rows}ull && column < {columns}ull) {destination} = {self.encode(out.dtype, "sum")};"""

        grid = ((columns + 31) // 32, (rows + 7) // 8, math.prod(batch_shape))
        self.launch([out], values, math.prod(out.shape), body, grid=grid)
        self.tiled_matmuls += 1
        self.matrix_contractions += 1
        return out

    def launch_source(self, launch: Launch, input_count: int) -> str:
        # Runtime buffers are weights, arena, error flag, inputs, then outputs.
        slots = {"weight": 0, "arena": 1, "error": 2, "input": 3, "output": 3 + input_count}
        bindings = []
        for index, binding in launch.bindings:
            if binding.kind == "arena":
                name = cast(str, binding.index)
                if name in self.output_buffers:
                    binding = Binding("output", self.output_buffers[name], binding.offset)
                else:
                    binding = Binding("arena", offset=self.allocations[name].offset + binding.offset)
            slot = slots[binding.kind] + (cast(int, binding.index) if binding.kind in {"input", "output"} else 0)
            bindings.append(f"{{{index}, {slot}, {binding.offset}}}")
        fields = [
            json.dumps(launch.name),
            "{" + ", ".join(bindings) + "}",
            f"MTLSizeMake({', '.join(map(str, launch.groups))})",
            f"MTLSizeMake({', '.join(map(str, launch.threads))})",
            "{" + ", ".join(map(str, launch.params)) + "}",
        ]
        return "{" + ", ".join(fields) + "}"

    def generate(self, inputs: list[TensorSpec], outputs: list[TensorSpec], stats: Stats) -> GeneratedCode:
        input_bytes, output_bytes = list(map(nbytes, inputs)), list(map(nbytes, outputs))
        stats = stats | {
            "backend": "metal",
            "bf16_kernels": self.bf16_kernels,
            "tiled_matmuls": self.tiled_matmuls,
            "parallel_reductions": self.parallel_reductions,
            "unique_kernels": len({kernel.name for kernel in self.launches}),
            "input_bytes": sum(input_bytes),
            "metal_buffer_bytes": stats["weight_bytes"] + stats["workspace_bytes"] + sum(input_bytes) + sum(output_bytes) + 4,
        }
        shader = "\n".join(
            [f'#line 1 "kernels.metal"\n{_KERNELS.read_text()}', '#line 1 "chroma-generated.metal"', *self.helpers, *self.shaders]
        )
        kernels = [self.launch_source(launch, len(inputs)) for launch in self.launches]
        source = generation_template(shader, stats, input_bytes, output_bytes, kernels)
        flags = ["-fobjc-arc", "-fblocks", "-mmacosx-version-min=15.0", "-framework", "Metal", "-framework", "Foundation"]
        return GeneratedCode({"model.mm": source, "model.metal": shader}, "model.mm", flags, stats, inputs, outputs, bytes(self.weights))
