import math
import sys
from collections.abc import Sequence
from typing import cast

import torch

from chroma._src.codegen.base import Codegen, GeneratedCode, literal
from chroma._src.program import Stats, TensorSpec
from chroma._src.value import DTYPES, Operand, Shape, Value, View, contraction, matrix_axes, strides

_UNARY = {
    "aten.reciprocal.default": "1.0f / x",
    "aten.rsqrt.default": "1.0f / std::sqrt(x)",
    "aten.sigmoid.default": "1.0f / (1.0f + std::exp(-x))",
    "aten.neg.default": "-x",
    "aten.sin.default": "std::sin(x)",
    "aten.cos.default": "std::cos(x)",
    "aten.exp.default": "std::exp(x)",
    "aten.abs.default": "std::abs(x)",
    "aten.square.default": "x * x",
}

_BINARY = {
    "aten.add.Tensor": "+",
    "aten.sub.Tensor": "-",
    "aten.mul.Tensor": "*",
    "aten.div.Tensor": "/",
}


def dims(shape: Sequence[int]) -> str:
    return "dims<" + ", ".join(map(str, shape)) + ">"


def operand(value: Operand) -> str:
    return value.name if isinstance(value, Value) else f"Scalar<float>{{{literal(value)}}}"


def encode(dtype: torch.dtype, expression: str) -> str:
    return f"astype<chroma::bfloat16_t>({expression})" if dtype == torch.bfloat16 else expression


def generation_template(stats: Stats, body: str) -> str:
    return f"""#include "codegen/cpu/kernels.h"
#include "core/bindings.h"
using namespace chroma;
struct ModelProgram {{
Buffer weights{{{stats["weight_bytes"]}}};
Buffer workspace{{{stats["workspace_bytes"]}}};
explicit ModelProgram(const char* filename) {{ read_weights(filename, weights.data(), weights.size()); }}
const char* device_name() const {{ return "CPU"; }}
void run(const void* const* inputs, void* const* outputs) {{
{body}
}}
}};
auto create_model(const char* filename) {{ return std::make_unique<ModelProgram>(filename); }}
"""


def contract_loops(out: Value, reduction_shape: Sequence[int], product: str, divisor: int = 1) -> str:
    lines = ["{"]
    for i, size in enumerate(out.shape):
        lines.append(f"for (size_t o{i} = 0; o{i} < {size}; ++o{i}) {{")
    lines.append("float sum = 0.0f;")
    for i, size in enumerate(reduction_shape):
        lines.append(f"for (size_t r{i} = 0; r{i} < {size}; ++r{i}) {{")
    lines.append(f"sum += {product};")
    lines.extend("}" for _ in reduction_shape)
    index = " + ".join(f"o{i} * {stride}" for i, stride in enumerate(strides(out.shape))) or "0"
    lines.append(f"{out.name}.data[{index}] = sum / {literal(divisor)};")
    lines.extend("}" for _ in out.shape)
    lines.append("}")
    return "\n".join(lines)


class CPUCodegen(Codegen):
    def __init__(self, *, blas: bool = True) -> None:
        super().__init__()
        self.blas = blas
        self.expert_contractions = 0
        self.lines: list[str | Value] = []
        self.gathers: dict[str, tuple[Value, Value]] = {}

    def extern(self, tensor: torch.Tensor, kind: str, offset: int) -> Value:
        name = f"v{next(self.counter)}"
        shape = tuple(tensor.shape)
        ctype = DTYPES[tensor.dtype][0]
        pointer = f"inputs[{offset}]" if kind == "input" else f"weights.data() + {offset}"
        self.lines.append(f"auto {name} = view<const {ctype}, {dims(shape)}>(reinterpret_cast<const {ctype}*>({pointer}));")
        return Value(name, shape, tensor.dtype, stored=True, view=View(None, strides(shape)))

    def expr(self, meta: torch.Tensor, expression: str, inputs: Sequence[Value]) -> Value:
        name = f"e{next(self.counter)}"
        self.lines.append(f"auto {name} = {expression};")
        value = self.expression_value(name, meta, inputs)
        return self.materialize(value) if value.depth >= 32 else value

    def allocate(self, shape: Shape, dtype: torch.dtype) -> Value:
        value = super().allocate(shape, dtype)
        self.lines.append(value)
        return value

    def record(self, code: str, inputs: Sequence[Value]) -> None:
        self.record_usage(inputs)
        self.lines.append(code)
        self.kernels += 1

    def materialize(self, value: Value) -> Value:
        if value.stored:
            return value
        if value.name in self.materialized:
            return self.materialized[value.name]
        out = self.allocate(value.shape, value.dtype)
        self.record(f"evaluate({out.name}, {value.name});", [value])
        self.materialized[value.name] = out
        return out

    def alias(self, meta: torch.Tensor, root: Value, offset: int) -> Value:
        shape = tuple(meta.shape)
        name = f"v{next(self.counter)}"
        ctype = DTYPES[meta.dtype][0]

        self.lines.append(f"auto {name} = view<const {ctype}, {dims(shape)}>({root.name}.data + {offset});")
        return Value(name, shape, meta.dtype, root.deps, stored=True, view=View(root, strides(shape), offset))

    def reshape(self, meta: torch.Tensor, x: Value) -> Value:
        if x.stored:
            view = cast(View, x.view)
            return self.alias(meta, view.base or x, view.offset)
        return self.expr(meta, f"reshape<{dims(tuple(meta.shape))}>({x.name})", [x])

    def reindex(self, meta: torch.Tensor, x: Value, mapping: Sequence[int], offset: int) -> Value:
        shape = tuple(meta.shape)
        return self.expr(meta, f"reindex<{dims(shape)}, {dims(mapping)}, {offset}>({x.name})", [x])

    def cast(self, meta: torch.Tensor, x: Value) -> Value:
        return self.expr(meta, f"astype<{DTYPES[meta.dtype][0]}>({x.name})", [x])

    def output(self, index: int, value: Value) -> None:
        if self.promote_output(index, value):
            return
        ctype = DTYPES[value.dtype][0]
        output = f"view<{ctype}, {dims(value.shape)}>(reinterpret_cast<{ctype}*>(outputs[{index}]))"
        self.record(f"evaluate({output}, {value.name});", [value])

    def elementwise(
        self,
        meta: torch.Tensor,
        target: str,
        args: tuple[Operand, ...],
        *,
        alpha: float = 1,
        low: float | None = None,
        high: float | None = None,
    ) -> Value:
        values = [value for value in args if isinstance(value, Value)]
        cpp = [operand(value) for value in args]
        if target in _BINARY:
            if alpha != 1:
                cpp[1] = f"({cpp[1]} * Scalar<float>{{{literal(alpha)}}})"
                cpp[1] = encode(meta.dtype, cpp[1])
            expression = f"{cpp[0]} {_BINARY[target]} {cpp[1]}"
        elif target == "aten.rsub.Scalar":
            product = f"({cpp[0]} * Scalar<float>{{{literal(alpha)}}})"
            product = encode(meta.dtype, product)
            expression = f"{cpp[1]} - {product}"
        elif target in {"aten.pow.Tensor_Scalar", "aten.pow.Scalar"}:
            expression = f"binary({cpp[0]}, {cpp[1]}, [](float a, float b) {{ return std::pow(a, b); }})"
        elif target == "aten.clamp.default":
            expression = "x"
            if low is not None:
                expression = f"std::max({expression}, {literal(low)})"
            if high is not None:
                expression = f"std::min({expression}, {literal(high)})"
            expression = f"unary({cpp[0]}, [](float x) {{ return {expression}; }})"
        else:
            expression = _UNARY[target]
            expression = f"unary({cpp[0]}, [](float x) {{ return {expression}; }})"
        return self.expr(meta, encode(meta.dtype, expression), values)

    def gather(self, meta: torch.Tensor, x: Value, indices: Value, negative: bool) -> Value:
        value = self.expr(
            meta,
            f"gather<{dims(tuple(meta.shape))}, {str(negative).lower()}>({x.name}, {indices.name})",
            [x, indices],
        )
        if negative and not value.stored:
            self.gathers[value.name] = x, indices
        return value

    def linear(self, meta: torch.Tensor, x: Value, weight: Value, bias: Value | None) -> Value:
        x, weight = self.materialize(x), self.materialize(weight)
        bias = self.materialize(bias) if bias is not None else None
        out = self.allocate(tuple(meta.shape), meta.dtype)
        rows, columns, inner = math.prod(x.shape[:-1]), weight.shape[0], weight.shape[1]
        if max(rows, columns, inner) > 2**31 - 1:
            raise ValueError("Matrix dimensions exceed the BLAS integer range")
        bias_pointer = f"{bias.name}.data" if bias else "nullptr"
        code = f"linear({out.name}.data, {x.name}.data, {weight.name}.data, {bias_pointer}, {rows}, {columns}, {inner});"
        self.record(code, [x, weight] + ([bias] if bias else []))
        return out

    def softmax(self, meta: torch.Tensor, x: Value, dim: int) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        self.record(f"softmax<{dim}>({out.name}, {x.name});", [x])
        return out

    def topk(self, meta: Sequence[torch.Tensor], x: Value, dim: int, largest: bool, sort: bool) -> tuple[Value, Value]:
        values, indices = [self.allocate(tuple(tensor.shape), tensor.dtype) for tensor in meta]
        code = f"topk<{dim}, {str(largest).lower()}, {str(sort).lower()}>({values.name}, {indices.name}, {x.name});"
        self.record(code, [x])
        return values, indices

    def concatenate(self, meta: torch.Tensor, values: Sequence[Value], dim: int) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        start, parts = 0, []
        for value in values:
            parts.append(f"concat_part<{dim}, {start}>({out.name}, {value.name});")
            start += value.shape[dim]
        self.record("\n".join(parts), values)
        return out

    def triangular(self, meta: torch.Tensor, x: Value, diagonal: int, upper: bool) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        self.record(f"triangular<{str(upper).lower()}, {diagonal}>({out.name}, {x.name});", [x])
        return out

    def norm(self, meta: torch.Tensor, x: Value, weight: Value, epsilon: float) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        self.record(f"rmsnorm({out.name}, {x.name}, {weight.name}, {literal(epsilon)});", [x, weight])
        return out

    def reduction(self, meta: torch.Tensor, x: Value, axes: Sequence[int], keepdim: bool, mean: bool) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        reduction_shape = tuple(x.shape[dim] for dim in axes)
        terms: list[str] = []
        out_dim = reduce_dim = 0
        for dim, stride in enumerate(strides(x.shape)):
            if dim in axes:
                terms.append(f"r{reduce_dim} * {stride}")
                reduce_dim += 1
            else:
                terms.append(f"o{out_dim} * {stride}")
            if keepdim or dim not in axes:
                out_dim += 1
        product = f"{x.name}[{' + '.join(terms) or '0'}]"
        divisor = math.prod(reduction_shape) if mean else 1
        self.record(contract_loops(out, reduction_shape, product, divisor), [x])
        return out

    def einsum(self, meta: torch.Tensor, eqn: str, values: Sequence[Value]) -> Value:
        labels, output, sizes, reduction = contraction(eqn, values)
        eqn = eqn.replace(" ", "")
        expert = self.expert_einsum(meta, eqn, labels, sizes, values)
        if expert is not None:
            return expert
        res = self.matrix(meta, labels, output, sizes, reduction, values)
        if res is not None:
            return res
        return self.loop_einsum(meta, labels, output, sizes, reduction, values)

    def expert_einsum(
        self,
        meta: torch.Tensor,
        eqn: str,
        labels: Sequence[str],
        sizes: dict[str, int],
        values: Sequence[Value],
    ) -> Value | None:
        if eqn not in {"beck,bk->bec", "beck,bek->bec"}:
            return None
        gather = self.gathers.get(values[0].name)
        if gather is None:
            return None
        for label, value in zip(labels, values, strict=True):
            if any(size != sizes[dim] for dim, size in zip(label, value.shape, strict=True)):
                return None

        weight, indices = gather
        weight, indices, x = map(self.materialize, (weight, indices, values[1]))
        out = self.allocate(tuple(meta.shape), meta.dtype)
        batches, experts, channels = out.shape
        inner = weight.shape[-1]
        if max(batches, experts, channels, inner, weight.shape[0]) > 2**31 - 1:
            raise ValueError("Expert dimensions exceed the BLAS integer range")
        per_expert_input = str(eqn == "beck,bek->bec").lower()
        code = (
            f"expert_linear({out.name}.data, {weight.name}.data, {indices.name}.data, "
            f"{x.name}.data, {batches}, {experts}, {weight.shape[0]}, {channels}, {inner}, {per_expert_input});"
        )
        self.record(code, [weight, indices, x])
        self.expert_contractions += 1
        return out

    def loop_einsum(
        self,
        meta: torch.Tensor,
        labels: Sequence[str],
        output: str,
        sizes: dict[str, int],
        reduction: Sequence[str],
        values: Sequence[Value],
    ) -> Value:
        coordinates = {label: f"o{i}" for i, label in enumerate(output)}
        coordinates |= {label: f"r{i}" for i, label in enumerate(reduction)}
        factors: list[str] = []
        for label, value in zip(labels, values, strict=True):
            terms = []
            for letter, size, stride in zip(label, value.shape, strides(value.shape), strict=True):
                if size != 1:
                    terms.append(f"{coordinates[letter]} * {stride}")
            index = " + ".join(terms) or "0"
            factors.append(f"{value.name}[{index}]")
        out = self.allocate(tuple(meta.shape), meta.dtype)
        reduction_shape = tuple(sizes[label] for label in reduction)
        self.record(contract_loops(out, reduction_shape, " * ".join(factors)), values)
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
        out_strides = dict(zip(output, strides(tuple(meta.shape))))
        rows = sizes[row_axis]
        columns = sizes[column_axis]
        inner = sizes[contract_axis]
        row_stride = out_strides[row_axis]
        if out_strides[column_axis] != 1 or max(rows, columns, inner, row_stride) >= 2**31:
            return None

        layouts: list[dict[str, int]] = []
        for label, value in zip(labels, values, strict=True):
            value_strides = value.view.strides if value.view else strides(value.shape)
            layout = dict(zip(label, value_strides, strict=True))
            layouts.append(layout)
            matrix_axes_for_value = {row_axis, column_axis, contract_axis} & set(label)
            if any(value.shape[label.index(axis)] != sizes[axis] for axis in matrix_axes_for_value):
                return None

        left_layout, right_layout = layouts
        if not (left_layout[contract_axis] == 1 or left_layout[row_axis] == 1):
            return None
        if not (right_layout[column_axis] == 1 or right_layout[contract_axis] == 1):
            return None

        trans_a = left_layout[contract_axis] != 1
        trans_b = right_layout[column_axis] != 1
        lda = left_layout[contract_axis if trans_a else row_axis]
        ldb = right_layout[column_axis if trans_b else contract_axis]
        if lda < sizes[row_axis if trans_a else contract_axis] or ldb < sizes[contract_axis if trans_b else column_axis]:
            return None
        if max(lda, ldb) >= 2**31:
            raise ValueError("Matrix strides exceed the BLAS integer range")

        operands = []
        for value in values:
            if value.view:
                operands.append(value.view.base or value)
            else:
                operands.append(self.materialize(value))
        left, right = operands
        out = self.allocate(tuple(meta.shape), meta.dtype)
        offsets: list[str] = []
        for label, value, layout in zip(labels, values, layouts, strict=True):
            offset = value.view.offset if value.view else 0
            terms = [str(offset)]
            for index, axis in enumerate(batch_axes):
                if value.shape[label.index(axis)] != 1:
                    terms.append(f"b{index} * {layout[axis]}")
            offsets.append(" + ".join(terms))

        destination = " + ".join(f"b{i} * {out_strides[axis]}" for i, axis in enumerate(batch_axes)) or "0"
        batch_loops = [f"for (size_t b{i} = 0; b{i} < {sizes[axis]}; ++b{i})" for i, axis in enumerate(batch_axes)]
        code = "\n".join(batch_loops)
        code += (
            f"\ngemm({out.name}.data + {destination}, {left.name}.data + {offsets[0]}, "
            f"{right.name}.data + {offsets[1]}, {rows}, {columns}, {inner}, "
            f"{lda}, {ldb}, {row_stride}, {str(trans_a).lower()}, {str(trans_b).lower()});"
        )
        self.record(code, [left, right])
        self.matrix_contractions += 1
        return out

    def render_line(self, line: str | Value) -> str:
        if isinstance(line, str):
            return line
        ctype = DTYPES[line.dtype][0]
        if line.name in self.output_buffers:
            pointer = f"outputs[{self.output_buffers[line.name]}]"
        else:
            pointer = f"workspace.data() + {self.allocations[line.name].offset}"
        return f"auto {line.name} = view<{ctype}, {dims(line.shape)}>(reinterpret_cast<{ctype}*>({pointer}));"

    def compiler_options(self) -> tuple[list[str], str]:
        if not self.blas:
            return [], "portable"
        if sys.platform == "darwin":
            return ["-DCHROMA_ACCELERATE", "-DACCELERATE_NEW_LAPACK", "-framework", "Accelerate"], "accelerate"
        return ["-DCHROMA_CBLAS", "-lblas"], "cblas"

    def generate(self, inputs: list[TensorSpec], outputs: list[TensorSpec], stats: Stats) -> GeneratedCode:
        flags, math_library = self.compiler_options()
        stats = stats | {"expert_contractions": self.expert_contractions, "backend": "cpu", "math_library": math_library}
        source = generation_template(stats, "\n".join(self.render_line(line) for line in self.lines))
        return GeneratedCode({"model.cc": source}, "model.cc", flags, stats, inputs, outputs, bytes(self.weights))
