import math
from collections.abc import Callable, Sequence
from typing import ClassVar, cast

import torch

from chroma._src.codegen.base import Codegen, literal
from chroma._src.value import Operand, Shape, Value, View, contraction, strides

BINARY = {
    "add": "({a} + {scaled_b})",
    "sub": "({a} - {scaled_b})",
    "mul": "(({a}) * ({b}))",
    "div": "(({a}) / ({b}))",
    "rsub": "({b} - {scaled_a})",
    "pow": "{pow}({a}, {b})",
}
UNARY = {
    "reciprocal": "(1.0f / ({a}))",
    "rsqrt": "(1.0f / {sqrt}({a}))",
    "sigmoid": "(1.0f / (1.0f + {exp}(-({a}))))",
    "sin": "{sin}({a})",
    "cos": "{cos}({a})",
    "exp": "{exp}({a})",
    "neg": "(-({a}))",
    "abs": "{abs}({a})",
    "square": "(({a}) * ({a}))",
}


def coordinate(index: str, shape: Sequence[int], dim: int) -> str:
    return f"(({index}) / {strides(shape)[dim]}ull % {shape[dim]}ull)"


def flat_index(coordinates: Sequence[str], shape: Sequence[int]) -> str:
    return " + ".join(f"({value}) * {stride}ull" for value, stride in zip(coordinates, strides(shape), strict=True)) or "0ull"


def compute_type(dtype: torch.dtype) -> str:
    return "int64_t" if dtype == torch.int64 else "float"


def row_loop(parallel: bool) -> tuple[str, int]:
    # A parallel row is strided across the 32 lanes of its group; otherwise one thread walks it.
    return ("tid", 32) if parallel else ("0", 1)


class ScalarReads:
    """Names each distinct read inside a fused helper so repeated operands are loaded once."""

    def __init__(self) -> None:
        self.names: dict[tuple[str, str], str] = {}
        self.declarations: list[str] = []

    def read(self, source: str, index: str, ctype: str, expression: str) -> str:
        key = source, index
        if key not in self.names:
            self.names[key] = f"s{len(self.names)}"
            self.declarations.append(f"const {ctype} {self.names[key]} = {expression};")
        return self.names[key]


class GPUCodegen(Codegen):
    """Expression fusion and kernel bodies shared by CUDA and Metal.

    Subclasses provide the dialect (types, math names, 32-lane intrinsics) and how kernels are dispatched.
    Fused expressions become device helpers; materialized values become one thread per element, or one
    32-lane group per row for reductions wide enough to use it.
    """

    backend: str
    ctypes: ClassVar[dict[torch.dtype, str]]
    functions: ClassVar[dict[str, str]]
    lanes: str  # Prefix of the 32-lane sum, max, and min reductions.
    any_lane: str
    all_lanes: str
    shuffle: str
    index_function: str
    helper_prefix: str
    pointer_prefix: str
    error_parameter: str

    def __init__(self) -> None:
        super().__init__()
        self.helpers: list[str] = []
        self.shaders: list[str] = []
        self.sources: dict[str, dict[str, torch.dtype]] = {}
        self.local_reads: ScalarReads | None = None
        self.parallel_reductions = 0

    def encode(self, dtype: torch.dtype, expression: str) -> str:
        raise NotImplementedError

    def decode(self, expression: str) -> str:
        raise NotImplementedError

    def launch(
        self, outputs: Sequence[Value], inputs: Sequence[Value], count: int, body: Callable[[], str], *, parallel: bool = False
    ) -> None:
        raise NotImplementedError

    def pointer(self, name: str, dtype: torch.dtype) -> str:
        return name

    def ctype(self, dtype: torch.dtype) -> str:
        if dtype not in self.ctypes:
            raise NotImplementedError(f"Unsupported {self.backend} dtype: {dtype}")
        return self.ctypes[dtype]

    def allocate(self, shape: Shape, dtype: torch.dtype) -> Value:
        self.ctype(dtype)
        value = super().allocate(shape, dtype)
        self.sources[value.name] = {value.name: dtype}
        return value

    def expr(self, meta: torch.Tensor, expression: Callable[[str], str], inputs: Sequence[Value]) -> Value:
        name = f"e{next(self.counter)}"
        sources = {source: dtype for value in inputs for source, dtype in self.sources[value.name].items()}
        self.sources[name] = sources
        # Function boundaries keep scalar and index expressions linear in graph size.
        reads = ScalarReads()
        self.local_reads = reads
        try:
            res = self.encode(meta.dtype, expression("i"))
        finally:
            self.local_reads = None
        parameters = [f"{self.pointer_prefix}const {self.ctype(dtype)}* {source}" for source, dtype in sources.items()]
        parameters += ["size_t i", self.error_parameter]
        self.helpers.append(
            f"{self.helper_prefix}{self.ctype(meta.dtype)} {name}({', '.join(parameters)}) {{\n"
            + "\n".join(reads.declarations)
            + f"\nreturn {res};\n}}"
        )
        value = self.expression_value(name, meta, inputs)
        return self.materialize(value) if value.depth >= 24 or len(sources) >= 24 else value

    def address(self, value: Value, index: str) -> str:
        return f"{self.pointer(value.name, value.dtype)}[{index}]"

    def access(self, value: Value, index: str) -> str:
        if value.stored:
            code = self.address(value, index)
        else:
            arguments = [self.pointer(source, dtype) for source, dtype in self.sources[value.name].items()]
            code = f"{value.name}({', '.join([*arguments, index, 'error'])})"
        if value.dtype == torch.bfloat16:
            code = self.decode(code)
        if self.local_reads is None:
            return code
        return self.local_reads.read(value.name, index, compute_type(value.dtype), code)

    def broadcast(self, value: Operand, index: str, shape: Shape) -> str:
        if not isinstance(value, Value):
            return literal(value)
        if value.shape == shape:
            return self.access(value, index)
        pad = len(shape) - len(value.shape)
        parts = [
            f"{coordinate(index, shape, dim + pad)} * {stride}ull"
            for dim, (size, stride) in enumerate(zip(value.shape, strides(value.shape), strict=True))
            if size != 1
        ]
        return self.access(value, " + ".join(parts) or "0ull")

    def materialize(self, value: Value) -> Value:
        if value.stored:
            return value
        if value.name not in self.materialized:
            out = self.allocate(value.shape, value.dtype)
            self.copy(out, value)
            self.materialized[value.name] = out
        return self.materialized[value.name]

    def copy(self, out: Value, value: Value, index: str = "gid") -> None:
        def body() -> str:
            return f"{self.address(out, index)} = {self.encode(out.dtype, self.access(value, 'gid'))};"

        self.launch([out], [value], math.prod(value.shape), body)

    def reshape(self, meta: torch.Tensor, x: Value) -> Value:
        if x.stored:
            view = cast(View, x.view)
            return self.alias(meta, view.base or x, view.offset)
        return self.cast(meta, x)

    def alias(self, meta: torch.Tensor, root: Value, offset: int) -> Value:
        raise NotImplementedError

    def reindex(self, meta: torch.Tensor, x: Value, mapping: Sequence[int], offset: int) -> Value:
        def reader(index: str) -> str:
            parts = [f"{coordinate(index, meta.shape, dim)} * {stride}ull" for dim, stride in enumerate(mapping) if stride]
            return self.access(x, " + ".join([f"{offset}ull", *parts]))

        return self.expr(meta, reader, [x])

    def cast(self, meta: torch.Tensor, x: Value) -> Value:
        return self.expr(meta, lambda index: self.access(x, index), [x])

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
        shape = tuple(meta.shape)
        operation = target.split(".")[1]

        def reader(index: str) -> str:
            a = self.broadcast(args[0], index, shape)
            if operation in BINARY:
                b = self.broadcast(args[1], index, shape)
                scaled_a, scaled_b = f"({a}) * {literal(alpha)}", f"({b}) * {literal(alpha)}"
                if meta.dtype == torch.bfloat16:
                    # Torch rounds the scaled operand to BF16 before the add.
                    scaled_a = self.decode(self.encode(meta.dtype, scaled_a))
                    scaled_b = self.decode(self.encode(meta.dtype, scaled_b))
                return BINARY[operation].format(a=a, b=b, scaled_a=scaled_a, scaled_b=scaled_b, pow=self.functions["pow"])
            if operation == "clamp":
                if low is not None:
                    a = f"(({a}) < {literal(low)} ? {literal(low)} : ({a}))"
                if high is not None:
                    a = f"(({a}) > {literal(high)} ? {literal(high)} : ({a}))"
                return a
            return UNARY[operation].format(a=a, **self.functions)

        return self.expr(meta, reader, [value for value in args if isinstance(value, Value)])

    def gather(self, meta: torch.Tensor, x: Value, indices: Value, negative: bool) -> Value:
        width = math.prod(x.shape[1:])

        def reader(index: str) -> str:
            selected = self.access(indices, f"({index}) / {width}ull")
            row = f"{self.index_function}({selected}, {x.shape[0]}ull, {str(negative).lower()}, error)"
            return self.access(x, f"{row} * {width}ull + ({index}) % {width}ull")

        return self.expr(meta, reader, [x, indices])

    def triangular(self, meta: torch.Tensor, x: Value, diagonal: int, upper: bool) -> Value:
        shape = tuple(meta.shape)
        sign = ">=" if upper else "<="

        def reader(index: str) -> str:
            column = f"int64_t(({index}) % {shape[-1]}ull)"
            row = f"int64_t(({index}) / {shape[-1]}ull % {shape[-2]}ull)"
            return f"({column} - {row} {sign} {diagonal} ? {self.access(x, index)} : 0)"

        return self.expr(meta, reader, [x])

    def concatenate(self, meta: torch.Tensor, values: Sequence[Value], dim: int) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        start, inner = 0, math.prod(out.shape[dim + 1 :])
        for value in values:
            block = value.shape[dim] * inner
            index = f"gid / {block}ull * {out.shape[dim] * inner}ull + {start * inner}ull + gid % {block}ull"
            self.copy(out, value, index)
            start += value.shape[dim]
        return out

    def scalar_linear(self, out: Value, x: Value, weight: Value, bias: Value | None) -> None:
        rows, columns, inner = math.prod(x.shape[:-1]), weight.shape[0], weight.shape[1]

        def body() -> str:
            a = self.access(x, f"(gid / {columns}ull) * {inner}ull + k")
            b = self.access(weight, f"(gid % {columns}ull) * {inner}ull + k")
            offset = f" + {self.access(bias, f'gid % {columns}ull')}" if bias is not None else ""
            res = self.encode(out.dtype, f"sum{offset}")
            return f"float sum = 0.0f;\nfor (size_t k = 0; k < {inner}ull; ++k) sum += ({a}) * ({b});\n{self.address(out, 'gid')} = {res};"

        self.launch([out], [x, weight] + ([bias] if bias is not None else []), rows * columns, body)

    def softmax(self, meta: torch.Tensor, x: Value, dim: int) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        width, inner = x.shape[dim], math.prod(x.shape[dim + 1 :])
        rows, parallel = math.prod(x.shape) // width, width >= 32
        element = f"(gid / {inner}ull) * {width * inner}ull + gid % {inner}ull + j * {inner}ull"
        start, step = row_loop(parallel)
        exp, fmax = self.functions["exp"], self.functions["fmax"]

        def body() -> str:
            source = self.access(x, element)
            maximum = f"maximum = {self.lanes}max(maximum);" if parallel else ""
            reduce = f"sum = {self.lanes}sum(sum);" if parallel else ""
            res = self.encode(out.dtype, f"{exp}({source} - maximum) / sum")
            return f"""float maximum = -INFINITY;
for (size_t j = {start}; j < {width}ull; j += {step}) maximum = {fmax}(maximum, {source});
{maximum}
float sum = 0.0f;
for (size_t j = {start}; j < {width}ull; j += {step}) sum += {exp}({source} - maximum);
{reduce}
for (size_t j = {start}; j < {width}ull; j += {step}) {self.address(out, element)} = {res};"""

        self.launch([out], [x], rows, body, parallel=parallel)
        self.parallel_reductions += parallel
        return out

    def topk(self, meta: Sequence[torch.Tensor], x: Value, dim: int, largest: bool, sort: bool) -> tuple[Value, Value]:
        # One 32-lane group per row selects k times; ties go to the lowest index, NaN ranks highest.
        out = self.allocate(tuple(meta[0].shape), meta[0].dtype)
        indices = self.allocate(tuple(meta[1].shape), meta[1].dtype)
        width, k, inner = x.shape[dim], meta[0].shape[dim], math.prod(x.shape[dim + 1 :])
        rows, chunks = math.prod(x.shape) // width, (width + 31) // 32
        source = f"(gid / {inner}ull) * {width * inner}ull + gid % {inner}ull + j * {inner}ull"
        destination = f"(gid / {inner}ull) * {k * inner}ull + gid % {inner}ull + selected * {inner}ull"
        compare = "(isnan(v) && !isnan(best)) || v > best" if largest else "(!isnan(v) && isnan(best)) || v < best"
        nan_reduce = self.any_lane.format("chosen >= 0 && isnan(best)") if largest else self.all_lanes.format("chosen < 0 || isnan(best)")
        extremum, empty = ("max", "-INFINITY") if largest else ("min", "INFINITY")
        lanes = self.lanes

        def body() -> str:
            res = self.encode(out.dtype, "value")
            return f"""float scratch[{chunks}];
bool used[{chunks}];
for (unsigned int c = 0; c < {chunks}; ++c) {{
size_t j = tid + c * 32;
used[c] = j >= {width};
scratch[c] = used[c] ? 0.0f : {self.access(x, source)};
}}
for (size_t selected = 0; selected < {k}ull; ++selected) {{
float best = 0.0f; int chosen = -1;
for (unsigned int c = 0; c < {chunks}; ++c) {{
float v = scratch[c];
if (!used[c] && (chosen < 0 || {compare})) {{ best = v; chosen = int(tid + c * 32); }}
}}
bool nan_wins = {nan_reduce};
float extreme = {lanes}{extremum}(chosen < 0 || isnan(best) ? {empty} : best);
unsigned int winner = {lanes}min(chosen >= 0 && (nan_wins ? isnan(best) : best == extreme) ? unsigned(chosen) : 0xffffffffu);
float value = {self.shuffle.format("best", "winner % 32")};
if (winner % 32 == tid) used[winner / 32] = true;
if (tid == 0) {{ {self.address(out, destination)} = {res}; {self.address(indices, destination)} = int64_t(winner); }}
}}"""

        self.launch([out, indices], [x], rows, body, parallel=True)
        return out, indices

    def norm(self, meta: torch.Tensor, x: Value, weight: Value, epsilon: float) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        columns = x.shape[-1]
        parallel = columns >= 32
        index = f"gid * {columns}ull + j"
        start, step = row_loop(parallel)

        def body() -> str:
            source = self.access(x, index)
            reduce = f"sum = {self.lanes}sum(sum);" if parallel else ""
            res = self.encode(out.dtype, f"{source} * scale * {self.access(weight, 'j')}")
            return f"""float sum = 0.0f;
for (size_t j = {start}; j < {columns}ull; j += {step}) {{ float v = {source}; sum += v * v; }}
{reduce}
float scale = 1.0f / {self.functions["sqrt"]}(sum / {literal(columns)} + {literal(epsilon)});
for (size_t j = {start}; j < {columns}ull; j += {step}) {self.address(out, index)} = {res};"""

        self.launch([out], [x, weight], math.prod(x.shape[:-1]), body, parallel=parallel)
        self.parallel_reductions += parallel
        return out

    def reduction(self, meta: torch.Tensor, x: Value, axes: Sequence[int], keepdim: bool, mean: bool) -> Value:
        out = self.allocate(tuple(meta.shape), meta.dtype)
        reduction_shape = tuple(x.shape[dim] for dim in axes)
        width = math.prod(reduction_shape)
        parallel = width >= 32
        coordinates = []
        out_dim = reduce_dim = 0
        for dim in range(len(x.shape)):
            if dim in axes:
                coordinates.append(coordinate("r", reduction_shape, reduce_dim))
                reduce_dim += 1
            else:
                coordinates.append(coordinate("gid", out.shape, out_dim))
            if keepdim or dim not in axes:
                out_dim += 1
        start, step = row_loop(parallel)

        def body() -> str:
            reduce = f"sum = {self.lanes}sum(sum);" if parallel else ""
            write = "if (tid == 0) " if parallel else ""
            res = self.encode(out.dtype, f"sum / {literal(width if mean else 1)}")
            return f"""float sum = 0.0f;
for (size_t r = {start}; r < {width}ull; r += {step}) sum += {self.access(x, flat_index(coordinates, x.shape))};
{reduce}
{write}{self.address(out, "gid")} = {res};"""

        self.launch([out], [x], math.prod(out.shape), body, parallel=parallel)
        self.parallel_reductions += parallel
        return out

    def einsum(self, meta: torch.Tensor, eqn: str, values: Sequence[Value]) -> Value:
        labels, output, sizes, reduction = contraction(eqn, values)
        materialized = []
        for label, value in zip(labels, values, strict=True):
            reused = math.prod(sizes[dim] for dim in output if dim not in label)
            if not value.stored and value.view is None and reused >= 8:
                value = self.materialize(value)
            materialized.append(value)
        values = materialized
        res = self.matrix(meta, labels, output, sizes, reduction, values)
        if res is not None:
            return res
        out = self.allocate(tuple(meta.shape), meta.dtype)
        reduction_shape = tuple(sizes[dim] for dim in reduction)
        width = math.prod(reduction_shape)
        parallel = width >= 512
        start, step = row_loop(parallel)

        def body() -> str:
            coordinates = {label: coordinate("gid", out.shape, i) for i, label in enumerate(output)}
            coordinates.update({label: coordinate("r", reduction_shape, i) for i, label in enumerate(reduction)})
            factors = []
            for label, value in zip(labels, values, strict=True):
                indices = [coordinates[letter] if size != 1 else "0ull" for letter, size in zip(label, value.shape, strict=True)]
                factors.append(self.access(value, flat_index(indices, value.shape)))
            reduce = f"sum = {self.lanes}sum(sum);" if parallel else ""
            write = "if (tid == 0) " if parallel else ""
            return f"""float sum = 0.0f;
for (size_t r = {start}; r < {width}ull; r += {step}) sum += ({factors[0]}) * ({factors[1]});
{reduce}
{write}{self.address(out, "gid")} = {self.encode(out.dtype, "sum")};"""

        self.launch([out], values, math.prod(out.shape), body, parallel=parallel)
        self.parallel_reductions += parallel
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
        return None
