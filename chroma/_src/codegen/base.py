import hashlib
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from chroma._src.memory import Allocation, plan_memory
from chroma._src.program import Stats, TensorSpec
from chroma._src.value import DTYPES, Shape, Value, View, strides


def literal(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "-INFINITY" if value < 0 else "INFINITY"
    return repr(value) + "f"


def nbytes(spec: TensorSpec) -> int:
    return math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize


@dataclass
class GeneratedCode:
    files: dict[str, str]
    source_name: str
    flags: list[str]
    stats: Stats
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    weights: bytes


class Codegen:
    def __init__(self) -> None:
        self.materialized: dict[str, Value] = {}
        self.allocations: dict[str, Allocation] = {}
        self.weights = bytearray()
        self.weight_cache: dict[tuple[torch.dtype, Shape, bytes], Value] = {}
        self.output_buffers: dict[str, int] = {}
        self.counter = itertools.count()
        self.kernels = 0
        self.expressions = 0
        self.matrix_contractions = 0

    def extern(self, tensor: torch.Tensor, kind: str, offset: int) -> Value:
        raise NotImplementedError

    def weight(self, tensor: torch.Tensor) -> Value:
        tensor = tensor.detach().contiguous()
        data = tensor.view(torch.uint16).numpy().tobytes() if tensor.dtype == torch.bfloat16 else tensor.numpy().tobytes()
        key = (tensor.dtype, tuple(tensor.shape), hashlib.sha256(data).digest())
        if key in self.weight_cache:
            return self.weight_cache[key]
        offset = len(self.weights)
        self.weights.extend(data)
        self.weights.extend(bytes(-len(self.weights) % 64))
        value = self.extern(tensor, "weight", offset)
        self.weight_cache[key] = value
        return value

    def allocate(self, shape: Shape, dtype: torch.dtype) -> Value:
        name = f"t{next(self.counter)}"
        size = math.prod(shape) * np.dtype(DTYPES[dtype][1]).itemsize
        self.allocations[name] = Allocation((size + 63) // 64 * 64, self.kernels, self.kernels)
        return Value(name, shape, dtype, frozenset({name}), stored=True, view=View(None, strides(shape)))

    def expression_value(self, name: str, meta: torch.Tensor, inputs: Sequence[Value]) -> Value:
        self.expressions += 1
        return Value(
            name,
            tuple(meta.shape),
            meta.dtype,
            frozenset().union(*(value.deps for value in inputs)),
            depth=1 + max((value.depth for value in inputs), default=0),
        )

    def finish(self, inputs: list[TensorSpec], outputs: list[TensorSpec], folded_constants: int) -> GeneratedCode:
        for name in self.output_buffers:
            del self.allocations[name]

        workspace_bytes = plan_memory(self.allocations)
        output_bytes = sum(map(nbytes, outputs))
        stats: Stats = {
            "kernels": self.kernels,
            "expressions": self.expressions,
            "folded_constants": folded_constants,
            "matrix_contractions": self.matrix_contractions,
            "workspace_bytes": workspace_bytes,
            "unreused_workspace_bytes": sum(allocation.size for allocation in self.allocations.values()),
            "weight_bytes": len(self.weights),
            "output_bytes": output_bytes,
        }
        return self.generate(inputs, outputs, stats)

    def record_usage(self, inputs: Sequence[Value]) -> None:
        for value in inputs:
            for name in value.deps:
                self.allocations[name].end = self.kernels

    def promote_output(self, index: int, value: Value) -> bool:
        if value.name not in self.allocations or value.name in self.output_buffers:
            return False
        self.output_buffers[value.name] = index
        return True

    def generate(self, inputs: list[TensorSpec], outputs: list[TensorSpec], stats: Stats) -> GeneratedCode:
        raise NotImplementedError
