import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from chroma._src.program import TensorSpec

type Shape = tuple[int, ...]
FLOAT_DTYPES = (torch.float32, torch.bfloat16)
DTYPES: dict[torch.dtype, tuple[str, Literal["float32", "bfloat16", "int64"]]] = {
    torch.float32: ("float", "float32"),
    torch.bfloat16: ("chroma::bfloat16_t", "bfloat16"),
    torch.int64: ("int64_t", "int64"),
}


@dataclass(frozen=True)
class View:
    base: Value | None
    strides: Shape
    offset: int = 0


@dataclass
class Value:
    name: str
    shape: Shape
    dtype: torch.dtype
    deps: frozenset[str] = frozenset()
    stored: bool = False
    depth: int = 0
    view: View | None = None

    def spec(self, name: str) -> TensorSpec:
        return {"name": name, "shape": list(self.shape), "dtype": DTYPES[self.dtype][1]}


type Operand = Value | float


def strides(shape: Sequence[int]) -> Shape:
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


def axis(value: int, rank: int) -> int:
    # A scalar only has axis 0, whatever got passed in.
    return value % rank if rank else 0


def contraction(eqn: str, values: Sequence[Value]) -> tuple[list[str], str, dict[str, int], list[str]]:
    eqn = eqn.replace(" ", "")
    if len(values) != 2 or "->" not in eqn or "..." in eqn:
        raise NotImplementedError(f"einsum {eqn!r}: two operands, explicit output, no ellipses")
    if any(v.dtype not in FLOAT_DTYPES for v in values):
        raise NotImplementedError("Einsum requires float tensors")

    inputs, output = eqn.split("->")
    labels = inputs.split(",")
    sizes = {}

    for label, value in zip(labels, values, strict=True):
        for letter, size in zip(label, value.shape, strict=True):
            sizes[letter] = max(size, sizes.get(letter, 1))
    return labels, output, sizes, sorted(set(sizes) - set(output))


def matrix_axes(labels: Sequence[str], output: str, reduction: Sequence[str]) -> tuple[list[str], str, str, str] | None:
    left, right = map(set, labels)
    rows, columns = left - right, right - left
    batch = [letter for letter in output if letter in left & right]

    if len(reduction) != 1 or len(rows) != 1 or len(columns) != 1:
        return None

    m, n, k = next(iter(rows)), next(iter(columns)), reduction[0]
    if set(output) != set(batch) | {m, n} or k not in left & right:
        return None

    return batch, m, n, k
