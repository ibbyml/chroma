from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from ml_dtypes import bfloat16
from torch.fx.node import Target

from chroma._src.value import FLOAT_DTYPES, Operand, Value, View, axis, strides

if TYPE_CHECKING:
    from chroma._src.codegen.cpu import CPUCodegen
    from chroma._src.codegen.cuda import CUDACodegen
    from chroma._src.codegen.metal import MetalCodegen

type Operation = Literal[
    "elementwise",
    "layout",
    "linear",
    "gather",
    "reduction",
    "softmax",
    "topk",
    "einsum",
    "cat",
    "triangular",
    "constant",
    "metadata",
]

OPERATIONS: dict[str, Operation] = {
    "aten.abs.default": "elementwise",
    "aten.add.Tensor": "elementwise",
    "aten.clamp.default": "elementwise",
    "aten.cos.default": "elementwise",
    "aten.div.Tensor": "elementwise",
    "aten.exp.default": "elementwise",
    "aten.mul.Tensor": "elementwise",
    "aten.neg.default": "elementwise",
    "aten.pow.Scalar": "elementwise",
    "aten.pow.Tensor_Scalar": "elementwise",
    "aten.reciprocal.default": "elementwise",
    "aten.rsub.Scalar": "elementwise",
    "aten.rsqrt.default": "elementwise",
    "aten.sigmoid.default": "elementwise",
    "aten.sin.default": "elementwise",
    "aten.square.default": "elementwise",
    "aten.sub.Tensor": "elementwise",
    "aten.view.default": "layout",
    "aten.reshape.default": "layout",
    "aten._unsafe_view.default": "layout",
    "aten.unsqueeze.default": "layout",
    "aten.expand.default": "layout",
    "aten.slice.Tensor": "layout",
    "aten.permute.default": "layout",
    "aten.transpose.int": "layout",
    "aten.t.default": "layout",
    "aten.to.dtype": "layout",
    "aten._to_copy.default": "layout",
    "aten.clone.default": "layout",
    "aten.contiguous.default": "layout",
    "aten.split.Tensor": "layout",
    "aten.split_with_sizes.default": "layout",
    "aten.linear.default": "linear",
    "aten.embedding.default": "gather",
    "aten.index.Tensor": "gather",
    "aten.einsum.default": "einsum",
    "aten.mean.dim": "reduction",
    "aten.mean.default": "reduction",
    "aten.sum.default": "reduction",
    "aten.sum.dim_IntList": "reduction",
    "aten.softmax.int": "softmax",
    "aten.topk.default": "topk",
    "aten.cat.default": "cat",
    "aten.arange.default": "constant",
    "aten.arange.start": "constant",
    "aten.arange.start_step": "constant",
    "aten.new_full.default": "constant",
    "aten.full.default": "constant",
    "aten.triu.default": "triangular",
    "aten.tril.default": "triangular",
    "aten._assert_tensor_metadata.default": "metadata",
}


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight


def argument(args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    return args[index] if len(args) > index else kwargs.get(name, default)


def layout_view(
    codegen: CPUCodegen | CUDACodegen | MetalCodegen, meta: torch.Tensor, x: Value, mapping: Sequence[int], offset: int
) -> Value:
    if x.view is None:
        return codegen.reindex(meta, x, mapping, offset)
    root = x.view.base or x
    offset += x.view.offset
    shape = tuple(meta.shape)
    if all(d == 1 or a == b for d, a, b in zip(shape, mapping, strides(shape))):
        return codegen.alias(meta, root, offset)
    res = codegen.reindex(meta, root, mapping, offset)
    res.view = View(root, tuple(mapping), offset)
    return res


def lower(
    codegen: CPUCodegen | CUDACodegen | MetalCodegen,
    target: Target,
    meta: torch.Tensor | Sequence[torch.Tensor],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Value | tuple[Value, ...]:
    if target == operator.getitem:
        return args[0][args[1]]

    name = str(target)
    operation = OPERATIONS.get(name)
    tensor = cast(torch.Tensor, meta)
    if target == _rmsnorm:
        return codegen.norm(tensor, args[0], args[1], args[2])

    if operation == "layout":
        x = cast(Value, args[0])
        if name in {"aten.to.dtype", "aten._to_copy.default"}:
            if x.dtype != tensor.dtype:
                if x.dtype not in FLOAT_DTYPES or tensor.dtype not in FLOAT_DTYPES:
                    raise NotImplementedError(f"{x.dtype} -> {tensor.dtype}")
                return codegen.cast(tensor, x)
            return x
        if name in {"aten.clone.default", "aten.contiguous.default"}:
            return codegen.materialize(x)
        if name in {"aten.view.default", "aten.reshape.default", "aten._unsafe_view.default"}:
            return codegen.reshape(tensor, x)

        mapping = list(x.view.strides if x.view is not None else strides(x.shape))
        offset = 0
        if name in {"aten.split.Tensor", "aten.split_with_sizes.default"}:
            dim = argument(args, kwargs, 2, "dim", 0)
            dim = axis(dim, len(x.shape))
            res = []
            for part in cast(Sequence[torch.Tensor], meta):
                res.append(layout_view(codegen, part, x, mapping, offset))
                offset += part.shape[dim] * mapping[dim]
            return tuple(res)

        shape = tuple(tensor.shape)
        if name == "aten.unsqueeze.default":
            mapping.insert(axis(args[1], len(shape)), 0)
        elif name == "aten.expand.default":
            pad = len(shape) - len(x.shape)
            mapping = [0] * pad + mapping
            input_shape = (1,) * pad + x.shape
            mapping = [s if a == b else 0 for s, a, b in zip(mapping, input_shape, shape)]
        elif name == "aten.slice.Tensor":
            dim = argument(args, kwargs, 1, "dim", 0)
            start = argument(args, kwargs, 2, "start")
            end = argument(args, kwargs, 3, "end")
            step = argument(args, kwargs, 4, "step", 1)
            dim = axis(dim, len(x.shape))
            start, _, step = slice(start, end, step).indices(x.shape[dim])
            if step <= 0:
                raise NotImplementedError(f"slice step {step}")
            offset = start * mapping[dim]
            mapping[dim] *= step
        else:
            order = list(range(len(x.shape)))
            if name == "aten.permute.default":
                order = [axis(d, len(x.shape)) for d in args[1]]
            elif len(order) > 1:
                if name == "aten.t.default":
                    a, b = 0, 1
                else:
                    a = axis(args[1], len(order))
                    b = axis(args[2], len(order))
                order[a], order[b] = order[b], order[a]
            mapping = [mapping[d] for d in order]
        return layout_view(codegen, tensor, x, mapping, offset)

    if operation == "elementwise":
        if tensor.dtype not in FLOAT_DTYPES:
            raise NotImplementedError(name)
        if name == "aten.rsub.Scalar":
            alpha = argument(args, kwargs, 2, "alpha", 1)
            kwargs = {**kwargs, "alpha": alpha}
        if name == "aten.clamp.default":
            low = argument(args, kwargs, 1, "min")
            high = argument(args, kwargs, 2, "max")
            kwargs = {"min": low, "max": high}
            if any(bound is not None and math.isnan(bound) for bound in (low, high)):
                name = "aten.mul.Tensor"
                args = (args[0], math.nan)
                kwargs = {}
        operands = cast(tuple[Operand, ...], tuple(value for value in args if value is not None))
        alpha = kwargs.get("alpha", 1)
        if tensor.dtype == torch.bfloat16 and name in {
            "aten.add.Tensor",
            "aten.sub.Tensor",
            "aten.rsub.Scalar",
            "aten.pow.Tensor_Scalar",
            "aten.pow.Scalar",
        }:
            operands = tuple(value if isinstance(value, Value) else float(bfloat16(value)) for value in operands)
            alpha = float(bfloat16(alpha))
        return codegen.elementwise(tensor, name, operands, alpha=alpha, low=kwargs.get("min"), high=kwargs.get("max"))

    if operation == "gather":
        negative = name == "aten.index.Tensor"
        indices = args[1]
        if negative:
            if len(indices) != 1 or not isinstance(indices[0], Value):
                raise NotImplementedError("index: one tensor, axis 0")
            indices = indices[0]
        if indices.dtype != torch.int64:
            raise NotImplementedError("index wants int64")
        return codegen.gather(tensor, args[0], indices, negative)

    if operation == "linear":
        if any(v.dtype not in FLOAT_DTYPES for v in args[:2]) or tensor.dtype not in FLOAT_DTYPES:
            raise NotImplementedError(name)
        bias = argument(args, kwargs, 2, "bias")
        return codegen.linear(tensor, args[0], args[1], bias)

    if operation == "reduction":
        x = cast(Value, args[0])
        if x.dtype not in FLOAT_DTYPES or tensor.dtype not in FLOAT_DTYPES:
            raise NotImplementedError(name)
        dimensions = argument(args, kwargs, 1, "dim", [])
        dimensions = dimensions or range(len(x.shape))
        axes = [] if not x.shape else sorted({axis(d, len(x.shape)) for d in dimensions})
        keepdim = argument(args, kwargs, 2, "keepdim", False)
        mean = name.startswith("aten.mean.")
        if not mean and x.dtype != tensor.dtype:
            x = codegen.cast(tensor.new_empty(x.shape), x)
        return codegen.reduction(tensor, x, axes, keepdim, mean)

    if operation == "einsum":
        return codegen.einsum(tensor, args[0], args[1])

    if operation == "softmax":
        x = cast(Value, args[0])
        if x.dtype not in FLOAT_DTYPES or tensor.dtype not in FLOAT_DTYPES:
            raise NotImplementedError(name)
        dim = axis(args[1], len(x.shape))
        if x.dtype != tensor.dtype:
            x = codegen.cast(tensor.new_empty(x.shape), x)
        if x.shape:
            return codegen.softmax(tensor, x, dim)
        vector = tensor.new_empty((1,))
        res = codegen.softmax(vector, codegen.reshape(vector, x), 0)
        return codegen.reshape(tensor, res)

    if operation == "topk":
        x = cast(Value, args[0])
        dim = argument(args, kwargs, 2, "dim", -1)
        dim = axis(dim, len(x.shape))
        width = x.shape[dim] if x.shape else 1
        if x.dtype not in FLOAT_DTYPES or width > 4096:
            raise NotImplementedError(f"topk axis {width}")
        largest = argument(args, kwargs, 3, "largest", True)
        sort = argument(args, kwargs, 4, "sorted", True)
        tensors = cast(Sequence[torch.Tensor], meta)
        if x.shape:
            return codegen.topk(tensors, x, dim, largest, sort)
        vector = tensors[0].new_empty((1,), dtype=x.dtype)
        values, indices = codegen.topk([part.new_empty((1,)) for part in tensors], codegen.reshape(vector, x), 0, largest, sort)
        return codegen.reshape(tensors[0], values), codegen.reshape(tensors[1], indices)

    if operation == "cat":
        dim = argument(args, kwargs, 1, "dim", 0)
        dim = axis(dim, len(tensor.shape))
        return codegen.concatenate(tensor, args[0], dim)

    if operation == "triangular":
        diagonal = argument(args, kwargs, 1, "diagonal", 0)
        upper = name == "aten.triu.default"
        return codegen.triangular(tensor, args[0], diagonal, upper)

    raise NotImplementedError(f"Unsupported runtime operation {target}")
