import operator
import os
from typing import TypeGuard, cast

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind, TensorArgument
from torch.fx import Node
from torch.utils._pytree import tree_leaves

from chroma._src.lowering import OPERATIONS, _rmsnorm
from chroma._src.value import DTYPES


def validate_tensor(tensor: torch.Tensor) -> None:
    if any(type(d) is not int for d in tensor.shape):
        raise ValueError(f"Dynamic shape {tuple(tensor.shape)}")
    if any(d <= 0 for d in tensor.shape):
        raise ValueError(f"empty dim in {tuple(tensor.shape)}")
    if tensor.numel() * tensor.element_size() >= 2**63:
        raise ValueError(f"{tensor.numel()} elements do not fit")
    if tensor.device.type != "cpu":
        raise ValueError(f"export is on {tensor.device.type}")
    if tensor.dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype {tensor.dtype}")


def prepare(program: ExportedProgram | str | os.PathLike[str]) -> ExportedProgram:
    if isinstance(program, (str, os.PathLike)):
        program = torch.export.load(program)
    program = program.run_decompositions({})

    signature = program.graph_signature
    for spec in signature.input_specs:
        if spec.kind == InputKind.USER_INPUT and not isinstance(spec.arg, TensorArgument):
            raise TypeError(f"Only tensor inputs are supported, got {spec.arg}")
    for spec in signature.output_specs:
        if spec.kind != OutputKind.USER_OUTPUT:
            raise ValueError(f"Export mutates {spec.target}; mutation of inputs and buffers is unsupported")
        if not isinstance(spec.arg, TensorArgument):
            raise TypeError(f"Only tensor outputs are supported, got {spec.arg}")

    for node in program.graph.nodes:
        if node.op == "call_function" and node.target != operator.getitem and str(node.target) not in OPERATIONS:
            raise NotImplementedError(f"{node.name}: unsupported operator {node.target}")
        tensors = tree_leaves(node.meta.get("val"))
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                validate_tensor(tensor)
        if node.op not in {"placeholder", "call_function", "output"}:
            raise NotImplementedError(f"{node.name}: unsupported node kind {node.op}")

    fuse_norms(program)

    return program


def fuse_norms(program: ExportedProgram) -> None:
    def is_op(n: object, name: str) -> TypeGuard[Node]:
        return isinstance(n, Node) and str(n.target) == name

    for node in program.graph.nodes:
        if not is_op(node, "aten.mul.Tensor"):
            continue
        product, weight = node.args

        if not is_op(product, "aten.mul.Tensor") or not isinstance(weight, Node):
            continue
        x, inverse = product.args

        if not is_op(inverse, "aten.rsqrt.default"):
            continue
        added = inverse.args[0]

        if not is_op(added, "aten.add.Tensor") or len(added.args) != 2 or added.kwargs:
            continue
        mean, epsilon = added.args

        if not is_op(mean, "aten.mean.dim") or list(mean.args[1:]) != [[-1], True] or not isinstance(epsilon, (int, float)):
            continue
        squared = mean.args[0]

        if not is_op(squared, "aten.pow.Tensor_Scalar") or squared.args != (x, 2):
            continue
        x = cast(Node, x)

        if any(part.meta["val"].dtype != torch.float32 for part in (x, squared, mean, added, inverse, product, node)):
            continue

        if tuple(weight.meta["val"].shape) != (x.meta["val"].shape[-1],):
            continue

        node.target, node.args = _rmsnorm, (x, weight, epsilon)

    program.graph.eliminate_dead_code()
