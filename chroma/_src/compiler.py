import operator
import os
import sys
from collections.abc import Callable, Mapping
from typing import Any, Self, cast

import torch
from torch.export import ExportedProgram
from torch.fx import Node, map_arg
from torch.utils._pytree import tree_leaves

from chroma._src.build import build
from chroma._src.codegen.base import GeneratedCode
from chroma._src.codegen.cpu import CPUCodegen
from chroma._src.codegen.cuda import CUDACodegen
from chroma._src.codegen.metal import MetalCodegen
from chroma._src.graph import prepare, validate_tensor
from chroma._src.lowering import OPERATIONS, lower
from chroma._src.program import Program, TensorSpec
from chroma._src.value import Value

type ExportSource = ExportedProgram | str | os.PathLike[str]
type Backend = CPUCodegen | CUDACodegen | MetalCodegen
type Lowered = Value | tuple[Value, ...]
_NOT_CONSTANT = object()


def consumers(node: Node) -> set[Node]:
    pending = list(node.users)
    users: set[Node] = set()
    visited: set[Node] = set()
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        operation = OPERATIONS.get(str(user.target))
        if operation == "layout" or user.target == operator.getitem:
            pending.extend(user.users)
        elif operation != "metadata":
            users.add(user)
    return users


def arguments(node: Node, values: Mapping[Node, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args = map_arg(node.args, values.__getitem__)
    kwargs = map_arg(node.kwargs, values.__getitem__)
    return cast(tuple[Any, ...], args), dict(kwargs)


class Compiler:
    def __init__(self, program: ExportedProgram, codegen: Backend) -> None:
        self.program = program
        self.codegen = codegen
        self.values: dict[Node, Lowered] = {}
        self.constants: dict[Node, Any] = {}
        self.state_constants: set[Node] = set()
        self.fold_bytes = 0
        self.folded: set[str] = set()
        self.inputs: list[TensorSpec] = []
        self.outputs: list[TensorSpec] = []

    @classmethod
    def from_export(cls, program: ExportSource, codegen: Backend) -> Self:
        compiler = cls(prepare(program), codegen)
        placeholders = [node for node in compiler.program.graph.nodes if node.op == "placeholder"]

        for node, spec in zip(placeholders, compiler.program.graph_signature.input_specs, strict=True):
            if spec.kind.name == "USER_INPUT":
                tensor = cast(torch.Tensor, node.meta["val"])
                value = codegen.extern(tensor, "input", len(compiler.inputs))
                compiler.inputs.append(value.spec(node.name))
                compiler.values[node] = value
            else:
                state = compiler.program.constants if spec.kind.name == "CONSTANT_TENSOR" else compiler.program.state_dict
                tensor = cast(torch.Tensor, state[spec.target]).detach()
                validate_tensor(tensor)
                compiler.constants[node] = tensor
                compiler.state_constants.add(node)

        return compiler

    def fold_constant(self, node: Node) -> Any:
        if node in self.constants:
            return self.constants[node]
        target = str(node.target)
        if node.op != "call_function" or OPERATIONS.get(target) == "metadata":
            return _NOT_CONSTANT

        size = 0
        if any(n in self.state_constants for n in node.all_input_nodes):
            self.state_constants.add(node)
            tensors = tree_leaves(node.meta.get("val"))
            size = sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))
            if size > 1024**2 or self.fold_bytes + size > 16 * 1024**2:
                return _NOT_CONSTANT

        if target == "aten.new_full.default":
            tensor = cast(torch.Tensor, node.meta["val"])
            return torch.full(tuple(tensor.shape), cast(float, node.args[2]), dtype=tensor.dtype)

        args, kwargs = arguments(node, self.constants)
        if any(value is _NOT_CONSTANT for value in tree_leaves((args, kwargs))):
            return _NOT_CONSTANT
        with torch.no_grad():
            value = cast(Callable[..., Any], node.target)(*args, **kwargs)
        self.fold_bytes += size
        return value

    def emit(self, node: Node) -> Lowered:
        constant = self.constants[node]
        target = str(node.target)
        is_layout = OPERATIONS.get(target) == "layout"
        if constant is not _NOT_CONSTANT and target in {"aten.to.dtype", "aten._to_copy.default"}:
            is_layout = False

        if not is_layout and isinstance(constant, torch.Tensor):
            if node.op == "call_function":
                self.folded.add(node.name)
            res: Lowered = self.codegen.weight(constant)
        elif not is_layout and isinstance(constant, (tuple, list)):
            res = tuple(self.codegen.weight(t) for t in constant)
        else:
            args, kwargs = arguments(node, self.values)
            res = lower(self.codegen, node.target, node.meta["val"], args, kwargs)
        if isinstance(res, Value) and OPERATIONS.get(target) == "elementwise" and len(consumers(node)) > 1:
            res = self.codegen.materialize(res)
        self.values[node] = res
        return res

    def lower(self) -> GeneratedCode:
        nodes = list(self.program.graph.nodes)
        for node in nodes:
            self.constants[node] = self.fold_constant(node)

        output = next(node for node in nodes if node.op == "output")
        needed = set(output.all_input_nodes)
        for node in reversed(nodes):
            if node in needed and (self.constants[node] is _NOT_CONSTANT or OPERATIONS.get(str(node.target)) == "layout"):
                needed.update(node.all_input_nodes)
        for node in nodes:
            if node in needed and node not in self.values:
                self.emit(node)

        for index, node in enumerate(cast(tuple[Node, ...], output.args[0])):
            value = cast(Value, self.values[node])
            self.outputs.append(value.spec(node.name))
            self.codegen.output(index, value)

        return self.codegen.finish(self.inputs, self.outputs, len(self.folded))


def compile_for_backend(program: ExportSource, output: str | os.PathLike[str], *, blas: bool = True, backend: str = "cpu") -> Program:
    if backend == "cpu":
        codegen: Backend = CPUCodegen(blas=blas)
    elif backend == "cuda":
        codegen = CUDACodegen(blas=blas)
    elif backend == "metal":
        if sys.platform != "darwin":
            raise RuntimeError("The Metal backend requires macOS")
        codegen = MetalCodegen()
    else:
        raise ValueError(f"Unknown runtime backend: {backend}")
    return build(Compiler.from_export(program, codegen).lower(), output)
