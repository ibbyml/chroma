from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from functools import partial, wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, overload

from chroma._src.program import Array

if TYPE_CHECKING:
    from torch import Tensor


type TensorFunction = Callable[..., Tensor | tuple[Tensor, ...]]
type CompiledFunction = Callable[..., Array | tuple[Array, ...]]


@overload
def jit(
    fn: TensorFunction,
    *,
    backend: Literal["cpu", "cuda", "metal"] = "cpu",
    blas: bool = True,
) -> CompiledFunction: ...


@overload
def jit(
    fn: None = None,
    *,
    backend: Literal["cpu", "cuda", "metal"] = "cpu",
    blas: bool = True,
) -> Callable[[TensorFunction], CompiledFunction]: ...


def jit(
    fn: TensorFunction | None = None,
    *,
    backend: Literal["cpu", "cuda", "metal"] = "cpu",
    blas: bool = True,
) -> CompiledFunction | Callable[[TensorFunction], CompiledFunction]:
    if fn is None:
        return partial(jit, backend=backend, blas=blas)
    if not callable(fn):
        raise TypeError(f"callable, got {type(fn).__name__}")

    import torch

    from chroma._src.compiler import compile_for_backend

    class Function(torch.nn.Module):
        def __init__(self, fn: TensorFunction) -> None:
            super().__init__()
            self.fn = fn

        def forward(self, *args: Tensor) -> Tensor | tuple[Tensor, ...]:
            return self.fn(*args)

    module = fn if isinstance(fn, torch.nn.Module) else Function(fn)
    directory = TemporaryDirectory(prefix="chroma-jit-")
    programs = {}
    lock = threading.Lock()

    @wraps(fn)
    def run(*args: Tensor, out: Array | Sequence[Array] | None = None) -> Array | tuple[Array, ...]:
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in args):
            raise TypeError("positional CPU Torch tensor")
        key = tuple((tuple(value.shape), value.dtype) for value in args)
        with lock:
            if key not in programs:
                with torch.no_grad():
                    examples = tuple(value.detach().contiguous() for value in args)
                    exported = torch.export.export(module, examples)
                programs[key] = compile_for_backend(exported, Path(directory.name) / str(len(programs)), backend=backend, blas=blas)
            program = programs[key]
        return program(*args, out=out)

    return run
