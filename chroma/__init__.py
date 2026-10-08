from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from chroma._src.decoder import Decoder
from chroma._src.jit import jit
from chroma._src.program import Program

if TYPE_CHECKING:
    from torch.export import ExportedProgram


def compile(
    input: ExportedProgram | str | Path,
    output: str | Path,
    *,
    blas: bool = True,
    backend: Literal["cpu", "cuda", "metal"] = "cpu",
) -> Program:
    from chroma._src.compiler import compile_for_backend

    return compile_for_backend(input, output, blas=blas, backend=backend)


__all__ = ["Decoder", "Program", "compile", "jit"]
