from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import sysconfig
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Protocol, Self, TypedDict, cast

import numpy as np
from ml_dtypes import bfloat16
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor


type Array = NDArray[Any]


class Stats(TypedDict):
    kernels: int
    expressions: int
    folded_constants: int
    matrix_contractions: int
    workspace_bytes: int
    unreused_workspace_bytes: int
    weight_bytes: int
    output_bytes: int
    backend: NotRequired[str]
    math_library: NotRequired[str]
    expert_contractions: NotRequired[int]
    bf16_kernels: NotRequired[int]
    tiled_matmuls: NotRequired[int]
    parallel_reductions: NotRequired[int]
    unique_kernels: NotRequired[int]
    input_bytes: NotRequired[int]
    metal_buffer_bytes: NotRequired[int]
    cuda_buffer_bytes: NotRequired[int]


class TensorSpec(TypedDict):
    name: str
    shape: list[int]
    dtype: Literal["float32", "bfloat16", "int64"]


class ArtifactManifest(TypedDict):
    version: int
    python_abi: str
    library: str
    weights: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    stats: Stats


def signature(inputs: list[TensorSpec], outputs: list[TensorSpec]) -> str:
    return hashlib.sha256(json.dumps({"inputs": inputs, "outputs": outputs}, sort_keys=True).encode()).hexdigest()


class ModelProgram(Protocol):
    @property
    def device(self) -> str: ...

    def run(self, inputs: list[Array], outputs: list[Array]) -> None: ...


class Program:
    def __init__(self, directory: str | Path) -> None:
        manifest, native = self._load(directory)
        self._lock = threading.Lock()
        self._model: ModelProgram | None = native
        self._inputs = manifest["inputs"]
        self._outputs = manifest["outputs"]
        self._stats = manifest["stats"]
        self._device = native.device

    @staticmethod
    def _load(directory: str | Path) -> tuple[ArtifactManifest, ModelProgram]:
        directory = Path(directory).resolve()
        manifest = cast(ArtifactManifest, json.loads((directory / "model.json").read_text()))
        if manifest["version"] != 2 or manifest.get("python_abi") != sysconfig.get_config_var("SOABI"):
            raise ValueError("Incompatible Chroma artifact; recompile it with this Python interpreter")
        name = manifest["library"].split(".")[0]
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, directory / manifest["library"])
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load the native model extension")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
        if module.signature != signature(manifest["inputs"], manifest["outputs"]):
            raise ValueError("Artifact tensor metadata does not match its native library")
        model = cast(Callable[[str], ModelProgram], module.Model)
        return manifest, model(str(directory / manifest["weights"]))

    @property
    def device(self) -> str:
        return self._device

    @property
    def inputs(self) -> list[TensorSpec]:
        return self._inputs

    @property
    def outputs(self) -> list[TensorSpec]:
        return self._outputs

    @property
    def stats(self) -> Stats:
        return self._stats

    @staticmethod
    def _array(value: Array | Tensor) -> Array:
        if isinstance(value, np.ndarray):
            array = value
        else:
            tensor = value.detach()
            if str(tensor.dtype) == "torch.bfloat16":
                import torch

                array = tensor.view(torch.uint16).numpy().view(bfloat16)
            else:
                array = tensor.numpy()
        return np.require(array, requirements=["C", "A"])

    def __call__(self, *inputs: Array | Tensor, out: Array | Sequence[Array] | None = None) -> Array | tuple[Array, ...]:
        arrays = [self._array(value) for value in inputs]
        if out is None:
            res: list[Array] = [np.empty(spec["shape"], dtype=spec["dtype"]) for spec in self._outputs]
        else:
            res = [out] if isinstance(out, np.ndarray) else list(out)
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model is closed")
            self._model.run(arrays, res)
        return res[0] if len(res) == 1 else tuple(res)

    def close(self) -> None:
        with self._lock:
            self._model = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.close()
