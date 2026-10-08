import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import uuid
from pathlib import Path

import pybind11

from chroma._src.codegen.base import GeneratedCode
from chroma._src.program import Program, TensorSpec, signature

_ROOT = Path(__file__).resolve().parent
_HOST_FLAGS = ["-fPIC", "-fvisibility=hidden", "-ffp-contract=off"]


def compiler_command(source_name: str) -> list[str]:
    if not source_name.endswith(".cu"):
        return [*shlex.split(os.environ.get("CXX", "c++")), "-std=c++20", "-O3", "-shared", *_HOST_FLAGS]
    if sys.platform != "linux":
        raise RuntimeError("CUDA compilation requires Linux")
    command = shlex.split(os.environ.get("NVCC", "nvcc"))
    if not command or shutil.which(command[0]) is None:
        raise RuntimeError("CUDA compilation requires nvcc; set NVCC or add it to PATH")
    return [*command, "-std=c++20", "-O3", "-shared", "--cudart=shared", "-Xcompiler", ",".join(_HOST_FLAGS)]


def specs(tensors: list[TensorSpec]) -> str:
    values = []
    for tensor in tensors:
        shape = ", ".join(map(str, tensor["shape"]))
        values.append(f"{{{{{shape}}}, {json.dumps(tensor['dtype'])}}}")
    return "{" + ", ".join(values) + "}"


def artifact_files(directory: Path) -> set[str]:
    try:
        manifest = json.loads((directory / "model.json").read_text())
        names = {manifest["library"], manifest["weights"]}
    except OSError, ValueError, KeyError, TypeError:
        return set()
    return {name for name in names if isinstance(name, str) and name == Path(name).name}


def build(generated: GeneratedCode, output: str | os.PathLike[str]) -> Program:
    inputs, outputs, weights = generated.inputs, generated.outputs, generated.weights
    files = dict(generated.files)
    files[generated.source_name] += f'''\nPYBIND11_MODULE(CHROMA_MODULE, m) {{
    m.attr("signature") = "{signature(inputs, outputs)}";
    chroma::bind_model<ModelProgram>(m, create_model, {specs(inputs)}, {specs(outputs)});
}}
'''

    module = f"_chroma_{uuid.uuid4().hex}"
    library = module + sysconfig.get_config_var("EXT_SUFFIX")
    weights_name = f"weights-{hashlib.sha256(weights).hexdigest()[:20]}.bin"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stale = artifact_files(output)

    with tempfile.TemporaryDirectory(prefix=".compile-", dir=output) as temporary:
        stage = Path(temporary)
        for name, text in files.items():
            (stage / name).write_text(text)
        command = [
            *compiler_command(generated.source_name),
            f"-DCHROMA_MODULE={module}",
            "-I",
            str(_ROOT),
            "-I",
            pybind11.get_include(),
            "-I",
            sysconfig.get_path("include"),
            str(stage / generated.source_name),
            "-o",
            str(stage / library),
            *generated.flags,
        ]
        if sys.platform == "darwin":
            command += ["-undefined", "dynamic_lookup"]
        res = subprocess.run(command, capture_output=True, text=True, check=False)
        if res.returncode:
            raise RuntimeError(f"Native compilation failed:\n{res.stderr}")

        (stage / weights_name).write_bytes(weights)
        manifest = {
            "version": 2,
            "python_abi": sysconfig.get_config_var("SOABI"),
            "library": library,
            "weights": weights_name,
            "inputs": inputs,
            "outputs": outputs,
            "stats": generated.stats,
        }
        (stage / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")

        model = Program(stage)
        try:
            for name in (library, weights_name, *files, "model.json"):
                os.replace(stage / name, output / name)
        except BaseException:
            model.close()
            raise

    # A loaded Program keeps its library mapped and weights in memory, so replaced files can go.
    for name in stale - {library, weights_name}:
        (output / name).unlink(missing_ok=True)
    return model
