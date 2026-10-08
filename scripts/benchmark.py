import argparse
import hashlib
import json
import os
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, TypedDict, cast

if TYPE_CHECKING:
    from chroma.models.gpt_oss import Transformer

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
type Backend = Literal["cpu", "cuda", "metal"]


class Timing(TypedDict):
    median_ms: float
    p10_ms: float
    p90_ms: float
    samples_ms: list[float]


def limit_threads(threads: int) -> None:
    # BLAS reads these when it loads, so this runs before NumPy, Torch, or Chroma are imported.
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(threads)


def default_backends() -> list[Backend]:
    return ["cpu", "metal"] if sys.platform == "darwin" else ["cpu"]


def summarize(samples: dict[str, list[float]]) -> dict[str, Timing]:
    import numpy as np

    return {
        name: {
            "median_ms": statistics.median(values),
            "p10_ms": float(np.percentile(values, 10)),
            "p90_ms": float(np.percentile(values, 90)),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }


def elapsed_ms(call: Callable[[], object]) -> float:
    start = time.perf_counter_ns()
    res = call()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    del res
    return elapsed


def source_identity() -> dict[str, str | bool | None]:
    paths = [path for path in (ROOT / "chroma").rglob("*") if path.suffix in {".py", ".h", ".metal"}]
    paths += list((ROOT / "scripts").glob("*.py"))
    paths += [ROOT / name for name in ("pyproject.toml", "uv.lock", "CMakeLists.txt", "CMakePresets.json")]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    return {
        "sha256": digest.hexdigest(),
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(status.stdout) if status.returncode == 0 else None,
    }


def load_model(weights: Path | None = None) -> Transformer:
    from safetensors.torch import load_file

    from scripts.export import make_dev_model

    model = make_dev_model(seed=0)
    if weights is not None:
        model.load_state_dict(load_file(str(weights)))
    return model


def weights_digest(model: Transformer) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def environment_info() -> dict[str, object]:
    import numpy as np
    import pybind11
    import torch

    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    environment: dict[str, object] = {
        "os": platform.platform(),
        "cpu": cpu,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pybind11": pybind11.__version__,
        "compiler": subprocess.check_output([*compiler, "--version"], text=True).splitlines()[0],
        "compiler_command": compiler,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "blas_thread_limits": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }
    nvcc = shlex.split(os.environ.get("NVCC", "nvcc"))
    for name, command in (
        ("nvcc", [*nvcc, "--version"]),
        ("nvidia_smi", ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
    ):
        if command and shutil.which(command[0]):
            res = subprocess.run(command, capture_output=True, text=True, check=False)
            environment[name] = res.stdout.strip() if res.returncode == 0 else None
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-shape FP32 inference against eager Torch.")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backends", nargs="+", choices=["cpu", "cuda", "metal"])
    parser.add_argument("--torch-gpu", action="store_true", help="Include eager Torch on each selected GPU with CPU inputs and outputs")
    parser.add_argument("--weights", type=Path, help="Use one exported weights.safetensors snapshot across machines")
    parser.add_argument("--output", type=Path, default=Path("build/benchmarks/results.json"))
    args = parser.parse_args()
    limit_threads(args.threads)

    import numpy as np
    import torch

    from chroma import compile
    from chroma.models.gpt_oss import Transformer
    from chroma.models.variants import dev

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    eager = load_model(args.weights)
    tokens = torch.arange(args.tokens, dtype=torch.int64) % dev.vocab_size
    array = tokens.numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        start = time.perf_counter()
        program = torch.export.export(eager, (tokens,))
        export_seconds = time.perf_counter() - start

    backends: dict[str, dict[str, object]] = {}
    report: dict[str, object] = {
        "measured_at": datetime.now(UTC).isoformat(),
        "source": source_identity(),
        "environment": environment_info(),
        "model": {
            "config": asdict(dev),
            "parameters": sum(p.numel() for p in eager.parameters()),
            "seed": 0 if args.weights is None else None,
            "tokens": args.tokens,
            "dtype": "float32",
            "weights_sha256": weights_digest(eager),
        },
        "runs": args.runs,
        "warmup": args.warmup,
        "export_seconds": export_seconds,
        "method": "Torch runs eagerly on CPU and optionally on the selected GPU with weights resident on that GPU. "
        "All GPU paths start with CPU inputs and return CPU outputs, including transfers and synchronization. "
        "CUDA Torch matmul uses IEEE FP32 (TF32 disabled). Calls use fixed inputs, warmed caches, and seeded interleaved order. "
        "Timing excludes compilation, export, warmup, validation, and output destruction. "
        "Fresh-output calls include allocation; Chroma out= calls are reported separately. "
        "GPU calls include copies, dispatch, and waiting for completion. Driver caches are not cleared.",
        "validation": {"input_sets": 2, "rtol": 2e-5, "atol": 2e-6},
        "backends": backends,
    }
    with ExitStack() as stack, torch.inference_mode():
        artifacts = Path(stack.enter_context(TemporaryDirectory(prefix="chroma-benchmark-")))
        calls: dict[str, Callable[[], object]] = {"torch_cpu_fresh": lambda: eager(tokens)}
        targets = cast(list[Backend], args.backends or default_backends())
        if args.torch_gpu and not any(target in {"cuda", "metal"} for target in targets):
            parser.error("--torch-gpu requires a CUDA or Metal backend")
        for backend in targets:
            start = time.perf_counter()
            native = stack.enter_context(compile(program, artifacts / backend, backend=backend))
            compile_seconds = time.perf_counter() - start
            for values in (tokens, (tokens * 7919 + 17) % dev.vocab_size):
                np.testing.assert_allclose(native(values.numpy()), eager(values).numpy(), rtol=2e-5, atol=2e-6)
            backends[backend] = {"device": native.device, "compile_and_load_seconds": compile_seconds, "stats": native.stats}
            calls[f"chroma_{backend}_fresh"] = lambda native=native: native(array)
            out = native(array)
            calls[f"chroma_{backend}_reuse"] = lambda native=native, out=out: native(array, out=out)
            print(f"{backend}: validated; compiled and loaded in {compile_seconds:.2f} s", flush=True)
            if args.torch_gpu and backend != "cpu":
                device = torch.device("mps" if backend == "metal" else "cuda")
                if backend == "cuda":
                    torch.backends.cuda.matmul.fp32_precision = "ieee"
                gpu_eager = Transformer(dev, device=device).float().eval().requires_grad_(False)
                gpu_eager.load_state_dict(eager.state_dict())
                for values in (tokens, (tokens * 7919 + 17) % dev.vocab_size):
                    np.testing.assert_allclose(gpu_eager(values.to(device)).cpu().numpy(), eager(values).numpy(), rtol=2e-5, atol=2e-6)
                calls[f"torch_{device.type}_fresh"] = lambda model=gpu_eager, device=device: model(tokens.to(device)).cpu().numpy()
                print(f"torch {device.type}: validated; CPU-to-CPU timing with resident weights", flush=True)

        order, rng = list(calls), random.Random(0)
        for _ in range(args.warmup):
            rng.shuffle(order)
            for name in order:
                calls[name]()
        samples: dict[str, list[float]] = {name: [] for name in calls}
        for _ in range(args.runs):
            rng.shuffle(order)
            for name in order:
                samples[name].append(elapsed_ms(calls[name]))
        timings = summarize(samples)
    report["timings"] = timings
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, timing in timings.items():
        print(f"{name}: {timing['median_ms']:.3f} ms (p10–p90 {timing['p10_ms']:.3f}–{timing['p90_ms']:.3f})")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
