"""Pool the reports in docs/benchmarks/ into summary.json and refresh the numbers the docs quote from it."""

import argparse
import json
import re
import statistics
import sys
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs/benchmarks"
SUMMARY = REPORTS / "summary.json"
README = ROOT / "README.md"
BENCHMARKS_DOC = ROOT / "docs/benchmarks.md"

# Target -> forward report, decode report, Chroma backend, eager Torch timing.
TARGETS = {
    "m3_cpu": ("benchmark", "decode-benchmark", "cpu", "torch_cpu_fresh"),
    "m3_metal": ("benchmark", "decode-benchmark", "metal", "torch_mps_fresh"),
    "a100": ("cuda-benchmark", "cuda-decode-benchmark", "cuda", "torch_cuda_fresh"),
}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}

type Json = dict[str, Any]


def text(value: float, places: int) -> str:
    return str(Decimal(repr(value)).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP))


def grouped(value: int) -> str:
    return f"{value:,}"


def stat(samples: list[float]) -> Json:
    median = statistics.median(samples)
    return {
        "median_ms": round(median, 7),
        "p10_ms": round(float(np.percentile(samples, 10)), 7),
        "p90_ms": round(float(np.percentile(samples, 90)), 7),
        "median": text(median, 2),
    }


def load(stem: str) -> tuple[Json, Json]:
    first, second = (json.loads((REPORTS / f"{stem}{suffix}.json").read_text()) for suffix in ("", "-repeat"))
    for key in ("environment", "model", "source"):
        if first[key] != second[key]:
            raise ValueError(f"{stem}.json and {stem}-repeat.json differ in {key}; rerun both")
    return first, second


def pooled(pair: tuple[Json, Json], *path: str) -> list[float]:
    samples = []
    for report in pair:
        for key in path:
            report = report[key]
        samples += report["samples_ms"]
    return samples


def measured(reports: tuple[Json, ...]) -> tuple[str, str]:
    days = sorted({datetime.fromisoformat(report["measured_at"]).date() for report in reports})
    first, last = days[0], days[-1]

    def day(d: date) -> str:
        return f"{d:%B} {d.day}"

    if first == last:
        label = f"{day(first)}, {first.year}"
    elif (first.year, first.month) == (last.year, last.month):
        label = f"{day(first)}–{last.day}, {first.year}"
    else:
        label = f"{day(first)} – {day(last)}, {last.year}"
    return last.isoformat(), label


def environment(report: Json) -> tuple[str, str]:
    env = report["environment"]
    compiler = env["compiler"].splitlines()[0]
    if match := re.match(r"Apple clang version (\d+)", compiler):
        compiler = f"Apple clang {match[1]}"
    elif match := re.search(r"(\d+)\.(\d+)\.\d+\s*$", compiler):
        compiler = f"GCC {match[1]}.{match[2]}"
    software = [f"Python {env['python']}", f"Torch {env['torch']}"]
    if match := re.search(r"release (\d+\.\d+)", env.get("nvcc", "")):
        software.append(f"CUDA {match[1]}")
    if "nvidia_smi" in env:
        hardware = env["nvidia_smi"].split(",")[0]
    else:
        hardware = f"{env['cpu']}, macOS {env['os'].split('-')[1]}"
    return hardware, ", ".join([*software, compiler])


def summarize() -> Json:
    forward_reports = {stem: load(stem) for stem, _, _, _ in TARGETS.values()}
    decode_reports = {stem: load(stem) for _, stem, _, _ in TARGETS.values()}
    first = forward_reports["benchmark"][0]
    if len({report[0]["model"]["weights_sha256"] for report in forward_reports.values()}) != 1:
        raise ValueError("Forward reports use different weights")

    forward: Json = {}
    decode: Json = {}
    stats = {}
    for name, (forward_stem, decode_stem, backend, torch_timing) in TARGETS.items():
        pair, decode_pair = forward_reports[forward_stem], decode_reports[decode_stem]
        hardware, software = environment(pair[0])
        if environment(decode_pair[0]) != (hardware, software):
            raise ValueError(f"{forward_stem} and {decode_stem} reports come from different environments")
        stats[name] = pair[0]["backends"][backend]["stats"]
        day, day_label = measured((*pair, *decode_pair))
        chroma = stat(pooled(pair, "timings", f"chroma_{backend}_fresh"))
        torch = stat(pooled(pair, "timings", torch_timing))
        compile_seconds = statistics.mean(report["backends"][backend]["compile_and_load_seconds"] for report in pair)
        forward[name] = {
            "hardware": hardware,
            "software": software,
            "measured_date": day,
            "measured": day_label,
            "chroma": chroma,
            "chroma_reuse": stat(pooled(pair, "timings", f"chroma_{backend}_reuse")),
            "torch": torch,
            "speedup_ratio": round(torch["median_ms"] / chroma["median_ms"], 4),
            "speedup": text(torch["median_ms"] / chroma["median_ms"], 1),
            "compile_s": round(compile_seconds, 2),
            "compile": text(compile_seconds, 0),
            "kernels": stats[name]["kernels"],
        }
        timings = {kind: stat(pooled(decode_pair, "backends", backend, "timings", kind)) for kind in ("cached", "uncached", "prefill")}
        decode_seconds = statistics.mean(report["backends"][backend]["compile_and_load_all_graphs_seconds"] for report in decode_pair)
        ratio = timings["uncached"]["median_ms"] / timings["cached"]["median_ms"]
        decode[name] = {
            "measured_date": day,
            "measured": day_label,
            **timings,
            "speedup_ratio": round(ratio, 4),
            "speedup": text(ratio, 1),
            "compile_s": round(decode_seconds, 2),
            "compile": text(decode_seconds, 0),
        }

    memory = {(s["workspace_bytes"], s["unreused_workspace_bytes"], s["weight_bytes"]) for s in stats.values()}
    if len(memory) != 1:
        raise ValueError(f"Backends plan different storage: {memory}")
    arena, unreused, weights = memory.pop()
    config = first["model"]["config"]
    decode_first = decode_reports["decode-benchmark"][0]
    return {
        "generated_by": "scripts/summarize_benchmarks.py from docs/benchmarks/*.json; do not edit",
        "model": {
            "parameter_count": first["model"]["parameters"],
            "parameters": grouped(first["model"]["parameters"]),
            "parameters_millions": text(first["model"]["parameters"] / 1e6, 2),
            "layers": config["num_hidden_layers"],
            "hidden_size": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "experts": config["num_experts"],
            "experts_per_token": config["experts_per_token"],
            "vocab_count": config["vocab_size"],
            "vocab": grouped(config["vocab_size"]),
        },
        "memory": {
            "arena_bytes": arena,
            "unreused_bytes": unreused,
            "weight_bytes": weights,
            "arena": f"{text(arena / 1024, 0)} KiB",
            "unreused": f"{text(unreused / 1024**2, 2)} MiB",
            "weights": f"{text(weights / 1024**2, 0)} MiB",
            "arena_exact": f"{grouped(arena)} bytes",
            "unreused_exact": f"{grouped(unreused)} bytes",
            "reduction": text(unreused / arena, 0),
        },
        "forward": {
            "tokens": first["model"]["tokens"],
            "runs": 2,
            "calls": first["runs"],
            "warmup": first["warmup"],
            "targets": forward,
        },
        "decode": {
            "context": decode_first["context"],
            "prompt": decode_first["prompt"],
            "steps": decode_first["steps"],
            "rounds": decode_first["rounds"],
            "runs": 2,
            "targets": decode,
        },
    }


def readme_block(s: Json) -> str:
    f, d, run = s["forward"]["targets"], s["decode"]["targets"], s["forward"]
    macos = f["m3_cpu"]["hardware"].split(", ")[1]
    return f"""Eight-token FP32 forward passes with identical weights, pooled from {WORDS.get(run["runs"], run["runs"])} runs of {run["calls"]} calls after {run["warmup"]} warmups:

| Target | Chroma | Torch eager |
| --- | ---: | ---: |
| Apple M3 CPU, {macos} | {f["m3_cpu"]["chroma"]["median"]} ms | {f["m3_cpu"]["torch"]["median"]} ms |
| Apple M3 Metal / MPS | {f["m3_metal"]["chroma"]["median"]} ms | {f["m3_metal"]["torch"]["median"]} ms |
| NVIDIA A100 | {f["a100"]["chroma"]["median"]} ms | {f["a100"]["torch"]["median"]} ms |

Calls return fresh NumPy arrays, and GPU times include host transfers and synchronization. On this graph, {s["memory"]["unreused"]} of distinct temporaries fit in a {s["memory"]["arena"]} arena.

With a {s["decode"]["context"]}-token KV cache, a decode step takes {d["m3_cpu"]["cached"]["median"]} ms on CPU, {d["m3_metal"]["cached"]["median"]} ms on Metal, and {d["a100"]["cached"]["median"]} ms on the A100, against {d["m3_cpu"]["uncached"]["median"]}, {d["m3_metal"]["uncached"]["median"]}, and {d["a100"]["uncached"]["median"]} ms for Chroma's own padded, uncached graph. Method, compile times, and the raw reports are in [docs/benchmarks.md](docs/benchmarks.md)."""


def environments_block(s: Json) -> str:
    f = s["forward"]["targets"]
    return f"""| Reports | Hardware | Software |
| --- | --- | --- |
| `benchmark*.json`, `decode-benchmark*.json` | {f["m3_cpu"]["hardware"]} | {f["m3_cpu"]["software"]} |
| `cuda-*.json` | {f["a100"]["hardware"]} | {f["a100"]["software"]} |"""


def forward_block(s: Json) -> str:
    f, m = s["forward"]["targets"], s["memory"]
    rows = (("Apple M3 CPU", "m3_cpu"), ("Apple M3 Metal (Torch: MPS)", "m3_metal"), ("NVIDIA A100", "a100"))
    table = "\n".join(
        f"| {label} | {f[k]['chroma']['median']} ms | {f[k]['chroma_reuse']['median']} ms | {f[k]['torch']['median']} ms |"
        for label, k in rows
    )
    return f"""| Median of {s["forward"]["runs"] * s["forward"]["calls"]} calls | Chroma | Chroma, reuse | Torch eager |
| --- | ---: | ---: | ---: |
{table}

Compiling took about {f["m3_cpu"]["compile"]} s on CPU, {f["m3_metal"]["compile"]} s on Metal, and {f["a100"]["compile"]} s with nvcc. The planned arena is {m["arena_exact"]}, against {m["unreused_exact"]} if no temporary were reused."""


def decode_block(s: Json) -> str:
    d = s["decode"]["targets"]
    rows = (("Apple M3 CPU", "m3_cpu"), ("Apple M3 Metal", "m3_metal"), ("NVIDIA A100", "a100"))
    table = "\n".join(
        f"| {label} | {d[k]['cached']['median']} ms | {d[k]['uncached']['median']} ms | {d[k]['prefill']['median']} ms |"
        for label, k in rows
    )
    return f"""| Median | Cached step | Uncached step | Prefill ({s["decode"]["prompt"]} tokens) |
| --- | ---: | ---: | ---: |
{table}

Compiling all three graphs took about {d["m3_cpu"]["compile"]} s on CPU, {d["m3_metal"]["compile"]} s on Metal, and {d["a100"]["compile"]} s with nvcc."""


def fill(document: str, name: str, content: str) -> str:
    start, end = f"<!-- benchmarks:{name} -->", f"<!-- /benchmarks:{name} -->"
    pattern = re.compile(re.escape(start) + r"\n.*?\n" + re.escape(end), re.DOTALL)
    if len(pattern.findall(document)) != 1:
        raise ValueError(f"Expected one {start} ... {end} block")
    return pattern.sub(lambda _: f"{start}\n{content}\n{end}", document)


def outputs() -> dict[Path, str]:
    s = summarize()
    doc = BENCHMARKS_DOC.read_text()
    for name, block in (("environments", environments_block), ("forward", forward_block), ("decode", decode_block)):
        doc = fill(doc, name, block(s))
    return {
        SUMMARY: json.dumps(s, indent=2, ensure_ascii=False) + "\n",
        README: fill(README.read_text(), "readme", readme_block(s)),
        BENCHMARKS_DOC: doc,
    }


def stale() -> list[Path]:
    return [path for path, content in outputs().items() if not path.exists() or path.read_text() != content]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail instead of writing if anything is out of date")
    args = parser.parse_args()
    if args.check:
        if paths := stale():
            sys.exit("Out of date: " + ", ".join(str(p.relative_to(ROOT)) for p in paths) + "; run scripts/summarize_benchmarks.py")
        return
    for path, content in outputs().items():
        path.write_text(content)
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
