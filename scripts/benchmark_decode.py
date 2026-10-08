import argparse
import json
import random
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from scripts.benchmark import (
    Backend,
    default_backends,
    elapsed_ms,
    environment_info,
    limit_threads,
    load_model,
    source_identity,
    summarize,
    weights_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare KV-cached decoding with a padded full-window graph.")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--prompt", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backends", nargs="+", choices=["cpu", "cuda", "metal"])
    parser.add_argument("--weights", type=Path, help="Use one exported weights.safetensors snapshot across machines")
    parser.add_argument("--output", type=Path, default=Path("build/benchmarks/decode.json"))
    args = parser.parse_args()
    limit_threads(args.threads)

    import numpy as np
    import torch

    from chroma import Decoder, compile
    from chroma.models.cached import export_cached, export_uncached
    from chroma.models.variants import dev

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    eager = load_model(args.weights)
    tokens = (np.arange(args.prompt + args.steps, dtype=np.int64) * 7919 + 17) % dev.vocab_size
    backends: dict[str, dict[str, object]] = {}
    report: dict[str, object] = {
        "measured_at": datetime.now(UTC).isoformat(),
        "source": source_identity(),
        "environment": environment_info(),
        "context": args.context,
        "prompt": args.prompt,
        "steps": args.steps,
        "rounds": args.rounds,
        "model": asdict(dev),
        "seed": 0 if args.weights is None else None,
        "weights_sha256": weights_digest(eager),
        "method": "One validation/warmup round, then seeded interleaved cached/uncached calls at each forced token. "
        "Both paths return a fresh last-logit row. Cached timings include host mask/cache work and GPU staging. "
        "Uncached input padding is prepared outside timing. Reset, prefill, validation, and output destruction are excluded from decode latency. "
        "Each round starts from the same prompt; no eviction occurs during this comparison.",
        "backends": backends,
    }
    programs = (
        export_uncached(eager, args.context),
        export_cached(eager, args.context, args.context),
        export_cached(eager, args.context, 1),
    )
    with TemporaryDirectory(prefix="chroma-decode-benchmark-") as temporary:
        for backend in cast(list[Backend], args.backends or default_backends()):
            with ExitStack() as stack:
                start = time.perf_counter()
                full, prefill, step = [
                    stack.enter_context(compile(program, Path(temporary) / backend / name, backend=backend))
                    for program, name in zip(programs, ("full", "prefill", "decode"), strict=True)
                ]
                cached = stack.enter_context(Decoder(prefill, step, sliding_window=dev.sliding_window))
                compile_seconds = time.perf_counter() - start
                padded = np.zeros(args.context, dtype=np.int64)
                last = np.array([args.prompt - 1], dtype=np.int64)
                samples: dict[str, list[float]] = {"uncached": [], "cached": [], "prefill": []}
                order, rng = ["uncached", "cached"], random.Random(0)
                for round_index in range(args.rounds + 1):
                    cached.reset()
                    padded.fill(0)
                    padded[: args.prompt] = tokens[: args.prompt]
                    last[0] = args.prompt - 1
                    if round_index:
                        samples["prefill"].append(elapsed_ms(partial(cached.append, tokens[: args.prompt])))
                    else:
                        np.testing.assert_allclose(cached.append(tokens[: args.prompt]), full(padded, last), rtol=2e-5, atol=2e-6)
                    for position in range(args.prompt, len(tokens)):
                        padded[position] = tokens[position]
                        last[0] = position
                        if round_index == 0:
                            np.testing.assert_allclose(
                                cached.append(tokens[position : position + 1]), full(padded, last), rtol=2e-5, atol=2e-6
                            )
                        else:
                            calls: dict[str, Callable[[], object]] = {
                                "uncached": partial(full, padded, last),
                                "cached": partial(cached.append, tokens[position : position + 1]),
                            }
                            rng.shuffle(order)
                            for name in order:
                                samples[name].append(elapsed_ms(calls[name]))
                timings = summarize(samples)
                backends[backend] = {
                    "device": cached.device,
                    "compile_and_load_all_graphs_seconds": compile_seconds,
                    "stats": cached.stats,
                    "timings": timings,
                }
                print(
                    f"{backend}: uncached {timings['uncached']['median_ms']:.3f} ms; cached {timings['cached']['median_ms']:.3f} ms; "
                    f"prefill {timings['prefill']['median_ms']:.3f} ms",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
