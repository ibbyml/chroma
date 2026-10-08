import argparse
import json
import time
from pathlib import Path

import numpy as np

from chroma import Program, compile


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a fixed-shape Torch export to native Chroma inference.")
    parser.add_argument("program", type=Path)
    parser.add_argument("--output", type=Path, help="Output directory (default: build/models/<backend>)")
    parser.add_argument("--verify", action="store_true", help="Compare against the saved export example inputs")
    parser.add_argument("--benchmark", type=int, default=0, metavar="RUNS")
    parser.add_argument("--no-blas", action="store_true", help="Use scalar matrix kernels (CPU and CUDA)")
    parser.add_argument("--backend", choices=["cpu", "cuda", "metal"], default="cpu")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("build/models") / args.backend
    if args.benchmark < 0:
        parser.error("--benchmark must be nonnegative")
    with compile(args.program, args.output, blas=not args.no_blas, backend=args.backend) as model:
        print(f"Device: {model.device}")
        print(json.dumps(model.stats, indent=2))
        if args.verify or args.benchmark:
            import torch
            from torch.utils._pytree import tree_leaves

            program = torch.export.load(args.program)
            inputs, kwargs = program.example_inputs
            if kwargs:
                raise ValueError("The verification CLI requires positional example inputs")
            if args.verify:
                with torch.no_grad():
                    expected = program.module()(*inputs)
                expected = tree_leaves(expected)
                actual = model(*inputs)
                actual = actual if isinstance(actual, tuple) else (actual,)
                for got, want in zip(actual, expected, strict=True):
                    if want.dtype == torch.bfloat16:
                        torch.testing.assert_close(torch.from_numpy(got.astype(np.float32)), want.float(), rtol=0.02, atol=0.005)
                    else:
                        torch.testing.assert_close(torch.from_numpy(got), want, rtol=1e-4, atol=1e-5)
                print("Native output matches the saved Torch export.")
            if args.benchmark:
                arrays = [Program._array(x) for x in inputs]
                output = model(*arrays)
                for _ in range(3):
                    model(*arrays, out=output)
                times: list[float] = []
                for _ in range(args.benchmark):
                    start = time.perf_counter()
                    model(*arrays, out=output)
                    times.append((time.perf_counter() - start) * 1000)
                print(f"Median: {np.median(times):.3f} ms over {args.benchmark} calls (reused output).")
        print(f"Artifact: {args.output.resolve()}")


if __name__ == "__main__":
    main()
