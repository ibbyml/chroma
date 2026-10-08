import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from chroma.models.gpt_oss import Transformer
from chroma.models.variants import dev

OUTPUT_DIR = Path("build/graph")


def make_dev_model(seed: int = 0) -> Transformer:
    # fork_rng keeps this helper from changing the caller's CPU random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Transformer(dev, device=torch.device("cpu")).float().eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(".scale"):
                    parameter.fill_(1.0)
                elif name.endswith(("bias", ".sinks")):
                    parameter.zero_()
                else:
                    parameter.normal_(mean=0.0, std=0.02)
        return model.requires_grad_(False)


@torch.no_grad()
def save_export(model: torch.nn.Module, examples: Sequence[torch.Tensor], path: Path, seed: int) -> None:
    tokens = examples[0]
    reference = model(tokens)
    if not torch.isfinite(reference).all():
        raise ValueError("The model produced non-finite outputs")
    program = torch.export.export(model, (tokens,))

    output = path.parent
    output.mkdir(parents=True, exist_ok=True)
    torch.export.save(program, path)
    restored = torch.export.load(path).module()
    for inputs in examples:
        torch.testing.assert_close(restored(inputs), model(inputs))

    save_file(model.state_dict(), output / "weights.safetensors")
    save_file({"tokens": tokens, "logits": reference.contiguous()}, output / "reference.safetensors")
    (output / "graph.txt").write_text(str(program.graph) + "\n")
    (output / "graph_signature.txt").write_text(str(program.graph_signature) + "\n")
    operators = sorted({str(node.target) for node in program.graph.nodes if node.op == "call_function"})
    (output / "operators.txt").write_text("\n".join(operators) + "\n")
    (output / "export.json").write_text(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "seed": seed,
                "num_tokens": tokens.numel(),
                "dtype": str(reference.dtype).removeprefix("torch."),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported {sum(p.numel() for p in model.parameters()):,} parameters to {output}")
    print(f"Input: {tokens.dtype} {list(tokens.shape)}; output: {reference.dtype} {list(reference.shape)}")
    print(f"{len(operators)} operator targets; saved graph matches eager PyTorch on {len(examples)} inputs")


def export_dev(output: Path, num_tokens: int = 8, seed: int = 0) -> None:
    tokens = torch.arange(num_tokens, dtype=torch.int64) % dev.vocab_size
    # Different token values exercise routing beyond the tracing example.
    save_export(make_dev_model(seed), (tokens, (tokens * 7919 + 17) % dev.vocab_size), output / "model.pt2", seed)
    (output / "config.json").write_text(json.dumps(asdict(dev), indent=2) + "\n")


def export_minimal_example(output: Path, num_tokens: int = 8, seed: int = 0) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Linear(num_tokens, 1).eval().requires_grad_(False)
    tokens = torch.arange(num_tokens, dtype=torch.float32)
    save_export(model, (tokens, tokens.flip(0)), output / "example.pt2", seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the GPT-OSS dev model or a single linear layer for Chroma.")
    parser.add_argument("--output", type=Path, help="Output directory (default: build/graph/dev, or build/graph/example with --example)")
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--example", action="store_true", help="Export a single linear layer")
    args = parser.parse_args()
    output = args.output or OUTPUT_DIR / ("example" if args.example else "dev")
    export = export_minimal_example if args.example else export_dev
    export(output, args.num_tokens, args.seed)


if __name__ == "__main__":
    main()
