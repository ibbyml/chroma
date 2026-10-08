from collections.abc import Callable

import torch

from chroma.models.gpt_oss import ModelConfig, Transformer


class Function[T](torch.nn.Module):
    """Wraps a plain function so torch.export can trace it."""

    def __init__(self, fn: Callable[..., T]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args: torch.Tensor | float) -> T:
        return self.fn(*args)


def random_transformer(config: ModelConfig, *, seed: int, std: float, dtype: torch.dtype = torch.float32) -> Transformer:
    """Unit norm scales and normal(0, std) elsewhere, drawn in FP32 without disturbing the global RNG."""
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        model = Transformer(config, device=torch.device("cpu")).float().eval().requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name.endswith(".scale"):
                parameter.fill_(1)
            else:
                parameter.normal_(std=std)
    return model.to(dtype)
