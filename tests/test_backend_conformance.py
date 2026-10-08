import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from chroma import compile

type Backend = Literal["cpu", "cuda", "metal"]


def backends() -> list[Backend]:
    result: list[Backend] = ["cpu"]
    if sys.platform == "darwin":
        result.append("metal")
    if os.environ.get("CHROMA_TEST_CUDA") == "1":
        result.append("cuda")
    return result


class DTypeSemantics(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return x.sum(-1, dtype=torch.bfloat16), x.mean(-1, dtype=torch.bfloat16), x.softmax(-1, dtype=torch.bfloat16)


class ScalarAxes(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values, indices = x.topk(1)
        return x.sum(0), x.softmax(-1), values, indices


class ExpertBroadcast(torch.nn.Module):
    def forward(
        self,
        weights: torch.Tensor,
        narrow_weights: torch.Tensor,
        indices: torch.Tensor,
        batch_x: torch.Tensor,
        narrow_x: torch.Tensor,
        x: torch.Tensor,
        batch_expert_x: torch.Tensor,
        narrow_expert_x: torch.Tensor,
        expert_x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        selected = weights[indices]
        narrow_selected = narrow_weights[indices]
        return (
            torch.einsum("beck,bk->bec", selected, batch_x),
            torch.einsum("beck,bk->bec", selected, narrow_x),
            torch.einsum("beck,bk->bec", narrow_selected, x),
            torch.einsum("beck,bek->bec", selected, batch_expert_x),
            torch.einsum("beck,bek->bec", selected, narrow_expert_x),
            torch.einsum("beck,bek->bec", narrow_selected, expert_x),
        )


class BackendConformanceTests(unittest.TestCase):
    def check(self, module: torch.nn.Module, *inputs: torch.Tensor) -> None:
        expected = module(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        exported = torch.export.export(module, inputs)
        with tempfile.TemporaryDirectory() as root:
            for backend in backends():
                with self.subTest(backend=backend), compile(exported, Path(root) / backend, backend=backend, blas=False) as model:
                    self.assertEqual(model.stats["backend"], backend)
                    if backend == "cpu":
                        self.assertEqual(model.stats["math_library"], "portable")
                    actual = model(*inputs)
                    actual = actual if isinstance(actual, tuple) else (actual,)
                    for result, reference in zip(actual, expected, strict=True):
                        if reference.dtype == torch.bfloat16:
                            np.testing.assert_array_equal(result.view(np.uint16), reference.view(torch.uint16).numpy())
                        else:
                            np.testing.assert_allclose(result, reference.numpy(), rtol=2e-5, atol=2e-6)

    def test_explicit_bfloat16_sum_mean_and_softmax(self) -> None:
        x = torch.tensor([[1.003, 1.003, -2.0], [10.01, -10.0, 0.01], [256.5, 256.0, -1000.0]])
        self.check(DTypeSemantics(), x)

    def test_scalar_dimension_operations(self) -> None:
        self.check(ScalarAxes(), torch.tensor(2.5))

    def test_expert_contraction_broadcast(self) -> None:
        weights = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
        narrow_weights = weights[:, :, :1].clone()
        indices = torch.tensor([[0], [1]])
        batch_x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        narrow_x = torch.tensor([[2.0], [3.0]])
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1
        batch_expert_x = batch_x.unsqueeze(1)
        narrow_expert_x = narrow_x.unsqueeze(1)
        expert_x = x.unsqueeze(1)
        self.check(
            ExpertBroadcast(),
            weights,
            narrow_weights,
            indices,
            batch_x,
            narrow_x,
            x,
            batch_expert_x,
            narrow_expert_x,
            expert_x,
        )


if __name__ == "__main__":
    unittest.main()
