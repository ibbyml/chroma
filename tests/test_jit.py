import inspect
import subprocess
import sys
import unittest
from types import FunctionType
from typing import Literal, cast
from unittest.mock import patch

import numpy as np
import torch

from chroma import jit
from chroma._src.compiler import compile_for_backend
from chroma._src.jit import TensorFunction
from chroma._src.program import Array


class JitTests(unittest.TestCase):
    def test_invalid_callables_and_inputs_fail_before_export(self) -> None:
        with self.assertRaisesRegex(TypeError, "callable"):
            jit(cast(TensorFunction, 1))
        compiled = jit(torch.nn.Identity())
        with patch("torch.export.export") as export:
            for value in ([1, 2], np.ones(2), torch.ones(2, device="meta")):
                with self.subTest(value=type(value)), self.assertRaisesRegex(TypeError, "CPU Torch tensor"):
                    compiled(value)
            export.assert_not_called()

    def test_caches_by_shape_and_dtype(self) -> None:
        def identity(x: torch.Tensor) -> torch.Tensor:
            """Return the input tensor."""
            return x

        with patch("chroma._src.compiler.compile_for_backend", wraps=compile_for_backend) as compile:
            compiled = jit(identity)
            self.assertIs(inspect.unwrap(compiled), identity)
            self.assertEqual(cast(FunctionType, compiled).__name__, identity.__name__)
            self.assertEqual(compiled.__doc__, identity.__doc__)
            for value in (torch.ones(4), torch.arange(4, dtype=torch.float32)):
                np.testing.assert_array_equal(compiled(value), value.numpy())
            self.assertEqual(compile.call_count, 1)
            np.testing.assert_array_equal(compiled(torch.ones(5)), np.ones(5))
            self.assertEqual(compile.call_count, 2)
            integers = torch.arange(5, dtype=torch.int64)
            np.testing.assert_array_equal(compiled(integers), integers.numpy())
            self.assertEqual(compile.call_count, 3)
            compiled(integers + 1)
            self.assertEqual(compile.call_count, 3)

    def test_function_backends_and_output_reuse(self) -> None:
        backends: tuple[Literal["cpu", "metal"], ...] = ("cpu", "metal") if sys.platform == "darwin" else ("cpu",)
        for backend in backends:
            with self.subTest(backend=backend):

                @jit(backend=backend, blas=False)
                def function(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    return x.sin() + y, x * y

                x = torch.linspace(-1, 1, 12).reshape(3, 4)
                y = torch.arange(4, dtype=torch.float32)
                outputs = cast(tuple[Array, Array], function(x, y))
                np.testing.assert_allclose(outputs[0], (x.sin() + y).numpy(), atol=1e-6)
                np.testing.assert_allclose(outputs[1], (x * y).numpy(), atol=1e-6)
                reused = cast(tuple[Array, Array], function(x + 0.25, y, out=outputs))
                self.assertIs(reused[0], outputs[0])
                self.assertIs(reused[1], outputs[1])
                np.testing.assert_allclose(reused[0], ((x + 0.25).sin() + y).numpy(), atol=1e-6)
                np.testing.assert_allclose(reused[1], ((x + 0.25) * y).numpy(), atol=1e-6)

    def test_export_uses_contiguous_inference_inputs(self) -> None:
        def function(x: torch.Tensor) -> torch.Tensor:
            return x + int(not x.is_contiguous() or x.requires_grad)

        x = torch.arange(9, dtype=torch.float32).reshape(3, 3).t().requires_grad_(True)
        with patch("chroma._src.compiler.compile_for_backend", wraps=compile_for_backend) as compile:
            compiled = jit(function)
            np.testing.assert_array_equal(compiled(x), x.detach().numpy())
            np.testing.assert_array_equal(compiled(x.detach().contiguous()), x.detach().numpy())
            self.assertEqual(compile.call_count, 1)

    def test_module_weights_are_captured_per_specialization(self) -> None:
        model = torch.nn.Linear(3, 2, bias=False).eval().requires_grad_(False)
        model.weight.fill_(0.5)
        compiled = jit(model)
        np.testing.assert_array_equal(compiled(torch.ones(1, 3)), np.full((1, 2), 1.5))
        model.weight.fill_(2.0)
        np.testing.assert_array_equal(compiled(torch.ones(1, 3)), np.full((1, 2), 1.5))
        np.testing.assert_array_equal(compiled(torch.ones(2, 3)), np.full((2, 2), 6.0))

    def test_public_import_does_not_load_torch(self) -> None:
        subprocess.run(
            [sys.executable, "-c", "import sys; from chroma import jit; assert 'torch' not in sys.modules"],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
