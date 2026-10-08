import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils._pytree import tree_leaves

from chroma import compile
from chroma._src.build import build
from chroma._src.codegen.cpu import CPUCodegen
from chroma._src.codegen.metal import MetalCodegen
from chroma._src.compiler import Compiler
from tests.helpers import Function


class CompilerTests(unittest.TestCase):
    def test_unsupported_input_types_fail_at_import(self) -> None:
        float16 = torch.export.export(torch.nn.Identity(), (torch.ones(4, dtype=torch.float16),))
        scalar = torch.export.export(Function(lambda x, scale: x * scale), (torch.ones(4), 2))
        for codegen in (CPUCodegen, MetalCodegen):
            with self.assertRaisesRegex(ValueError, "Unsupported dtype torch.float16"):
                Compiler.from_export(float16, codegen())
            with self.assertRaisesRegex(TypeError, "Only tensor inputs"):
                Compiler.from_export(scalar, codegen())

    def test_lower_then_build_with_views_constants_and_repeated_outputs(self) -> None:
        class Module(torch.nn.Module):
            weight: torch.Tensor

            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("weight", torch.arange(6, dtype=torch.float32).reshape(2, 3))

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                first, second = x.sigmoid().split(2)
                result = first + second
                return result, result, x.t(), self.weight.sin()

        module = Module()
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        program = torch.export.export(module, (x,))
        for cls in (CPUCodegen, MetalCodegen) if sys.platform == "darwin" else (CPUCodegen,):
            codegen = cls()
            compiler = Compiler.from_export(program, codegen)
            generated = compiler.lower()
            sources = dict(generated.files)
            self.assertEqual(generated.stats["folded_constants"], 1)
            with (
                tempfile.TemporaryDirectory() as root,
                build(generated, root) as model,
            ):
                actual = model(x)
                for got, want in zip(actual, module(x), strict=True):
                    np.testing.assert_allclose(got, want.numpy(), atol=2e-6, rtol=2e-5)
                self.assertFalse(np.shares_memory(actual[0], actual[1]))
            self.assertEqual(generated.files, sources)

    def test_nan_clamp_bounds_preserve_index_errors(self) -> None:
        def fn(x: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return x.clamp(min=float("nan")), x.clamp(max=float("nan")), x.clamp(float("nan"), 1), x[indices].clamp(max=float("nan"))

        module = Function(fn)
        x = torch.tensor([-float("inf"), -1, 0, 1, float("inf"), float("nan")])
        indices = torch.tensor([0, 2, 4])
        program = torch.export.export(module, (x, indices))
        with tempfile.TemporaryDirectory() as root:
            for backend in ("cpu", "metal") if sys.platform == "darwin" else ("cpu",):
                with compile(program, Path(root) / backend, backend=backend) as model:
                    for actual, expected in zip(model(x, indices), module(x, indices), strict=True):
                        np.testing.assert_allclose(actual, expected.numpy(), equal_nan=True)
                    with self.assertRaises(IndexError):
                        model(x, torch.tensor([6, 0, 0]))

    def test_shared_dispatch_preserves_scalar_arguments(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
            z = torch.add(x, y, alpha=0.5)
            return (z.square() + 0.5).rsqrt(), torch.rsub(z, 3.0, alpha=0.25), z.clamp(-1, 0.75), torch.pow(2.0, z), torch.tril(z, 1)

        module = Function(fn)
        inputs = (torch.linspace(-2, 2, 6).reshape(2, 3), torch.ones(3))
        program = torch.export.export(module, inputs)
        with tempfile.TemporaryDirectory() as root:
            for backend in ("cpu", "metal") if sys.platform == "darwin" else ("cpu",):
                with compile(program, Path(root) / backend, backend=backend) as model:
                    for actual, expected in zip(model(*inputs), module(*inputs), strict=True):
                        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=2e-5)

    def test_metal_expression_size_and_nonfinite_clamps(self) -> None:
        def chain(count: int) -> Function[torch.Tensor]:
            def fn(x: torch.Tensor) -> torch.Tensor:
                for _ in range(count):
                    x = x.clamp(-1, 1)
                return x

            return Function(fn)

        x = torch.tensor([-float("inf"), -2, 0.5, float("inf"), float("nan")])
        sizes = []
        for count in (7, 20):
            compiler = Compiler.from_export(torch.export.export(chain(count), (x,)), MetalCodegen())
            generated = compiler.lower()
            source = generated.files["model.metal"].split('#line 1 "chroma-generated.metal"\n', 1)[1]
            sizes.append(len(source))
        self.assertLess(sizes[1], 4 * sizes[0])
        self.assertLess(sizes[1], 10_000)
        if sys.platform == "darwin":
            with tempfile.TemporaryDirectory() as root, compile(torch.export.export(chain(80), (x,)), root, backend="metal") as model:
                np.testing.assert_allclose(model(x), chain(80)(x).numpy(), equal_nan=True)

    def test_folded_state_is_stored_once_and_large_folds_are_bounded(self) -> None:
        class Constant(torch.nn.Module):
            weight: torch.Tensor

            def __init__(self, size: int) -> None:
                super().__init__()
                self.register_buffer("weight", torch.arange(size, dtype=torch.float32))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.weight.sin()

        for cls in (CPUCodegen, MetalCodegen):
            small = Constant(8)
            codegen = cls()
            compiler = Compiler.from_export(torch.export.export(small, (torch.ones(8),)), codegen)
            compiler.lower()
            self.assertEqual(len(codegen.weights), 64)
            np.testing.assert_array_equal(np.frombuffer(codegen.weights, dtype=np.float32)[:8], small.weight.sin().numpy())
            self.assertTrue(all(len(key[2]) == 32 for key in codegen.weight_cache))
            large = Constant(262145)
            compiler = Compiler.from_export(torch.export.export(large, (torch.ones(262145),)), cls())
            compiler.lower()
            self.assertEqual(compiler.folded, set())

    def test_reshape_linear_does_not_allocate_an_input_copy(self) -> None:
        x, w = torch.randn(3, 5, 17), torch.randn(7, 17)
        fn = Function(lambda x, w: torch.nn.functional.linear(x.reshape(-1, 17), w))
        for backend in ("cpu", "metal") if sys.platform == "darwin" else ("cpu",):
            with tempfile.TemporaryDirectory() as root, compile(torch.export.export(fn, (x, w)), root, backend=backend) as model:
                self.assertEqual(model.stats["workspace_bytes"], 0)
                np.testing.assert_allclose(model(x, w), fn(x, w).numpy(), atol=5e-6, rtol=2e-5)

    def test_matrix_labels_broadcasts_transposes_and_odd_tails(self) -> None:
        torch.manual_seed(7)
        cases = [
            ("mk,nk->mn", (35, 67), (33, 67)),
            ("qhd,khd->hqk", (35, 3, 17), (67, 3, 17)),
            ("hqk,khd->qhd", (3, 35, 67), (67, 3, 33)),
            ("bik,bkj->bij", (2, 35, 67), (1, 67, 33)),
        ]
        backends = ("cpu", "metal") if sys.platform == "darwin" else ("cpu",)
        with tempfile.TemporaryDirectory() as root:
            for i, (eqn, a_shape, b_shape) in enumerate(cases):
                a, b = torch.randn(a_shape), torch.randn(b_shape)
                fn = Function(lambda a, b, eqn=eqn: torch.einsum(eqn, a, b))
                program = torch.export.export(fn, (a, b))
                for backend in backends:
                    with (
                        self.subTest(eqn=eqn, backend=backend),
                        compile(program, Path(root) / f"{i}-{backend}", backend=backend) as model,
                    ):
                        self.assertEqual(model.stats["matrix_contractions"], 1)
                        np.testing.assert_allclose(model(a, b), fn(a, b).numpy(), atol=2e-5, rtol=2e-4)
                with compile(program, Path(root) / f"{i}-portable", blas=False) as model:
                    np.testing.assert_allclose(model(a, b), fn(a, b).numpy(), atol=2e-5, rtol=2e-4)

    @unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
    def test_parallel_topk_strides_ties_nan_and_infinity(self) -> None:
        torch.manual_seed(8)
        x = torch.randn(257, 3)
        x[::17] = float("nan")
        x[1::17] = float("inf")
        x[2::17] = -float("inf")
        with tempfile.TemporaryDirectory() as root:
            for largest in (True, False):
                fn = Function(lambda x, largest=largest: torch.topk(x, 40, dim=0, largest=largest))
                with compile(torch.export.export(fn, (x,)), Path(root) / str(largest), backend="metal") as model:
                    values, indices = model(x)
                    np.testing.assert_allclose(values, fn(x).values.numpy(), equal_nan=True)
                    np.testing.assert_allclose(
                        values, np.take_along_axis(x.numpy(), cast(NDArray[np.int64], indices), axis=0), equal_nan=True
                    )
                    self.assertTrue(all(len(set(column)) == 40 for column in indices.T))

    def test_expanded_matrix_axes_do_not_become_invalid_blas_strides(self) -> None:
        fn = Function(lambda a, b: torch.einsum("mk,nk->mn", a.expand(35, 67), b.expand(33, 67)))
        with tempfile.TemporaryDirectory() as root:
            for i, (a_shape, b_shape) in enumerate([((35, 1), (33, 67)), ((1, 67), (33, 67)), ((35, 67), (1, 67))]):
                a, b = torch.randn(a_shape), torch.randn(b_shape)
                program = torch.export.export(fn, (a, b))
                for backend in ("cpu", "metal") if sys.platform == "darwin" else ("cpu",):
                    with compile(program, Path(root) / f"{i}-{backend}", backend=backend) as model:
                        np.testing.assert_allclose(model(a, b), fn(a, b).numpy(), atol=2e-5, rtol=2e-4)

    @unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
    def test_long_dot_and_strided_parallel_reductions(self) -> None:
        a, b = torch.randn(4, 1025), torch.randn(1025)
        dot = Function(lambda a, b: torch.einsum("ij,j->i", a, b))
        rows = Function(lambda x: (x.softmax(0), x.mean(0)))
        x = torch.randn(1025, 3)
        x[17, 1], x[25, 2] = float("nan"), -float("inf")
        with tempfile.TemporaryDirectory() as root:
            for name, fn, inputs in (("dot", dot, (a, b)), ("rows", rows, (x,))):
                with compile(torch.export.export(fn, inputs), Path(root) / name, backend="metal") as model:
                    expected = tree_leaves(fn(*inputs))
                    actual = model(*inputs)
                    for got, want in zip(actual if isinstance(actual, tuple) else (actual,), expected, strict=True):
                        np.testing.assert_allclose(got, want.numpy(), atol=2e-5, rtol=2e-4, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
