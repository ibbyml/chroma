import dataclasses
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import torch
from ml_dtypes import bfloat16
from torch.utils._pytree import tree_leaves

from chroma import Decoder, Program, compile, jit
from chroma._src.build import compiler_command
from chroma._src.codegen.base import GeneratedCode
from chroma._src.codegen.cuda import CUDACodegen
from chroma._src.compiler import Compiler
from chroma._src.program import Array
from chroma.models.cached import export_cached
from chroma.models.gpt_oss import RMSNorm, Transformer
from chroma.models.variants import dev
from tests.helpers import Function


def lower(module: torch.nn.Module, *inputs: torch.Tensor, blas: bool = True) -> GeneratedCode:
    return Compiler.from_export(torch.export.export(module, inputs), CUDACodegen(blas=blas)).lower()


def small_model(dtype: torch.dtype = torch.float32) -> Transformer:
    config = dataclasses.replace(
        dev,
        num_hidden_layers=1,
        num_experts=2,
        experts_per_token=2,
        vocab_size=17,
        hidden_size=8,
        intermediate_size=8,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        sliding_window=3,
    )
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(5)
        model = Transformer(config, device=torch.device("cpu")).to(dtype).eval().requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name.endswith(".scale"):
                parameter.fill_(1)
            else:
                parameter.normal_(std=0.1)
        return model


def numpy_bf16(tensor: torch.Tensor) -> Array:
    return tensor.detach().view(torch.uint16).numpy().view(bfloat16)


def assert_close(actual: Array, expected: torch.Tensor, *, rtol: float = 2e-5, atol: float = 5e-6) -> None:
    dtypes = {torch.float32: np.dtype(np.float32), torch.bfloat16: np.dtype(bfloat16), torch.int64: np.dtype(np.int64)}
    np.testing.assert_equal(actual.dtype, dtypes[expected.dtype])
    if expected.dtype == torch.bfloat16:
        np.testing.assert_allclose(actual.astype(np.float32), expected.detach().float().numpy(), rtol=rtol, atol=atol, equal_nan=True)
    elif expected.dtype == torch.int64:
        np.testing.assert_array_equal(actual, expected.numpy())
    else:
        np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=rtol, atol=atol, equal_nan=True)


def warp_operations(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return x.sum(1), x.mean((0, 2), keepdim=True), x.softmax(1)


def contractions(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    e: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.einsum("ij,j->i", a, b),
        torch.einsum("bik,bkj->bij", c, d),
        torch.einsum("km,kn->mn", e, f),
        torch.einsum("ij,ik->ijk", g, h),
    )


def contraction_inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.linspace(-1, 1, 4 * 1025).reshape(4, 1025),
        torch.linspace(0.5, 1.5, 1025),
        torch.linspace(-1, 1, 2 * 3 * 17).reshape(2, 3, 17),
        torch.linspace(1, -1, 17 * 5).reshape(1, 17, 5),
        torch.linspace(-0.5, 0.5, 17 * 3).reshape(17, 3),
        torch.linspace(0.75, -0.75, 17 * 5).reshape(17, 5),
        torch.linspace(-1, 1, 2 * 3).reshape(2, 3),
        torch.linspace(1, -1, 2 * 4).reshape(2, 4),
    )


class CUDACodegenTests(unittest.TestCase):
    def test_fusion_views_constants_and_int64_outputs(self) -> None:
        module = Function(lambda x, y, ids: ((x.sin() + y).t()[:, 1::2], ids.t(), torch.arange(4), x, x))
        generated = lower(module, torch.ones(5, 3), torch.ones(3), torch.ones(2, 3, dtype=torch.int64))
        source = generated.files["model.cu"]
        self.assertEqual(generated.source_name, "model.cu")
        self.assertEqual(generated.stats["backend"], "cuda")
        self.assertEqual(generated.stats["workspace_bytes"], 0)
        self.assertEqual(
            generated.stats["cuda_buffer_bytes"],
            generated.stats["weight_bytes"]
            + generated.stats["workspace_bytes"]
            + generated.stats["input_bytes"]
            + generated.stats["output_bytes"]
            + 4,
        )
        self.assertEqual(generated.stats["kernels"], 5)
        self.assertEqual(generated.stats["folded_constants"], 1)
        self.assertIn("const int64_t*", source)
        self.assertIn("sinf(", source)
        self.assertIn("__global__ void", source)
        self.assertIn("runtime.stream()", source)
        self.assertEqual(generated.outputs[0]["shape"], [3, 2])
        self.assertIn("--generate-code=arch=compute_75,code=sm_75", generated.flags)
        self.assertIn("--generate-code=arch=compute_80,code=[sm_80,compute_80]", generated.flags)
        self.assertIn("--fmad=false", generated.flags)
        self.assertNotIn("--use_fast_math", generated.flags)

    def test_linear_uses_cublas_and_promotes_output_storage(self) -> None:
        module = torch.nn.Linear(17, 7).eval()
        for blas in (True, False):
            with self.subTest(blas=blas):
                generated = lower(module, torch.ones(3, 5, 17), blas=blas)
                source = generated.files["model.cu"]
                self.assertEqual(generated.stats["workspace_bytes"], 0)
                self.assertEqual(generated.stats["kernels"], 2 if blas else 1)
                self.assertEqual("runtime.linear(" in source, blas)
                self.assertEqual("for (size_t k" in source, not blas)
                self.assertIn("runtime.output(0)", source)
                self.assertEqual(generated.stats["output_bytes"], 3 * 5 * 7 * 4)

    def test_bfloat16_linear_uses_scalar_float_accumulation(self) -> None:
        module = torch.nn.Linear(17, 7).bfloat16().eval()
        for blas in (True, False):
            with self.subTest(blas=blas):
                generated = lower(module, torch.ones(3, 5, 17, dtype=torch.bfloat16), blas=blas)
                source = generated.files["model.cu"]
                self.assertNotIn("runtime.linear(", source)
                self.assertIn("float sum = 0.0f", source)
                self.assertEqual(generated.inputs[0]["dtype"], "bfloat16")
                self.assertEqual(generated.outputs[0]["dtype"], "bfloat16")
                self.assertEqual(generated.stats["output_bytes"], 3 * 5 * 7 * 2)

    def test_shared_expressions_and_workspace_reuse(self) -> None:
        def fn(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, ...]:
            first = torch.nn.functional.linear(x, w)
            shared = first.sin()
            second = torch.nn.functional.linear(shared + 1, w)
            third = torch.nn.functional.linear(second + shared, w)
            return third, first[:, 1:], shared

        compiler = Compiler.from_export(torch.export.export(Function(fn), (torch.ones(3, 17), torch.eye(17))), CUDACodegen())
        generated = compiler.lower()
        self.assertGreater(generated.stats["workspace_bytes"], 0)
        allocations = list(compiler.codegen.allocations.values())
        for i, a in enumerate(allocations):
            for b in allocations[i + 1 :]:
                if a.start <= b.end and b.start <= a.end:
                    self.assertTrue(a.offset + a.size <= b.offset or b.offset + b.size <= a.offset)
        self.assertLess(generated.stats["workspace_bytes"], generated.stats["unreused_workspace_bytes"])

    def test_long_expressions_stay_bounded(self) -> None:
        def fn(x: torch.Tensor) -> torch.Tensor:
            for _ in range(80):
                x = x.clamp(-1, 1)
            return x

        generated = lower(Function(fn), torch.ones(257))
        self.assertEqual(generated.stats["kernels"], 4)
        self.assertLess(len(generated.files["model.cu"]), 35_000)

    def test_all_operations_lower_for_float32_and_bfloat16(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
            selected = x[ids]
            values, indices = selected.topk(2, dim=0, largest=False, sorted=False)
            return (
                selected.sum(0),
                selected.mean(1, keepdim=True),
                selected.softmax(0),
                values,
                indices,
                torch.cat((selected, y), dim=0),
                torch.triu(selected, 1),
                torch.tril(selected, -1),
                torch.einsum("ij,kj->ik", selected, y),
            )

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                generated = lower(
                    Function(fn),
                    torch.randn(6, 4, dtype=dtype),
                    torch.randn(2, 4, dtype=dtype),
                    torch.tensor([4, -1, 0], dtype=torch.int64),
                )
                self.assertEqual(generated.stats["backend"], "cuda")
                self.assertEqual(generated.outputs[3]["dtype"], str(dtype).removeprefix("torch."))
                self.assertEqual(generated.outputs[4]["dtype"], "int64")

                norm = RMSNorm(17).to(dtype).eval()
                norm_generated = lower(norm, torch.randn(3, 17, dtype=dtype))
                self.assertEqual(norm_generated.outputs[0]["dtype"], str(dtype).removeprefix("torch."))

    def test_int64_copy_gather_concatenate_and_triangular_lower(self) -> None:
        def fn(x: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
            selected = x[ids]
            return x, selected, torch.cat((selected, x[:1]), dim=0), torch.triu(x, 1), torch.tril(x, -1)

        generated = lower(Function(fn), torch.arange(12, dtype=torch.int64).reshape(4, 3), torch.tensor([3, -1, 0], dtype=torch.int64))
        self.assertTrue(all(output["dtype"] == "int64" for output in generated.outputs))

    def test_warp_reductions_norm_and_general_contractions_lower(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.linspace(-3, 3, 3 * 65 * 17).reshape(3, 65, 17).to(dtype)
                reductions = lower(Function(warp_operations), x)
                self.assertEqual(reductions.stats["parallel_reductions"], 3)
                norm = lower(RMSNorm(65).to(dtype).eval(), torch.randn(4, 65, dtype=dtype))
                self.assertEqual(norm.stats["parallel_reductions"], 1)

        generated = lower(Function(contractions), *contraction_inputs())
        self.assertEqual(len(generated.outputs), 4)
        self.assertGreaterEqual(generated.stats["parallel_reductions"], 1)

    def test_full_transformer_and_cached_graphs_lower_for_both_float_dtypes(self) -> None:
        tokens = torch.tensor([1, 4, 2, 7], dtype=torch.int64)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = small_model(dtype)
                generated = lower(model, tokens)
                self.assertEqual(generated.outputs[0]["dtype"], str(dtype).removeprefix("torch."))
                self.assertGreater(generated.stats["kernels"], 0)
                for steps in (1, 5):
                    cached = Compiler.from_export(export_cached(model, 5, steps), CUDACodegen()).lower()
                    self.assertEqual([output["dtype"] for output in cached.outputs], [str(dtype).removeprefix("torch.")] * 3)

    def test_float16_remains_out_of_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported dtype torch.float16"):
            lower(torch.nn.Identity(), torch.ones(4, dtype=torch.float16))

    def test_build_selects_nvcc_and_reports_missing_toolchain(self) -> None:
        with patch("chroma._src.build.sys.platform", "linux"), patch.dict(os.environ, {"NVCC": "/opt/cuda/bin/nvcc -ccbin g++"}):
            with patch("chroma._src.build.shutil.which", return_value="/opt/cuda/bin/nvcc"):
                command = compiler_command("model.cu")
                self.assertEqual(command[:3], ["/opt/cuda/bin/nvcc", "-ccbin", "g++"])
                self.assertIn("-Xcompiler", command)
                self.assertIn("--cudart=shared", command)
                self.assertIn("-std=c++20", command)
            with patch("chroma._src.build.shutil.which", return_value=None), self.assertRaisesRegex(RuntimeError, "requires nvcc"):
                compiler_command("model.cu")
        with patch("chroma._src.build.sys.platform", "darwin"), self.assertRaisesRegex(RuntimeError, "requires Linux"):
            compiler_command("model.cu")
        with patch.dict(os.environ, {"CXX": "clang++"}):
            self.assertEqual(compiler_command("model.cc")[0], "clang++")

    def test_public_backend_dispatch(self) -> None:
        exported = torch.export.export(torch.nn.Identity(), (torch.ones(4),))
        with patch("chroma._src.compiler.build") as build:
            compile(exported, "unused", backend="cuda")
            self.assertEqual(build.call_args.args[0].stats["backend"], "cuda")


@unittest.skipUnless(os.environ.get("CHROMA_TEST_CUDA") == "1", "Set CHROMA_TEST_CUDA=1 on a Linux CUDA host")
class CUDARuntimeTests(unittest.TestCase):
    def check(
        self,
        module: torch.nn.Module,
        *inputs: torch.Tensor,
        blas: bool = True,
        rtol: float = 2e-5,
        atol: float = 5e-6,
    ) -> None:
        exported = torch.export.export(module, inputs)
        with tempfile.TemporaryDirectory() as root:
            with compile(exported, root, backend="cuda", blas=blas) as model:
                for values in (inputs, tuple(x + 1 for x in inputs)):
                    with torch.no_grad():
                        expected = tree_leaves(module(*values))
                    actual = model(*values)
                    for got, want in zip(tree_leaves(actual), expected, strict=True):
                        assert_close(got, want, rtol=rtol, atol=atol)
                    reused = model(*values, out=actual)
                    for got, original in zip(tree_leaves(reused), tree_leaves(actual), strict=True):
                        self.assertIs(got, original)
                saved = [array.copy() for array in tree_leaves(actual)]
                model(*inputs)
                for got, want in zip(tree_leaves(actual), saved, strict=True):
                    np.testing.assert_array_equal(got, want)
            for got, want in zip(tree_leaves(actual), saved, strict=True):
                np.testing.assert_array_equal(got, want)
            with Program(root) as loaded:
                for got, want in zip(tree_leaves(loaded(*inputs)), tree_leaves(module(*inputs)), strict=True):
                    assert_close(got, want, rtol=rtol, atol=atol)

    def test_fused_views_and_wide_int64(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
            z = (x.sin() + y).clamp(-1, 1)
            return z, z, z.t().reshape(3, 514)[:, 1::2], x, ids.t(), ids[1:, 1:]

        ids = torch.tensor([[2**60 + 7, -(2**60) - 3, 2**40], [-11, 2**62 - 1, 0]], dtype=torch.int64)
        self.check(Function(fn), torch.linspace(-3, 3, 1542).reshape(514, 3), torch.ones(3), ids)

    def test_cublas_and_scalar_linear(self) -> None:
        torch.manual_seed(42)

        def fn(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
            y = torch.nn.functional.linear(x[:, ::2], w.t(), b)
            return y.sigmoid(), y, torch.nn.functional.linear(x[:, ::2] + 1, w.t())

        inputs = (torch.randn(35, 34), torch.randn(17, 7), torch.randn(7))
        for blas in (True, False):
            with self.subTest(blas=blas):
                self.check(Function(fn), *inputs, blas=blas)
        self.check(torch.nn.Linear(17, 7).eval(), torch.randn(17))
        self.check(torch.nn.Linear(17, 7, bias=False).eval(), torch.randn(2, 3, 17))

    def test_aliases_and_live_workspace_across_linear_layers(self) -> None:
        def fn(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, ...]:
            first = torch.nn.functional.linear(x, w)
            shared = first.sin()
            second = torch.nn.functional.linear(shared + 1, w)
            third = torch.nn.functional.linear(second + shared, w)
            return third, first[:, 1:], shared

        self.check(Function(fn), torch.randn(3, 17), torch.eye(17))

    def test_reductions_softmax_gather_cat_triangular_norm_and_einsum(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
            selected = x[ids]
            values, indices = selected.topk(2, dim=0, largest=False)
            return (
                selected.sum(0),
                selected.mean(1, keepdim=True),
                selected.softmax(0),
                values,
                indices,
                torch.cat((selected, y), dim=0),
                torch.triu(selected, 1),
                torch.tril(selected, -1),
                torch.einsum("ij,kj->ik", selected, y),
            )

        torch.manual_seed(9)
        ids = torch.tensor([4, -1, 0], dtype=torch.int64)
        for dtype, rtol, atol in ((torch.float32, 2e-4, 2e-5), (torch.bfloat16, 0.03, 0.01)):
            with self.subTest(dtype=dtype):
                self.check(
                    Function(fn),
                    torch.arange(-12, 12, dtype=torch.float32).reshape(6, 4).to(dtype),
                    torch.randn(2, 4, dtype=dtype),
                    ids,
                    rtol=rtol,
                    atol=atol,
                )
                self.check(RMSNorm(17).to(dtype).eval(), torch.randn(5, 17, dtype=dtype), rtol=rtol, atol=atol)

    def test_int64_gather_cat_masks_bounds_and_recovery(self) -> None:
        def fn(x: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
            selected = x[ids]
            return selected, torch.cat((selected, x[:1]), dim=0), torch.triu(x, 1), torch.tril(x, -1)

        x = torch.arange(12, dtype=torch.int64).reshape(4, 3)
        ids = torch.tensor([3, -1, 0], dtype=torch.int64)
        exported = torch.export.export(Function(fn), (x, ids))
        with tempfile.TemporaryDirectory() as root, compile(exported, root, backend="cuda") as model:
            expected = fn(x, ids)
            for got, want in zip(cast(tuple[Array, ...], model(x, ids)), expected, strict=True):
                assert_close(got, want)
            for bad in (-5, 4, 2**40):
                with self.assertRaises(IndexError):
                    model(x, torch.tensor([bad, 0, 0], dtype=torch.int64))
                for got, want in zip(cast(tuple[Array, ...], model(x, ids)), expected, strict=True):
                    assert_close(got, want)

    def test_strided_axes_nonfinite_values_and_topk_ties(self) -> None:
        def strided(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            y = x.t()[1::2]
            values, indices = y.topk(3, dim=1, largest=False)
            return y, y.sum(0), y.mean(1), y.softmax(0), values, indices, torch.tril(y, 1)

        x = torch.randn(17, 8)
        x[2, 1] = float("nan")
        x[5, 3] = float("inf")
        x[9, 5] = -float("inf")
        self.check(Function(strided), x, rtol=2e-4, atol=2e-5)

        tied = torch.tensor([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [1.0, 2.0, 3.0], [0.0, 3.0, 3.0], [0.0, 3.0, 3.0]])
        with tempfile.TemporaryDirectory() as root:
            for largest in (True, False):
                module = Function(lambda x, largest=largest: torch.topk(x, 4, dim=0, largest=largest, sorted=False))
                with (
                    self.subTest(largest=largest),
                    compile(torch.export.export(module, (tied,)), Path(root) / str(largest), backend="cuda") as model,
                ):
                    values, indices = cast(tuple[Array, Array], model(tied))
                    expected = module(tied).values.numpy()
                    np.testing.assert_allclose(np.sort(values, axis=0), np.sort(expected, axis=0))
                    np.testing.assert_array_equal(values, np.take_along_axis(tied.numpy(), indices, axis=0))
                    self.assertTrue(all(len(set(column)) == 4 for column in indices.T))

    def test_warp_reductions_norm_and_general_contractions(self) -> None:
        for dtype, rtol, atol in ((torch.float32, 2e-4, 2e-5), (torch.bfloat16, 0.03, 0.01)):
            with self.subTest(dtype=dtype):
                x = torch.linspace(-3, 3, 3 * 65 * 17).reshape(3, 65, 17).to(dtype)
                self.check(Function(warp_operations), x, rtol=rtol, atol=atol)
                self.check(RMSNorm(65).to(dtype).eval(), torch.randn(4, 65, dtype=dtype), rtol=rtol, atol=atol)

        self.check(Function(contractions), *contraction_inputs(), rtol=5e-4, atol=2e-4)

    def test_bfloat16_storage_rounding_and_float_accumulation(self) -> None:
        def rounding(x: torch.Tensor, y: torch.Tensor, one: torch.Tensor) -> tuple[torch.Tensor, ...]:
            rounded = y.bfloat16()
            return x.sin() + x, rounded, rounded.float(), (one + 2**-8) + 2**-8, (one * 0.333984375) - 0.333

        x = torch.linspace(-1, 1, 16).bfloat16()
        y = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, -1 - 2**-8, -1 - 3 * 2**-8, float("inf"), float("nan")])
        one = torch.ones(1, dtype=torch.bfloat16)
        module = Function(rounding)
        with tempfile.TemporaryDirectory() as root, compile(torch.export.export(module, (x, y, one)), root, backend="cuda") as model:
            actual = cast(tuple[Array, ...], model(numpy_bf16(x), y.numpy(), numpy_bf16(one)))
            for got, want in zip(actual, module(x, y, one), strict=True):
                assert_close(got, want, rtol=0, atol=0)
            self.assertEqual([value.itemsize for value in actual], [2, 2, 4, 2, 2])
            np.testing.assert_array_equal(actual[3].astype(np.float32), np.ones(1, dtype=np.float32))

        def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.nn.functional.linear(x, weight, bias), torch.einsum("mk,nk->mn", x, weight), weight.sum(dim=1)

        values = (
            torch.ones(2, 17, dtype=torch.bfloat16),
            torch.full((5, 17), 2**-8, dtype=torch.bfloat16),
            torch.full((5,), 2**-5, dtype=torch.bfloat16),
        )
        values[1][:, 0] = 1
        for blas in (True, False):
            with self.subTest(blas=blas):
                self.check(Function(linear), *values, blas=blas, rtol=0, atol=0)

    def test_small_transformer_and_decoder_match_torch(self) -> None:
        context = 5
        tokens = (torch.arange(14) * 7 + 3) % 17
        with tempfile.TemporaryDirectory() as directory:
            for dtype, rtol, atol in ((torch.float32, 2e-4, 2e-5), (torch.bfloat16, 0.03, 0.01)):
                with self.subTest(dtype=dtype):
                    eager = small_model(dtype)
                    expected = eager(tokens)
                    full = compile(torch.export.export(eager, (tokens[:4],)), Path(directory) / f"full-{dtype}", backend="cuda")
                    programs = [export_cached(eager, context, steps) for steps in (context, 1)]
                    prefill = compile(programs[0], Path(directory) / f"prefill-{dtype}", backend="cuda")
                    step = compile(programs[1], Path(directory) / f"step-{dtype}", backend="cuda")
                    with full, Decoder(prefill, step, sliding_window=3) as decoder:
                        self.assertTrue(full.device)
                        assert_close(cast(Array, full(tokens[:4])), expected[:4], rtol=rtol, atol=atol)
                        for bad in (-1, 17, 2**40):
                            with self.assertRaises(IndexError):
                                full(torch.full((4,), bad, dtype=torch.int64))
                            assert_close(cast(Array, full(tokens[:4])), expected[:4], rtol=rtol, atol=atol)
                        position = 0
                        for size in (2, 1, 4, 5, 2):
                            logits = decoder.append(tokens[position : position + size].numpy())
                            position += size
                            assert_close(logits, expected[position - 1 : position], rtol=rtol, atol=atol)
                        preserved = logits.copy()
                        decoder.reset()
                        self.assertEqual(decoder.position, 0)
                        assert_close(decoder.append(tokens.numpy()), expected[-1:], rtol=rtol, atol=atol)
                        np.testing.assert_array_equal(logits, preserved)

    def test_nonfinite_math_scalars_and_long_chains(self) -> None:
        def fn(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            z = x.clamp(-1, 1)
            for _ in range(80):
                z = z.clamp(-1, 1)
            return z, torch.add(x, x, alpha=0.25), torch.rsub(x, 3, alpha=0.5), 2.0**x, x.clamp(max=float("nan"))

        self.check(Function(fn), torch.tensor([-float("inf"), -2, 0.5, float("inf"), float("nan")]))
        self.check(Function(lambda x: x.sin() + 1), torch.tensor(0.5))
        self.check(Function(lambda x: (torch.arange(4), x)), torch.ones(1))

    def test_jit_and_runtime_only_artifact(self) -> None:
        @jit(backend="cuda")
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x.sin() + x

        x = torch.arange(8, dtype=torch.float32)
        np.testing.assert_allclose(fn(x), (x.sin() + x).numpy(), rtol=2e-5, atol=2e-6)
        with tempfile.TemporaryDirectory() as root:
            with compile(torch.export.export(torch.nn.Identity(), (x,)), root, backend="cuda") as model:
                np.save(Path(root) / "expected.npy", model(x))
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys, numpy as np; from chroma import Program; "
                        "model = Program(sys.argv[1]); "
                        "np.testing.assert_array_equal(model(np.arange(8, dtype=np.float32)), np.load(sys.argv[1] + '/expected.npy')); "
                        "assert 'torch' not in sys.modules and 'pybind11' not in sys.modules"
                    ),
                    root,
                ],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
