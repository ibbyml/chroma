import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from ml_dtypes import bfloat16

from chroma import Decoder, compile, jit
from chroma._src.program import Array, ArtifactManifest
from chroma.models.cached import export_cached
from chroma.models.gpt_oss import Transformer
from chroma.models.variants import dev
from tests.helpers import Function, random_transformer

BACKENDS: tuple[Literal["cpu", "metal"], ...] = ("cpu", "metal") if sys.platform == "darwin" else ("cpu",)


def numpy_bf16(tensor: torch.Tensor) -> Array:
    return tensor.detach().view(torch.uint16).numpy().view(bfloat16)


def close(
    actual: Array | tuple[Array, ...], expected: torch.Tensor | tuple[torch.Tensor, ...], *, rtol: float = 0.02, atol: float = 0.01
) -> None:
    results = actual if isinstance(actual, tuple) else (actual,)
    tensors = expected if isinstance(expected, tuple) else (expected,)
    dtypes = {torch.bfloat16: np.dtype(bfloat16), torch.float32: np.dtype(np.float32), torch.int64: np.dtype(np.int64)}
    for result, tensor in zip(results, tensors, strict=True):
        np.testing.assert_equal(result.dtype, dtypes[tensor.dtype])
        if tensor.dtype == torch.int64:
            np.testing.assert_array_equal(result, tensor.numpy())
        else:
            np.testing.assert_allclose(result.astype(np.float32), tensor.detach().float().numpy(), rtol=rtol, atol=atol)


def small_model() -> Transformer:
    config = dataclasses.replace(
        dev,
        num_hidden_layers=1,
        num_experts=2,
        experts_per_token=2,
        hidden_size=8,
        intermediate_size=8,
        vocab_size=17,
        head_dim=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        sliding_window=3,
    )
    return random_transformer(config, seed=5, std=0.1, dtype=torch.bfloat16)


class BFloat16Tests(unittest.TestCase):
    def test_storage_casts_views_reuse_and_torch_free_reload(self) -> None:
        class Buffers(torch.nn.Module):
            weight: torch.Tensor

            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("weight", torch.linspace(-1, 1, 128).reshape(4, 32).bfloat16())

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
                rounded = y.bfloat16()
                return x + self.weight, x.t(), x.reshape(-1), rounded, rounded.float(), self.weight

        module = Buffers()
        x = torch.linspace(-0.5, 0.5, 128).reshape(4, 32).bfloat16()
        y = torch.linspace(0.75, 1.25, 128).reshape(4, 32)
        y.view(-1)[:4] = torch.tensor([1 + 2**-8, 1 + 3 * 2**-8, -1 - 2**-8, -1 - 3 * 2**-8])
        y.view(-1)[4:9] = torch.tensor([0.0, -0.0, torch.inf, -torch.inf, torch.nan])
        array = numpy_bf16(x)
        expected = module(x, y)
        exported = torch.export.export(module, (x, y))
        with tempfile.TemporaryDirectory() as directory:
            for backend in BACKENDS:
                with self.subTest(backend=backend):
                    path = Path(directory) / backend
                    with compile(exported, path, backend=backend) as model:
                        self.assertEqual(model.inputs[0]["dtype"], "bfloat16")
                        self.assertEqual(model.stats["weight_bytes"], 256)
                        self.assertEqual(model.stats["output_bytes"], 1792)
                        manifest = cast(ArtifactManifest, json.loads((path / "model.json").read_text()))
                        weight_file = path / manifest["weights"]
                        self.assertEqual(weight_file.stat().st_size, 256)
                        np.testing.assert_array_equal(
                            np.fromfile(weight_file, dtype=bfloat16).astype(np.float32), module.weight.float().numpy().ravel()
                        )
                        result = cast(tuple[Array, ...], model(array, y.numpy()))
                        close(result, expected, rtol=0, atol=0)
                        self.assertTrue(np.signbit(result[3].astype(np.float32).ravel()[5]))
                        self.assertTrue(all(value.itemsize == 2 for i, value in enumerate(result) if i != 4))
                        self.assertFalse(np.shares_memory(result[1], result[2]))
                        close(model(x.t().contiguous().t(), y), expected, rtol=0, atol=0)
                        preserved = tuple(value.copy() for value in result)
                        model(x * 2, y + 1)
                        for value, before in zip(result, preserved, strict=True):
                            np.testing.assert_array_equal(value.view(np.uint8), before.view(np.uint8))
                        out: tuple[Array, ...] = tuple(np.empty_like(value) for value in result)
                        reused = cast(tuple[Array, ...], model(x, y, out=out))
                        close(reused, expected, rtol=0, atol=0)
                        for value, destination in zip(reused, out, strict=True):
                            self.assertIs(value, destination)
                        for invalid in (array.view(np.uint16), array.astype(np.float16), array.view("V2")):
                            with self.assertRaisesRegex(ValueError, "dtype"):
                                model(cast(Array, invalid), y.numpy())
                            wrong_outputs = (cast(Array, invalid.copy()), *out[1:])
                            with self.assertRaisesRegex(ValueError, "dtype"):
                                model(array, y.numpy(), out=wrong_outputs)
                    close(result, expected, rtol=0, atol=0)
                    np.save(path / "input.npy", array.view(np.uint16))
                    np.save(path / "float-input.npy", y.numpy())
                    np.save(path / "expected.npy", expected[0].float().numpy())
                    code = """
import sys
import numpy as np
from ml_dtypes import bfloat16
from chroma import Program
assert 'torch' not in sys.modules
path = sys.argv[1]
x = np.load(path + '/input.npy').view(bfloat16)
y = np.load(path + '/float-input.npy')
with Program(path) as model:
    result = model(x, y)
    assert result[0].dtype == np.dtype(bfloat16)
    np.testing.assert_array_equal(result[0].astype(np.float32), np.load(path + '/expected.npy'))
assert 'torch' not in sys.modules
"""
                    subprocess.run([sys.executable, "-c", code, str(path)], check=True)

    def test_operations_and_fused_rounding(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor, full: torch.Tensor, ids: torch.Tensor, one: torch.Tensor) -> tuple[torch.Tensor, ...]:
            values, indices = x.topk(3, dim=-1)
            return (
                x + y,
                x - y,
                x * y,
                x / y,
                x.sin(),
                x.cos(),
                x.exp(),
                x.sigmoid(),
                x.abs(),
                -x,
                x.square(),
                x.clamp(-0.25, 0.5),
                x.pow(2),
                (x.square() + 1).rsqrt(),
                x.sum(dim=1),
                x.mean(dim=0),
                x.sum(dtype=torch.float32),
                x.softmax(dim=-1),
                values,
                indices,
                x[ids],
                torch.cat((x, y.unsqueeze(0)), dim=0),
                torch.triu(x),
                torch.tril(x, diagonal=-1),
                x + full,
                (one + 2**-8) + 2**-8,
                (one * 0.333984375) - 0.333,
                torch.add(one * -0.30078125, one * 3, alpha=0.1),
            )

        module = Function(fn)
        inputs = (
            torch.linspace(-1, 1, 96).reshape(3, 32).bfloat16(),
            torch.linspace(0.5, 1.5, 32).bfloat16(),
            torch.linspace(-0.1, 0.1, 96).reshape(3, 32),
            torch.tensor([-1, 0], dtype=torch.int64),
            torch.ones(1, dtype=torch.bfloat16),
        )
        expected = fn(*inputs)
        exported = torch.export.export(module, inputs)
        with tempfile.TemporaryDirectory() as directory:
            for backend in BACKENDS:
                with self.subTest(backend=backend), compile(exported, Path(directory) / backend, backend=backend) as model:
                    actual = cast(tuple[Array, ...], model(*inputs))
                    close(actual, expected)
                    mixed, rounded, cancelled, scaled = actual[-4:]
                    np.testing.assert_array_equal(rounded.astype(np.float32), np.ones(1, dtype=np.float32))
                    np.testing.assert_allclose(mixed.astype(np.float32), expected[-4].numpy(), rtol=2e-5, atol=2e-6)
                    close((cancelled, scaled), expected[-2:], rtol=0, atol=0)
                    with self.assertRaises(IndexError):
                        model(*inputs[:3], torch.tensor([-4, 0]), inputs[4])
                    close(model(*inputs), expected)

    def test_linear_and_einsum_accumulate_in_float32(self) -> None:
        def fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.nn.functional.linear(x, weight, bias), torch.einsum("mk,nk->mn", x, weight), weight.sum(dim=1)

        x = torch.ones(2, 17, dtype=torch.bfloat16)
        weight = torch.full((5, 17), 2**-8, dtype=torch.bfloat16)
        weight[:, 0] = 1
        bias = torch.full((5,), 2**-5, dtype=torch.bfloat16)
        inputs = (x, weight, bias)
        exported = torch.export.export(Function(fn), inputs)
        targets: tuple[tuple[Literal["cpu", "metal"], bool], ...] = (("cpu", False), ("cpu", True))
        if "metal" in BACKENDS:
            targets += (("metal", True),)
        with tempfile.TemporaryDirectory() as directory:
            for backend, blas in targets:
                with (
                    self.subTest(backend=backend, blas=blas),
                    compile(exported, Path(directory) / f"{backend}-{blas}", backend=backend, blas=blas) as model,
                ):
                    actual = cast(tuple[Array, ...], model(*inputs))
                    close(actual, fn(*inputs), rtol=0, atol=0)
                    np.testing.assert_array_equal(actual[0].astype(np.float32), np.full((2, 5), 1.09375))
                    np.testing.assert_array_equal(actual[1].astype(np.float32), np.full((2, 5), 1.0625))
                    np.testing.assert_array_equal(actual[2].astype(np.float32), np.full(5, 1.0625))

    def test_small_transformer_with_bfloat16_weights(self) -> None:
        model = small_model()
        inputs = torch.tensor([1, 4, 2, 7])
        exported = torch.export.export(model, (inputs,))
        with tempfile.TemporaryDirectory() as directory:
            for backend in BACKENDS:
                with self.subTest(backend=backend), compile(exported, Path(directory) / backend, backend=backend) as native:
                    for tokens in (inputs, torch.tensor([0, 2, 9, 16])):
                        close(native(tokens), model(tokens), rtol=0.03, atol=0.005)

    def test_cached_prefill_decode_wrap_and_reset(self) -> None:
        model = small_model()
        context = 5
        tokens = (torch.arange(14) * 7 + 3) % 17
        expected = model(tokens)
        programs = [export_cached(model, context, steps) for steps in (context, 1)]
        with tempfile.TemporaryDirectory() as directory:
            for backend in BACKENDS:
                path = Path(directory) / backend
                with (
                    self.subTest(backend=backend),
                    compile(programs[0], path / "prefill", backend=backend) as prefill,
                    compile(programs[1], path / "decode", backend=backend) as decode,
                    Decoder(prefill, decode, sliding_window=3) as cached,
                ):
                    self.assertEqual(cached.stats["cache_bytes"], 2 * context * 4 * 2)
                    self.assertEqual(cached._state.keys.dtype, np.dtype(bfloat16))
                    position = 0
                    for size in (2, 1, 4, 5, 2):
                        logits = cached.append(tokens[position : position + size].numpy())
                        position += size
                        close(logits, expected[position - 1 : position], rtol=0.03, atol=0.005)
                        self.assertEqual(cached.position, position)
                    preserved = logits.copy()
                    cached.reset()
                    self.assertEqual(cached.position, 0)
                    self.assertEqual(cached._state.keys.dtype, np.dtype(bfloat16))
                    close(cached.append(tokens.numpy()), expected[-1:], rtol=0.03, atol=0.005)
                    np.testing.assert_array_equal(logits, preserved)

    def test_jit_bfloat16_tensor_input(self) -> None:
        @jit
        def function(x: torch.Tensor) -> torch.Tensor:
            return x.sin() + x

        x = torch.linspace(-1, 1, 16).bfloat16()
        close(function(x), x.sin() + x)
        output = cast(Array, function(x))
        self.assertEqual(output.dtype, np.dtype(bfloat16))
        self.assertIs(function(x * 2, out=output), output)
        close(output, (x * 2).sin() + x * 2)


if __name__ == "__main__":
    unittest.main()
