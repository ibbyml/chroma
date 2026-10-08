import dataclasses
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar, Literal, cast

import numpy as np
import torch
from torch.utils._pytree import tree_leaves

from chroma import Program, compile
from chroma.models.cached import export_uncached
from chroma.models.gpt_oss import RMSNorm, Transformer
from chroma.models.variants import dev
from scripts.export import make_dev_model
from tests.helpers import Function, random_transformer


def check(native: Program, module: torch.nn.Module, *inputs: torch.Tensor) -> None:
    with torch.no_grad():
        expected = tree_leaves(module(*inputs))
    actual = native(*inputs)
    actual = actual if isinstance(actual, tuple) else (actual,)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(got), want, rtol=2e-5, atol=2e-6, equal_nan=True)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class MetalTests(unittest.TestCase):
    temporary: ClassVar[tempfile.TemporaryDirectory[str]]
    root: ClassVar[Path]
    eager: ClassVar[Transformer]
    path: ClassVar[Path]
    native: ClassVar[Program]

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.eager = make_dev_model(0)
        cls.path = cls.root / "dev"
        cls.native = compile(torch.export.export(cls.eager, (torch.arange(8, dtype=torch.int64),)), cls.path, backend="metal")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.native.close()
        cls.temporary.cleanup()

    def build(self, name: str, module: torch.nn.Module, *inputs: torch.Tensor) -> Program:
        module.eval()
        return compile(torch.export.export(module, inputs), self.root / name, backend="metal")

    def test_full_dev_model_runs_on_gpu(self) -> None:
        self.assertIn("Apple", self.native.device)
        self.assertEqual(self.native.stats["backend"], "metal")
        self.assertGreaterEqual(self.native.stats["parallel_reductions"], 2 * dev.num_hidden_layers + 1)
        self.assertGreater(self.native.stats["kernels"], self.native.stats["bf16_kernels"])
        # Eight-token, width-128 dev graph: keep the arena within the CPU-validated budget.
        self.assertLessEqual(self.native.stats["workspace_bytes"], 74 * 1024)
        self.assertLess(self.native.stats["workspace_bytes"], self.native.stats["unreused_workspace_bytes"])
        for tokens in (torch.arange(8), torch.arange(8) * 7919 + 17, torch.zeros(8, dtype=torch.int64), torch.arange(8).flip(0)):
            check(self.native, self.eager, tokens)
        reference = self.native(torch.arange(8))
        out = np.empty_like(reference)
        self.assertIs(self.native(torch.arange(8), out=out), out)
        np.testing.assert_array_equal(out, reference)
        self.native(torch.arange(8) + 1)
        np.testing.assert_array_equal(out, reference)

    def test_invalid_indices_recovery_and_thread_safety(self) -> None:
        for bad in (-1, dev.vocab_size, 2**40):
            with self.assertRaisesRegex(IndexError, "index"):
                self.native(np.full(8, bad, dtype=np.int64))
        expected = self.native(torch.arange(8))
        with ThreadPoolExecutor(max_workers=3) as pool:
            for output in pool.map(self.native, [torch.arange(8)] * 3):
                np.testing.assert_array_equal(output, expected)
        with Program(self.path) as another:
            np.testing.assert_array_equal(another(torch.arange(8)), expected)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            another(torch.arange(8))

    def test_reload_without_torch(self) -> None:
        code = """
import sys
import numpy as np
from chroma import Program
assert 'torch' not in sys.modules
with Program(sys.argv[1]) as model:
    result = model(np.arange(8, dtype=np.int64))
    assert np.isfinite(result).all()
    assert 'Apple' in model.device
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code, str(self.path)], check=True)

    def test_odd_shape_linear_and_aligned_bf16_linear(self) -> None:
        for width in (3, 17, 128, 132):
            module = torch.nn.Linear(width, 7).eval()
            with torch.no_grad():
                module.weight.mul_(0.1)
                if width >= 128:
                    module.weight.copy_(module.weight.bfloat16().float())
                    module.bias.copy_(module.bias.bfloat16().float())
            inputs = torch.randn(2, 3, width)
            with self.build(f"linear-{width}", module, inputs) as native:
                check(native, module, inputs)
                self.assertEqual(native.stats["bf16_kernels"], int(width >= 128))

    def test_tiled_linear_partial_rows_columns_and_inner(self) -> None:
        module = torch.nn.Linear(17, 67).eval()
        inputs = torch.randn(5, 17)
        with self.build("tiled", module, inputs) as native:
            self.assertEqual(native.stats["tiled_matmuls"], 1)
            check(native, module, inputs)
            check(native, module, inputs * 3)

    def test_norm_dispatch_preserves_fp32_weights(self) -> None:
        for width, exact in ((16, True), (17, True), (16, False), (64, False), (4096, True)):
            module = RMSNorm(width)
            if not exact:
                with torch.no_grad():
                    module.scale.uniform_(0.1, 1.0)
            inputs = torch.randn(5, width)
            with self.build(f"norm-{width}-{exact}", module, inputs) as native:
                check(native, module, inputs)
                self.assertEqual(native.stats["bf16_kernels"], int(width >= 4096 and exact))

    def test_views_broadcast_reductions_and_aliases(self) -> None:
        def fn(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
            z = (x + y).clamp(-3, 3)
            p = z.transpose(0, 1).reshape(2, 6)[:, 1::2]
            return p.square().mean(0), z, z, z.sum((0, 1), keepdim=True), y, (x + 2).sum()

        module = Function(fn)
        inputs = (torch.randn(3, 4), torch.randn(4))
        with self.build("views", module, *inputs) as native:
            check(native, module, *inputs)
            check(native, module, *(x * 5 for x in inputs))

    def test_topk_gather_nonfinite_softmax_and_einsum(self) -> None:
        def fn(x: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
            selected = x[ids]
            values, indices = torch.topk(selected, 3, dim=0, largest=False)
            return values, indices, selected.softmax(1), torch.tril(selected, -1)

        module = Function(fn)
        inputs = (torch.randn(6, 4), torch.tensor([4, -1, 0, 2], dtype=torch.int64))
        with self.build("select", module, *inputs) as native:
            check(native, module, *inputs)
            x = inputs[0].clone()
            x[0, 1] = float("nan")
            x[2, 2] = float("-inf")
            check(native, module, x, inputs[1])
        module = Function(lambda a, b: torch.einsum("bij,bjk->bik", a, b))
        inputs = (torch.randn(2, 3, 5), torch.randn(1, 5, 7))
        with self.build("einsum", module, *inputs) as native:
            check(native, module, *inputs)

    def test_general_shape_full_transformer(self) -> None:
        config = dataclasses.replace(
            dev,
            num_hidden_layers=2,
            hidden_size=17,
            intermediate_size=15,
            head_dim=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=257,
            sliding_window=3,
        )
        eager = random_transformer(config, seed=42, std=0.04)
        tokens = torch.arange(5, dtype=torch.int64)
        with self.build("odd-transformer", eager, tokens) as native:
            self.assertEqual(native.stats["bf16_kernels"], 0)
            check(native, eager, tokens)
            check(native, eager, tokens * 17 + 23)

    def test_chat_logits_match_unpadded_torch(self) -> None:
        with compile(export_uncached(self.eager, 32), self.root / "chat", backend="metal") as native:
            self.assertEqual(native.stats["output_bytes"], dev.vocab_size * 4)
            for length in (1, 8, 32):
                tokens = torch.arange(length, dtype=torch.int64) * 7919 % dev.vocab_size
                with torch.no_grad():
                    expected = self.eager(tokens)[-1:]
                padded = np.full(32, 199999, dtype=np.int64)
                padded[:length] = tokens.numpy()
                position = np.array([length - 1], dtype=np.int64)
                actual = native(padded, position)
                torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=2e-5, atol=2e-6)
                padded[length:] = 12345
                np.testing.assert_array_equal(native(padded, position), actual)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            compile(cast(torch.export.ExportedProgram, None), self.root / "invalid", backend=cast(Literal["cpu", "metal"], "missing"))


if __name__ == "__main__":
    unittest.main()
