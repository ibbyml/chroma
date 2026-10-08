import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import torch
from torch.utils._pytree import tree_leaves

from chroma import Program, compile
from chroma._src.memory import Allocation, plan_memory
from chroma._src.program import Array
from chroma.models.gpt_oss import Transformer
from chroma.models.variants import dev
from scripts.export import make_dev_model
from tests.helpers import Function, random_transformer


def export(module: torch.nn.Module, *inputs: torch.Tensor) -> torch.export.ExportedProgram:
    with torch.no_grad():
        return torch.export.export(module.eval(), inputs)


def close(
    actual: Array | tuple[Array, ...], expected: torch.Tensor | tuple[torch.Tensor, ...], *, atol: float = 2e-6, rtol: float = 2e-5
) -> None:
    actual = actual if isinstance(actual, tuple) else (actual,)
    for a, e in zip(actual, tree_leaves(expected), strict=True):
        torch.testing.assert_close(torch.from_numpy(a), e, atol=atol, rtol=rtol, equal_nan=True)


class RuntimeTests(unittest.TestCase):
    temporary: ClassVar[tempfile.TemporaryDirectory[str]]
    root: ClassVar[Path]
    torch_model: ClassVar[Transformer]
    tokens: ClassVar[torch.Tensor]
    program: ClassVar[torch.export.ExportedProgram]
    path: ClassVar[Path]
    native: ClassVar[Program]

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.torch_model = make_dev_model(0)
        cls.tokens = torch.arange(8, dtype=torch.int64)
        cls.program = export(cls.torch_model, cls.tokens)
        cls.path = cls.root / "gpt"
        cls.native = compile(cls.program, cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.native.close()
        cls.temporary.cleanup()

    def test_full_gpt_oss_changed_routing_and_repeated_calls(self) -> None:
        for tokens in [self.tokens, (self.tokens * 7919 + 17) % dev.vocab_size, torch.zeros_like(self.tokens), self.tokens.flip(0)]:
            with torch.no_grad():
                expected = self.torch_model(tokens)
            close(self.native(tokens), expected)
        first = cast(Array, self.native(self.tokens))
        preserved = first.copy()
        self.native(self.tokens + 1)
        np.testing.assert_array_equal(first, preserved)
        self.assertEqual(self.native.stats["expert_contractions"], 2 * dev.num_hidden_layers)
        # Eight-token, width-128 dev graph: the freshly validated arena is 75,776 bytes.
        self.assertLessEqual(self.native.stats["workspace_bytes"], 74 * 1024)
        self.assertLess(self.native.stats["workspace_bytes"], self.native.stats["unreused_workspace_bytes"])

    def test_torch_free_reload(self) -> None:
        code = """
import sys
import numpy as np
from chroma import Program
assert 'torch' not in sys.modules
with Program(sys.argv[1]) as model:
    result = model(np.arange(8, dtype=np.int64))
    np.save(sys.argv[2], result)
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code, str(self.path), str(self.root / "result.npy")], check=True)
        np.testing.assert_array_equal(np.load(self.root / "result.npy"), self.native(self.tokens))

    def test_input_validation_output_reuse_and_threads(self) -> None:
        expected = self.native(self.tokens)
        result = np.empty_like(expected)
        self.assertIs(self.native(self.tokens, out=result), result)
        np.testing.assert_array_equal(result, expected)
        for invalid in [np.arange(8, dtype=np.int32), np.arange(7, dtype=np.int64)]:
            with self.assertRaises(ValueError):
                self.native(cast(Array, invalid))
        with self.assertRaises(AttributeError):
            self.native(cast(Array, list(range(8))))
        with self.assertRaises(ValueError):
            self.native()
        with self.assertRaises(IndexError):
            self.native(np.full(8, dev.vocab_size, dtype=np.int64))
        with self.assertRaises(IndexError):
            self.native(np.full(8, -1, dtype=np.int64))
        result.setflags(write=False)
        with self.assertRaises(ValueError):
            self.native(self.tokens, out=result)
        noncontiguous = np.arange(16, dtype=np.int64)[::2]
        close(self.native(noncontiguous), self.torch_model(torch.from_numpy(noncontiguous)))
        with ThreadPoolExecutor(max_workers=3) as pool:
            outputs = list(pool.map(self.native, [self.tokens] * 3))
        for output in outputs:
            np.testing.assert_array_equal(output, expected)
        with Program(self.path) as other:
            np.testing.assert_array_equal(other(self.tokens), expected)
        other.close()
        with self.assertRaises(RuntimeError):
            other(self.tokens)

    def build_function[T](
        self, name: str, function: Callable[..., T], inputs: tuple[torch.Tensor, ...], blas: bool = False
    ) -> tuple[Function[T], Program]:
        module = Function(function)
        program = export(module, *inputs)
        return module, compile(program, self.root / name, blas=blas)

    def test_broadcast_views_reduction_and_tuple_alias_lifetimes(self) -> None:
        def function(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, ...]:
            z = x + y * 2
            r = z.transpose(0, 1).reshape(2, 6)
            s = r[:, 1::2]
            return ((s.sin() * s).mean(dim=0), z, z, z.sum(dim=(0, 1), keepdim=True), y)

        inputs = (torch.randn(3, 4), torch.randn(4))
        module, native = self.build_function("views", function, inputs)
        with native:
            close(native(*inputs), module(*inputs))
            close(native(*(x * 5 for x in inputs)), module(*(x * 5 for x in inputs)))
            with self.assertRaises(ValueError):
                outputs = list(native(*inputs))
                outputs[2] = outputs[1]
                native(*inputs, out=outputs)

    def test_inplace_is_functionalized_without_corrupting_alias(self) -> None:
        def function(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            z = x * 2
            alias = z[:, ::2]
            z.add_(3)
            return z, alias, x

        inputs = (torch.randn(3, 4),)
        module, native = self.build_function("inplace", function, inputs)
        with native:
            close(native(*inputs), module(*inputs))

    def test_indexing_softmax_topk_and_nonfinite_values(self) -> None:
        def function(x: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
            selected = x[indices]
            values, positions = torch.topk(selected, 3, dim=0, largest=False)
            return values, positions, selected.softmax(dim=1), torch.triu(selected, diagonal=-1)

        inputs = (torch.randn(6, 4), torch.tensor([4, -1, 0, 2], dtype=torch.int64))
        module, native = self.build_function("routing", function, inputs)
        with native:
            close(native(*inputs), module(*inputs))
            x = inputs[0].clone()
            x[0, 1] = float("nan")
            x[2, 2] = float("-inf")
            close(native(x, inputs[1]), module(x, inputs[1]))
            with self.assertRaises(IndexError):
                native(inputs[0], torch.tensor([6, 0, 0, 0]))

    def test_topk_ties_and_partial_sort(self) -> None:
        for width in (12, 256):

            def function(x: torch.Tensor) -> torch.return_types.topk:
                return torch.topk(x, 4)

            inputs = (torch.zeros(2, width),)
            module, native = self.build_function(f"topk-{width}", function, inputs)
            with native:
                # Tie order is unspecified by Torch; selected values and valid unique indices are required.
                values, indices = native(*inputs)
                np.testing.assert_array_equal(values, 0)
                self.assertTrue(all(len(set(row)) == 4 for row in indices))
                x = torch.rand(2, width)
                close(native(x), module(x))

    def test_portable_linear_and_scalar_outputs(self) -> None:
        module = torch.nn.Linear(8, 3).eval()
        inputs = (torch.randn(8),)
        with compile(export(module, *inputs), self.root / "linear", blas=False) as native:
            close(native(*inputs), module(*inputs))
        module, native = self.build_function("scalar", lambda x: ((x + 1).mean(), x * 2), (torch.tensor(3.0),))
        with native:
            close(native(torch.tensor(3.0)), module(torch.tensor(3.0)))

    def test_general_einsum_and_constant_only_graph(self) -> None:
        inputs = (torch.randn(2, 3, 4), torch.randn(1, 4, 5))
        module, native = self.build_function("einsum", lambda a, b: torch.einsum("bij,bjk->bik", a, b), inputs)
        with native:
            close(native(*inputs), module(*inputs))
        module, native = self.build_function("constant", lambda: torch.arange(7).to(torch.float32).cos(), ())
        with native:
            close(native(), module())

    def test_lengths_sliding_window_and_yarn(self) -> None:
        for length in (1, 17):
            config = dataclasses.replace(
                dev,
                num_hidden_layers=2,
                hidden_size=16,
                head_dim=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=257,
                sliding_window=3,
                initial_context_length=4096,
                rope_scaling_factor=32.0,
            )
            model = random_transformer(config, seed=length, std=0.04)
            tokens = torch.arange(length, dtype=torch.int64)
            with compile(export(model, tokens), self.root / f"yarn-{length}") as native:
                close(native(tokens), model(tokens), atol=4e-6)
                tokens = (tokens * 19 + 23) % config.vocab_size
                close(native(tokens), model(tokens), atol=4e-6)

    def test_long_expression_chain(self) -> None:
        def function(x: torch.Tensor) -> torch.Tensor:
            for _ in range(180):
                x = (x * 0.99 + 0.02).sin()
            return x

        inputs = (torch.randn(4, 5),)
        module, native = self.build_function("deep", function, inputs)
        with native:
            close(native(*inputs), module(*inputs))
            self.assertLess(native.stats["kernels"], 25)

    def test_reject_unsupported_export_and_corrupt_weights(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "unsupported operator"):
            compile(export(Function(lambda x: torch.linalg.det(x)), torch.eye(3)), self.root / "unsupported")
        with self.assertRaisesRegex(ValueError, "dtype"):
            compile(export(torch.nn.Linear(2, 2).double(), torch.ones(2, dtype=torch.float64)), self.root / "double")

        class Mutation(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.add_(1)

        class BufferMutation(torch.nn.Module):
            count: torch.Tensor

            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("count", torch.zeros(2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.count.add_(1)
                return x + self.count

        for module in (Mutation(), BufferMutation()):
            with self.assertRaisesRegex(ValueError, "mutation"):
                compile(export(module, torch.ones(2)), self.root / "mutation")
        with self.assertRaisesRegex(TypeError, "Only tensor outputs"):
            compile(export(Function(lambda x: (x, 3)), torch.ones(2)), self.root / "constant-output")
        dynamic = torch.export.export(
            Function(lambda x: x.sin()), (torch.ones(4),), dynamic_shapes=(({0: torch.export.Dim("length", min=1, max=16)},),)
        )
        with self.assertRaisesRegex(ValueError, "Dynamic"):
            compile(dynamic, self.root / "dynamic")
        bad = self.root / "bad"
        bad.mkdir()
        manifest = json.loads((self.path / "model.json").read_text())
        os.symlink(self.path / manifest["library"], bad / manifest["library"])
        (bad / "model.json").write_text(json.dumps(manifest))
        (bad / manifest["weights"]).write_bytes(b"bad")
        with self.assertRaisesRegex(RuntimeError, "byte size"):
            Program(bad)
        manifest["inputs"][0]["shape"] = [1]
        (bad / "model.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "metadata"):
            Program(bad)


class MemoryTests(unittest.TestCase):
    def test_inclusive_lifetimes_alignment_and_coalescing(self) -> None:
        allocations = {
            "a": Allocation(64, 0, 1),
            "b": Allocation(128, 1, 2),
            "c": Allocation(64, 2, 3),
            "d": Allocation(192, 4, 5),
        }
        self.assertEqual(plan_memory(allocations), 192)
        self.assertEqual(allocations["a"].offset, allocations["c"].offset)
        self.assertEqual(allocations["d"].offset, 0)
        for left in allocations.values():
            for right in allocations.values():
                if left is right or left.end < right.start or right.end < left.start:
                    continue
                self.assertTrue(left.offset + left.size <= right.offset or right.offset + right.size <= left.offset)


if __name__ == "__main__":
    unittest.main()
