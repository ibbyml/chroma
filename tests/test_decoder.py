import copy
import dataclasses
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Never, cast
from unittest.mock import Mock, patch

import numpy as np
import torch

from chroma import Decoder, Program, compile
from chroma._src.program import Array
from chroma.models.cached import export_cached
from chroma.models.gpt_oss import Transformer, TransformerBlock
from chroma.models.variants import dev
from tests.helpers import random_transformer


def small_model(head_dim: int = 8) -> Transformer:
    config = dataclasses.replace(
        dev,
        num_hidden_layers=2,
        hidden_size=16,
        intermediate_size=16,
        vocab_size=31,
        num_experts=4,
        experts_per_token=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        sliding_window=3,
        initial_context_length=4096,
        rope_scaling_factor=32.0 if head_dim == 64 else 1.0,
    )
    return random_transformer(config, seed=5, std=0.03)


class DecoderTests(unittest.TestCase):
    def test_incompatible_program_pairs_fail_at_construction(self) -> None:
        def program(steps: int, capacity: int = 7, vocabulary: int = 5) -> Program:
            cache = [2, capacity, 1, 4]
            inputs = [
                ([steps], "int64"),
                ([steps], "float32"),
                (cache, "float32"),
                (cache, "float32"),
                ([steps, capacity + steps], "float32"),
                ([steps, capacity + steps], "float32"),
                ([1], "int64"),
            ]
            outputs = [([1, vocabulary], "float32"), ([2, steps, 1, 4], "float32"), ([2, steps, 1, 4], "float32")]
            return cast(
                Program,
                Mock(
                    spec=Program,
                    device="CPU",
                    inputs=[{"name": str(i), "shape": shape, "dtype": dtype} for i, (shape, dtype) in enumerate(inputs)],
                    outputs=[{"name": str(i), "shape": shape, "dtype": dtype} for i, (shape, dtype) in enumerate(outputs)],
                ),
            )

        with Decoder(program(3), program(1)) as decoder:
            self.assertEqual((decoder.context, decoder.vocab_size), (7, 5))
        for prefill, decode in ((program(3, capacity=4), program(1)), (program(3, vocabulary=3), program(1)), (program(3), program(2))):
            with self.subTest(prefill=prefill.inputs, decode=decode.inputs), self.assertRaisesRegex(ValueError, "Decoder"):
                Decoder(prefill, decode)
        broken = program(1)
        cast(Mock, broken).inputs = []
        with self.assertRaisesRegex(ValueError, "seven inputs"):
            Decoder(program(3), broken)

    def test_reject_invalid_sliding_window(self) -> None:
        program = Mock(spec=Program)
        for window in (-1, -7):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "sliding_window"):
                Decoder(program, program, sliding_window=window)

    def test_prefill_decode_wrap_and_reset_match_causal_reference(self) -> None:
        context = 7
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for head_dim in (8, 64):
                eager = small_model(head_dim)
                reference = copy.deepcopy(eager)
                for module in reference.block:
                    block = cast(TransformerBlock, module)
                    block.attn.sliding_window = min(block.attn.sliding_window or context, context)
                programs = [export_cached(eager, context, steps) for steps in (context, 1)]
                tokens = (torch.arange(23) * 7 + 3) % 31
                with torch.no_grad():
                    expected = reference(tokens).numpy()
                for backend in ("cpu", "metal") if sys.platform == "darwin" else ("cpu",):
                    path = root / f"{backend}-{head_dim}"
                    prefill = compile(programs[0], path / "prefill", backend=backend)
                    decode = compile(programs[1], path / "decode", backend=backend)
                    with Decoder(prefill, decode, sliding_window=3) as cached:
                        position = 0
                        for size in (3, 1, 5, 7, 1, 6):
                            logits = cached.append(tokens[position : position + size].numpy())
                            position += size
                            np.testing.assert_allclose(logits, expected[position - 1 : position], rtol=2e-5, atol=2e-6)
                            self.assertEqual(cached.position, position)
                        preserved = logits.copy()
                        cached.reset()
                        np.testing.assert_allclose(cached.append(tokens.numpy()), expected[-1:], rtol=2e-5, atol=2e-6)
                        cached.reset()
                        self.assertEqual(cached.position, 0)
                        np.testing.assert_allclose(cached.append(tokens[:3].numpy()), expected[2:3], rtol=2e-5, atol=2e-6)
                        np.testing.assert_array_equal(logits, preserved)
                        for invalid in ([-1], [31], [2**40]):
                            with self.assertRaises(IndexError):
                                cached.append(invalid)
                            self.assertEqual(cached.position, 3)
                        with self.assertRaises(TypeError):
                            cached.append(cast(Sequence[int], [1.5]))
                        with self.assertRaises(ValueError):
                            cached.append([])
                        before = cached._state
                        step = cached._models[1]

                        def interrupted(*args: Array, step: Program = step, out: Array | Sequence[Array] | None = None) -> Never:
                            step(*args, out=out)
                            raise KeyboardInterrupt

                        with patch.object(cached, "_models", (cached._models[0], interrupted)), self.assertRaises(KeyboardInterrupt):
                            cached.append([int(tokens[3])])
                        self.assertIs(cached._state, before)
                        np.save(path / "expected.npy", cached.append([int(tokens[3])]))
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        cached.append([1])
                    if head_dim == 8:
                        code = """
import sys
import numpy as np
from chroma import Program
from chroma import Decoder
assert 'torch' not in sys.modules
with Decoder(Program(sys.argv[1] + '/prefill'), Program(sys.argv[1] + '/decode'), sliding_window=3) as model:
    model.append([3, 10, 17])
    np.testing.assert_allclose(model.append([24]), np.load(sys.argv[1] + '/expected.npy'), rtol=2e-5, atol=2e-6)
assert 'torch' not in sys.modules
"""
                        subprocess.run([sys.executable, "-c", code, str(path)], check=True)


if __name__ == "__main__":
    unittest.main()
