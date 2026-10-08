import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar, Never, cast
from unittest.mock import patch

import numpy as np
import tiktoken
import torch
from numpy.typing import NDArray

from chroma import Decoder, compile
from chroma.models.cached import export_uncached
from chroma.models.variants import dev
from scripts.chat import Chat, interact, load_model, main, sample, terminal_text
from scripts.export import make_dev_model


class FakeModel:
    def __init__(self, tokenizer: tiktoken.Encoding, tokens: Sequence[int], context: int = 256) -> None:
        self.context, self.vocab_size, self.position = context, tokenizer.n_vocab, 0
        self.next_tokens = iter(tokens)
        self.end = tokenizer.encode_single_token("<|return|>")
        self.calls: list[tuple[NDArray[np.int64], int]] = []

    def reset(self) -> None:
        self.position = 0

    def append(self, tokens: Sequence[int] | NDArray[np.int64]) -> NDArray[np.float32]:
        self.calls.append((np.array(tokens), self.position))
        self.position += len(tokens)
        out = np.full((1, self.vocab_size), -10, dtype=np.float32)
        out[0, next(self.next_tokens, self.end)] = 10
        return out


class ChatTests(unittest.TestCase):
    def test_reject_negative_window_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.pt2"
            source.touch()
            source.with_name("config.json").write_text(json.dumps({"sliding_window": -1}))
            with patch("scripts.chat.compile") as build, self.assertRaisesRegex(ValueError, "sliding_window"):
                load_model(source, 32)
            build.assert_not_called()

    encoding: ClassVar[tiktoken.Encoding]

    @classmethod
    def setUpClass(cls) -> None:
        cls.encoding = tiktoken.get_encoding("o200k_harmony")

    def make_chat(self, text: str = " OK", *, context: int = 256, max_new_tokens: int = 64) -> tuple[Chat, FakeModel]:
        tokens = self.encoding.encode_ordinary(text) + [self.encoding.encode_single_token("<|return|>")]
        model = FakeModel(self.encoding, tokens * 4, context)
        chat = Chat(cast(Decoder, model), self.encoding, temperature=0, max_new_tokens=max_new_tokens)
        return chat, model

    def test_multi_turn_history_only_sends_new_tokens(self) -> None:
        chat, model = self.make_chat()
        self.assertEqual("".join(chat.reply("Hello")), " OK")
        previous = list(chat.history)
        self.assertEqual("".join(chat.reply("What about that?")), " OK")
        self.assertEqual(list(chat.history)[: len(previous)], previous)
        position = 0
        for tokens, start in model.calls:
            self.assertEqual(start, position)
            position += len(tokens)
        self.assertEqual(model.calls[2][0][0], chat.finish)
        text = "Literal <|return|> and <|start|> user text"
        chat.reset()
        list(chat.reply(text))
        self.assertIn(self.encoding.encode_ordinary(text)[0], chat.history)
        expected = [*chat.user_header, *self.encoding.encode_ordinary(text), chat.end, *chat.assistant_header]
        np.testing.assert_array_equal(model.calls[-2][0][: len(expected)], expected)

    def test_window_rolls_and_maximum_reply_length(self) -> None:
        chat, model = self.make_chat(" output" * 30, context=16, max_new_tokens=5)
        text = "message " * 100
        prompt = [*chat.user_header, *self.encoding.encode_ordinary(text), chat.end, *chat.assistant_header]
        list(chat.reply(text))
        self.assertEqual(len(model.calls), 5)
        self.assertEqual(len(chat.history), 16)
        np.testing.assert_array_equal(model.calls[0][0], prompt)
        self.assertTrue(all(len(tokens) == 1 for tokens, _ in model.calls[1:]))
        self.assertGreater(model.position, chat.context)
        self.assertEqual(chat.history[-1], chat.finish)

    def test_unicode_stream_and_interrupted_turn(self) -> None:
        chat, model = self.make_chat("🍀你好")
        self.assertEqual("".join(chat.reply("Hi")), "🍀你好")
        stream = chat.reply("Continue")
        next(stream)
        stream.close()
        self.assertEqual(chat.history[-1], chat.finish)
        pending = chat._pending[1]
        processed = model.position - chat._pending[0]
        self.assertEqual(pending[processed:][-1], chat.finish)
        self.assertEqual(terminal_text("hello\x1b[31m\x07\r\n世界\t"), "hello[31m\n世界\t")

    def test_interrupt_after_cache_commit_does_not_repeat_prompt(self) -> None:
        chat, model = self.make_chat()
        original = model.append

        def interrupted(tokens: Sequence[int] | NDArray[np.int64]) -> Never:
            original(tokens)
            raise KeyboardInterrupt

        with patch.object(model, "append", side_effect=interrupted), self.assertRaises(KeyboardInterrupt):
            list(chat.reply("First"))
        position = model.position
        list(chat.reply("Next"))
        self.assertEqual(model.calls[1][1], position)
        np.testing.assert_array_equal(
            model.calls[1][0],
            [chat.finish, *chat.user_header, *chat.tokenizer.encode_ordinary("Next"), chat.end, *chat.assistant_header],
        )

    def test_sampling_controls_and_special_tokens(self) -> None:
        allowed = np.array([0, 2, 3])
        logits = np.array([1.0, 1000.0, 2.0, 3.0])
        rng = np.random.default_rng(0)
        self.assertEqual(sample(logits, allowed, rng, 0, 3), 3)
        self.assertEqual(sample(logits, allowed, rng, 0.8, 1), 3)
        self.assertEqual(sample(logits, allowed, rng, 1e-300, 3), 3)
        self.assertTrue(all(sample(logits, allowed, rng, 1, 2) in {2, 3} for _ in range(20)))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            sample(np.array([np.nan]), np.array([0]), rng, 1, 1)
        chat, _ = self.make_chat()
        self.assertIn(chat.finish, chat.allowed)
        self.assertNotIn(chat.start, chat.allowed)
        self.assertNotIn(self.encoding.encode_single_token("<|channel|>"), chat.allowed)
        for arguments in (("--temperature", "-1"), ("--temperature", "nan"), ("--top-k", "0"), ("--max-new-tokens", "0")):
            with (
                patch.object(sys, "argv", ["chat.py", *arguments]),
                patch("scripts.chat.load_model") as load,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main()
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()

    def test_terminal_conversation_reset_quit_and_eof(self) -> None:
        chat, model = self.make_chat()
        output = io.StringIO()
        with (
            patch("builtins.input", side_effect=["", "Hello", "And then?", "/reset", "Fresh start", "/quit"]),
            contextlib.redirect_stdout(output),
        ):
            interact(chat)
        self.assertEqual(output.getvalue().count("Model:  OK"), 3)
        self.assertIn("Conversation cleared.", output.getvalue())
        expected = [*chat.user_header, *self.encoding.encode_ordinary("Fresh start"), chat.end, *chat.assistant_header]
        np.testing.assert_array_equal(model.calls[-2][0][: len(expected)], expected)
        with patch("builtins.input", side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()):
            interact(chat)

    def test_native_selected_logits_match_unpadded_torch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eager = make_dev_model(0)
            source = root / "model.pt2"
            torch.export.save(torch.export.export(eager, (torch.arange(8, dtype=torch.int64),)), source)
            (root / "config.json").write_text(json.dumps(asdict(dev)))
            program = export_uncached(eager, 32)
            with compile(program, root / "native") as native:
                self.assertEqual(native.stats["output_bytes"], dev.vocab_size * 4)
                self.assertLess(native.stats["workspace_bytes"], 1024 * 1024)
                for length in (1, 8, 32):
                    tokens = (torch.arange(length, dtype=torch.int64) * 7919 + 17) % dev.vocab_size
                    with torch.no_grad():
                        expected = eager(tokens)[-1:]
                    padded = np.full(32, self.encoding.encode_single_token("<|endoftext|>"), dtype=np.int64)
                    padded[:length] = tokens.numpy()
                    position = np.array([length - 1], dtype=np.int64)
                    actual = native(padded, position)
                    torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=2e-5, atol=2e-6)
                    padded[length:] = 12345
                    np.testing.assert_array_equal(native(padded, position), actual)
            with patch("scripts.chat.CACHE_ROOT", root / "cache"), load_model(source, 32) as decoder:
                self.assertEqual(decoder.stats["cache_bytes"], 2 * dev.num_hidden_layers * 32 * dev.num_key_value_heads * dev.head_dim * 4)
                self.assertEqual(decoder._models[1].inputs[0]["shape"], [1])
                for length in (1, 8, 32):
                    decoder.reset()
                    tokens = (torch.arange(length, dtype=torch.int64) * 7919 + 17) % dev.vocab_size
                    with torch.no_grad():
                        expected = eager(tokens)[-1:].numpy()
                    np.testing.assert_allclose(decoder.append(tokens.numpy()), expected, rtol=2e-5, atol=2e-6)
                chat = Chat(decoder, self.encoding, max_new_tokens=3, seed=42)
                first = "".join(chat.reply("Hello"))
                second = "".join(chat.reply("Tell me more"))
                self.assertTrue(first and second)
                self.assertEqual(chat.history[-1], chat.finish)
            code = """
import sys
from pathlib import Path
import scripts.chat as chat
chat.CACHE_ROOT = Path(sys.argv[2])
assert 'torch' not in sys.modules
with chat.load_model(Path(sys.argv[1]), 32) as model:
    assert model.append([1, 2, 3]).shape == (1, 201088)
assert 'torch' not in sys.modules
"""
            subprocess.run([sys.executable, "-c", code, str(source), str(root / "cache")], check=True)


if __name__ == "__main__":
    unittest.main()
