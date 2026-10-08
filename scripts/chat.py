from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import math
import os
import platform
import sysconfig
from collections import deque
from collections.abc import Generator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tiktoken
from numpy.typing import NDArray

from chroma import Decoder, Program, compile
from chroma._src.program import Array

if TYPE_CHECKING:
    from chroma.models.gpt_oss import Transformer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = ROOT / "build/graph/dev/model.pt2"
CACHE_ROOT = ROOT / "build/cache/chat"


def load_transformer(source: Path) -> Transformer:
    import torch

    from chroma.models.gpt_oss import ModelConfig, Transformer

    config = ModelConfig(**json.loads(source.with_name("config.json").read_text()))
    original = torch.export.load(source)
    model = Transformer(config, device=torch.device("cpu")).eval().requires_grad_(False)
    model.load_state_dict(original.state_dict, assign=True)
    return model


def load_model(source: Path, context: int, backend: Literal["cpu", "cuda", "metal"] = "cpu") -> Decoder:
    source = source.resolve()
    if not source.exists() and source == DEFAULT_EXPORT:
        from scripts.export import export_dev

        print("Creating the random-weight dev export...", flush=True)
        export_dev(source.parent)
    config = json.loads(source.with_name("config.json").read_text())
    if config["sliding_window"] < 0:
        raise ValueError(f"sliding_window {config['sliding_window']}")
    compiler = os.environ.get("NVCC", "nvcc") if backend == "cuda" else os.environ.get("CXX", "c++")
    fingerprint = hashlib.sha256(f"{context}:{backend}:{platform.platform()}:{sysconfig.get_config_var('SOABI')}:{compiler}".encode())
    package = sorted(path for path in (ROOT / "chroma").rglob("*") if path.suffix in {".py", ".h", ".metal"})
    files = [source, source.with_name("config.json"), Path(__file__), *package]
    for filename in files:
        with filename.open("rb") as file:
            fingerprint.update(hashlib.file_digest(file, "sha256").digest())
    directory = CACHE_ROOT / fingerprint.hexdigest()[:16]
    with ExitStack() as stack:
        models: list[Program] = []
        transformer: Transformer | None = None
        for name, steps in (("prefill", context), ("decode", 1)):
            path = directory / name
            if (path / "model.json").exists():
                model = Program(path)
            else:
                from chroma.models.cached import export_cached

                if transformer is None:
                    transformer = load_transformer(source)
                print(f"Compiling {backend} {name} with a {context}-token KV cache...", flush=True)
                model = compile(export_cached(transformer, context, steps), path, backend=backend)
            models.append(stack.enter_context(model))
        decoder = Decoder(models[0], models[1], sliding_window=config["sliding_window"])
        stack.pop_all()
        return decoder


def sample(
    logits: Array,
    allowed: NDArray[np.int64],
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
) -> int:
    scores = logits[allowed].astype(np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("The model produced non-finite logits")
    if temperature == 0:
        return int(allowed[np.argmax(scores)])
    count = min(top_k, len(scores))
    indices = np.argpartition(scores, len(scores) - count)[-count:]
    scores = scores[indices]
    scores = np.exp((scores - scores.max()) / temperature)
    return int(allowed[rng.choice(indices, p=scores / scores.sum())])


class Chat:
    def __init__(
        self,
        model: Decoder,
        tokenizer: tiktoken.Encoding,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.context = model.context
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)
        self.history: deque[int] = deque(maxlen=self.context)
        token = tokenizer.encode_single_token
        self.start = token("<|start|>")
        self.separator = token("<|message|>")
        self.end = token("<|end|>")
        self.finish = token("<|return|>")
        self.pad = token("<|endoftext|>")
        self.stop = {self.end, self.finish, self.pad}
        self.user_header = [self.start, *tokenizer.encode_ordinary("user"), self.separator]
        self.assistant_header = [
            self.start,
            *tokenizer.encode_ordinary("assistant"),
            token("<|channel|>"),
            *tokenizer.encode_ordinary("final"),
            self.separator,
        ]
        self.allowed = self._allowed_tokens(tokenizer, self.stop)
        self.reset()

    @staticmethod
    def _allowed_tokens(tokenizer: tiktoken.Encoding, stop: set[int]) -> NDArray[np.int64]:
        special = {tokenizer.encode_single_token(text) for text in tokenizer.special_tokens_set}
        allowed: list[int] = []
        for i in range(tokenizer.n_vocab):
            if i in special and i not in stop:
                continue
            try:
                tokenizer.decode_single_token_bytes(i)
            except KeyError:  # Vocabulary IDs can contain gaps.
                continue
            allowed.append(i)
        return np.array(allowed, dtype=np.int64)

    def reset(self) -> None:
        self.model.reset()
        self.history.clear()
        self._pending: tuple[int, list[int]] = (0, [])

    def _queue(self, tokens: Sequence[int]) -> None:
        start, pending = self._pending
        # The native call may have committed before Ctrl-C reached the generator.
        self._pending = (self.model.position, pending[self.model.position - start :] + list(tokens))
        self.history.extend(tokens)

    def reply(self, text: str) -> Generator[str]:
        self._queue([*self.user_header, *self.tokenizer.encode_ordinary(text), self.end, *self.assistant_header])
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            for _ in range(self.max_new_tokens):
                start, pending = self._pending
                logits = self.model.append(pending[self.model.position - start :])
                token = sample(logits[0], self.allowed, self.rng, self.temperature, self.top_k)
                if token in self.stop:
                    break
                self._queue([token])
                yield decoder.decode(self.tokenizer.decode_single_token_bytes(token))
            yield decoder.decode(b"", final=True)
        finally:
            self._queue([self.finish])


def terminal_text(text: str) -> str:
    return "".join(c for c in text if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) <= 159))


def interact(chat: Chat) -> None:
    print(f"Chat ready — KV cache retains {chat.context} tokens. /reset clears history; /quit exits.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            return
        if text in {"/quit", "/exit"}:
            return
        if text == "/reset":
            chat.reset()
            print("Conversation cleared.")
            continue
        if not text:
            continue
        print("Model: ", end="", flush=True)
        stream = chat.reply(text)
        try:
            for piece in stream:
                print(terminal_text(piece), end="", flush=True)
        except KeyboardInterrupt:
            print(" [stopped]", end="")
        except (IndexError, RuntimeError) as error:
            print(f" [generation failed: {error}]", end="")
        finally:
            stream.close()
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with a GPT-OSS export through the KV-cached Chroma decoder.")
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT, help="GPT-OSS .pt2 export with adjacent config.json")
    parser.add_argument("--context", type=int, default=256, help="Rolling KV cache capacity (default: 256)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum reply length (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature; 0 selects greedily")
    parser.add_argument("--top-k", type=int, default=40, help="Sample among the highest-scoring tokens (default: 40)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--backend", choices=["cpu", "cuda", "metal"], default="cpu", help="Native execution backend")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    for name in ("context", "max_new_tokens", "top_k"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.export.resolve() == DEFAULT_EXPORT:
        print("GPT-OSS dev chat — the default export has random weights, so replies will be gibberish.", flush=True)
    tokenizer = tiktoken.get_encoding("o200k_harmony")
    with load_model(args.export, args.context, args.backend) as model:
        if model.vocab_size != tokenizer.n_vocab:
            parser.error(f"vocab {model.vocab_size}, tokenizer {tokenizer.n_vocab}")
        print(f"Device: {model.device}", flush=True)
        chat = Chat(model, tokenizer, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, seed=args.seed)
        interact(chat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
