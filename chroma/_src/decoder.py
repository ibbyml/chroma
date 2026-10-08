import threading
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self, TypedDict

import numpy as np
from numpy.typing import NDArray

from chroma._src.program import Array, Program, Stats, TensorSpec


@dataclass(frozen=True)
class CacheState:
    keys: Array
    values: Array
    slots: NDArray[np.int64]  # Absolute position held by each cache row, or -1.
    position: int


class DecoderStats(TypedDict):
    cache_bytes: int
    prefill: Stats
    decode: Stats


def validate(prefill: Program, decode: Program) -> tuple[TensorSpec, TensorSpec]:
    # export_cached lays these out as tokens, positions, K, V, mask, window mask, logit index.
    for name, model in (("prefill", prefill), ("decode", decode)):
        if len(model.inputs) != 7 or len(model.outputs) != 3:
            raise ValueError(f"Decoder {name} wants seven inputs and three outputs, not {len(model.inputs)} and {len(model.outputs)}")
    cache, logits = decode.inputs[2], decode.outputs[0]
    steps = prefill.inputs[0]["shape"][0]
    if (
        decode.inputs[0]["shape"] != [1]
        or prefill.inputs[2]["shape"] != cache["shape"]
        or prefill.outputs[0]["shape"] != logits["shape"]
        or not 1 <= steps <= cache["shape"][1]
    ):
        raise ValueError("Decoder prefill and decode were not exported as a pair")
    return cache, logits


class Decoder:
    def __init__(self, prefill: Program, decode: Program, *, sliding_window: int = 0) -> None:
        if sliding_window < 0:
            raise ValueError(f"sliding_window must be nonnegative, got {sliding_window}")
        cache, logits = validate(prefill, decode)
        self._models = (prefill, decode)
        self._shape = tuple(cache["shape"])
        self._cache_dtype = np.dtype(cache["dtype"])
        self._logit_dtype = np.dtype(logits["dtype"])
        self.context = self._shape[1]
        self.vocab_size = logits["shape"][1]
        self.device = decode.device
        self._window = min(sliding_window or self.context, self.context)  # 0 means the whole cache.
        self._buffers = [
            (
                [np.empty(spec["shape"], dtype=spec["dtype"]) for spec in model.inputs],
                [np.empty(spec["shape"], dtype=spec["dtype"]) for spec in model.outputs[1:]],
            )
            for model in self._models
        ]
        self._lock = threading.Lock()
        self._closed = False
        self._commit(self._empty())

    def _empty(self) -> CacheState:
        keys = np.zeros(self._shape, dtype=self._cache_dtype)
        return CacheState(keys, np.zeros_like(keys), np.full(self.context, -1, dtype=np.int64), 0)

    def _commit(self, state: CacheState) -> None:
        # Swapping in a whole new state keeps the cache consistent even if Ctrl-C lands mid-update.
        self._state = state
        for args, _ in self._buffers:
            args[2:4] = state.keys, state.values

    @property
    def position(self) -> int:
        return self._state.position

    @property
    def stats(self) -> DecoderStats:
        return {
            "cache_bytes": self._state.keys.nbytes + self._state.values.nbytes,
            "prefill": self._models[0].stats,
            "decode": self._models[1].stats,
        }

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Decoder is closed")
            self._commit(self._empty())

    def append(self, tokens: Sequence[int] | NDArray[np.int64]) -> Array:
        """Consume new token IDs and return the logits after the last one."""
        ids = np.asarray(tokens)
        if ids.ndim != 1 or not len(ids):
            raise ValueError(f"append wants a nonempty 1-D sequence of token IDs, got shape {ids.shape}")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"Token IDs must be integers, got {ids.dtype}")
        ids = ids.astype(np.int64)
        with self._lock:
            if self._closed:
                raise RuntimeError("Decoder is closed")
            offset = 0
            while offset < len(ids):
                remaining = len(ids) - offset
                index = 1 if remaining == 1 else 0  # One token uses the decode graph; anything else is chunked prefill.
                args, updates = self._buffers[index]
                steps = len(args[0])
                count = min(steps, remaining)

                state = self._state
                times = state.position + np.arange(steps, dtype=np.int64)
                key_times = np.concatenate((state.slots, times))
                valid = np.concatenate((state.slots >= 0, np.ones(steps, dtype=bool)))
                causal = valid[None, :] & (key_times[None, :] <= times[:, None])

                token_ids, positions, _, _, full_mask, window_mask, logit_index = args
                token_ids.fill(0)
                token_ids[:count] = ids[offset : offset + count]
                positions[:] = times
                for mask, window in ((full_mask, self.context), (window_mask, self._window)):
                    mask.fill(-np.inf)
                    mask[causal & (key_times[None, :] > times[:, None] - window)] = 0
                logit_index[0] = count - 1
                logits = np.empty((1, self.vocab_size), dtype=self._logit_dtype)
                self._models[index](*args, out=[logits, *updates])

                rows = times[:count] % self.context
                keys, values, slots = state.keys.copy(), state.values.copy(), state.slots.copy()
                keys[:, rows] = updates[0][:, :count]
                values[:, rows] = updates[1][:, :count]
                slots[rows] = times[:count]
                self._commit(CacheState(keys, values, slots, state.position + count))
                offset += count
            return logits

    def close(self) -> None:
        with self._lock:
            for model in self._models:
                model.close()
            self._closed = True
            self._buffers.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.close()
