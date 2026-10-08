from typing import cast

import torch
from torch import Tensor, nn

from chroma.models.gpt_oss import Transformer, TransformerBlock


class CachedTransformer(nn.Module):
    """One prefill chunk or decode step against explicit K/V cache inputs; returns logits and the new K/V rows."""

    def __init__(self, model: Transformer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor,
        window_mask: Tensor,
        last: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = self.model.embedding(tokens)

        new_keys: list[Tensor] = []
        new_values: list[Tensor] = []

        for layer, module in enumerate(self.model.block):
            block = cast(TransformerBlock, module)
            attention = block.attn
            q, k, v = attention.project(x, positions)
            new_keys.append(k.unsqueeze(0))
            new_values.append(v.unsqueeze(0))
            past_shape = (keys.shape[1], attention.num_key_value_heads, attention.head_dim)
            all_k = torch.cat((keys[layer : layer + 1].reshape(past_shape), k), dim=0)
            all_v = torch.cat((values[layer : layer + 1].reshape(past_shape), v), dim=0)
            n, heads, groups, _ = q.shape
            all_k = all_k[:, :, None, :].expand(-1, -1, groups, -1)
            all_v = all_v[:, :, None, :].expand(-1, -1, groups, -1)
            scores = torch.einsum("qhmd,khmd->hmqk", q, all_k) * attention.sm_scale
            scores = scores + (window_mask if attention.sliding_window else mask)[None, None, :, :]
            sinks = attention.sinks.reshape(heads, groups, 1, 1).expand(-1, -1, n, -1)
            probabilities = torch.cat((scores, sinks), dim=-1).softmax(-1)[..., :-1]
            attended = torch.einsum("hmqk,khmd->qhmd", probabilities, all_v).reshape(n, -1)
            x = block.mlp(x + attention.out(attended))

        logits = self.model.unembedding(self.model.norm(x)[last])
        return logits, torch.cat(new_keys, dim=0), torch.cat(new_values, dim=0)


class UncachedTransformer(nn.Module):
    """A full forward over a padded window that projects only the selected position into the vocabulary."""

    def __init__(self, model: Transformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: Tensor, position: Tensor) -> Tensor:
        x = self.model.embedding(tokens)
        for block in self.model.block:
            x = block(x)
        return self.model.unembedding(self.model.norm(x)[position])


def export_cached(model: Transformer, context: int, steps: int) -> torch.export.ExportedProgram:
    if context < 1 or not 1 <= steps <= context:
        raise ValueError(f"{steps} steps, context {context}")
    first = cast(TransformerBlock, model.block[0]).attn
    shape = (len(model.block), context, first.num_key_value_heads, first.head_dim)
    dtype = first.qkv.weight.dtype
    inputs = (
        torch.zeros(steps, dtype=torch.int64),
        torch.arange(steps, dtype=torch.float32),
        torch.zeros(shape, dtype=dtype),
        torch.zeros(shape, dtype=dtype),
        torch.zeros(steps, context + steps, dtype=dtype),
        torch.zeros(steps, context + steps, dtype=dtype),
        torch.tensor([steps - 1], dtype=torch.int64),
    )
    with torch.no_grad():
        return torch.export.export(CachedTransformer(model).eval(), inputs)


def export_uncached(model: Transformer, context: int) -> torch.export.ExportedProgram:
    """The padded, uncached counterpart of export_cached, used as the reference for parity and benchmarks."""
    tokens = torch.arange(context, dtype=torch.int64) % model.embedding.num_embeddings
    with torch.no_grad():
        return torch.export.export(UncachedTransformer(model).eval(), (tokens, torch.tensor([context - 1], dtype=torch.int64)))
