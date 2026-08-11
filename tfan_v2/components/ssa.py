"""Selective self-attention with dynamic temperature control."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = "Tensor"  # type: ignore
    nn = object  # type: ignore


def _no_grad():  # pragma: no cover - helper for optional torch
    if torch is None:
        def decorator(fn):
            return fn
        return decorator
    return torch.no_grad()


@dataclass
class SSAConfig:
    """Configuration container for :class:`SelectiveSelfAttention`."""

    embed_dim: int
    num_heads: int = 4
    topk_ratio: float = 0.5
    base_temperature: float = 1.0
    temperature_momentum: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1]")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")


class SelectiveSelfAttention(nn.Module):
    """Scaled dot product attention with lightweight selection."""

    def __init__(self, config: SSAConfig):
        if torch is None:  # pragma: no cover - guard for optional dependency
            raise ImportError("PyTorch is required to use SelectiveSelfAttention")
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.register_buffer("temperature", torch.tensor(config.base_temperature))

    def _shape(self, tensor: Tensor, seq_len: int, batch: int) -> Tuple[Tensor, Tensor, Tensor]:
        reshaped = (
            tensor.view(batch, seq_len, 3, self.config.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, batch * self.config.num_heads, seq_len, -1)
        )
        return reshaped[0], reshaped[1], reshaped[2]

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Optional[Tensor] = None,
        temperature: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform the selective self-attention operation."""

        batch, seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        q, k, v = self._shape(qkv, seq_len, batch)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        temp = temperature if temperature is not None else self.temperature
        probs = self._sparse_softmax(scores, temp)
        context = torch.matmul(probs, v)
        context = (
            context.view(batch, self.config.num_heads, seq_len, -1)
            .permute(0, 2, 1, 3)
            .reshape(batch, seq_len, -1)
        )
        return self.out_proj(context), probs

    def _sparse_softmax(self, scores: Tensor, temperature: Tensor) -> Tensor:
        """Apply a softmax with top-k masking."""

        k = max(1, int(scores.size(-1) * self.config.topk_ratio))
        topk_scores, _ = torch.topk(scores, k=k, dim=-1)
        kth_values = topk_scores[..., -1:, :]
        masked = torch.where(scores >= kth_values, scores, torch.full_like(scores, float("-inf")))
        scaled = masked / torch.clamp(temperature, min=1e-5)
        return torch.softmax(scaled, dim=-1)

    @_no_grad()
    def update_temperature(self, gradient_norm: float) -> float:
        """Momentum update for the internal temperature state."""

        target = self.config.base_temperature * (1 + gradient_norm)
        new_temp = (1 - self.config.temperature_momentum) * float(self.temperature) + self.config.temperature_momentum * target
        self.temperature.copy_(self.temperature.new_tensor(new_temp))
        return float(self.temperature)


__all__ = ["SSAConfig", "SelectiveSelfAttention"]
