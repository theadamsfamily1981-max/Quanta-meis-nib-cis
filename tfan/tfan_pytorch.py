"""PyTorch implementation of the Temporal Fractal Attention Network (T-FAN)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TFANConfig:
    hidden_size: int = 64
    fractal_depth: int = 3
    dropout: float = 0.1


class FractalAttention(nn.Module):
    def __init__(self, config: TFANConfig, input_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, config.hidden_size)
        self.key = nn.Linear(input_dim, config.hidden_size)
        self.value = nn.Linear(input_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = config.hidden_size ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        return self.dropout(torch.matmul(weights, v))


class TFANBlock(nn.Module):
    def __init__(self, config: TFANConfig, input_dim: int):
        super().__init__()
        self.attention = FractalAttention(config, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        return self.norm(x + attended)


class TFAN(nn.Module):
    """Stacked TFAN blocks with residual connections."""

    def __init__(self, config: TFANConfig, input_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TFANBlock(config, input_dim) for _ in range(config.fractal_depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
