#!/usr/bin/env python
"""Text Adapter - processes text into embeddings."""

import torch
import torch.nn as nn


class TextAdapter(nn.Module):
    """Text modality adapter using transformer embeddings."""

    def __init__(self, feature_dim: int = 768, vocab_size: int = 50257):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding = nn.Embedding(vocab_size, feature_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Process token IDs [batch, seq_len] → [batch, seq_len, dim]."""
        return self.embedding(tokens)
