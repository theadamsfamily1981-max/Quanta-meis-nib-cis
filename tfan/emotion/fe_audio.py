#!/usr/bin/env python
"""Audio Feature Extractor for PAD estimation."""

import torch
import torch.nn as nn


class AudioFeatureExtractor(nn.Module):
    """Extracts features from audio for PAD prediction."""

    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
        # Stub: would use Wav2Vec2/HuBERT
        self.encoder = nn.Linear(80, feature_dim)  # Mel-spec input

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract audio features [batch, time] → [batch, feature_dim]."""
        # Stub: random features
        batch = audio.shape[0]
        return torch.randn(batch, self.feature_dim, device=audio.device)
