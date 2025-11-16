#!/usr/bin/env python
"""Facial Affect Feature Extractor for PAD estimation."""

import torch
import torch.nn as nn


class FaceFeatureExtractor(nn.Module):
    """Extracts facial features for PAD prediction."""

    def __init__(self, feature_dim: int = 512, num_landmarks: int = 68):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_landmarks * 2, 256),  # x,y coords
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Extract face features [batch, 68, 2] → [batch, feature_dim]."""
        batch = landmarks.shape[0]
        flat = landmarks.view(batch, -1)
        return self.encoder(flat)
