#!/usr/bin/env python
"""PAD Regressor - Multi-modal → PAD [Pleasure, Arousal, Dominance]."""

import torch
import torch.nn as nn


class PADRegressor(nn.Module):
    """Regresses PAD values from audio + face features."""

    def __init__(self, audio_dim: int = 768, face_dim: int = 512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(audio_dim + face_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # P, A, D
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, audio_feat: torch.Tensor, face_feat: torch.Tensor) -> torch.Tensor:
        """Predict PAD [batch, 3] from audio + face features."""
        combined = torch.cat([audio_feat, face_feat], dim=-1)
        return self.fusion(combined)

    def loss(self, pred_pad: torch.Tensor, true_pad: torch.Tensor) -> torch.Tensor:
        """Compute MAE loss."""
        return torch.abs(pred_pad - true_pad).mean()
