#!/usr/bin/env python
"""
PAD Gate - Emotion-based Attention Modulation

Maps PAD (Pleasure-Arousal-Dominance) emotion vectors to:
- Temperature T for softmax (controls exploration)
- Keep ratio k for sparse attention (controls focus)

PAD space:
- Pleasure: [-1, 1] (negative to positive affect)
- Arousal: [-1, 1] (low to high energy)
- Dominance: [-1, 1] (submissive to dominant)

Mapping rules:
- High Arousal → High Temperature (more exploration)
- High Dominance → Low keep_ratio (more selective attention)

Usage:
    gate = PADGate(t_bounds=(0.6, 1.8), k_bounds=(0.2, 0.5))

    pad = torch.tensor([[0.5, 0.8, -0.3]])  # [batch, 3]
    T, k = gate.schedule(pad)

    # Use T and k in model
    model.set_temperature(T)
    model.set_keep_ratio(k)
"""

import torch
import torch.nn as nn
from typing import Tuple


class PADGate:
    """
    PAD-based scheduler for temperature and attention keep ratio.
    """

    def __init__(
        self,
        t_bounds: Tuple[float, float] = (0.6, 1.8),
        k_bounds: Tuple[float, float] = (0.2, 0.5)
    ):
        """
        Initialize PAD gate.

        Args:
            t_bounds: (min_temp, max_temp) for temperature
            k_bounds: (min_keep, max_keep) for keep ratio
        """
        self.t_lo, self.t_hi = t_bounds
        self.k_lo, self.k_hi = k_bounds

        print(f"✓ PADGate initialized")
        print(f"  Temperature bounds: {t_bounds}")
        print(f"  Keep ratio bounds: {k_bounds}")

    def schedule(self, pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute temperature and keep ratio from PAD vector.

        Args:
            pad: PAD vector [batch, 3] in range [-1, 1]
                 Format: [Pleasure, Arousal, Dominance]

        Returns:
            T: Temperature [batch]
            k: Keep ratio [batch]
        """
        # Extract components
        P, A, D = pad.unbind(-1)

        # Arousal → Temperature
        # High arousal (excited/energized) → higher temperature (more exploration)
        # Low arousal (calm/relaxed) → lower temperature (more exploitation)
        A_norm = (A.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
        T = self.t_lo + A_norm * (self.t_hi - self.t_lo)

        # Dominance → Keep ratio
        # High dominance (confident/in-control) → lower keep ratio (more selective)
        # Low dominance (uncertain/cautious) → higher keep ratio (more inclusive)
        D_norm = (D.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
        k = self.k_hi - D_norm * (self.k_hi - self.k_lo)

        return T, k

    def schedule_with_pleasure(
        self,
        pad: torch.Tensor,
        use_pleasure: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced scheduling that also considers Pleasure.

        Pleasure can modulate both T and k:
        - Positive pleasure → slightly higher T (openness to positive stimuli)
        - Negative pleasure → slightly lower k (cautious filtering)

        Args:
            pad: PAD vector [batch, 3]
            use_pleasure: Whether to incorporate Pleasure

        Returns:
            T: Temperature [batch]
            k: Keep ratio [batch]
        """
        P, A, D = pad.unbind(-1)

        # Base scheduling from Arousal and Dominance
        T, k = self.schedule(pad)

        if use_pleasure:
            # Modulate with Pleasure
            P_norm = (P.clamp(-1, 1) + 1) / 2  # [0, 1]

            # Positive pleasure → +10% temperature boost
            # Negative pleasure → -10% temperature penalty
            T_mod = 1.0 + 0.2 * (P_norm - 0.5)
            T = T * T_mod

            # Negative pleasure → reduce keep ratio (more selective filtering)
            k_mod = 1.0 - 0.2 * (0.5 - P_norm).clamp(0, 1)
            k = k * k_mod

            # Clamp to bounds
            T = T.clamp(self.t_lo, self.t_hi)
            k = k.clamp(self.k_lo, self.k_hi)

        return T, k


class LearnablePADGate(nn.Module):
    """
    Learnable PAD gate with trainable mapping.

    Instead of fixed rules, learns optimal mapping from PAD to T/k.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        t_bounds: Tuple[float, float] = (0.6, 1.8),
        k_bounds: Tuple[float, float] = (0.2, 0.5)
    ):
        """
        Initialize learnable PAD gate.

        Args:
            hidden_dim: Hidden dimension for MLP
            t_bounds: Temperature bounds
            k_bounds: Keep ratio bounds
        """
        super().__init__()

        self.t_lo, self.t_hi = t_bounds
        self.k_lo, self.k_hi = k_bounds

        # MLP: PAD [3] → (T, k) [2]
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute T and k from PAD.

        Args:
            pad: PAD vector [batch, 3]

        Returns:
            T: Temperature [batch]
            k: Keep ratio [batch]
        """
        # Pass through MLP
        output = self.mlp(pad)  # [batch, 2] in [0, 1]

        # Scale to bounds
        T = self.t_lo + output[:, 0] * (self.t_hi - self.t_lo)
        k = self.k_lo + output[:, 1] * (self.k_hi - self.k_lo)

        return T, k
