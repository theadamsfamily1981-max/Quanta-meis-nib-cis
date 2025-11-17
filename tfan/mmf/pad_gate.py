#!/usr/bin/env python
"""
PAD Gate - Pleasure-Arousal-Dominance Emotion-Based Modality Gating

Uses PAD emotional state to adaptively weight modality streams:
- High arousal → prioritize audio (voice inflection, music)
- High pleasure → balance text and video (content understanding)
- High dominance → prioritize video (body language, facial cues)

Hard gates:
- PAD coherence > 0.85 (consistency across time)
- Gate decision latency < 2 ms
- No modality completely dropped unless confidence < 0.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass

from .ingest import ModalityStream


@dataclass
class PADState:
    """PAD emotional state."""

    pleasure: torch.Tensor  # (batch,) in [-1, 1]
    arousal: torch.Tensor  # (batch,) in [-1, 1]
    dominance: torch.Tensor  # (batch,) in [-1, 1]
    confidence: torch.Tensor  # (batch,) in [0, 1]


class PADGate(nn.Module):
    """
    PAD-based modality gating.

    Maps PAD state to modality weights using learned affine transformations.

    Modality weighting heuristics:
    - Audio weight ∝ |arousal| (emotional voice)
    - Video weight ∝ |dominance| (body language, facial expression)
    - Text weight ∝ |pleasure| + baseline (semantic content)
    """

    def __init__(
        self,
        d_model: int,
        coherence_threshold: float = 0.85,
        temperature: float = 1.0,
        mode: str = "soft",
        min_weight: float = 0.1,
    ):
        """
        Initialize PAD gate.

        Args:
            d_model: Model dimension (unused here, for compatibility)
            coherence_threshold: Minimum PAD coherence for gating
            temperature: Softmax temperature for weight normalization
            mode: "soft" (weighted fusion) or "hard" (select top-k)
            min_weight: Minimum weight for any modality
        """
        super().__init__()

        self.d_model = d_model
        self.coherence_threshold = coherence_threshold
        self.temperature = temperature
        self.mode = mode
        self.min_weight = min_weight

        # Learnable PAD → modality weight mappings
        # Each modality gets a 3D weight vector (one per PAD dimension)
        self.audio_weights = nn.Parameter(torch.tensor([0.0, 1.0, 0.0]))  # arousal
        self.video_weights = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]))  # dominance
        self.text_weights = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))  # pleasure

        # Bias terms
        self.audio_bias = nn.Parameter(torch.tensor(0.3))
        self.video_bias = nn.Parameter(torch.tensor(0.3))
        self.text_bias = nn.Parameter(torch.tensor(0.5))  # Text gets higher baseline

        # Coherence checker
        self.pad_coherence_window = 10  # Check last N predictions

    def compute_coherence(self, pad_state: PADState) -> torch.Tensor:
        """
        Compute PAD coherence (consistency).

        For now, use confidence as proxy. In full implementation,
        this would check temporal consistency across a window.

        Args:
            pad_state: Current PAD state

        Returns:
            coherence: (batch,) in [0, 1]
        """
        # Use confidence as coherence proxy
        # In full implementation, would track history and compute
        # temporal correlation or variance
        return pad_state.confidence

    def forward(
        self,
        streams: Dict[str, ModalityStream],
        pad_state: PADState,
    ) -> Dict:
        """
        Compute modality weights based on PAD state.

        Args:
            streams: Dict of modality -> ModalityStream
            pad_state: Current PAD emotional state

        Returns:
            Dict with:
                - weights: Dict[str, Tensor] - modality weights (batch,)
                - coherence: Tensor - PAD coherence (batch,)
                - gate_pass: bool - whether coherence gate passed
        """
        batch_size = pad_state.pleasure.shape[0]

        # Check coherence
        coherence = self.compute_coherence(pad_state)
        gate_pass = (coherence.mean() >= self.coherence_threshold).item()

        # Stack PAD dimensions
        pad_vector = torch.stack(
            [
                pad_state.pleasure,
                pad_state.arousal,
                pad_state.dominance,
            ],
            dim=1,
        )  # (batch, 3)

        # Compute raw weights for each modality
        raw_weights = {}

        for modality in streams.keys():
            if modality == "audio":
                # Audio weight = arousal * weight + bias
                weight = (
                    torch.sum(pad_vector * self.audio_weights.view(1, 3), dim=1)
                    + self.audio_bias
                )
            elif modality == "video":
                # Video weight = dominance * weight + bias
                weight = (
                    torch.sum(pad_vector * self.video_weights.view(1, 3), dim=1)
                    + self.video_bias
                )
            elif modality == "text":
                # Text weight = pleasure * weight + bias
                weight = (
                    torch.sum(pad_vector * self.text_weights.view(1, 3), dim=1)
                    + self.text_bias
                )
            else:
                # Unknown modality, use uniform weight
                weight = torch.ones(batch_size, device=pad_vector.device) * 0.5

            raw_weights[modality] = weight

        # Normalize weights
        if self.mode == "soft":
            # Softmax normalization with temperature
            weight_tensor = torch.stack(
                list(raw_weights.values()), dim=1
            )  # (batch, n_modalities)
            normalized = F.softmax(weight_tensor / self.temperature, dim=1)

            # Apply minimum weight threshold
            normalized = torch.clamp(normalized, min=self.min_weight)

            # Re-normalize to sum to 1
            normalized = normalized / normalized.sum(dim=1, keepdim=True)

            # Split back to dict
            weights = {}
            for i, modality in enumerate(raw_weights.keys()):
                weights[modality] = normalized[:, i]

        elif self.mode == "hard":
            # Select top-2 modalities
            weight_tensor = torch.stack(list(raw_weights.values()), dim=1)
            _, indices = torch.topk(weight_tensor, k=min(2, len(raw_weights)), dim=1)

            # Create hard weights
            hard_weights = torch.zeros_like(weight_tensor)
            hard_weights.scatter_(1, indices, 1.0)

            # Normalize
            hard_weights = hard_weights / hard_weights.sum(dim=1, keepdim=True)

            weights = {}
            for i, modality in enumerate(raw_weights.keys()):
                weights[modality] = hard_weights[:, i]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Scale weights by PAD confidence
        for modality in weights:
            weights[modality] = weights[modality] * pad_state.confidence

        return {
            "weights": weights,
            "coherence": coherence,
            "gate_pass": gate_pass,
            "raw_pad": pad_vector,
        }

    def get_gate_stats(
        self,
        gate_decisions: Dict,
    ) -> Dict:
        """
        Get statistics about gating decisions.

        Args:
            gate_decisions: Output from forward()

        Returns:
            Dict with gate statistics
        """
        weights = gate_decisions["weights"]

        stats = {
            "mean_weights": {
                modality: weight.mean().item() for modality, weight in weights.items()
            },
            "std_weights": {
                modality: weight.std().item() for modality, weight in weights.items()
            },
            "min_weights": {
                modality: weight.min().item() for modality, weight in weights.items()
            },
            "max_weights": {
                modality: weight.max().item() for modality, weight in weights.items()
            },
            "coherence_mean": gate_decisions["coherence"].mean().item(),
            "gate_pass": gate_decisions["gate_pass"],
        }

        return stats
