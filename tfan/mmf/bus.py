#!/usr/bin/env python
"""
Multimodal Fusion Bus

Central hub for multimodal stream fusion with:
- Stream registration and synchronization
- Adapter dispatch to modality-specific processors
- Late fusion with learnable weights
- PAD-based attention modulation

Usage:
    bus = FusionBus()

    bus.register_stream('audio', AudioAdapter())
    bus.register_stream('video', VideoAdapter())
    bus.register_stream('text', TextAdapter())

    fused = bus.fuse(timestamp=1.234, pad_vector=pad)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class StreamConfig:
    """Configuration for a stream."""
    name: str
    modality: str  # 'audio', 'video', 'text', 'imu'
    feature_dim: int
    sample_rate: float  # Hz
    adapter: nn.Module


class FusionBus(nn.Module):
    """
    Multimodal fusion bus for real-time stream integration.
    """

    def __init__(
        self,
        fusion_dim: int = 512,
        enable_ttw: bool = True,
        enable_pad: bool = True
    ):
        """
        Initialize fusion bus.

        Args:
            fusion_dim: Dimension of fused representation
            enable_ttw: Enable time warping alignment
            enable_pad: Enable PAD emotion gating
        """
        super().__init__()

        self.fusion_dim = fusion_dim
        self.enable_ttw = enable_ttw
        self.enable_pad = enable_pad

        # Registered streams
        self.streams: Dict[str, StreamConfig] = {}

        # Fusion weights (learnable)
        self.fusion_weights = nn.ParameterDict()

        # Projection layers per modality
        self.projections = nn.ModuleDict()

        # Late fusion layer
        self.fusion_layer = None  # Initialized when streams registered

        print(f"✓ FusionBus initialized")
        print(f"  Fusion dim: {fusion_dim}")
        print(f"  TTW enabled: {enable_ttw}")
        print(f"  PAD enabled: {enable_pad}")

    def register_stream(
        self,
        name: str,
        config: StreamConfig
    ):
        """
        Register a new stream.

        Args:
            name: Stream identifier
            config: Stream configuration
        """
        if name in self.streams:
            raise ValueError(f"Stream {name} already registered")

        self.streams[name] = config

        # Create projection layer for this modality
        self.projections[name] = nn.Sequential(
            nn.Linear(config.feature_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU()
        )

        # Initialize fusion weight
        self.fusion_weights[name] = nn.Parameter(
            torch.ones(1) / len(self.streams)
        )

        # Reinitialize fusion layer
        self._init_fusion_layer()

        print(f"✓ Registered stream: {name} ({config.modality})")

    def _init_fusion_layer(self):
        """Initialize late fusion layer."""
        if len(self.streams) == 0:
            return

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )

    def fuse(
        self,
        features: Dict[str, torch.Tensor],
        pad_vector: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        keep_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Fuse multimodal features.

        Args:
            features: Dict mapping stream names to feature tensors
            pad_vector: Optional PAD emotion vector [batch, 3]
            temperature: Softmax temperature for fusion weights
            keep_ratio: Token keep ratio for sparse attention

        Returns:
            fused: Fused representation [batch, fusion_dim]
            info: Dict with fusion statistics
        """
        batch_size = next(iter(features.values())).shape[0]

        # Project each modality
        projected = {}
        for name, feat in features.items():
            if name not in self.streams:
                raise ValueError(f"Unknown stream: {name}")

            proj = self.projections[name](feat)
            projected[name] = proj

        # Compute fusion weights (with temperature)
        weights = {}
        weight_logits = torch.stack([
            self.fusion_weights[name] for name in projected.keys()
        ])

        weight_probs = torch.softmax(weight_logits / temperature, dim=0)

        for i, name in enumerate(projected.keys()):
            weights[name] = weight_probs[i]

        # Weighted fusion
        fused = torch.zeros(batch_size, self.fusion_dim, device=next(iter(features.values())).device)

        for name, proj_feat in projected.items():
            fused += weights[name] * proj_feat

        # Apply fusion layer
        if self.fusion_layer is not None:
            fused = self.fusion_layer(fused)

        # Gather info
        info = {
            'fusion_weights': {name: w.item() for name, w in weights.items()},
            'temperature': temperature,
            'keep_ratio': keep_ratio,
            'num_streams': len(projected)
        }

        return fused, info

    def get_stream_stats(self) -> Dict:
        """Get statistics about registered streams."""
        return {
            'num_streams': len(self.streams),
            'stream_names': list(self.streams.keys()),
            'modalities': [s.modality for s in self.streams.values()],
            'feature_dims': {name: s.feature_dim for name, s in self.streams.items()}
        }
