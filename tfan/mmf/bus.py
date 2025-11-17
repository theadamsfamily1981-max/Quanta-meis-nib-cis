#!/usr/bin/env python
"""
MMF Bus - Multimodal Fusion Bus with TTW-PAD Gating

Orchestrates the complete multimodal fusion pipeline:
1. Modality ingestion → ModalityStreams
2. TTW alignment → aligned timebases
3. PAD-gated fusion → unified representation
4. TLS landmark selection → sparse attention prep

Hard gates:
- TTW p95 latency < 5 ms
- TTW coverage ≥ 90%
- PAD gate coherence > 0.85
- Fusion overhead < 10% vs single-modal
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from .ingest import ModalityStream, ModalityAdapter
from .align import align_streams, validate_alignment
from .fuse import MultiModalFuser, FusedRepresentation
from .pad_gate import PADGate, PADState
from ..emotion import EmotionHead


@dataclass
class MMFBusConfig:
    """Configuration for MMF Bus."""
    d_model: int = 768
    modalities: List[str] = None  # Will default to ["text", "audio", "video"]

    # TTW alignment
    ttw_max_iter: int = 50
    ttw_p95_latency_ms: float = 5.0
    ttw_coverage_target: float = 0.90

    # PAD gating
    pad_coherence_threshold: float = 0.85
    pad_temperature: float = 1.0
    pad_mode: str = "soft"  # "soft" or "hard"

    # Fusion
    fusion_keep_ratio: float = 0.33
    fusion_alpha: float = 0.7
    n_heads: int = 12

    # Monitoring
    enable_profiling: bool = False
    log_alignment_metrics: bool = True


@dataclass
class MMFBusOutput:
    """Output from MMF Bus."""
    fused: FusedRepresentation
    pad_state: PADState
    alignment_metrics: Dict
    gate_decisions: Dict
    profiling: Optional[Dict] = None


class MMFBus(nn.Module):
    """
    Multimodal Fusion Bus.

    Orchestrates:
    - Modality ingestion
    - TTW temporal alignment
    - PAD emotional gating
    - Multi-modal fusion
    - TLS landmark selection

    Example:
        bus = MMFBus(config)
        output = bus(raw_inputs, emotion_context)
        fused_tokens = output.fused.tokens  # Ready for transformer
    """

    def __init__(
        self,
        config: Optional[MMFBusConfig] = None,
        adapters: Optional[Dict[str, ModalityAdapter]] = None,
    ):
        """
        Initialize MMF Bus.

        Args:
            config: Bus configuration
            adapters: Pre-initialized modality adapters (optional)
        """
        super().__init__()

        self.config = config or MMFBusConfig()

        if self.config.modalities is None:
            self.config.modalities = ["text", "audio", "video"]

        # Modality adapters
        if adapters is None:
            # Will be registered externally or use defaults
            self.adapters = nn.ModuleDict()
        else:
            self.adapters = nn.ModuleDict(adapters)

        # PAD gate
        self.pad_gate = PADGate(
            d_model=self.config.d_model,
            coherence_threshold=self.config.pad_coherence_threshold,
            temperature=self.config.pad_temperature,
            mode=self.config.pad_mode,
        )

        # Fusion module
        self.fuser = MultiModalFuser(
            d_model=self.config.d_model,
            modalities=self.config.modalities,
            n_heads=self.config.n_heads,
            keep_ratio=self.config.fusion_keep_ratio,
            alpha=self.config.fusion_alpha,
        )

        # Emotion head for PAD prediction (optional, can use pre-computed)
        self.emotion_head = EmotionHead(
            d_model=self.config.d_model,
            mode="PAD",
        )

        # Metrics tracking
        self.register_buffer("total_calls", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_latency_ms", torch.tensor(0.0))
        self.register_buffer("ttw_violations", torch.tensor(0, dtype=torch.long))

    def register_adapter(self, modality: str, adapter: ModalityAdapter):
        """
        Register a modality adapter.

        Args:
            modality: Modality name
            adapter: ModalityAdapter instance
        """
        self.adapters[modality] = adapter

    def forward(
        self,
        raw_inputs: Dict[str, torch.Tensor],
        emotion_context: Optional[torch.Tensor] = None,
        pad_override: Optional[PADState] = None,
        return_intermediates: bool = False,
    ) -> MMFBusOutput:
        """
        Process multi-modal inputs through the fusion bus.

        Args:
            raw_inputs: Dict of modality -> raw input tensor
            emotion_context: Optional pre-computed emotion embedding
            pad_override: Optional PAD state override
            return_intermediates: Return intermediate representations

        Returns:
            MMFBusOutput with fused representation and metadata
        """
        profiling = {} if self.config.enable_profiling else None
        t_start = time.perf_counter()

        # Step 1: Ingest modalities → ModalityStreams
        streams = {}
        for modality, raw_input in raw_inputs.items():
            if modality in self.adapters:
                t_ingest_start = time.perf_counter()
                streams[modality] = self.adapters[modality](raw_input)

                if profiling is not None:
                    profiling[f"ingest_{modality}_ms"] = (
                        time.perf_counter() - t_ingest_start
                    ) * 1000

        if not streams:
            raise ValueError("No valid modality streams produced")

        # Step 2: TTW Alignment
        t_align_start = time.perf_counter()

        aligned_streams, alignment_metrics = align_streams(
            streams,
            max_iter=self.config.ttw_max_iter,
            p95_latency_ms=self.config.ttw_p95_latency_ms,
            coverage_target=self.config.ttw_coverage_target,
        )

        # Validate alignment
        validation = validate_alignment(aligned_streams)
        if not all(validation.values()):
            raise RuntimeError(f"Alignment validation failed: {validation}")

        t_align_ms = (time.perf_counter() - t_align_start) * 1000

        if profiling is not None:
            profiling["align_ms"] = t_align_ms

        # Check TTW latency gate
        if t_align_ms > self.config.ttw_p95_latency_ms:
            self.ttw_violations += 1
            if self.training:
                raise RuntimeError(
                    f"TTW latency violation: {t_align_ms:.2f} ms > "
                    f"{self.config.ttw_p95_latency_ms} ms threshold"
                )

        # Step 3: PAD Gating
        t_pad_start = time.perf_counter()

        # Compute or use pre-computed PAD state
        if pad_override is not None:
            pad_state = pad_override
        else:
            # Extract representative features for PAD prediction
            # Use mean-pooled features from all modalities
            rep_features = []
            for stream in aligned_streams.values():
                pooled = stream.features.mean(dim=1)  # (batch, feat_dim)
                rep_features.append(pooled)

            # Average across modalities
            combined_features = torch.stack(rep_features).mean(dim=0)  # (batch, feat_dim)

            # Project to d_model if needed
            if combined_features.shape[-1] != self.config.d_model:
                # Simple linear projection
                if not hasattr(self, 'pad_projection'):
                    self.pad_projection = nn.Linear(
                        combined_features.shape[-1],
                        self.config.d_model
                    ).to(combined_features.device)
                combined_features = self.pad_projection(combined_features)

            # Predict PAD
            emotion_pred = self.emotion_head(combined_features.unsqueeze(1))

            pad_state = PADState(
                pleasure=emotion_pred.pleasure.squeeze(1) if emotion_pred.pleasure is not None else emotion_pred.valence.squeeze(1),
                arousal=emotion_pred.arousal.squeeze(1),
                dominance=emotion_pred.dominance.squeeze(1),
                confidence=emotion_pred.confidence.squeeze(1) if emotion_pred.confidence is not None else torch.ones_like(emotion_pred.arousal.squeeze(1)),
            )

        # Apply PAD gate to select/weight modalities
        gate_decisions = self.pad_gate(aligned_streams, pad_state)

        if profiling is not None:
            profiling["pad_ms"] = (time.perf_counter() - t_pad_start) * 1000

        # Step 4: Fusion
        t_fuse_start = time.perf_counter()

        # Apply gate weights to streams
        weighted_streams = {}
        for modality, stream in aligned_streams.items():
            weight = gate_decisions["weights"][modality]

            # Apply weight to features
            weighted_features = stream.features * weight.view(-1, 1, 1)

            weighted_streams[modality] = ModalityStream(
                features=weighted_features,
                timestamps=stream.timestamps,
                modality=stream.modality,
                confidence=stream.confidence * weight.item(),
                metadata=stream.metadata,
            )

        # Fuse streams
        fused = self.fuser(weighted_streams)

        if profiling is not None:
            profiling["fuse_ms"] = (time.perf_counter() - t_fuse_start) * 1000

        # Total time
        t_total_ms = (time.perf_counter() - t_start) * 1000

        if profiling is not None:
            profiling["total_ms"] = t_total_ms

        # Update metrics
        self.total_calls += 1
        self.total_latency_ms += t_total_ms

        # Build output
        output = MMFBusOutput(
            fused=fused,
            pad_state=pad_state,
            alignment_metrics=alignment_metrics,
            gate_decisions=gate_decisions,
            profiling=profiling,
        )

        return output

    def get_metrics(self) -> Dict:
        """Get bus performance metrics."""
        return {
            "total_calls": self.total_calls.item(),
            "avg_latency_ms": (
                self.total_latency_ms / self.total_calls
            ).item() if self.total_calls > 0 else 0.0,
            "ttw_violations": self.ttw_violations.item(),
            "ttw_violation_rate": (
                self.ttw_violations / self.total_calls
            ).item() if self.total_calls > 0 else 0.0,
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self.total_calls.zero_()
        self.total_latency_ms.zero_()
        self.ttw_violations.zero_()
