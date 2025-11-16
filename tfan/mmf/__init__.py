"""
TF-A-N Multimodal Fusion Bus (MMF)

Real-time fusion of audio/video/text/IMU streams with:
- Trainable Time Warping (TTW) alignment
- PAD (Pleasure-Arousal-Dominance) emotion gating
- Modality-specific adapters
- Late fusion with attention modulation

Hard gates:
- Alignment p95 < 5ms per stream pair (TTW)
- Late-fusion AUROC +≥3% vs no-TTW baseline
- PAD latency to scheduler < 20ms
"""

from .bus import FusionBus, StreamConfig
from .align import TTWAligner, align_streams
from .pad_gate import PADGate

__all__ = [
    'FusionBus',
    'StreamConfig',
    'TTWAligner',
    'align_streams',
    'PADGate'
]
