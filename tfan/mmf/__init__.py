"""MMF - Multimodal Fusion Bus with TTW-PAD Gating."""

from .bus import MMFBus, MMFBusConfig, MMFBusOutput
from .pad_gate import PADGate, PADState
from .adapters import AudioProsodyAdapter, VideoOpticalAdapter, TextEntityAdapter

__all__ = [
    "MMFBus",
    "MMFBusConfig",
    "MMFBusOutput",
    "PADGate",
    "PADState",
    "AudioProsodyAdapter",
    "VideoOpticalAdapter",
    "TextEntityAdapter",
]
