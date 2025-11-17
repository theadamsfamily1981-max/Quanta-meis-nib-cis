"""
Enhanced Modality Adapters for MMF Bus.

Specialized adapters with:
- Prosody extraction for audio
- Optical flow for video
- Entity linking for text
- Emotion-aware feature extraction
"""

from .audio_prosody import AudioProsodyAdapter
from .video_optical import VideoOpticalAdapter
from .text_entity import TextEntityAdapter

__all__ = [
    "AudioProsodyAdapter",
    "VideoOpticalAdapter",
    "TextEntityAdapter",
]
