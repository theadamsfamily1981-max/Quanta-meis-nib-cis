"""
Modality-specific adapters for MMF bus.

Each adapter processes raw modality data into feature vectors.
"""

from .audio import AudioAdapter
from .video import VideoAdapter
from .text import TextAdapter
from .imu import IMUAdapter

__all__ = ['AudioAdapter', 'VideoAdapter', 'TextAdapter', 'IMUAdapter']
