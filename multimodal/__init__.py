"""Multimodal research utilities package."""

from .consolidation import svd_consolidate
from .emotion import (
    EmotionClassifierHead,
    MFCCFeatureExtractor,
    PADHead,
    ProsodyFeatureExtractor,
)
from .shared import SharedProjection, CrossModalAttention, ModalityDropout

__all__ = [
    "SharedProjection",
    "CrossModalAttention",
    "ModalityDropout",
    "ProsodyFeatureExtractor",
    "MFCCFeatureExtractor",
    "PADHead",
    "EmotionClassifierHead",
    "svd_consolidate",
]
