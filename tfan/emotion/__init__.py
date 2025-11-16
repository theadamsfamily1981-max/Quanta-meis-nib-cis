"""
Emotion prediction and neuromod components.

Includes:
- Original: EmotionHead, EmotionController
- PAD Engine v2: Audio/Face feature extractors, PAD regressor, trainer hooks
"""

from .head import EmotionHead
from .controller import EmotionController
from .fe_audio import AudioFeatureExtractor
from .fe_face import FaceFeatureExtractor
from .pad_regressor import PADRegressor
from .hooks import on_batch_metrics, PADTrainerHook

__all__ = [
    "EmotionHead",
    "EmotionController",
    "AudioFeatureExtractor",
    "FaceFeatureExtractor",
    "PADRegressor",
    "on_batch_metrics",
    "PADTrainerHook"
]
