"""Emotion prediction and neuromodulation components."""

from .head import EmotionHead, EmotionPrediction, concordance_correlation_coefficient
from .controller import EmotionController
from .fe_audio import AudioEmotionFeatureExtractor
from .fe_face import FacialEmotionFeatureExtractor
from .pad_regressor import PADRegressor, PADPrediction
from .hooks import (
    PADEmotionHook,
    EmotionHookRegistry,
    create_emotion_aware_trainer_hook,
    create_emotion_aware_inference_hook,
)

__all__ = [
    # V1 Components
    "EmotionHead",
    "EmotionPrediction",
    "EmotionController",
    "concordance_correlation_coefficient",
    # PAD V2 Components
    "AudioEmotionFeatureExtractor",
    "FacialEmotionFeatureExtractor",
    "PADRegressor",
    "PADPrediction",
    "PADEmotionHook",
    "EmotionHookRegistry",
    "create_emotion_aware_trainer_hook",
    "create_emotion_aware_inference_hook",
]
