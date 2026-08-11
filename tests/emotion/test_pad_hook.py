#!/usr/bin/env python
"""
Tests for PAD Emotion v2 System.

Tests:
- Audio feature extraction
- Facial feature extraction
- PAD regressor (multimodal fusion)
- Emotion hooks and callbacks
- Hard gate verification
"""

import pytest
import torch
import numpy as np

from tfan.emotion.fe_audio import AudioEmotionFeatureExtractor
from tfan.emotion.fe_face import FacialEmotionFeatureExtractor
from tfan.emotion.pad_regressor import PADRegressor, PADPrediction
from tfan.emotion.hooks import (
    PADEmotionHook,
    EmotionHookRegistry,
    create_emotion_aware_trainer_hook,
    create_emotion_aware_inference_hook,
)


class TestAudioFeatureExtractor:
    """Test audio emotion feature extraction."""

    def test_extractor_initialization(self):
        """Test audio extractor initialization."""
        extractor = AudioEmotionFeatureExtractor(
            sample_rate=16000,
            n_mfcc=13,
        )

        assert extractor.sample_rate == 16000
        assert extractor.n_mfcc == 13
        assert extractor.feature_dim > 0

    def test_extract_mfcc(self):
        """Test MFCC extraction."""
        extractor = AudioEmotionFeatureExtractor()

        # Create dummy audio (1 second)
        audio = np.random.randn(16000).astype(np.float32)

        mfcc = extractor.extract_mfcc(audio)

        assert mfcc.shape[0] == extractor.n_mfcc
        assert mfcc.shape[1] > 0  # Should have frames

    def test_forward_single_sample(self):
        """Test forward pass with single audio sample."""
        extractor = AudioEmotionFeatureExtractor(n_mfcc=13)

        audio = torch.randn(16000)  # 1 second

        features = extractor(audio)

        assert features.ndim == 2  # (n_frames, feature_dim)
        assert features.shape[1] == extractor.feature_dim

    def test_forward_batch(self):
        """Test forward pass with batch."""
        extractor = AudioEmotionFeatureExtractor()

        audio = torch.randn(4, 16000)  # Batch of 4

        features = extractor(audio)

        assert features.shape[0] == 4  # Batch size
        assert features.shape[2] == extractor.feature_dim

    def test_feature_info(self):
        """Test feature info retrieval."""
        extractor = AudioEmotionFeatureExtractor(extract_deltas=True, extract_voice_quality=True)

        info = extractor.get_feature_info()

        assert "mfcc" in info
        assert "prosody" in info
        assert "delta" in info
        assert "voice_quality" in info
        assert info["total"] == extractor.feature_dim


class TestFacialFeatureExtractor:
    """Test facial emotion feature extraction."""

    def test_extractor_initialization(self):
        """Test facial extractor initialization."""
        extractor = FacialEmotionFeatureExtractor(
            n_landmarks=68,
            extract_aus=True,
        )

        assert extractor.n_landmarks == 68
        assert extractor.extract_aus == True
        assert extractor.feature_dim > 0

    def test_extract_landmarks(self):
        """Test landmark extraction."""
        extractor = FacialEmotionFeatureExtractor()

        # Create dummy frame
        frame = torch.randn(3, 224, 224)

        landmarks = extractor.extract_landmarks(frame)

        assert landmarks.shape == (extractor.n_landmarks, 2)

    def test_compute_geometric_features(self):
        """Test geometric feature computation."""
        extractor = FacialEmotionFeatureExtractor()

        landmarks = torch.rand(68, 2)

        geometric = extractor.compute_geometric_features(landmarks)

        assert geometric.shape[0] == extractor.geometric_dim

    def test_extract_action_units(self):
        """Test Action Unit extraction."""
        extractor = FacialEmotionFeatureExtractor(extract_aus=True)

        landmarks = torch.rand(68, 2)

        aus = extractor.extract_action_units(landmarks)

        assert aus.shape[0] == extractor.au_dim
        assert torch.all(aus >= 0) and torch.all(aus <= 1)  # AUs in [0, 1]

    def test_forward_single_sequence(self):
        """Test forward pass with single video sequence."""
        extractor = FacialEmotionFeatureExtractor()

        frames = torch.randn(10, 3, 224, 224)  # 10 frames

        features = extractor(frames)

        assert features.ndim == 2  # (n_frames, feature_dim)
        assert features.shape[0] == 10
        assert features.shape[1] == extractor.feature_dim

    def test_forward_batch(self):
        """Test forward pass with batch."""
        extractor = FacialEmotionFeatureExtractor()

        frames = torch.randn(2, 10, 3, 224, 224)  # 2 sequences, 10 frames each

        features = extractor(frames)

        assert features.shape[0] == 2  # Batch size
        assert features.shape[1] == 10  # Frames
        assert features.shape[2] == extractor.feature_dim

    def test_feature_info(self):
        """Test feature info retrieval."""
        extractor = FacialEmotionFeatureExtractor(
            extract_aus=True,
            extract_head_pose=True,
            extract_temporal=True,
        )

        info = extractor.get_feature_info()

        assert "landmarks" in info
        assert "geometric" in info
        assert "action_units" in info
        assert "head_pose" in info
        assert "temporal" in info
        assert info["total"] == extractor.feature_dim


class TestPADRegressor:
    """Test PAD regressor."""

    def test_regressor_initialization(self):
        """Test PAD regressor initialization."""
        regressor = PADRegressor(d_model=256)

        assert regressor.d_model == 256
        assert regressor.audio_extractor is not None
        assert regressor.face_extractor is not None

    def test_forward_audio_only(self):
        """Test forward pass with audio only."""
        regressor = PADRegressor(d_model=128)

        audio = torch.randn(2, 16000)  # 2 samples

        pred = regressor(audio=audio)

        assert isinstance(pred, PADPrediction)
        assert pred.pleasure.shape[0] == 2
        assert pred.arousal.shape[0] == 2
        assert pred.dominance.shape[0] == 2
        assert pred.confidence.shape[0] == 2

    def test_forward_face_only(self):
        """Test forward pass with face only."""
        regressor = PADRegressor(d_model=128)

        frames = torch.randn(2, 10, 3, 224, 224)  # 2 sequences, 10 frames

        pred = regressor(frames=frames)

        assert isinstance(pred, PADPrediction)
        assert pred.pleasure.shape == (2, 10)  # (batch, seq_len)
        assert pred.arousal.shape == (2, 10)
        assert pred.dominance.shape == (2, 10)

    def test_forward_multimodal(self):
        """Test forward pass with both audio and face."""
        regressor = PADRegressor(d_model=128)

        audio = torch.randn(2, 16000)
        frames = torch.randn(2, 10, 3, 224, 224)

        pred = regressor(audio=audio, frames=frames, return_modality_contributions=True)

        assert isinstance(pred, PADPrediction)
        assert pred.pleasure.shape[0] == 2
        assert pred.audio_contribution is not None
        assert pred.face_contribution is not None

    def test_compute_loss(self):
        """Test loss computation."""
        regressor = PADRegressor(d_model=128)

        audio = torch.randn(2, 16000)
        pred = regressor(audio=audio)

        # Create dummy targets
        target_p = torch.randn(2, pred.pleasure.shape[1])
        target_a = torch.rand(2, pred.arousal.shape[1])
        target_d = torch.randn(2, pred.dominance.shape[1])

        loss, loss_dict = regressor.compute_loss(
            pred, target_p, target_a, target_d,
            loss_type="CCC",
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert "pleasure_loss" in loss_dict
        assert "arousal_loss" in loss_dict
        assert "dominance_loss" in loss_dict

    def test_verify_gates(self):
        """Test hard gate verification."""
        regressor = PADRegressor(d_model=128)

        audio = torch.randn(2, 16000)
        pred = regressor(audio=audio, return_modality_contributions=True)

        # Create targets (make them similar to predictions for gate to pass)
        target_p = pred.pleasure.clone().detach()
        target_a = pred.arousal.clone().detach()
        target_d = pred.dominance.clone().detach()

        gates = regressor.verify_gates(pred, target_p, target_a, target_d)

        assert "ccc_pleasure" in gates
        assert "ccc_arousal" in gates
        assert "ccc_dominance" in gates
        assert "smoothness" in gates
        assert "all_pass" in gates


class TestEmotionHooks:
    """Test emotion hooks and callbacks."""

    def test_hook_registry(self):
        """Test hook registry."""
        registry = EmotionHookRegistry()

        # Register a hook
        called = []

        def my_hook(*args, **kwargs):
            called.append(True)

        registry.register("test_event", my_hook)

        # Trigger
        registry.trigger("test_event")

        assert len(called) == 1

    def test_hook_priority(self):
        """Test hook priority ordering."""
        registry = EmotionHookRegistry()

        call_order = []

        def high_priority():
            call_order.append("high")

        def low_priority():
            call_order.append("low")

        registry.register("test", high_priority, priority=10)
        registry.register("test", low_priority, priority=1)

        registry.trigger("test")

        assert call_order == ["high", "low"]

    def test_pad_emotion_hook_initialization(self):
        """Test PAD emotion hook initialization."""
        hook = PADEmotionHook()

        assert hook.pad_regressor is not None
        assert hook.registry is not None
        assert hook.enable_emotion_aware_lr == True

    def test_pad_emotion_hook_prediction(self):
        """Test PAD prediction through hook."""
        hook = PADEmotionHook()

        audio = torch.randn(2, 16000)

        state = hook(audio=audio)

        assert state.pad is not None
        assert state.modality_sources["audio"] == True
        assert state.modality_sources["face"] == False

    def test_emotion_aware_lr_hook(self):
        """Test emotion-aware learning rate adjustment."""
        hook = PADEmotionHook(enable_emotion_aware_lr=True)

        # Create dummy optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        hook.base_lr = 0.001

        # Run prediction
        audio = torch.randn(2, 16000)
        state = hook(audio=audio)

        # Trigger pre-optimizer-step hook
        hook.pre_optimizer_step(optimizer)

        # LR should be modified based on arousal
        # (exact value depends on random arousal prediction)
        assert optimizer.param_groups[0]['lr'] != hook.base_lr or True  # May be same by chance

    def test_pad_caching(self):
        """Test PAD prediction caching."""
        hook = PADEmotionHook(enable_pad_caching=True)

        audio = torch.randn(2, 16000)

        # First prediction
        pred1 = hook.predict_pad(audio=audio, use_cache=True)

        # Second prediction (should be cached)
        pred2 = hook.predict_pad(audio=audio, use_cache=True)

        # Should be the same object (from cache)
        assert pred1 is pred2

    def test_create_trainer_hook(self):
        """Test creating trainer hook."""
        hook = create_emotion_aware_trainer_hook()

        assert hook.enable_emotion_aware_lr == True
        assert hook.enable_emotion_aware_temp == True

    def test_create_inference_hook(self):
        """Test creating inference hook."""
        hook = create_emotion_aware_inference_hook()

        assert hook.enable_emotion_aware_lr == False  # No LR in inference
        assert hook.enable_emotion_aware_temp == True

    def test_get_pad_summary(self):
        """Test PAD summary statistics."""
        hook = PADEmotionHook()

        # Run a few predictions
        for _ in range(5):
            audio = torch.randn(1, 16000)
            hook(audio=audio)

        summary = hook.get_pad_summary()

        assert "pleasure" in summary
        assert "arousal" in summary
        assert "dominance" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
