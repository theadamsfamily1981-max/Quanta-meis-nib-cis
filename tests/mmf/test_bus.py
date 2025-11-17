#!/usr/bin/env python
"""
Integration tests for MMF Bus.

Tests:
- PAD gate weighting
- TTW alignment integration
- Fusion pipeline
- End-to-end bus operation
- Hard gate verification
"""

import pytest
import torch
import numpy as np

from tfan.mmf import (
    MMFBus,
    MMFBusConfig,
    PADGate,
    PADState,
    AudioProsodyAdapter,
    VideoOpticalAdapter,
    TextEntityAdapter,
)
from tfan.mm import ModalityStream


class TestPADGate:
    """Test PAD-based modality gating."""

    def test_pad_gate_forward(self):
        """Test PAD gate forward pass."""
        gate = PADGate(d_model=256)

        # Create dummy streams
        streams = {
            "text": ModalityStream(
                features=torch.randn(2, 10, 256),
                timestamps=torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1),
                modality="text",
            ),
            "audio": ModalityStream(
                features=torch.randn(2, 10, 256),
                timestamps=torch.linspace(0, 1, 10).unsqueeze(0).expand(2, -1),
                modality="audio",
            ),
        }

        # Create PAD state
        pad_state = PADState(
            pleasure=torch.tensor([0.5, -0.3]),
            arousal=torch.tensor([0.8, 0.2]),
            dominance=torch.tensor([0.1, 0.9]),
            confidence=torch.tensor([0.9, 0.95]),
        )

        # Forward
        output = gate(streams, pad_state)

        assert "weights" in output
        assert "coherence" in output
        assert "gate_pass" in output

        # Check weights
        assert "text" in output["weights"]
        assert "audio" in output["weights"]

        # Weights should be in [0, 1]
        for weight in output["weights"].values():
            assert torch.all(weight >= 0)
            assert torch.all(weight <= 1)

    def test_pad_gate_soft_mode(self):
        """Test soft gating mode."""
        gate = PADGate(d_model=256, mode="soft")

        streams = {
            "text": ModalityStream(
                features=torch.randn(1, 5, 256),
                timestamps=torch.linspace(0, 1, 5).unsqueeze(0),
                modality="text",
            ),
            "audio": ModalityStream(
                features=torch.randn(1, 5, 256),
                timestamps=torch.linspace(0, 1, 5).unsqueeze(0),
                modality="audio",
            ),
            "video": ModalityStream(
                features=torch.randn(1, 5, 256),
                timestamps=torch.linspace(0, 1, 5).unsqueeze(0),
                modality="video",
            ),
        }

        pad_state = PADState(
            pleasure=torch.tensor([0.5]),
            arousal=torch.tensor([0.8]),
            dominance=torch.tensor([0.3]),
            confidence=torch.tensor([0.9]),
        )

        output = gate(streams, pad_state)

        # In soft mode, all weights should be non-zero
        for weight in output["weights"].values():
            assert torch.all(weight > 0)

    def test_pad_gate_hard_mode(self):
        """Test hard gating mode."""
        gate = PADGate(d_model=256, mode="hard")

        streams = {
            "text": ModalityStream(
                features=torch.randn(1, 5, 256),
                timestamps=torch.linspace(0, 1, 5).unsqueeze(0),
                modality="text",
            ),
            "audio": ModalityStream(
                features=torch.randn(1, 5, 256),
                timestamps=torch.linspace(0, 1, 5).unsqueeze(0),
                modality="audio",
            ),
            "video": ModalityStream(
                features=torch.randn(1, 5, 256),
                timestamps=torch.linspace(0, 1, 5).unsqueeze(0),
                modality="video",
            ),
        }

        pad_state = PADState(
            pleasure=torch.tensor([0.5]),
            arousal=torch.tensor([0.8]),
            dominance=torch.tensor([0.3]),
            confidence=torch.tensor([0.9]),
        )

        output = gate(streams, pad_state)

        # In hard mode, only top-2 should have non-zero weights
        weights_list = list(output["weights"].values())
        non_zero_count = sum(1 for w in weights_list if w.item() > 0)

        assert non_zero_count <= 2, "Hard mode should select at most 2 modalities"


class TestModalityAdapters:
    """Test modality adapters."""

    def test_audio_prosody_adapter(self):
        """Test audio prosody adapter."""
        adapter = AudioProsodyAdapter(output_dim=256, sample_rate=16000)

        # Create dummy audio
        audio = torch.randn(2, 16000)  # 2 batches, 1 second each

        stream = adapter(audio)

        assert isinstance(stream, ModalityStream)
        assert stream.features.shape[0] == 2  # Batch size
        assert stream.features.shape[2] == 256  # Output dim
        assert stream.modality == "audio"

    def test_video_optical_adapter(self):
        """Test video optical adapter."""
        adapter = VideoOpticalAdapter(output_dim=256, frame_rate=30.0)

        # Create dummy video
        video = torch.randn(2, 30, 3, 224, 224)  # 2 batches, 30 frames

        stream = adapter(video)

        assert isinstance(stream, ModalityStream)
        assert stream.features.shape[0] == 2  # Batch size
        assert stream.features.shape[1] == 30  # Frames
        assert stream.features.shape[2] == 256  # Output dim
        assert stream.modality == "video"

    def test_text_entity_adapter(self):
        """Test text entity adapter."""
        adapter = TextEntityAdapter(output_dim=256, vocab_size=50000)

        # Create dummy text
        text = torch.randint(0, 50000, (2, 50))  # 2 batches, 50 tokens

        stream = adapter(text)

        assert isinstance(stream, ModalityStream)
        assert stream.features.shape[0] == 2  # Batch size
        assert stream.features.shape[1] == 50  # Tokens
        assert stream.features.shape[2] == 256  # Output dim
        assert stream.modality == "text"


class TestMMFBus:
    """Test MMF Bus end-to-end."""

    def test_bus_initialization(self):
        """Test bus initialization."""
        config = MMFBusConfig(d_model=256, modalities=["text", "audio"])
        bus = MMFBus(config=config)

        assert bus.config.d_model == 256
        assert bus.config.modalities == ["text", "audio"]

    def test_bus_forward_single_modality(self):
        """Test bus with single modality."""
        config = MMFBusConfig(d_model=256, modalities=["text"])
        bus = MMFBus(config=config)

        # Register adapter
        bus.register_adapter("text", TextEntityAdapter(output_dim=256))

        # Create input
        inputs = {
            "text": torch.randint(0, 50000, (2, 30))
        }

        # Forward
        output = bus(inputs)

        assert output.fused is not None
        assert output.pad_state is not None
        assert output.alignment_metrics is not None
        assert output.gate_decisions is not None

    def test_bus_forward_multimodal(self):
        """Test bus with multiple modalities."""
        config = MMFBusConfig(d_model=256, modalities=["text", "audio", "video"])
        bus = MMFBus(config=config)

        # Register adapters
        bus.register_adapter("text", TextEntityAdapter(output_dim=256))
        bus.register_adapter("audio", AudioProsodyAdapter(output_dim=256))
        bus.register_adapter("video", VideoOpticalAdapter(output_dim=256))

        # Create inputs
        inputs = {
            "text": torch.randint(0, 50000, (2, 30)),
            "audio": torch.randn(2, 16000),
            "video": torch.randn(2, 10, 3, 224, 224),
        }

        # Forward
        output = bus(inputs)

        assert output.fused is not None
        assert output.fused.tokens.shape[0] == 2  # Batch size
        assert output.pad_state is not None
        assert "weights" in output.gate_decisions

    def test_bus_with_pad_override(self):
        """Test bus with PAD state override."""
        config = MMFBusConfig(d_model=256, modalities=["text", "audio"])
        bus = MMFBus(config=config)

        bus.register_adapter("text", TextEntityAdapter(output_dim=256))
        bus.register_adapter("audio", AudioProsodyAdapter(output_dim=256))

        inputs = {
            "text": torch.randint(0, 50000, (2, 30)),
            "audio": torch.randn(2, 16000),
        }

        # Override PAD state
        pad_override = PADState(
            pleasure=torch.tensor([0.9, 0.5]),
            arousal=torch.tensor([0.2, 0.8]),
            dominance=torch.tensor([0.5, 0.5]),
            confidence=torch.tensor([1.0, 1.0]),
        )

        output = bus(inputs, pad_override=pad_override)

        # Check that override was used
        assert torch.allclose(output.pad_state.pleasure, pad_override.pleasure)
        assert torch.allclose(output.pad_state.arousal, pad_override.arousal)

    def test_bus_metrics(self):
        """Test bus metrics tracking."""
        config = MMFBusConfig(d_model=256, modalities=["text"])
        bus = MMFBus(config=config)

        bus.register_adapter("text", TextEntityAdapter(output_dim=256))

        # Reset metrics
        bus.reset_metrics()

        # Run a few forward passes
        for _ in range(5):
            inputs = {"text": torch.randint(0, 50000, (2, 30))}
            bus(inputs)

        # Check metrics
        metrics = bus.get_metrics()

        assert metrics["total_calls"] == 5
        assert metrics["avg_latency_ms"] > 0
        assert "ttw_violation_rate" in metrics

    def test_bus_profiling(self):
        """Test bus profiling."""
        config = MMFBusConfig(
            d_model=256,
            modalities=["text"],
            enable_profiling=True,
        )
        bus = MMFBus(config=config)

        bus.register_adapter("text", TextEntityAdapter(output_dim=256))

        inputs = {"text": torch.randint(0, 50000, (2, 30))}

        output = bus(inputs)

        # Check profiling data
        assert output.profiling is not None
        assert "total_ms" in output.profiling
        assert "align_ms" in output.profiling


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
