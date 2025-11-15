"""
Multi-Modal Ingestion Adapters
Converts raw modality inputs to (features, timestamps) pairs.

Each adapter returns:
    - features: np.ndarray of shape [T, D] where T is time steps, D is feature dim
    - timestamps: np.ndarray of shape [T] in seconds

Adapters:
    - TextAdapter: tokens from text
    - AudioAdapter: log-mel spectrograms + prosody features
    - VideoAdapter: ViT-style patches or frame features
    - IMUAdapter: sensor time series (optional)
"""

import numpy as np
from typing import Protocol, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Base Protocol
# ============================================================================

class BaseAdapter(Protocol):
    """Protocol for all modality adapters."""

    def __call__(self, x: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process input and return (features, timestamps).

        Args:
            x: Raw modality input

        Returns:
            features: [T, D] array
            timestamps: [T] array in seconds
        """
        ...


# ============================================================================
# Text Adapter
# ============================================================================

@dataclass
class TextAdapterConfig:
    """Configuration for text adapter."""
    max_length: int = 512
    add_timestamps: bool = True
    timestamp_scale: float = 0.1  # Seconds per token (reading time)


class TextAdapter:
    """
    Text adapter using HuggingFace tokenizers.

    Features: Token embeddings from tokenizer
    Timestamps: Synthetic reading time (0.1s per token default)
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[TextAdapterConfig] = None
    ):
        self.tokenizer = tokenizer
        self.config = config or TextAdapterConfig()
        logger.info(f"TextAdapter initialized with max_length={self.config.max_length}")

    def __call__(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize text and generate features.

        Args:
            text: Input text string

        Returns:
            features: [N, D] token IDs (as features)
            timestamps: [N] synthetic reading time in seconds
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            return_tensors='np'
        )

        input_ids = encoding['input_ids'][0]  # [N]
        N = len(input_ids)

        # Create features (token IDs as features for now)
        # In practice, you'd use token embeddings
        features = input_ids.reshape(-1, 1).astype(np.float32)

        # Generate synthetic timestamps
        if self.config.add_timestamps:
            timestamps = np.arange(N, dtype=np.float32) * self.config.timestamp_scale
        else:
            timestamps = np.zeros(N, dtype=np.float32)

        logger.debug(f"TextAdapter: {N} tokens, {timestamps[-1]:.2f}s duration")

        return features, timestamps


# ============================================================================
# Audio Adapter
# ============================================================================

@dataclass
class AudioAdapterConfig:
    """Configuration for audio adapter."""
    sr: int = 16000  # Sampling rate
    frame_hop: float = 0.02  # Frame hop in seconds (20ms)
    n_mels: int = 64  # Number of mel bands
    n_fft: int = 512  # FFT size
    prosody: bool = True  # Include prosody features
    prosody_dim: int = 3  # Pitch, energy, duration


class AudioAdapter:
    """
    Audio adapter with log-mel spectrograms + prosody features.

    Features: Log-mel spectrogram [T, n_mels] + prosody [T, 3]
    Timestamps: Frame centers in seconds
    """

    def __init__(self, config: Optional[AudioAdapterConfig] = None):
        self.config = config or AudioAdapterConfig()

        # Frame hop in samples
        self.hop_length = int(self.config.sr * self.config.frame_hop)

        logger.info(
            f"AudioAdapter initialized: sr={self.config.sr}, "
            f"hop={self.config.frame_hop}s, n_mels={self.config.n_mels}"
        )

    def __call__(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract audio features.

        Args:
            waveform: [num_samples] audio waveform

        Returns:
            features: [T, D] where D = n_mels + (prosody_dim if enabled)
            timestamps: [T] frame centers in seconds
        """
        # Ensure 1D
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        # Compute mel spectrogram
        mel_spec = self._compute_mel_spectrogram(waveform)  # [T, n_mels]

        features = [mel_spec]

        # Add prosody features
        if self.config.prosody:
            prosody = self._compute_prosody(waveform, mel_spec.shape[0])  # [T, 3]
            features.append(prosody)

        features = np.concatenate(features, axis=-1).astype(np.float32)

        # Generate timestamps (frame centers)
        T = features.shape[0]
        timestamps = np.arange(T, dtype=np.float32) * self.config.frame_hop

        logger.debug(
            f"AudioAdapter: {T} frames, {timestamps[-1]:.2f}s duration, "
            f"features dim={features.shape[1]}"
        )

        return features, timestamps

    def _compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram."""
        # Simple STFT + mel filterbank (placeholder)
        # In production, use librosa or torchaudio

        # STFT
        stft = np.abs(np.fft.rfft(
            self._frame(waveform, self.config.n_fft, self.hop_length),
            axis=-1
        ))

        # Mel filterbank (simplified - should use proper mel filters)
        n_freqs = stft.shape[-1]
        mel_filters = self._create_mel_filters(n_freqs, self.config.n_mels)

        # Apply mel filters
        mel_spec = np.dot(stft, mel_filters.T)

        # Log scale
        log_mel = np.log(mel_spec + 1e-8)

        return log_mel

    def _frame(self, signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        """Frame signal into overlapping windows."""
        n_frames = 1 + (len(signal) - frame_length) // hop_length
        frames = np.zeros((n_frames, frame_length))

        for i in range(n_frames):
            start = i * hop_length
            frames[i] = signal[start:start + frame_length]

        return frames

    def _create_mel_filters(self, n_freqs: int, n_mels: int) -> np.ndarray:
        """Create mel filterbank (simplified)."""
        # Simplified triangular filters
        # In production, use proper mel scale conversion
        filters = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            center = int((i + 1) * n_freqs / (n_mels + 1))
            width = max(1, n_freqs // (2 * n_mels))

            start = max(0, center - width)
            end = min(n_freqs, center + width)

            # Triangular filter
            for j in range(start, end):
                if j < center:
                    filters[i, j] = (j - start) / (center - start)
                else:
                    filters[i, j] = (end - j) / (end - center)

        return filters

    def _compute_prosody(self, waveform: np.ndarray, n_frames: int) -> np.ndarray:
        """
        Compute prosody features (pitch, energy, duration).

        Returns:
            [T, 3] array with pitch, energy, duration
        """
        prosody = np.zeros((n_frames, self.config.prosody_dim), dtype=np.float32)

        # Frame waveform
        frames = self._frame(waveform, self.config.n_fft, self.hop_length)

        # Ensure we have the right number of frames
        n_frames = min(n_frames, frames.shape[0])

        # Pitch (simplified autocorrelation)
        for i in range(n_frames):
            frame = frames[i]
            # Simple energy-based pitch proxy
            prosody[i, 0] = np.mean(np.abs(frame))

        # Energy (RMS)
        for i in range(n_frames):
            prosody[i, 1] = np.sqrt(np.mean(frames[i] ** 2))

        # Duration (constant for now)
        prosody[:, 2] = self.config.frame_hop

        # Normalize
        for dim in range(self.config.prosody_dim):
            mean = prosody[:, dim].mean()
            std = prosody[:, dim].std() + 1e-8
            prosody[:, dim] = (prosody[:, dim] - mean) / std

        return prosody


# ============================================================================
# Video Adapter
# ============================================================================

@dataclass
class VideoAdapterConfig:
    """Configuration for video adapter."""
    fps: int = 25  # Frames per second
    img_size: int = 224  # Image size
    patch_size: int = 16  # ViT patch size
    channels: int = 3  # RGB


class VideoAdapter:
    """
    Video adapter with ViT-style patches.

    Features: Flattened patches [T, num_patches * (patch_size^2 * 3)]
    Timestamps: Frame time in seconds
    """

    def __init__(self, config: Optional[VideoAdapterConfig] = None):
        self.config = config or VideoAdapterConfig()

        # Calculate patch dimensions
        self.num_patches_per_side = self.config.img_size // self.config.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_dim = (self.config.patch_size ** 2) * self.config.channels

        logger.info(
            f"VideoAdapter initialized: fps={self.config.fps}, "
            f"img_size={self.config.img_size}, patches={self.num_patches}"
        )

    def __call__(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract video features.

        Args:
            frames: [T, H, W, C] video frames

        Returns:
            features: [T, num_patches * patch_dim] flattened patches
            timestamps: [T] frame times in seconds
        """
        T = frames.shape[0]

        # Extract patches from each frame
        all_patches = []

        for t in range(T):
            frame = frames[t]  # [H, W, C]

            # Resize to img_size if needed
            if frame.shape[0] != self.config.img_size:
                frame = self._resize(frame, self.config.img_size)

            # Extract patches
            patches = self._extract_patches(frame)  # [num_patches, patch_dim]

            # Flatten patches for this frame
            frame_features = patches.flatten()
            all_patches.append(frame_features)

        features = np.stack(all_patches).astype(np.float32)  # [T, num_patches * patch_dim]

        # Generate timestamps
        timestamps = np.arange(T, dtype=np.float32) / self.config.fps

        logger.debug(
            f"VideoAdapter: {T} frames, {timestamps[-1]:.2f}s duration, "
            f"features dim={features.shape[1]}"
        )

        return features, timestamps

    def _resize(self, frame: np.ndarray, size: int) -> np.ndarray:
        """Resize frame (simplified - use cv2 in production)."""
        # Simplified nearest-neighbor resizing
        H, W, C = frame.shape

        if H == size and W == size:
            return frame

        # Simple decimation/interpolation
        scale_h = H / size
        scale_w = W / size

        resized = np.zeros((size, size, C), dtype=frame.dtype)

        for i in range(size):
            for j in range(size):
                src_i = min(int(i * scale_h), H - 1)
                src_j = min(int(j * scale_w), W - 1)
                resized[i, j] = frame[src_i, src_j]

        return resized

    def _extract_patches(self, frame: np.ndarray) -> np.ndarray:
        """Extract non-overlapping patches."""
        H, W, C = frame.shape
        p = self.config.patch_size

        patches = []

        for i in range(0, H, p):
            for j in range(0, W, p):
                patch = frame[i:i+p, j:j+p, :]
                # Flatten patch
                patches.append(patch.flatten())

        return np.array(patches)  # [num_patches, patch_dim]


# ============================================================================
# IMU Adapter
# ============================================================================

@dataclass
class IMUAdapterConfig:
    """Configuration for IMU adapter."""
    rate: int = 100  # Hz
    features: list = None  # Feature names (accel_x, accel_y, accel_z, gyro_x, ...)

    def __post_init__(self):
        if self.features is None:
            # Default: 3-axis accel + 3-axis gyro
            self.features = ['accel_x', 'accel_y', 'accel_z',
                           'gyro_x', 'gyro_y', 'gyro_z']


class IMUAdapter:
    """
    IMU sensor adapter.

    Features: Raw sensor readings [T, num_features]
    Timestamps: Sensor time in seconds
    """

    def __init__(self, config: Optional[IMUAdapterConfig] = None):
        self.config = config or IMUAdapterConfig()

        logger.info(
            f"IMUAdapter initialized: rate={self.config.rate}Hz, "
            f"features={len(self.config.features)}"
        )

    def __call__(self, sensor_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process IMU data.

        Args:
            sensor_data: [T, D] sensor readings

        Returns:
            features: [T, D] normalized sensor readings
            timestamps: [T] sensor time in seconds
        """
        T, D = sensor_data.shape

        assert D == len(self.config.features), \
            f"Expected {len(self.config.features)} features, got {D}"

        # Normalize features
        features = sensor_data.astype(np.float32)

        for dim in range(D):
            mean = features[:, dim].mean()
            std = features[:, dim].std() + 1e-8
            features[:, dim] = (features[:, dim] - mean) / std

        # Generate timestamps
        timestamps = np.arange(T, dtype=np.float32) / self.config.rate

        logger.debug(
            f"IMUAdapter: {T} samples, {timestamps[-1]:.2f}s duration"
        )

        return features, timestamps


# ============================================================================
# Adapter Registry
# ============================================================================

ADAPTER_REGISTRY = {
    'text': TextAdapter,
    'audio': AudioAdapter,
    'video': VideoAdapter,
    'imu': IMUAdapter
}


def create_adapter(modality: str, **kwargs) -> BaseAdapter:
    """
    Factory function to create adapters.

    Args:
        modality: Modality name ('text', 'audio', 'video', 'imu')
        **kwargs: Configuration parameters

    Returns:
        Adapter instance
    """
    if modality not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown modality: {modality}. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )

    adapter_class = ADAPTER_REGISTRY[modality]
    return adapter_class(**kwargs)
