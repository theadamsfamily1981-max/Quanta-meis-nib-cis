#!/usr/bin/env python
"""
Audio Prosody Adapter

Extracts prosodic features from audio:
- Pitch (F0) contours
- Energy envelopes
- Speaking rate
- Voice quality metrics

These features are crucial for emotion recognition and PAD prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import warnings

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not available. AudioProsodyAdapter will use stubs.")

from ..ingest import ModalityAdapter, ModalityStream


class AudioProsodyAdapter(ModalityAdapter):
    """
    Audio adapter with prosody extraction.

    Extracts mel-spectrograms + prosodic features:
    - F0 (pitch) contours
    - RMS energy
    - Zero-crossing rate
    - Spectral centroid
    """

    def __init__(
        self,
        output_dim: int = 256,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length: int = 160,  # 10ms at 16kHz
        n_fft: int = 400,
        extract_prosody: bool = True,
        deterministic: bool = True,
    ):
        """
        Initialize audio prosody adapter.

        Args:
            output_dim: Output feature dimension
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            hop_length: Hop length for STFT
            n_fft: FFT size
            extract_prosody: Whether to extract prosody features
            deterministic: Ensure deterministic output
        """
        super().__init__(
            modality_name="audio",
            output_dim=output_dim,
            deterministic=deterministic,
        )

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.extract_prosody = extract_prosody

        # Mel-spectrogram projection
        mel_dim = n_mels
        if extract_prosody:
            mel_dim += 4  # Add 4 prosody features

        self.proj = nn.Linear(mel_dim, output_dim)

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mel-spectrogram from audio.

        Args:
            audio: Audio waveform (samples,)

        Returns:
            (mel_spec, timestamps)
        """
        if not HAS_LIBROSA:
            # Stub: return dummy mel-spec
            n_frames = len(audio) // self.hop_length
            mel_spec = np.random.randn(self.n_mels, n_frames).astype(np.float32)
            timestamps = np.arange(n_frames) * (self.hop_length / self.sample_rate)
            return mel_spec, timestamps

        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )

        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Timestamps (center of each frame)
        n_frames = mel_spec.shape[1]
        timestamps = librosa.frames_to_time(
            np.arange(n_frames),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        return mel_spec, timestamps

    def extract_prosody(
        self,
        audio: np.ndarray,
        mel_spec: np.ndarray,
    ) -> np.ndarray:
        """
        Extract prosodic features.

        Args:
            audio: Audio waveform
            mel_spec: Mel-spectrogram (n_mels, n_frames)

        Returns:
            prosody: (4, n_frames) - [f0, energy, zcr, centroid]
        """
        n_frames = mel_spec.shape[1]

        if not HAS_LIBROSA:
            # Stub: return dummy prosody
            return np.random.randn(4, n_frames).astype(np.float32)

        # F0 (pitch) using piptrack
        f0, _ = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        f0 = np.max(f0, axis=0)  # Take max across frequency bins
        f0 = np.log(f0 + 1e-6)  # Log scale

        # RMS energy
        energy = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length,
        )[0]
        energy = np.log(energy + 1e-6)

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length,
        )[0]

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )[0]
        centroid = np.log(centroid + 1e-6)

        # Stack prosody features
        prosody = np.stack([f0, energy, zcr, centroid], axis=0)  # (4, n_frames)

        return prosody

    def forward(
        self,
        audio: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> ModalityStream:
        """
        Process audio input.

        Args:
            audio: Audio tensor (batch, samples) or numpy array
            timestamps: Optional timestamps (batch, samples)

        Returns:
            ModalityStream with audio features
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        batch_size = audio_np.shape[0]

        all_features = []
        all_timestamps = []

        for i in range(batch_size):
            # Extract mel-spectrogram
            mel_spec, ts = self.extract_mel_spectrogram(audio_np[i])

            # Extract prosody
            if self.extract_prosody:
                prosody = self.extract_prosody(audio_np[i], mel_spec)

                # Concatenate mel + prosody
                features = np.concatenate(
                    [mel_spec, prosody], axis=0
                )  # (n_mels + 4, n_frames)
            else:
                features = mel_spec  # (n_mels, n_frames)

            all_features.append(features.T)  # (n_frames, n_mels + 4)
            all_timestamps.append(ts)

        # Find max length
        max_len = max(f.shape[0] for f in all_features)

        # Pad to max length
        padded_features = []
        padded_timestamps = []

        for features, ts in zip(all_features, all_timestamps):
            if features.shape[0] < max_len:
                pad_len = max_len - features.shape[0]
                features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
                ts = np.pad(ts, (0, pad_len), mode="edge")

            padded_features.append(features)
            padded_timestamps.append(ts)

        # Stack and convert to torch
        features_tensor = torch.from_numpy(
            np.stack(padded_features, axis=0)
        ).float()  # (batch, n_frames, n_mels + 4)

        timestamps_tensor = torch.from_numpy(
            np.stack(padded_timestamps, axis=0)
        ).float()  # (batch, n_frames)

        # Project to output_dim
        features_projected = self.proj(features_tensor)  # (batch, n_frames, output_dim)

        return ModalityStream(
            features=features_projected,
            timestamps=timestamps_tensor,
            modality="audio",
            confidence=1.0,
            metadata={"prosody_enabled": self.extract_prosody},
        )
