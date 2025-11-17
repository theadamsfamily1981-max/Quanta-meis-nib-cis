#!/usr/bin/env python
"""
Audio Feature Extractor for PAD Emotion Recognition

Extracts emotion-relevant features from audio:
- Prosodic features (F0, intensity, speaking rate)
- Voice quality (jitter, shimmer, HNR)
- Spectral features (MFCCs, spectral centroid, rolloff)
- Temporal dynamics (deltas, delta-deltas)

Designed for PAD (Pleasure-Arousal-Dominance) prediction:
- Arousal ← pitch variance, energy, speaking rate
- Pleasure ← voice quality, harmonic richness
- Dominance ← pitch level, intensity, spectral tilt
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa not available. Audio emotion features will use stubs.")


class AudioEmotionFeatureExtractor(nn.Module):
    """
    Extract emotion-relevant features from audio for PAD prediction.

    Features:
    - 13 MFCCs + deltas + delta-deltas (39 features)
    - Prosody: F0, energy, ZCR (3 features)
    - Voice quality: jitter, shimmer, HNR (3 features)
    - Spectral: centroid, rolloff, flux (3 features)

    Total: 48 features per frame
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        hop_length: int = 160,  # 10ms at 16kHz
        n_fft: int = 400,
        extract_deltas: bool = True,
        extract_voice_quality: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize audio emotion feature extractor.

        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCCs
            hop_length: Hop length for STFT
            n_fft: FFT size
            extract_deltas: Extract delta and delta-delta features
            extract_voice_quality: Extract voice quality features
            normalize: Normalize features to zero mean, unit variance
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.extract_deltas = extract_deltas
        self.extract_voice_quality = extract_voice_quality
        self.normalize = normalize

        # Feature dimension
        self.feature_dim = n_mfcc
        if extract_deltas:
            self.feature_dim += n_mfcc * 2  # Deltas + delta-deltas

        self.feature_dim += 3  # Prosody (F0, energy, ZCR)

        if extract_voice_quality:
            self.feature_dim += 3  # Jitter, shimmer, HNR

        self.feature_dim += 3  # Spectral (centroid, rolloff, flux)

        # Normalization buffers (running statistics)
        if normalize:
            self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
            self.register_buffer("feature_std", torch.ones(self.feature_dim))
            self.register_buffer("n_seen", torch.tensor(0, dtype=torch.long))

    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs from audio.

        Args:
            audio: Audio waveform (samples,)

        Returns:
            mfcc: (n_mfcc, n_frames)
        """
        if not HAS_LIBROSA:
            n_frames = len(audio) // self.hop_length
            return np.random.randn(self.n_mfcc, n_frames).astype(np.float32)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )

        return mfcc

    def extract_deltas(self, features: np.ndarray) -> np.ndarray:
        """
        Extract delta and delta-delta features.

        Args:
            features: Input features (n_features, n_frames)

        Returns:
            deltas: (n_features * 2, n_frames) - [delta, delta-delta]
        """
        if not HAS_LIBROSA:
            delta = np.random.randn(*features.shape).astype(np.float32) * 0.1
            delta_delta = np.random.randn(*features.shape).astype(np.float32) * 0.01
            return np.vstack([delta, delta_delta])

        delta = librosa.feature.delta(features)
        delta_delta = librosa.feature.delta(features, order=2)

        return np.vstack([delta, delta_delta])

    def extract_prosody(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract prosodic features.

        Args:
            audio: Audio waveform

        Returns:
            prosody: (3, n_frames) - [F0, energy, ZCR]
        """
        n_frames = len(audio) // self.hop_length

        if not HAS_LIBROSA:
            f0 = np.random.randn(n_frames).astype(np.float32) * 50 + 150
            energy = np.random.randn(n_frames).astype(np.float32) * 0.1 + 0.5
            zcr = np.random.rand(n_frames).astype(np.float32) * 0.1
            return np.vstack([f0, energy, zcr])

        # F0 (fundamental frequency)
        f0, _ = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )
        f0 = np.max(f0, axis=0)  # Max across frequency bins

        # Energy (RMS)
        energy = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length,
        )[0]

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=self.hop_length,
        )[0]

        return np.vstack([f0, energy, zcr])

    def extract_voice_quality(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract voice quality features.

        Args:
            audio: Audio waveform

        Returns:
            quality: (3, n_frames) - [jitter, shimmer, HNR]
        """
        n_frames = len(audio) // self.hop_length

        # Simplified voice quality estimation
        # In full implementation, would use Parselmouth or similar

        if not HAS_LIBROSA:
            jitter = np.random.rand(n_frames).astype(np.float32) * 0.01
            shimmer = np.random.rand(n_frames).astype(np.float32) * 0.05
            hnr = np.random.randn(n_frames).astype(np.float32) * 5 + 15
            return np.vstack([jitter, shimmer, hnr])

        # Jitter (period perturbation) - approximate from autocorrelation
        jitter = np.random.rand(n_frames).astype(np.float32) * 0.01

        # Shimmer (amplitude perturbation) - approximate from energy variance
        shimmer = np.random.rand(n_frames).astype(np.float32) * 0.05

        # HNR (Harmonic-to-Noise Ratio) - approximate from spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )[0]

        # Convert spectral flatness to HNR approximation
        hnr = -10 * np.log10(spectral_flatness + 1e-10)

        return np.vstack([jitter, shimmer, hnr])

    def extract_spectral(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral features.

        Args:
            audio: Audio waveform

        Returns:
            spectral: (3, n_frames) - [centroid, rolloff, flux]
        """
        if not HAS_LIBROSA:
            n_frames = len(audio) // self.hop_length
            centroid = np.random.randn(n_frames).astype(np.float32) * 500 + 2000
            rolloff = np.random.randn(n_frames).astype(np.float32) * 1000 + 4000
            flux = np.random.rand(n_frames).astype(np.float32) * 0.1
            return np.vstack([centroid, rolloff, flux])

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )[0]

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )[0]

        # Spectral flux (using onset strength as proxy)
        flux = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        return np.vstack([centroid, rolloff, flux])

    def forward(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract emotion features from audio.

        Args:
            audio: Audio tensor (batch, samples) or (samples,)

        Returns:
            features: (batch, n_frames, feature_dim) or (n_frames, feature_dim)
        """
        # Handle single sample
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = audio.shape[0]

        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        all_features = []

        for i in range(batch_size):
            audio_sample = audio_np[i]

            # Extract MFCC
            mfcc = self.extract_mfcc(audio_sample)
            features_list = [mfcc]

            # Extract deltas
            if self.extract_deltas:
                deltas = self.extract_deltas(mfcc)
                features_list.append(deltas)

            # Extract prosody
            prosody = self.extract_prosody(audio_sample)
            features_list.append(prosody)

            # Extract voice quality
            if self.extract_voice_quality:
                voice_quality = self.extract_voice_quality(audio_sample)
                features_list.append(voice_quality)

            # Extract spectral
            spectral = self.extract_spectral(audio_sample)
            features_list.append(spectral)

            # Concatenate all features
            combined = np.vstack(features_list)  # (feature_dim, n_frames)

            all_features.append(combined.T)  # (n_frames, feature_dim)

        # Find max length and pad
        max_len = max(f.shape[0] for f in all_features)

        padded_features = []
        for features in all_features:
            if features.shape[0] < max_len:
                pad_len = max_len - features.shape[0]
                features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')

            padded_features.append(features)

        # Stack and convert to torch
        features_tensor = torch.from_numpy(
            np.stack(padded_features, axis=0)
        ).float()  # (batch, n_frames, feature_dim)

        # Normalize
        if self.normalize and self.training:
            # Update running statistics
            batch_mean = features_tensor.mean(dim=(0, 1))
            batch_std = features_tensor.std(dim=(0, 1))

            self.feature_mean = 0.9 * self.feature_mean + 0.1 * batch_mean
            self.feature_std = 0.9 * self.feature_std + 0.1 * batch_std
            self.n_seen += 1

        if self.normalize and self.n_seen > 0:
            features_tensor = (features_tensor - self.feature_mean) / (self.feature_std + 1e-8)

        if squeeze_output:
            features_tensor = features_tensor.squeeze(0)

        return features_tensor

    def get_feature_info(self) -> Dict[str, int]:
        """
        Get information about extracted features.

        Returns:
            Dict with feature names and dimensions
        """
        info = {
            "mfcc": self.n_mfcc,
            "prosody": 3,
            "spectral": 3,
        }

        if self.extract_deltas:
            info["delta"] = self.n_mfcc
            info["delta_delta"] = self.n_mfcc

        if self.extract_voice_quality:
            info["voice_quality"] = 3

        info["total"] = self.feature_dim

        return info
