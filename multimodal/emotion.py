"""Audio emotion utilities with lightweight feature extraction."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn


def _frame_signal(waveform: Tensor, frame_length: int, hop_length: int) -> Tensor:
    batch, time = waveform.shape
    if time < frame_length:
        pad = frame_length - time
    else:
        remainder = (time - frame_length) % hop_length
        pad = (hop_length - remainder) % hop_length
    if pad > 0:
        waveform = nn.functional.pad(waveform, (0, pad))
    return waveform.unfold(dimension=1, size=frame_length, step=hop_length)


class ProsodyFeatureExtractor(nn.Module):
    """Extract coarse prosodic features (energy, pitch, zero-cross rate)."""

    def __init__(self, sample_rate: int = 16000, frame_length: int = 400, hop_length: int = 160) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must be (batch, time)")
        frames = _frame_signal(waveform, self.frame_length, self.hop_length)
        energy = frames.pow(2).mean(dim=-1)
        zero_cross = (frames[..., 1:] * frames[..., :-1] < 0).float().mean(dim=-1)

        # Pitch estimate using max spectrum bin per frame
        window = torch.hann_window(self.frame_length, device=waveform.device)
        spectrum = torch.fft.rfft(frames * window, dim=-1)
        magnitudes = spectrum.abs()
        freqs = torch.fft.rfftfreq(self.frame_length, d=1.0 / self.sample_rate).to(waveform.device)
        peak_indices = magnitudes.argmax(dim=-1)
        pitch = freqs[peak_indices]

        features = torch.stack([energy, zero_cross, pitch], dim=-1)
        return features.mean(dim=1)


def _mel_filterbank(n_fft: int, n_mels: int, sample_rate: int) -> Tensor:
    f_min, f_max = 0.0, sample_rate / 2
    mel_min, mel_max = 2595.0 * math.log10(1 + f_min / 700.0), 2595.0 * math.log10(1 + f_max / 700.0)
    mels = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz = 700.0 * (10 ** (mels / 2595.0) - 1)
    bins = torch.floor((n_fft + 1) * hz / sample_rate).long()
    fbanks = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        left, center, right = bins[i : i + 3]
        if center == left:
            center += 1
        if right == center:
            right += 1
        fbanks[i, left:center] = torch.linspace(0, 1, center - left)
        fbanks[i, center:right] = torch.linspace(1, 0, right - center)
    return fbanks


class MFCCFeatureExtractor(nn.Module):
    """Compute Mel-frequency cepstral coefficients using torch primitives."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_mels: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft

        fbanks = _mel_filterbank(n_fft, n_mels, sample_rate)
        self.register_buffer("mel_filterbank", fbanks, persistent=False)

        mel_indices = torch.arange(n_mels).float()
        mfcc_indices = torch.arange(n_mfcc).float().unsqueeze(1)
        dct = torch.cos(math.pi / n_mels * (mel_indices + 0.5) * mfcc_indices)
        self.register_buffer("dct", dct, persistent=False)

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must be (batch, time)")
        window = torch.hann_window(self.win_length, device=waveform.device)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        power = spec.abs().pow(2)
        mel_spec = torch.matmul(power.transpose(1, 2), self.mel_filterbank.t().to(power.device))
        mel_spec = mel_spec.clamp_min_(1e-10).log()
        mfcc = torch.matmul(mel_spec, self.dct.to(mel_spec.device).t())
        return mfcc.mean(dim=1)


class PADHead(nn.Module):
    """Predict PAD (Pleasure/Arousal/Dominance) scores from features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        logits = self.net(features)
        valence, arousal, dominance = logits.unbind(dim=-1)
        return {
            "valence": torch.tanh(valence),
            "arousal": torch.sigmoid(arousal) * 2 - 1,
            "dominance": torch.tanh(dominance),
        }


class EmotionClassifierHead(nn.Module):
    """Simple linear classifier for discrete emotion labels."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(features)
