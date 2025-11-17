#!/usr/bin/env python
"""
PAD Regressor - Multimodal Emotion Prediction

Combines audio and facial features to predict PAD (Pleasure-Arousal-Dominance)
emotional state with high accuracy and temporal stability.

Architecture:
- Separate encoders for audio and face features
- Cross-modal attention fusion
- LSTM for temporal modeling
- PAD prediction heads with CCC loss

Hard gates:
- CCC (Concordance Correlation Coefficient) > 0.7 on each PAD dimension
- Temporal smoothness: std(delta) < 0.1
- Cross-modal consistency: correlation > 0.6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .fe_audio import AudioEmotionFeatureExtractor
from .fe_face import FacialEmotionFeatureExtractor
from .head import concordance_correlation_coefficient


@dataclass
class PADPrediction:
    """PAD prediction with metadata."""
    pleasure: torch.Tensor  # (batch, seq_len) or (batch,) in [-1, 1]
    arousal: torch.Tensor  # (batch, seq_len) or (batch,) in [0, 1]
    dominance: torch.Tensor  # (batch, seq_len) or (batch,) in [-1, 1]
    confidence: torch.Tensor  # (batch, seq_len) or (batch,) in [0, 1]

    # Modality-specific predictions (for cross-modal consistency check)
    audio_contribution: Optional[torch.Tensor] = None
    face_contribution: Optional[torch.Tensor] = None


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing audio and facial features.

    Allows each modality to attend to the other.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head attention
        self.audio_to_face = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.face_to_audio = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Layer norms
        self.norm_audio = nn.LayerNorm(d_model)
        self.norm_face = nn.LayerNorm(d_model)

    def forward(
        self,
        audio_features: torch.Tensor,
        face_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.

        Args:
            audio_features: (batch, seq_len, d_model)
            face_features: (batch, seq_len, d_model)

        Returns:
            (fused_audio, fused_face): Cross-attended features
        """
        # Audio attends to face
        audio_attended, _ = self.audio_to_face(
            audio_features,
            face_features,
            face_features,
        )
        audio_fused = self.norm_audio(audio_features + audio_attended)

        # Face attends to audio
        face_attended, _ = self.face_to_audio(
            face_features,
            audio_features,
            audio_features,
        )
        face_fused = self.norm_face(face_features + face_attended)

        return audio_fused, face_fused


class PADRegressor(nn.Module):
    """
    Multimodal PAD emotion regressor.

    Combines audio and facial features to predict PAD dimensions.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.2,
        use_pretrained_extractors: bool = False,
    ):
        """
        Initialize PAD regressor.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            use_pretrained_extractors: Use pretrained feature extractors
        """
        super().__init__()

        self.d_model = d_model

        # Feature extractors
        self.audio_extractor = AudioEmotionFeatureExtractor()
        self.face_extractor = FacialEmotionFeatureExtractor()

        # Get feature dimensions
        audio_feat_dim = self.audio_extractor.feature_dim
        face_feat_dim = self.face_extractor.feature_dim

        # Projection layers to d_model
        self.audio_proj = nn.Linear(audio_feat_dim, d_model)
        self.face_proj = nn.Linear(face_feat_dim, d_model)

        # Cross-modal attention
        self.cross_modal = CrossModalAttention(d_model, n_heads)

        # Temporal modeling (LSTM)
        self.lstm = nn.LSTM(
            d_model * 2,  # Concatenated audio + face
            d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # PAD prediction heads
        lstm_output_dim = d_model * 2  # Bidirectional

        self.pleasure_head = nn.Sequential(
            nn.Linear(lstm_output_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        self.arousal_head = nn.Sequential(
            nn.Linear(lstm_output_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        self.dominance_head = nn.Sequential(
            nn.Linear(lstm_output_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Modality-specific heads (for consistency check)
        self.audio_pad_head = nn.Linear(d_model, 3)  # [P, A, D]
        self.face_pad_head = nn.Linear(d_model, 3)

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        frames: Optional[torch.Tensor] = None,
        return_modality_contributions: bool = False,
    ) -> PADPrediction:
        """
        Predict PAD from audio and/or facial input.

        Args:
            audio: Audio tensor (batch, samples) or None
            frames: Video frames (batch, n_frames, C, H, W) or None
            return_modality_contributions: Return per-modality predictions

        Returns:
            PADPrediction with P/A/D values and confidence
        """
        if audio is None and frames is None:
            raise ValueError("At least one modality (audio or frames) must be provided")

        # Extract features
        if audio is not None:
            audio_features = self.audio_extractor(audio)  # (batch, n_frames, audio_feat_dim)
            audio_features = self.audio_proj(audio_features)  # (batch, n_frames, d_model)
        else:
            # Create dummy audio features
            batch_size = frames.shape[0]
            seq_len = frames.shape[1]
            audio_features = torch.zeros(batch_size, seq_len, self.d_model, device=frames.device)

        if frames is not None:
            face_features = self.face_extractor(frames)  # (batch, n_frames, face_feat_dim)
            face_features = self.face_proj(face_features)  # (batch, n_frames, d_model)
        else:
            # Create dummy face features
            batch_size = audio.shape[0]
            seq_len = audio_features.shape[1]
            face_features = torch.zeros(batch_size, seq_len, self.d_model, device=audio.device)

        # Cross-modal attention
        audio_fused, face_fused = self.cross_modal(audio_features, face_features)

        # Modality-specific predictions (for consistency check)
        if return_modality_contributions:
            audio_pad = self.audio_pad_head(audio_fused.mean(dim=1))  # (batch, 3)
            face_pad = self.face_pad_head(face_fused.mean(dim=1))  # (batch, 3)
        else:
            audio_pad = None
            face_pad = None

        # Concatenate modalities
        combined = torch.cat([audio_fused, face_fused], dim=-1)  # (batch, seq_len, d_model * 2)

        # Temporal modeling
        lstm_out, _ = self.lstm(combined)  # (batch, seq_len, d_model * 2)

        # Predict PAD
        pleasure = self.pleasure_head(lstm_out).squeeze(-1)  # (batch, seq_len)
        arousal = self.arousal_head(lstm_out).squeeze(-1)
        dominance = self.dominance_head(lstm_out).squeeze(-1)
        confidence = self.confidence_head(lstm_out).squeeze(-1)

        return PADPrediction(
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
            confidence=confidence,
            audio_contribution=audio_pad,
            face_contribution=face_pad,
        )

    def compute_loss(
        self,
        pred: PADPrediction,
        target_pleasure: torch.Tensor,
        target_arousal: torch.Tensor,
        target_dominance: torch.Tensor,
        loss_type: str = "CCC",
        temporal_smoothness_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PAD loss.

        Args:
            pred: PAD prediction
            target_pleasure: Target pleasure values
            target_arousal: Target arousal values
            target_dominance: Target dominance values
            loss_type: "CCC" or "MSE"
            temporal_smoothness_weight: Weight for temporal smoothness penalty

        Returns:
            (total_loss, loss_dict)
        """
        if loss_type == "CCC":
            # CCC loss (lower is better)
            loss_p = concordance_correlation_coefficient(pred.pleasure, target_pleasure)
            loss_a = concordance_correlation_coefficient(pred.arousal, target_arousal)
            loss_d = concordance_correlation_coefficient(pred.dominance, target_dominance)
        elif loss_type == "MSE":
            loss_p = F.mse_loss(pred.pleasure, target_pleasure)
            loss_a = F.mse_loss(pred.arousal, target_arousal)
            loss_d = F.mse_loss(pred.dominance, target_dominance)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Main loss
        main_loss = loss_p + loss_a + loss_d

        # Temporal smoothness (penalize rapid changes)
        if pred.pleasure.ndim == 2:  # (batch, seq_len)
            smoothness_p = torch.mean(torch.abs(pred.pleasure[:, 1:] - pred.pleasure[:, :-1]))
            smoothness_a = torch.mean(torch.abs(pred.arousal[:, 1:] - pred.arousal[:, :-1]))
            smoothness_d = torch.mean(torch.abs(pred.dominance[:, 1:] - pred.dominance[:, :-1]))

            smoothness_loss = smoothness_p + smoothness_a + smoothness_d
        else:
            smoothness_loss = torch.tensor(0.0, device=pred.pleasure.device)

        # Total loss
        total_loss = main_loss + temporal_smoothness_weight * smoothness_loss

        loss_dict = {
            "pleasure_loss": loss_p.item(),
            "arousal_loss": loss_a.item(),
            "dominance_loss": loss_d.item(),
            "smoothness_loss": smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else 0.0,
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict

    def verify_gates(
        self,
        pred: PADPrediction,
        target_pleasure: torch.Tensor,
        target_arousal: torch.Tensor,
        target_dominance: torch.Tensor,
    ) -> Dict[str, bool]:
        """
        Verify hard gates for PAD prediction.

        Gates:
        - CCC > 0.7 on each PAD dimension
        - Temporal smoothness: std(delta) < 0.1
        - Cross-modal consistency: correlation > 0.6 (if both modalities)

        Args:
            pred: PAD prediction
            target_pleasure, target_arousal, target_dominance: Ground truth

        Returns:
            Dict with gate verification results
        """
        gates = {}

        # CCC gates
        ccc_p = 1.0 - concordance_correlation_coefficient(pred.pleasure, target_pleasure).item()
        ccc_a = 1.0 - concordance_correlation_coefficient(pred.arousal, target_arousal).item()
        ccc_d = 1.0 - concordance_correlation_coefficient(pred.dominance, target_dominance).item()

        gates["ccc_pleasure"] = ccc_p > 0.7
        gates["ccc_arousal"] = ccc_a > 0.7
        gates["ccc_dominance"] = ccc_d > 0.7
        gates["ccc_all"] = gates["ccc_pleasure"] and gates["ccc_arousal"] and gates["ccc_dominance"]

        # Temporal smoothness gate
        if pred.pleasure.ndim == 2:
            delta_p = torch.std(pred.pleasure[:, 1:] - pred.pleasure[:, :-1]).item()
            delta_a = torch.std(pred.arousal[:, 1:] - pred.arousal[:, :-1]).item()
            delta_d = torch.std(pred.dominance[:, 1:] - pred.dominance[:, :-1]).item()

            gates["smoothness"] = (delta_p < 0.1) and (delta_a < 0.1) and (delta_d < 0.1)
        else:
            gates["smoothness"] = True  # N/A for single-frame

        # Cross-modal consistency gate
        if pred.audio_contribution is not None and pred.face_contribution is not None:
            # Compute correlation between audio and face predictions
            corr_p = torch.corrcoef(torch.stack([
                pred.audio_contribution[:, 0],
                pred.face_contribution[:, 0]
            ]))[0, 1].item()

            gates["cross_modal_consistency"] = corr_p > 0.6
        else:
            gates["cross_modal_consistency"] = True  # N/A for single modality

        # Overall gate
        gates["all_pass"] = (
            gates["ccc_all"]
            and gates["smoothness"]
            and gates["cross_modal_consistency"]
        )

        return gates
