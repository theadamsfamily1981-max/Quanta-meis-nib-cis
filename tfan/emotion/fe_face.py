#!/usr/bin/env python
"""
Facial Feature Extractor for PAD Emotion Recognition

Extracts emotion-relevant features from facial videos:
- Facial landmarks (68 points)
- Action Units (AU) intensities
- Geometric features (distances, angles)
- Temporal dynamics (velocities, accelerations)

Designed for PAD (Pleasure-Arousal-Dominance) prediction:
- Arousal ← eye openness, mouth opening, AU intensities
- Pleasure ← smile intensity (AU6+AU12), eyebrow position
- Dominance ← head pose, gaze direction, facial tension
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import warnings


class FacialEmotionFeatureExtractor(nn.Module):
    """
    Extract emotion-relevant features from facial videos for PAD prediction.

    Features:
    - 68 facial landmarks (x, y) = 136 features
    - Geometric features: 20 features (distances, ratios, angles)
    - Action Units: 17 AU intensities
    - Head pose: 3 features (yaw, pitch, roll)
    - Gaze: 2 features (horizontal, vertical)
    - Temporal: velocities and accelerations

    Total: ~200 features per frame
    """

    def __init__(
        self,
        n_landmarks: int = 68,
        extract_aus: bool = True,
        extract_head_pose: bool = True,
        extract_gaze: bool = True,
        extract_temporal: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize facial emotion feature extractor.

        Args:
            n_landmarks: Number of facial landmarks
            extract_aus: Extract Action Unit intensities
            extract_head_pose: Extract head pose (yaw, pitch, roll)
            extract_gaze: Extract gaze direction
            extract_temporal: Extract temporal derivatives
            normalize: Normalize features
        """
        super().__init__()

        self.n_landmarks = n_landmarks
        self.extract_aus = extract_aus
        self.extract_head_pose = extract_head_pose
        self.extract_gaze = extract_gaze
        self.extract_temporal = extract_temporal
        self.normalize = normalize

        # Feature dimension
        self.landmark_dim = n_landmarks * 2  # (x, y) for each landmark
        self.geometric_dim = 20  # Predefined geometric features
        self.au_dim = 17 if extract_aus else 0  # Standard AUs
        self.head_pose_dim = 3 if extract_head_pose else 0
        self.gaze_dim = 2 if extract_gaze else 0

        self.base_feature_dim = (
            self.landmark_dim
            + self.geometric_dim
            + self.au_dim
            + self.head_pose_dim
            + self.gaze_dim
        )

        if extract_temporal:
            self.feature_dim = self.base_feature_dim * 3  # Original + velocity + acceleration
        else:
            self.feature_dim = self.base_feature_dim

        # Action Unit definitions (simplified)
        # AU -> muscle/region mapping
        self.au_regions = {
            1: "inner_brow_raiser",
            2: "outer_brow_raiser",
            4: "brow_lowerer",
            5: "upper_lid_raiser",
            6: "cheek_raiser",
            7: "lid_tightener",
            9: "nose_wrinkler",
            10: "upper_lip_raiser",
            12: "lip_corner_puller",
            14: "dimpler",
            15: "lip_corner_depressor",
            17: "chin_raiser",
            20: "lip_stretcher",
            23: "lip_tightener",
            24: "lip_pressor",
            25: "lips_part",
            26: "jaw_drop",
        }

        # Normalization buffers
        if normalize:
            self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
            self.register_buffer("feature_std", torch.ones(self.feature_dim))

    def extract_landmarks(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Extract facial landmarks from frame.

        Args:
            frame: Video frame (C, H, W)

        Returns:
            landmarks: (n_landmarks, 2) - (x, y) coordinates
        """
        # In full implementation, would use MediaPipe, dlib, or similar
        # For now, simulate with dummy landmarks

        # Generate plausible landmark positions
        # Landmarks are normalized to [0, 1]
        landmarks = torch.rand(self.n_landmarks, 2)

        return landmarks

    def compute_geometric_features(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Compute geometric features from landmarks.

        Features:
        - Eye aspect ratios (left, right)
        - Mouth aspect ratio
        - Eyebrow heights (left, right)
        - Inter-landmark distances
        - Facial symmetry

        Args:
            landmarks: (n_landmarks, 2)

        Returns:
            geometric: (20,) geometric features
        """
        # Simplified landmark indices (68-point model)
        # Eyes: 36-41 (left), 42-47 (right)
        # Mouth: 48-67
        # Eyebrows: 17-21 (left), 22-26 (right)

        features = []

        # Eye Aspect Ratios
        # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        # Approximation with random values for stub
        left_ear = torch.rand(1) * 0.3 + 0.15  # Typical range [0.15, 0.45]
        right_ear = torch.rand(1) * 0.3 + 0.15
        features.extend([left_ear, right_ear])

        # Mouth Aspect Ratio
        mar = torch.rand(1) * 0.6  # Range [0, 0.6]
        features.append(mar)

        # Eyebrow heights (relative to eye)
        left_brow_height = torch.rand(1) * 0.1 + 0.05
        right_brow_height = torch.rand(1) * 0.1 + 0.05
        features.extend([left_brow_height, right_brow_height])

        # Smile width (mouth corners distance)
        smile_width = torch.rand(1) * 0.3 + 0.2
        features.append(smile_width)

        # Face width/height ratio
        face_ratio = torch.rand(1) * 0.3 + 0.7  # Typical 0.7-1.0
        features.append(face_ratio)

        # Facial symmetry (left-right difference)
        symmetry = torch.rand(1) * 0.1  # Low values = more symmetric
        features.append(symmetry)

        # Additional inter-landmark distances and ratios (12 more features)
        for _ in range(12):
            features.append(torch.rand(1))

        return torch.cat(features)

    def extract_action_units(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Extract Action Unit intensities from landmarks.

        Args:
            landmarks: (n_landmarks, 2)

        Returns:
            aus: (17,) AU intensities in [0, 1]
        """
        # In full implementation, would use OpenFace or similar
        # For now, simulate with plausible AU activations

        aus = torch.rand(len(self.au_regions))

        # Make some AUs more likely (realistic distributions)
        # AU12 (lip corner puller / smile) - common
        aus[8] = torch.rand(1) * 0.7 + 0.3

        # AU26 (jaw drop) - variable
        aus[16] = torch.rand(1) * 0.5

        return aus

    def extract_head_pose(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Extract head pose (yaw, pitch, roll).

        Args:
            landmarks: (n_landmarks, 2)

        Returns:
            pose: (3,) - [yaw, pitch, roll] in degrees
        """
        # In full implementation, would use PnP algorithm
        # For now, simulate with small rotations around neutral

        yaw = torch.randn(1) * 15  # ±15 degrees
        pitch = torch.randn(1) * 10  # ±10 degrees
        roll = torch.randn(1) * 5  # ±5 degrees

        return torch.cat([yaw, pitch, roll])

    def extract_gaze(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Extract gaze direction.

        Args:
            landmarks: (n_landmarks, 2)

        Returns:
            gaze: (2,) - [horizontal, vertical] in [-1, 1]
        """
        # In full implementation, would use eye landmarks and iris detection
        # For now, simulate with small deviations from center

        horizontal = torch.randn(1) * 0.3  # ±0.3
        vertical = torch.randn(1) * 0.2  # ±0.2

        return torch.cat([horizontal, vertical])

    def compute_temporal_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute temporal derivatives (velocity, acceleration).

        Args:
            features: (n_frames, feature_dim)

        Returns:
            temporal: (n_frames, feature_dim * 3) - [features, velocity, acceleration]
        """
        n_frames = features.shape[0]

        # Velocity (first-order difference)
        velocity = torch.zeros_like(features)
        velocity[1:] = features[1:] - features[:-1]

        # Acceleration (second-order difference)
        acceleration = torch.zeros_like(features)
        acceleration[1:] = velocity[1:] - velocity[:-1]

        # Concatenate
        temporal = torch.cat([features, velocity, acceleration], dim=1)

        return temporal

    def forward(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract emotion features from facial video.

        Args:
            frames: Video frames (batch, n_frames, C, H, W) or (n_frames, C, H, W)

        Returns:
            features: (batch, n_frames, feature_dim) or (n_frames, feature_dim)
        """
        # Handle single sequence
        if frames.ndim == 4:
            frames = frames.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, n_frames = frames.shape[:2]

        all_features = []

        for b in range(batch_size):
            frame_features = []

            for f in range(n_frames):
                frame = frames[b, f]

                # Extract landmarks
                landmarks = self.extract_landmarks(frame)

                # Geometric features
                geometric = self.compute_geometric_features(landmarks)

                # Flatten landmarks
                landmarks_flat = landmarks.flatten()

                # Combine base features
                features_list = [landmarks_flat, geometric]

                # Action Units
                if self.extract_aus:
                    aus = self.extract_action_units(landmarks)
                    features_list.append(aus)

                # Head pose
                if self.extract_head_pose:
                    head_pose = self.extract_head_pose(landmarks)
                    features_list.append(head_pose)

                # Gaze
                if self.extract_gaze:
                    gaze = self.extract_gaze(landmarks)
                    features_list.append(gaze)

                # Concatenate frame features
                frame_feat = torch.cat(features_list)
                frame_features.append(frame_feat)

            # Stack frame features
            sequence_features = torch.stack(frame_features, dim=0)  # (n_frames, base_feature_dim)

            # Compute temporal features
            if self.extract_temporal:
                sequence_features = self.compute_temporal_features(sequence_features)

            all_features.append(sequence_features)

        # Stack batch
        features_tensor = torch.stack(all_features, dim=0)  # (batch, n_frames, feature_dim)

        # Normalize
        if self.normalize:
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
            "landmarks": self.landmark_dim,
            "geometric": self.geometric_dim,
        }

        if self.extract_aus:
            info["action_units"] = self.au_dim

        if self.extract_head_pose:
            info["head_pose"] = self.head_pose_dim

        if self.extract_gaze:
            info["gaze"] = self.gaze_dim

        info["base_total"] = self.base_feature_dim

        if self.extract_temporal:
            info["temporal"] = self.base_feature_dim * 2  # velocity + acceleration
            info["total"] = self.feature_dim
        else:
            info["total"] = self.base_feature_dim

        return info

    def get_au_names(self) -> Dict[int, str]:
        """Get Action Unit names."""
        return self.au_regions
