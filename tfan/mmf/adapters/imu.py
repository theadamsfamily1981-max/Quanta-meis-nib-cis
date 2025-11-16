#!/usr/bin/env python
"""IMU Adapter - processes IMU sensor data (accelerometer, gyroscope)."""

import torch
import torch.nn as nn


class IMUAdapter(nn.Module):
    """IMU modality adapter for motion/gesture data."""

    def __init__(self, feature_dim: int = 128, num_sensors: int = 6):
        super().__init__()
        self.feature_dim = feature_dim
        # 6 sensors: accel_xyz + gyro_xyz
        self.encoder = nn.Sequential(
            nn.Linear(num_sensors, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """Process IMU readings [batch, time, 6] → [batch, time, dim]."""
        return self.encoder(imu_data)
