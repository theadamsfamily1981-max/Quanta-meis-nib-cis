"""Topology fusion primitives for the T-FAN stack."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TopoFusion:
    """Combine curvature and flow tensors into a unified representation."""

    curvature_weight: float = 0.6
    flow_weight: float = 0.4

    def fuse(self, curvature_map, flow_map):
        curvature, flow = self._prepare_inputs(curvature_map, flow_map)
        curvature_energy = np.abs(curvature)
        flow_energy = np.clip(flow, a_min=0.0, a_max=None)

        curvature_norm = self._normalise(curvature_energy)
        flow_norm = self._normalise(flow_energy)

        curv_scaled = np.multiply(curvature_norm, self.curvature_weight)
        flow_scaled = np.multiply(flow_norm, self.flow_weight)
        return np.add(curv_scaled, flow_scaled)

    def _prepare_inputs(self, curvature_map, flow_map):
        curvature = np.asarray(curvature_map, dtype=np.float64)
        flow = np.asarray(flow_map, dtype=np.float64)
        if np.shape(curvature) != np.shape(flow):
            raise ValueError("Curvature and flow maps must have identical shapes")
        if len(np.shape(curvature)) < 2:
            raise ValueError("Expected tensors with at least two dimensions")
        return curvature, flow

    def _normalise(self, tensor):
        total = np.sum(tensor, axis=-1, keepdims=True)
        total = np.add(total, 1e-9)
        return np.true_divide(tensor, total)


__all__ = ["TopoFusion"]
