"""Curvature and fluctuation diagnostics for TLS scheduling."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

try:  # pragma: no cover
    import torch
    from torch import Tensor
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = "Tensor"  # type: ignore
    nn = object  # type: ignore


@dataclass
class HutchinsonCurvatureEstimator:
    """Estimate the trace of a Hessian using Hutchinson's trick."""

    model: nn.Module  # type: ignore[valid-type]
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    num_samples: int = 4

    def __post_init__(self) -> None:
        if torch is None:  # pragma: no cover - optional dependency guard
            raise ImportError("PyTorch is required to use HutchinsonCurvatureEstimator")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")

    def __call__(self, inputs: Tensor, targets: Tensor) -> float:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is not available")
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        params = tuple(self.model.parameters())
        grad = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grad])

        trace_estimate = 0.0
        for _ in range(self.num_samples):
            noise = torch.randint(0, 2, flat_grad.shape, device=flat_grad.device, dtype=flat_grad.dtype) * 2 - 1
            grad_v = torch.dot(flat_grad, noise)
            hvp = torch.autograd.grad(grad_v, params, retain_graph=True)
            hvp_flat = torch.cat([h.reshape(-1) for h in hvp])
            trace_estimate += float(torch.dot(noise, hvp_flat))
        trace_estimate /= self.num_samples
        return trace_estimate


@dataclass
class WeightFluctuationTracker:
    """Monitor moving averages of parameter magnitudes."""

    beta: float = 0.9
    epsilon: float = 1e-8
    running_mean: float = field(default=0.0, init=False)
    running_var: float = field(default=0.0, init=False)

    def update(self, parameters: Iterable[Tensor]) -> float:
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required to track weight fluctuations")
        values = torch.cat([p.detach().reshape(-1) for p in parameters])
        mean = float(values.abs().mean())
        var = float(values.pow(2).mean())
        self.running_mean = self.beta * self.running_mean + (1 - self.beta) * mean
        self.running_var = self.beta * self.running_var + (1 - self.beta) * var
        return math.sqrt(max(self.running_var - self.running_mean**2, 0.0) + self.epsilon)


@dataclass
class FDRControlSignal:
    """Container for scheduling signals."""

    curvature: float
    fluctuation: float
    lr_scale: float
    temperature_scale: float

    def combine(self, other: "FDRControlSignal", weight: float = 0.5) -> "FDRControlSignal":
        return FDRControlSignal(
            curvature=self.curvature * (1 - weight) + other.curvature * weight,
            fluctuation=self.fluctuation * (1 - weight) + other.fluctuation * weight,
            lr_scale=self.lr_scale * (1 - weight) + other.lr_scale * weight,
            temperature_scale=self.temperature_scale * (1 - weight) + other.temperature_scale * weight,
        )


def control_from_metrics(
    curvature: float,
    fluctuation: float,
    *,
    lr_base: float,
    temperature_base: float,
    curvature_clip: Optional[float] = None,
    fluctuation_clip: Optional[float] = None,
) -> FDRControlSignal:
    """Translate metrics into actionable scales."""

    if curvature_clip is not None:
        curvature = max(-curvature_clip, min(curvature, curvature_clip))
    if fluctuation_clip is not None:
        fluctuation = max(-fluctuation_clip, min(fluctuation, fluctuation_clip))

    lr_scale = 1.0 / (1.0 + max(curvature, 0.0))
    temperature_scale = 1.0 + fluctuation
    return FDRControlSignal(
        curvature=curvature,
        fluctuation=fluctuation,
        lr_scale=lr_base * lr_scale,
        temperature_scale=temperature_base * temperature_scale,
    )


__all__ = [
    "HutchinsonCurvatureEstimator",
    "WeightFluctuationTracker",
    "FDRControlSignal",
    "control_from_metrics",
]
