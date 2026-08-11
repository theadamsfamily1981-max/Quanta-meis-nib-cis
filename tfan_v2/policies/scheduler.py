"""Schedulers and early stopping policies powered by FDR signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ..components.fdr import FDRControlSignal


if torch is not None:  # pragma: no cover
    _BaseScheduler = torch.optim.lr_scheduler._LRScheduler  # type: ignore[attr-defined]
else:  # pragma: no cover
    class _BaseScheduler:  # type: ignore[misc]
        def __init__(self, optimizer) -> None:
            self.optimizer = optimizer

        def step(self) -> None:  # noqa: D401 - compatibility stub
            """Placeholder step when torch is unavailable."""

        def get_lr(self):
            raise NotImplementedError


class FDRLRScheduler(_BaseScheduler):
    """Learning rate scheduler modulated by :class:`FDRControlSignal`."""

    def __init__(self, optimizer, base_lr: float):
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required to build FDRLRScheduler")
        self.base_lr = base_lr
        self.control_signal: Optional[FDRControlSignal] = None
        super().__init__(optimizer)

    def update(self, signal: FDRControlSignal) -> None:
        self.control_signal = signal
        self.step()

    def get_lr(self):  # pragma: no cover - invoked by base class
        if self.control_signal is None:
            return [self.base_lr for _ in self.optimizer.param_groups]
        return [self.control_signal.lr_scale for _ in self.optimizer.param_groups]


@dataclass
class EarlyStopController:
    patience: int = 5
    threshold: float = 1e-4

    def __post_init__(self) -> None:
        self.best_metric = float("inf")
        self.bad_steps = 0

    def step(self, metric: float) -> bool:
        """Update the controller and return ``True`` when stopping is advised."""

        if metric + self.threshold < self.best_metric:
            self.best_metric = metric
            self.bad_steps = 0
            return False
        self.bad_steps += 1
        return self.bad_steps >= self.patience


__all__ = ["FDRLRScheduler", "EarlyStopController"]
