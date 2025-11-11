"""Surprise-prioritised replay buffer utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


SurpriseFn = Callable[[Any], float]
HookFn = Callable[[str, Dict[str, Any]], None]


@dataclass
class SurpriseReplayBuffer:
    """A simple replay buffer prioritised by surprise."""

    capacity: int
    surprise_fn: Optional[SurpriseFn] = None
    device: torch.device = torch.device("cpu")
    epsilon: float = 1e-6
    hooks: List[HookFn] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._storage: List[Any] = []
        self._priorities: Tensor = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self._position: int = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._storage)

    def _compute_surprise(self, transition: Any, surprise: Optional[float]) -> float:
        if surprise is not None:
            return float(surprise)
        if self.surprise_fn is not None:
            return float(self.surprise_fn(transition))
        if isinstance(transition, Mapping):
            return float(transition.get("surprise", 1.0))
        return 1.0

    def _notify(self, event: str, payload: Dict[str, Any]) -> None:
        for hook in self.hooks:
            hook(event, payload)

    def register_hook(self, hook: HookFn) -> None:
        """Register a callback invoked on buffer mutations."""

        self.hooks.append(hook)

    def push(self, transition: Any, surprise: Optional[float] = None) -> None:
        """Insert a transition and update its priority."""

        s_val = max(self._compute_surprise(transition, surprise), self.epsilon)
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._position] = transition
        self._priorities[self._position] = s_val
        index = self._position
        self._position = (self._position + 1) % self.capacity
        self._notify("push", {"index": index, "surprise": s_val})

    def sample(self, batch_size: int) -> Tuple[List[Any], Tensor, Tensor]:
        """Sample transitions proportional to surprise."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._storage:
            raise RuntimeError("Cannot sample from an empty buffer")

        priorities = self._priorities[: len(self._storage)].clone()
        total_priority = priorities.sum()
        if total_priority <= self.epsilon:
            priorities.fill_(1.0)
            total_priority = priorities.sum()
        probs = priorities / total_priority
        indices = torch.multinomial(probs, num_samples=batch_size, replacement=len(self._storage) < batch_size)
        samples = [self._storage[i] for i in indices.tolist()]
        weights = (len(self._storage) * probs[indices]).reciprocal()
        self._notify("sample", {"indices": indices})
        return samples, indices, weights

    def update_priorities(self, indices: Sequence[int], surprises: Sequence[float]) -> None:
        """Update stored priorities after learning."""

        if len(indices) != len(surprises):
            raise ValueError("indices and surprises must have matching lengths")
        for idx, surprise in zip(indices, surprises):
            if idx >= len(self._storage):
                raise IndexError("index out of range for buffer")
            value = max(float(surprise), self.epsilon)
            self._priorities[idx] = value
        self._notify("update", {"indices": list(indices), "surprises": list(map(float, surprises))})
