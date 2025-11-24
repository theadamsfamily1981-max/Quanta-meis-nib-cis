"""Nightly proxy vs exact topology gap computation."""
from __future__ import annotations

from typing import Sequence

from tfan.topo import persistent_surprise, proxy_persistence_diagram


def compare(reference: Sequence[float], observation: Sequence[float]) -> float:
    ref_diag = proxy_persistence_diagram(reference)
    obs_diag = proxy_persistence_diagram(observation)
    return persistent_surprise(ref_diag, obs_diag)


__all__ = ["compare"]


if __name__ == "__main__":
    demo_reference = [0.1, 0.4, 0.2, 0.5]
    demo_observation = [0.3, 0.2, 0.6, 0.7]
    gap = compare(demo_reference, demo_observation)
    print(f"proxy surprise: {gap:.4f}")
