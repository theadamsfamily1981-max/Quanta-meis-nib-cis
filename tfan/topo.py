"""Topological proxy utilities.

Only simplified stand-ins of the research code are provided.  They expose a
consistent API that allows the benchmarking scripts to run in a deterministic
fashion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ProxyPersistenceDiagram:
    """Simple proxy for a persistence diagram consisting of (birth, death) pairs."""

    features: Tuple[Tuple[float, float], ...]

    def lifetimes(self) -> Tuple[float, ...]:
        return tuple(death - birth for birth, death in self.features)


def proxy_persistence_diagram(signal: Sequence[float], window: int = 4) -> ProxyPersistenceDiagram:
    """Return a lightweight proxy for the persistence diagram of ``signal``.

    The function partitions ``signal`` into overlapping windows and records the
    minimum/maximum within the window as a *birth/death* pair.  The resulting
    diagram behaves similarly to the one produced by the heavy persistent
    homology pipeline but is cheap to compute and deterministic.
    """

    if window <= 0:
        raise ValueError("window must be positive")
    values = [float(v) for v in signal]
    if len(values) < 2:
        raise ValueError("signal must contain at least two samples")

    features: List[Tuple[float, float]] = []
    for start in range(0, len(values) - 1):
        end = min(len(values), start + window)
        birth = min(values[start:end])
        death = max(values[start:end])
        if death > birth:
            features.append((birth, death))
    return ProxyPersistenceDiagram(tuple(features))


def persistent_surprise(
    reference: ProxyPersistenceDiagram,
    observation: ProxyPersistenceDiagram,
) -> float:
    """Return the average absolute difference in lifetimes.

    The metric loosely measures how surprising ``observation`` is relative to the
    ``reference`` diagram.
    """

    ref = reference.lifetimes()
    obs = observation.lifetimes()
    if not ref and not obs:
        return 0.0
    if not ref:
        return sum(obs) / len(obs)
    if not obs:
        return sum(ref) / len(ref)
    size = max(len(ref), len(obs))
    padded_ref = list(ref) + [0.0] * (size - len(ref))
    padded_obs = list(obs) + [0.0] * (size - len(obs))
    return sum(abs(r - o) for r, o in zip(padded_ref, padded_obs)) / size


__all__ = ["ProxyPersistenceDiagram", "proxy_persistence_diagram", "persistent_surprise"]
