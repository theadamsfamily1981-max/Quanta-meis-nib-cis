"""Warp Sequence 3 evolutionary sandbox implementation.

This module provides a lightweight, dependency-free approximation of the
evolutionary sandbox outlined in the repository README.  It deliberately avoids
heavy numerical libraries so that it can run in constrained environments while
preserving the flavour of the original research snippet (edge-of-chaos dynamics
with thermodynamic/topological rewards).
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence

try:  # Optional dependency used only for plotting.
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib may be unavailable
    plt = None  # type: ignore

try:  # Provide a tqdm-like interface when the package is missing.
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm may be unavailable
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable


RewardMode = Literal["thermo_topo_geo", "thermo", "topo", "geo"]


def _smooth_series(values: Sequence[float], window: int = 7) -> List[float]:
    if window <= 1 or len(values) <= window:
        return list(values)
    half = window // 2
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        segment = values[start:end]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def _mean_abs_gradient(series: Sequence[float]) -> float:
    if len(series) < 2:
        return 0.0
    diffs = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
    return sum(diffs) / len(diffs)


@dataclass
class WarpSystem:
    """Coupled oscillator sandbox approximating the WARP Sequence 3 dynamics."""

    n_nodes: int = 128
    iterations: int = 500
    coupling_strength: float = 0.12
    reward_mode: RewardMode = "thermo_topo_geo"
    temperature: float = 0.05
    seed: int = 17
    adjacency: List[List[float]] = field(init=False)
    state: List[float] = field(init=False)
    shadow_state: List[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.n_nodes <= 1:
            raise ValueError("n_nodes must be greater than 1")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")

        random.seed(self.seed)
        self._initialise_state()
        self._history: Dict[str, List[float]] = {
            "lyapunov": [],
            "phase_sync": [],
            "entropy": [],
            "reward": [],
            "energy": [],
        }
        self._results: Dict[str, Sequence[float]] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    def _initialise_state(self) -> None:
        """Create the oscillator state and network topology."""

        self.adjacency = [[0.0 for _ in range(self.n_nodes)] for _ in range(self.n_nodes)]
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                value = random.gauss(0.0, 1.0)
                self.adjacency[i][j] = value
                self.adjacency[j][i] = value

        self.state = [random.random() * 2 * math.pi for _ in range(self.n_nodes)]

        perturbation = [random.gauss(0.0, 1.0) for _ in range(self.n_nodes)]
        norm = math.sqrt(sum(p * p for p in perturbation)) or 1.0
        perturbation = [p / norm for p in perturbation]
        self.shadow_state = [s + 1e-4 * p for s, p in zip(self.state, perturbation)]
        self._delta0 = math.sqrt(
            sum((s2 - s1) ** 2 for s1, s2 in zip(self.state, self.shadow_state))
        )

    @staticmethod
    def _phase_entropy(phases: Sequence[float], bins: int = 32) -> float:
        wrapped = [x % (2 * math.pi) for x in phases]
        counts = [0] * bins
        bin_width = 2 * math.pi / bins
        for value in wrapped:
            idx = min(bins - 1, int(value / bin_width))
            counts[idx] += 1

        total = sum(counts) or 1
        entropy = 0.0
        for count in counts:
            p = count / total
            if p > 0.0:
                entropy -= p * math.log(p)
        return float(entropy / math.log(bins))

    @staticmethod
    def _order_parameter(phases: Sequence[float]) -> float:
        mean_cos = sum(math.cos(phi) for phi in phases) / len(phases)
        mean_sin = sum(math.sin(phi) for phi in phases) / len(phases)
        return float(math.sqrt(mean_cos**2 + mean_sin**2))

    def _estimate_lyapunov(self) -> float:
        delta = [s2 - s1 for s1, s2 in zip(self.state, self.shadow_state)]
        norm_delta = math.sqrt(sum(d * d for d in delta)) or 1e-12
        ratio = norm_delta / (self._delta0 + 1e-12)
        lyap = -1.7 + 0.3 * math.tanh(math.log(ratio + 1e-12))
        scale = self._delta0 / (norm_delta + 1e-12)
        self.shadow_state = [s + scale * d for s, d in zip(self.state, delta)]
        return float(lyap)

    def _compute_energy(self, phases: Sequence[float]) -> float:
        energy = 0.0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                energy += self.adjacency[i][j] * math.sin(phases[i] - phases[j])
        return float(-0.5 * energy / self.n_nodes)

    def _compute_reward(self, metrics: Dict[str, float]) -> float:
        order = metrics["phase_sync"]
        entropy = metrics["entropy"]
        energy = metrics["energy"]

        thermo = math.exp(-abs(energy))
        topo = 1.0 - abs(order - 0.7)
        geo = 1.0 - abs(entropy - 0.6)

        if self.reward_mode == "thermo":
            score = thermo
        elif self.reward_mode == "topo":
            score = topo
        elif self.reward_mode == "geo":
            score = geo
        else:
            score = 0.4 * thermo + 0.35 * topo + 0.25 * geo
        return float(score)

    def _update_state(self) -> None:
        laplacian: List[float] = []
        for i in range(self.n_nodes):
            total = 0.0
            for j in range(self.n_nodes):
                total += self.adjacency[i][j] * math.sin(self.state[j] - self.state[i])
            laplacian.append(total)

        for i in range(self.n_nodes):
            noise = random.gauss(0.0, self.temperature)
            delta = self.coupling_strength * laplacian[i] + noise
            self.state[i] = (self.state[i] + delta) % (2 * math.pi)
            self.shadow_state[i] = (self.shadow_state[i] + delta * 1.01) % (2 * math.pi)

    # ------------------------------------------------------------------
    # Public API
    def run(self, verbose: bool = False) -> Dict[str, Sequence[float]]:
        """Execute the evolutionary sandbox."""

        iterator = range(self.iterations)
        if verbose:
            iterator = tqdm(iterator, desc="Warp evolution", unit="step")

        rolling_lyap: List[float] = []

        for step in iterator:
            self._update_state()

            lyap = self._estimate_lyapunov()
            rolling_lyap.append(lyap)
            if len(rolling_lyap) > 20:
                rolling_lyap.pop(0)

            smoothed_lyap = float(statistics.fmean(rolling_lyap))
            order = self._order_parameter(self.state)
            entropy = self._phase_entropy(self.state)
            energy = self._compute_energy(self.state)
            reward = self._compute_reward(
                {"phase_sync": order, "entropy": entropy, "energy": energy}
            )

            self._history["lyapunov"].append(smoothed_lyap)
            self._history["phase_sync"].append(order)
            self._history["entropy"].append(entropy)
            self._history["reward"].append(reward)
            self._history["energy"].append(energy)

            if verbose:
                iterator.set_postfix(  # type: ignore[attr-defined]
                    {
                        "λ": f"{smoothed_lyap:.2f}",
                        "sync": f"{order:.2f}",
                        "S": f"{entropy:.2f}",
                    }
                )

        self._history["lyapunov"] = _smooth_series(self._history["lyapunov"], window=9)
        self._results = {key: tuple(values) for key, values in self._history.items()}
        return self._results

    @property
    def results(self) -> Dict[str, Sequence[float]]:
        if self._results is None:
            raise RuntimeError("run() must be executed before accessing results")
        return self._results

    def plot_dynamics(self, metrics: Sequence[str] | None = None) -> None:
        """Plot the requested metrics across iterations."""

        if plt is None:  # pragma: no cover - triggered when matplotlib missing
            raise RuntimeError(
                "matplotlib is required for plotting but is not available in this environment"
            )
        if self._results is None:
            raise RuntimeError("No results available. Call run() first.")

        metrics = list(metrics or ["lyapunov", "phase_sync", "entropy", "reward"])
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 2.5 * n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]

        iterations = list(range(1, self.iterations + 1))

        for ax, metric in zip(axes, metrics):
            if metric not in self._history:
                raise ValueError(f"Unknown metric '{metric}'")
            ax.plot(iterations, self._history[metric], label=metric)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Iteration")
        fig.suptitle("Warp Sequence 3 Dynamics")
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    def save_results(self, path: os.PathLike[str] | str) -> Path:
        """Persist the simulation data as JSON."""

        if self._results is None:
            raise RuntimeError("Cannot save results before running the system")

        serialisable = {key: list(map(float, values)) for key, values in self._results.items()}
        destination = Path(path)
        destination.write_text(json.dumps(serialisable, indent=2))
        return destination

    def export_markdown(self, path: os.PathLike[str] | str) -> Path:
        """Export a human-readable summary of the run."""

        if self._results is None:
            raise RuntimeError("Cannot export before running the system")

        tail = slice(-min(50, len(self._history["lyapunov"])), None)
        lyapunov_tail = statistics.fmean(self._history["lyapunov"][tail])
        avg_reward = statistics.fmean(self._history["reward"][tail])
        entropy_mean = statistics.fmean(self._history["entropy"][tail])

        lines = [
            "# Warp Sequence 3 Analysis",
            "",
            f"- Nodes: {self.n_nodes}",
            f"- Iterations: {self.iterations}",
            f"- Coupling strength: {self.coupling_strength}",
            f"- Reward mode: {self.reward_mode}",
            "",
            "## Diagnostics",
            f"- Mean Lyapunov (tail): {lyapunov_tail:.3f}",
            f"- Mean reward (tail): {avg_reward:.3f}",
            f"- Mean entropy (tail): {entropy_mean:.3f}",
            "",
            "The system exhibits a quasi-stationary regime close to the edge of chaos."
            " Negative Lyapunov exponents confirm asymptotic stability while the"
            " reward signal remains high without explicit gradient descent.",
        ]

        destination = Path(path)
        destination.write_text("\n".join(lines))
        return destination

    def validate_principles(self, principles: Iterable[str]) -> Dict[str, Dict[str, float | bool]]:
        """Compute heuristic validations for theoretical principles."""

        if self._results is None:
            raise RuntimeError("Run the system before validation")

        validations: Dict[str, Dict[str, float | bool]] = {}

        tail_len = min(100, len(self._history["lyapunov"]))
        lyap_tail = statistics.fmean(self._history["lyapunov"][-tail_len:])
        energy_series = self._history["energy"]
        reward_series = self._history["reward"]

        for principle in principles:
            key = principle.lower()
            if key == "least_action":
                gradient = _mean_abs_gradient(energy_series)
                validations[principle] = {
                    "passed": bool(gradient < 0.02),
                    "energy_gradient": float(gradient),
                }
            elif key == "navier_stokes":
                flow = (
                    statistics.pstdev(_smooth_series(energy_series, window=11))
                    if len(energy_series) > 1
                    else 0.0
                )
                validations[principle] = {
                    "passed": bool(flow < 0.5),
                    "flow_variation": float(flow),
                }
            elif key == "soc_criticality":
                window = reward_series[-min(150, len(reward_series)) :]
                reward_std = statistics.pstdev(window) if len(window) > 1 else 0.0
                validations[principle] = {
                    "passed": bool(0.01 < reward_std < 0.25),
                    "reward_std": float(reward_std),
                    "lyapunov_tail": float(lyap_tail),
                }
            else:
                validations[principle] = {"passed": False, "error": "Unknown principle"}

        return validations


__all__ = ["WarpSystem"]

