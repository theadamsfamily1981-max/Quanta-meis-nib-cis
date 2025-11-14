"""
TTW (Topological Transition Watch) Sentry for TFAN.
Minimal microbench detecting precursors (VFE spikes, entropy jumps).
Gate: p95 < 5ms, coverage ≥ 90%
"""
import time
import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Tuple, List
import numpy as np


class TTWSentry(nn.Module):
    """
    Lightweight topological transition detector.
    Monitors VFE (Variational Free Energy) spikes and entropy jumps.
    """
    def __init__(self, window_size: int = 50, vfe_threshold: float = 2.0,
                 entropy_threshold: float = 0.3, device: str = "cpu"):
        super().__init__()
        self.window_size = window_size
        self.vfe_threshold = vfe_threshold
        self.entropy_threshold = entropy_threshold
        self.device = device

        # Rolling windows
        self.vfe_history = deque(maxlen=window_size)
        self.entropy_history = deque(maxlen=window_size)

        # Statistics
        self.num_triggers = 0
        self.latencies_ms = []

    def compute_vfe(self, x: torch.Tensor) -> float:
        """
        Compute variational free energy proxy.
        VFE ≈ reconstruction_error + KL_divergence
        Simplified as variance for speed.
        """
        return x.var().item()

    def compute_entropy(self, x: torch.Tensor) -> float:
        """
        Compute entropy proxy from attention patterns.
        H(p) = -sum(p * log(p))
        """
        # Normalize to probability distribution
        x_flat = x.flatten()
        p = torch.softmax(x_flat, dim=0)
        # Clip to avoid log(0)
        p = torch.clamp(p, min=1e-10)
        entropy = -(p * torch.log(p)).sum().item()
        return entropy

    def detect_precursor(self, x: torch.Tensor) -> Tuple[bool, dict]:
        """
        Detect topological precursor events.

        Args:
            x: Input tensor (e.g., attention weights or activations)

        Returns:
            (triggered, stats) where triggered is True if precursor detected
        """
        t0 = time.perf_counter()

        # Compute current metrics
        vfe = self.compute_vfe(x)
        entropy = self.compute_entropy(x)

        # Update histories
        self.vfe_history.append(vfe)
        self.entropy_history.append(entropy)

        # Compute baselines (if enough history)
        triggered = False
        vfe_spike = False
        entropy_jump = False

        if len(self.vfe_history) >= 10:
            vfe_baseline = np.mean(list(self.vfe_history)[:-1])
            vfe_spike = (vfe / (vfe_baseline + 1e-8)) > self.vfe_threshold

            entropy_baseline = np.mean(list(self.entropy_history)[:-1])
            entropy_jump = abs(entropy - entropy_baseline) > self.entropy_threshold

            triggered = vfe_spike or entropy_jump

        if triggered:
            self.num_triggers += 1

        # Measure latency
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(latency_ms)

        stats = {
            "vfe": vfe,
            "entropy": entropy,
            "vfe_spike": vfe_spike,
            "entropy_jump": entropy_jump,
            "latency_ms": latency_ms
        }

        return triggered, stats

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return np.percentile(self.latencies_ms, 95)

    def get_coverage(self, total_episodes: int) -> float:
        """Get coverage ratio of triggered episodes."""
        if total_episodes == 0:
            return 0.0
        return self.num_triggers / total_episodes

    def reset_stats(self):
        """Reset statistics."""
        self.num_triggers = 0
        self.latencies_ms = []
        self.vfe_history.clear()
        self.entropy_history.clear()


def benchmark_ttw_sentry(num_iterations: int = 1000,
                         batch_size: int = 4,
                         seq_len: int = 512,
                         embed_dim: int = 256,
                         device: str = "cpu") -> dict:
    """
    Benchmark TTW-Sentry performance.

    Gate: p95 < 5ms, coverage ≥ 90% of flagged episodes

    Returns:
        dict with benchmark results
    """
    sentry = TTWSentry(window_size=50, device=device)

    print(f"Benchmarking TTW-Sentry ({num_iterations} iterations)...")
    print(f"  Batch: {batch_size}, SeqLen: {seq_len}, Dim: {embed_dim}")

    # Simulate attention patterns with occasional spikes
    triggers = []

    for i in range(num_iterations):
        # Generate attention-like tensor
        if i % 10 == 0:
            # Inject spike every 10 iterations
            x = torch.randn(batch_size, seq_len, embed_dim, device=device) * 3.0
        else:
            x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        triggered, stats = sentry.detect_precursor(x)
        triggers.append(triggered)

    # Compute statistics
    p95_latency = sentry.get_p95_latency()

    # Expected triggers: ~100 (every 10th iteration)
    expected_triggers = num_iterations // 10
    coverage = sentry.num_triggers / expected_triggers if expected_triggers > 0 else 0.0

    results = {
        "num_iterations": num_iterations,
        "p95_latency_ms": p95_latency,
        "num_triggers": sentry.num_triggers,
        "expected_triggers": expected_triggers,
        "coverage": coverage,
        "p95_gate_pass": p95_latency < 5.0,
        "coverage_gate_pass": coverage >= 0.90,
        "all_gates_pass": p95_latency < 5.0 and coverage >= 0.90
    }

    print(f"\nResults:")
    print(f"  P95 latency: {p95_latency:.3f} ms (gate: < 5 ms) {'✓' if results['p95_gate_pass'] else '✗'}")
    print(f"  Triggers: {sentry.num_triggers}/{expected_triggers}")
    print(f"  Coverage: {coverage:.2%} (gate: ≥ 90%) {'✓' if results['coverage_gate_pass'] else '✗'}")

    return results
