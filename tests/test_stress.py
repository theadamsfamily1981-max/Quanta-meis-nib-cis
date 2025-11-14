"""
Comprehensive stress tests for TFAN production system.
Tests scalability, memory efficiency, and performance under load.
"""
import pytest
import torch
import torch.nn as nn
import time
import psutil
import os
from pathlib import Path

from tfan.attention import SparseMultiHeadAttention
from tfan.trainer import FDTScheduler, TFANTrainer
from tfan.pgu import PGUCache
from tfan.ttw import TTWSentry
from tfan.topo import PersistenceLandscape, compute_persistence_diagram


class DummyModel(nn.Module):
    """Simple model for stress testing."""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.attn = SparseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            keep_ratio=0.33,
            use_flash=False
        )
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.attn(x)
        return self.fc(x)


@pytest.mark.stress
class TestAttentionStress:
    """Stress tests for sparse attention."""

    def test_long_sequence_scalability(self):
        """Test attention on progressively longer sequences."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_dim = 512
        num_heads = 8

        attn = SparseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            keep_ratio=0.33,
            use_flash=False
        ).to(device)

        sequence_lengths = [4000, 8000, 16000, 32000]
        times = []
        memory_usage = []

        for T in sequence_lengths:
            x = torch.randn(1, T, embed_dim, device=device)

            # Warmup
            with torch.no_grad():
                _ = attn(x)

            # Measure time
            if device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                out = attn(x)

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            # Measure memory
            if device == "cuda":
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            else:
                process = psutil.Process(os.getpid())
                mem_allocated = process.memory_info().rss / (1024 ** 2)  # MB

            memory_usage.append(mem_allocated)

            print(f"T={T}: {elapsed*1000:.2f}ms, {mem_allocated:.1f}MB")

            # Validate output
            assert out.shape == (1, T, embed_dim)

        # Check sub-quadratic time scaling (should be better than O(T^2))
        # For O(N log N), time ratio should be ~2.08 for 2x length
        # For O(N^2), time ratio would be ~4.0
        time_ratio_8k_to_4k = times[1] / times[0]
        time_ratio_32k_to_16k = times[3] / times[2]

        print(f"Time ratio (8k/4k): {time_ratio_8k_to_4k:.2f}")
        print(f"Time ratio (32k/16k): {time_ratio_32k_to_16k:.2f}")

        # Should be closer to 2 than to 4 (sub-quadratic)
        assert time_ratio_8k_to_4k < 3.5, "Scaling is too close to O(N^2)"
        assert time_ratio_32k_to_16k < 3.5, "Scaling is too close to O(N^2)"

    def test_attention_speedup_gate(self):
        """Validate ≥3× speedup gate for 16k and 32k sequences."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_dim = 512
        num_heads = 8

        for T in [16000, 32000]:
            x = torch.randn(1, T, embed_dim, device=device)

            # Sparse attention
            sparse_attn = SparseMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                keep_ratio=0.33,
                use_flash=False
            ).to(device)

            # Warmup
            with torch.no_grad():
                _ = sparse_attn(x)

            # Time sparse
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = sparse_attn(x)
            if device == "cuda":
                torch.cuda.synchronize()
            sparse_time = time.perf_counter() - t0

            # Dense attention
            dense_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            ).to(device)

            # Warmup
            with torch.no_grad():
                _ = dense_attn(x, x, x)

            # Time dense
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = dense_attn(x, x, x)
            if device == "cuda":
                torch.cuda.synchronize()
            dense_time = time.perf_counter() - t0

            speedup = dense_time / sparse_time

            print(f"\nT={T}:")
            print(f"  Sparse: {sparse_time*1000:.2f}ms")
            print(f"  Dense: {dense_time*1000:.2f}ms")
            print(f"  Speedup: {speedup:.2f}×")

            # Gate: ≥3× speedup
            assert speedup >= 3.0, f"Speedup {speedup:.2f}× below 3× gate for T={T}"


@pytest.mark.stress
class TestFDTStress:
    """Stress tests for FDT homeostatic training."""

    def test_epr_cv_convergence(self):
        """Test EPR-CV convergence under various conditions."""
        device = "cpu"  # Use CPU for reproducibility

        model = DummyModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        scheduler = FDTScheduler(
            optimizer=optimizer,
            target_epr_cv=0.15,
            adaptive_gains=True
        )

        # Simulate training with noisy losses
        num_steps = 200
        epr_values = []

        for step in range(num_steps):
            # Simulated loss with decreasing trend + noise
            base_loss = 1.0 - (step / num_steps) * 0.5
            noise = torch.randn(1).item() * 0.1
            epr_value = max(0.1, base_loss + noise)

            fdt_stats = scheduler.step(epr_value)
            epr_values.append(fdt_stats["epr_cv"])

        # Check final EPR-CV is near target
        final_epr_cv = epr_values[-10:]  # Last 10 values
        avg_final_epr_cv = sum(final_epr_cv) / len(final_epr_cv)

        print(f"Final EPR-CV (avg last 10): {avg_final_epr_cv:.4f}")
        print(f"Target: 0.15")

        # Should converge to within 50% of target
        assert avg_final_epr_cv < 0.225, f"EPR-CV {avg_final_epr_cv:.4f} did not converge"

    def test_fdt_under_distribution_shift(self):
        """Test FDT adaptation to sudden distribution shifts."""
        device = "cpu"

        model = DummyModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        scheduler = FDTScheduler(
            optimizer=optimizer,
            target_epr_cv=0.15,
            adaptive_gains=True,
            epr_smoothing=0.85
        )

        # Phase 1: Stable losses
        for _ in range(50):
            epr_value = 0.5 + torch.randn(1).item() * 0.05
            scheduler.step(epr_value)

        # Phase 2: Sudden shift (simulating new data distribution)
        shift_responses = []
        for step in range(50):
            # Sudden increase in loss
            epr_value = 1.5 + torch.randn(1).item() * 0.2
            fdt_stats = scheduler.step(epr_value)
            shift_responses.append(fdt_stats["lr"])

        # Phase 3: Stabilization
        for _ in range(50):
            epr_value = 1.2 + torch.randn(1).item() * 0.05
            scheduler.step(epr_value)

        # Check that LR adapted (decreased) after shift
        pre_shift_lr = shift_responses[0]
        post_shift_lr = shift_responses[-1]

        print(f"LR before shift: {pre_shift_lr:.2e}")
        print(f"LR after adaptation: {post_shift_lr:.2e}")

        # LR should decrease in response to higher EPR
        assert post_shift_lr < pre_shift_lr * 0.9, "FDT did not adapt to distribution shift"


@pytest.mark.stress
class TestPGUStress:
    """Stress tests for PGU cache under load."""

    def test_pgu_high_throughput(self):
        """Test PGU under high request throughput."""
        pgu = PGUCache(
            timeout_ms=120,
            max_cache_size=1000,
            mode="inference"
        )

        num_requests = 1000
        latencies = []

        # Generate diverse proof requests
        for i in range(num_requests):
            formula = f"x_{i % 100} > 0"

            t0 = time.perf_counter()
            result = pgu.check(formula)
            latency = (time.perf_counter() - t0) * 1000  # ms

            latencies.append(latency)

        # Compute p95 latency
        latencies_sorted = sorted(latencies)
        p95_idx = int(len(latencies) * 0.95)
        p95_latency = latencies_sorted[p95_idx]

        hit_rate = pgu.get_hit_rate()

        print(f"Requests: {num_requests}")
        print(f"p95 latency: {p95_latency:.2f}ms")
        print(f"Hit rate: {hit_rate*100:.1f}%")

        # Gate: p95 ≤ 200ms
        assert p95_latency <= 200.0, f"p95 latency {p95_latency:.1f}ms exceeds 200ms gate"

        # Gate: hit rate ≥ 50% (with 100 unique formulas, expect ~90% hits)
        assert hit_rate >= 0.50, f"Hit rate {hit_rate*100:.1f}% below 50% gate"


@pytest.mark.stress
class TestMemoryScaling:
    """Test memory scaling properties."""

    def test_sublinear_memory_scaling(self):
        """Verify sub-linear memory growth with sequence length."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            pytest.skip("Memory scaling test requires CUDA")

        embed_dim = 512
        num_heads = 8

        attn = SparseMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            keep_ratio=0.33,
            use_flash=False
        ).to(device)

        sequence_lengths = [4000, 8000, 16000]
        memory_usage = []

        for T in sequence_lengths:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(1, T, embed_dim, device=device)

            with torch.no_grad():
                out = attn(x)

            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            memory_usage.append(peak_mem)

            print(f"T={T}: {peak_mem:.1f}MB peak")

        # Check memory scaling
        # For dense attention: memory ~ O(T^2)
        # For sparse: memory should be ~ O(T * sparsity) = O(T) effectively
        mem_ratio_8k_to_4k = memory_usage[1] / memory_usage[0]
        mem_ratio_16k_to_8k = memory_usage[2] / memory_usage[1]

        print(f"Memory ratio (8k/4k): {mem_ratio_8k_to_4k:.2f}")
        print(f"Memory ratio (16k/8k): {mem_ratio_16k_to_8k:.2f}")

        # For 2× length increase:
        # O(T^2) would give ~4× memory
        # O(T) would give ~2× memory
        # We expect between linear and slightly super-linear
        assert mem_ratio_8k_to_4k < 3.0, "Memory scaling too close to O(T^2)"
        assert mem_ratio_16k_to_8k < 3.0, "Memory scaling too close to O(T^2)"


@pytest.mark.stress
class TestTopologyStress:
    """Stress tests for topological computations."""

    def test_persistence_computation_large_datasets(self):
        """Test persistence computation on large point clouds."""
        try:
            import gudhi
        except ImportError:
            pytest.skip("GUDHI not available")

        num_points_list = [500, 1000, 2000]
        times = []

        for num_points in num_points_list:
            # Generate random point cloud
            points = torch.randn(num_points, 32).numpy()

            t0 = time.perf_counter()
            pd = compute_persistence_diagram(points, max_dim=1, engine="gudhi")
            elapsed = time.perf_counter() - t0

            times.append(elapsed)
            print(f"Points={num_points}: {elapsed:.3f}s")

            assert len(pd) > 0, "Empty persistence diagram"

        # Check that computation time is reasonable
        # Should be roughly O(N^2.4) for Rips complex
        assert times[-1] < 30.0, "Persistence computation too slow for large data"


if __name__ == "__main__":
    # Run stress tests
    pytest.main([__file__, "-v", "-m", "stress"])
