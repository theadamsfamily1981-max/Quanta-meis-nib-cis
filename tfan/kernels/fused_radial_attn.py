#!/usr/bin/env python
"""
Fused Sparse-Radial Attention - PyTorch Wrapper

Provides high-level interface to CUDA fused kernel for radial attention.
Falls back to unfused PyTorch implementation if CUDA unavailable.

Usage:
    from tfan.kernels.fused_radial_attn import FusedRadialAttention

    attn = FusedRadialAttention(num_heads=8, head_dim=64)
    output = attn(Q, K, V, landmark_indices, radii)

Hard gates:
- ≥2× speedup vs unfused (measured in benchmark)
- <1e-3 numerical error vs reference
- -20% VRAM usage (no QK^T materialization)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings

# Try to import CUDA extension
try:
    from torch.utils.cpp_extension import load
    import os
    from pathlib import Path

    # Find kernel source directory
    kernel_dir = Path(__file__).parent.parent.parent / "kernels" / "ssa"

    # JIT compile if not already compiled
    if (kernel_dir / "fused_radial_attention.cu").exists():
        fused_radial_cuda = load(
            name="fused_radial_cuda",
            sources=[
                str(kernel_dir / "fused_radial_attention.cu"),
                str(kernel_dir / "fused_radial_binding.cpp")
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False
        )
        CUDA_AVAILABLE = True
    else:
        CUDA_AVAILABLE = False
        warnings.warn("CUDA kernel source not found, using unfused fallback")

except Exception as e:
    CUDA_AVAILABLE = False
    warnings.warn(f"Failed to load CUDA extension: {e}, using unfused fallback")


class FusedRadialAttention(nn.Module):
    """
    Fused Sparse-Radial Attention with topological landmarks.

    Computes attention only within radial neighborhoods defined by
    topological landmark selection (TLS). Fuses QK^T + softmax + V
    into a single kernel to reduce memory traffic.

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        use_cuda: Use CUDA fused kernel if available (default: True)
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        use_cuda: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.use_cuda = use_cuda and CUDA_AVAILABLE

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        landmark_indices: torch.Tensor,
        radii: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fused radial attention.

        Args:
            Q: Query tensor [batch, num_heads, seq_len, head_dim]
            K: Key tensor [batch, num_heads, seq_len, head_dim]
            V: Value tensor [batch, num_heads, seq_len, head_dim]
            landmark_indices: Landmark positions [seq_len, num_landmarks] (int32)
            radii: Radial distance per query [seq_len] (int32)

        Returns:
            output: Attention output [batch, num_heads, seq_len, head_dim]
        """
        if self.use_cuda and Q.is_cuda:
            return self._forward_cuda(Q, K, V, landmark_indices, radii)
        else:
            return self._forward_unfused(Q, K, V, landmark_indices, radii)

    def _forward_cuda(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        landmark_indices: torch.Tensor,
        radii: torch.Tensor
    ) -> torch.Tensor:
        """CUDA fused kernel path."""
        # Convert to FP16 for kernel
        Q_fp16 = Q.half()
        K_fp16 = K.half()
        V_fp16 = V.half()

        # Ensure landmark_indices and radii are int32
        landmark_indices = landmark_indices.to(dtype=torch.int32, device=Q.device)
        radii = radii.to(dtype=torch.int32, device=Q.device)

        # Call CUDA kernel
        output_fp16 = fused_radial_cuda.fused_radial_attention_forward(
            Q_fp16, K_fp16, V_fp16, landmark_indices, radii
        )

        # Convert back to original dtype
        return output_fp16.to(Q.dtype)

    def _forward_unfused(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        landmark_indices: torch.Tensor,
        radii: torch.Tensor
    ) -> torch.Tensor:
        """
        Unfused PyTorch reference implementation.

        Uses standard QK^T → softmax → V with sparse masking.
        Higher memory usage due to materializing QK^T.
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape

        # Compute QK^T with scaling
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, seq_len]

        # Create radial attention mask
        mask = self._create_radial_mask(
            seq_len, landmark_indices, radii, device=Q.device
        )  # [seq_len, seq_len]

        # Apply mask (set non-attending positions to -inf)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Handle NaN from -inf rows (no valid attention targets)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Compute output
        output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, head_dim]

        return output

    def _create_radial_mask(
        self,
        seq_len: int,
        landmark_indices: torch.Tensor,
        radii: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create radial attention mask.

        For each query position, allow attention to:
        - All landmark positions
        - Positions within radius of each landmark

        Args:
            seq_len: Sequence length
            landmark_indices: [seq_len, num_landmarks]
            radii: [seq_len]
            device: Target device

        Returns:
            mask: [seq_len, seq_len] binary mask (1 = attend, 0 = mask)
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # For each query position
        for q_idx in range(seq_len):
            landmarks = landmark_indices[q_idx]  # [num_landmarks]
            radius = radii[q_idx].item()

            # Allow attention to landmarks and their neighborhoods
            for landmark in landmarks:
                landmark = landmark.item()
                start = max(0, landmark - radius)
                end = min(seq_len, landmark + radius + 1)
                mask[q_idx, start:end] = True

        return mask


def benchmark_fused_vs_unfused(
    batch_size: int = 4,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    num_landmarks: int = 16,
    avg_radius: int = 32,
    num_iterations: int = 100
) -> dict:
    """
    Benchmark fused vs unfused radial attention.

    Returns:
        stats: Dictionary with speedup, error, memory usage
    """
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create random inputs
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Random landmarks and radii
    landmark_indices = torch.randint(0, seq_len, (seq_len, num_landmarks), device=device)
    radii = torch.randint(avg_radius // 2, avg_radius * 2, (seq_len,), device=device)

    # Initialize modules
    attn_fused = FusedRadialAttention(num_heads, head_dim, use_cuda=True).to(device)
    attn_unfused = FusedRadialAttention(num_heads, head_dim, use_cuda=False).to(device)

    # Warmup
    for _ in range(10):
        if CUDA_AVAILABLE and device.type == 'cuda':
            _ = attn_fused(Q, K, V, landmark_indices, radii)
        _ = attn_unfused(Q, K, V, landmark_indices, radii)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark unfused
    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(num_iterations):
        out_unfused = attn_unfused(Q, K, V, landmark_indices, radii)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    time_unfused = (time.time() - start) / num_iterations
    mem_unfused = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0

    # Benchmark fused
    if CUDA_AVAILABLE and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(num_iterations):
            out_fused = attn_fused(Q, K, V, landmark_indices, radii)
        torch.cuda.synchronize()
        time_fused = (time.time() - start) / num_iterations
        mem_fused = torch.cuda.max_memory_allocated()

        # Compute error
        error = (out_fused.float() - out_unfused.float()).abs().max().item()

        speedup = time_unfused / time_fused
        mem_reduction = (mem_unfused - mem_fused) / mem_unfused if mem_unfused > 0 else 0.0

        return {
            'time_fused_ms': time_fused * 1000,
            'time_unfused_ms': time_unfused * 1000,
            'speedup': speedup,
            'max_error': error,
            'mem_fused_mb': mem_fused / 1024**2,
            'mem_unfused_mb': mem_unfused / 1024**2,
            'mem_reduction': mem_reduction,
            'passes_speedup_gate': speedup >= 2.0,
            'passes_error_gate': error < 1e-3,
            'passes_mem_gate': mem_reduction >= 0.20
        }
    else:
        return {
            'time_fused_ms': 0,
            'time_unfused_ms': time_unfused * 1000,
            'speedup': 0,
            'max_error': 0,
            'mem_fused_mb': 0,
            'mem_unfused_mb': mem_unfused / 1024**2 if device.type == 'cuda' else 0,
            'mem_reduction': 0,
            'passes_speedup_gate': False,
            'passes_error_gate': False,
            'passes_mem_gate': False,
            'cuda_available': False
        }


if __name__ == '__main__':
    # Run benchmark
    print("=" * 60)
    print("Fused Radial Attention Benchmark")
    print("=" * 60)

    stats = benchmark_fused_vs_unfused(
        batch_size=4,
        num_heads=8,
        seq_len=512,
        head_dim=64,
        num_landmarks=16,
        avg_radius=32,
        num_iterations=100
    )

    print(f"\nTiming:")
    print(f"  Fused:   {stats['time_fused_ms']:.3f} ms")
    print(f"  Unfused: {stats['time_unfused_ms']:.3f} ms")
    print(f"  Speedup: {stats['speedup']:.2f}×")

    print(f"\nAccuracy:")
    print(f"  Max error: {stats['max_error']:.6f}")

    print(f"\nMemory:")
    print(f"  Fused:     {stats['mem_fused_mb']:.2f} MB")
    print(f"  Unfused:   {stats['mem_unfused_mb']:.2f} MB")
    print(f"  Reduction: {stats['mem_reduction']:.1%}")

    print(f"\nGate checks:")
    print(f"  {'✓' if stats['passes_speedup_gate'] else '✗'} Speedup ≥2×: {stats['speedup']:.2f}×")
    print(f"  {'✓' if stats['passes_error_gate'] else '✗'} Error <1e-3: {stats['max_error']:.6f}")
    print(f"  {'✓' if stats['passes_mem_gate'] else '✗'} Memory -20%: {stats['mem_reduction']:.1%}")

    all_pass = all([
        stats['passes_speedup_gate'],
        stats['passes_error_gate'],
        stats['passes_mem_gate']
    ])

    print(f"\n{'✓' if all_pass else '✗'} Overall: {'PASS' if all_pass else 'FAIL'}")
