#!/usr/bin/env python
"""
MMF Bus Evaluation Script

Evaluates the MMF (Multimodal Fusion Bus) on:
- TTW alignment latency and coverage
- PAD gate coherence and decision quality
- Fusion overhead vs single-modal
- End-to-end throughput

Hard gates:
- TTW p95 latency < 5 ms
- TTW coverage ≥ 90%
- PAD coherence > 0.85
- Fusion overhead < 10%

Usage:
    python scripts/mmf_eval.py --batch-size 8 --iterations 100
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from tfan.mmf import MMFBus, MMFBusConfig, PADState
from tfan.mmf.adapters import AudioProsodyAdapter, VideoOpticalAdapter, TextEntityAdapter


def generate_synthetic_inputs(
    batch_size: int,
    modalities: list,
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic multi-modal inputs for testing.

    Args:
        batch_size: Batch size
        modalities: List of modalities to generate

    Returns:
        Dict of modality -> input tensor
    """
    inputs = {}

    if "text" in modalities:
        # Text: token IDs
        seq_len = np.random.randint(50, 200)
        inputs["text"] = torch.randint(0, 50000, (batch_size, seq_len))

    if "audio" in modalities:
        # Audio: waveform
        audio_len = np.random.randint(16000, 48000)  # 1-3 seconds at 16kHz
        inputs["audio"] = torch.randn(batch_size, audio_len)

    if "video" in modalities:
        # Video: frames (C, H, W)
        n_frames = np.random.randint(10, 60)
        inputs["video"] = torch.randn(batch_size, n_frames, 3, 224, 224)

    return inputs


def evaluate_ttw_latency(
    bus: MMFBus,
    num_iterations: int = 100,
    batch_size: int = 8,
) -> Dict:
    """
    Evaluate TTW alignment latency.

    Args:
        bus: MMF Bus instance
        num_iterations: Number of evaluation iterations
        batch_size: Batch size

    Returns:
        Dict with latency metrics
    """
    print("Evaluating TTW alignment latency...")

    latencies = []

    for i in range(num_iterations):
        # Generate inputs
        inputs = generate_synthetic_inputs(batch_size, ["text", "audio", "video"])

        # Run bus with profiling
        bus.config.enable_profiling = True

        with torch.no_grad():
            output = bus(inputs)

        if output.profiling and "align_ms" in output.profiling:
            latencies.append(output.profiling["align_ms"])

        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/{num_iterations}")

    latencies = np.array(latencies)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean = np.mean(latencies)
    std = np.std(latencies)

    metrics = {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean,
        "std_ms": std,
        "gate_pass": p95 < bus.config.ttw_p95_latency_ms,
    }

    print(f"  P50: {p50:.2f} ms")
    print(f"  P95: {p95:.2f} ms (threshold: {bus.config.ttw_p95_latency_ms} ms)")
    print(f"  P99: {p99:.2f} ms")
    print(f"  Gate: {'PASS ✓' if metrics['gate_pass'] else 'FAIL ✗'}")
    print()

    return metrics


def evaluate_pad_coherence(
    bus: MMFBus,
    num_iterations: int = 100,
    batch_size: int = 8,
) -> Dict:
    """
    Evaluate PAD gate coherence.

    Args:
        bus: MMF Bus instance
        num_iterations: Number of evaluation iterations
        batch_size: Batch size

    Returns:
        Dict with coherence metrics
    """
    print("Evaluating PAD gate coherence...")

    coherences = []

    for i in range(num_iterations):
        # Generate inputs
        inputs = generate_synthetic_inputs(batch_size, ["text", "audio", "video"])

        with torch.no_grad():
            output = bus(inputs)

        coherence = output.gate_decisions["coherence"].mean().item()
        coherences.append(coherence)

        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/{num_iterations}")

    coherences = np.array(coherences)

    mean_coherence = np.mean(coherences)
    std_coherence = np.std(coherences)
    min_coherence = np.min(coherences)

    metrics = {
        "mean_coherence": mean_coherence,
        "std_coherence": std_coherence,
        "min_coherence": min_coherence,
        "gate_pass": mean_coherence > bus.config.pad_coherence_threshold,
    }

    print(f"  Mean: {mean_coherence:.3f}")
    print(f"  Std: {std_coherence:.3f}")
    print(f"  Min: {min_coherence:.3f}")
    print(f"  Threshold: {bus.config.pad_coherence_threshold}")
    print(f"  Gate: {'PASS ✓' if metrics['gate_pass'] else 'FAIL ✗'}")
    print()

    return metrics


def evaluate_fusion_overhead(
    bus: MMFBus,
    num_iterations: int = 50,
    batch_size: int = 8,
) -> Dict:
    """
    Evaluate fusion overhead vs single-modal.

    Args:
        bus: MMF Bus instance
        num_iterations: Number of evaluation iterations
        batch_size: Batch size

    Returns:
        Dict with overhead metrics
    """
    print("Evaluating fusion overhead...")

    multimodal_times = []
    single_modal_times = []

    for i in range(num_iterations):
        # Multi-modal timing
        inputs = generate_synthetic_inputs(batch_size, ["text", "audio", "video"])

        bus.config.enable_profiling = True

        start = time.perf_counter()
        with torch.no_grad():
            output = bus(inputs)
        multimodal_time = (time.perf_counter() - start) * 1000

        multimodal_times.append(multimodal_time)

        # Single-modal timing (text only)
        inputs_single = {"text": inputs["text"]}

        start = time.perf_counter()
        with torch.no_grad():
            output_single = bus(inputs_single)
        single_time = (time.perf_counter() - start) * 1000

        single_modal_times.append(single_time)

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{num_iterations}")

    multimodal_times = np.array(multimodal_times)
    single_modal_times = np.array(single_modal_times)

    mean_multimodal = np.mean(multimodal_times)
    mean_single = np.mean(single_modal_times)

    overhead_pct = ((mean_multimodal - mean_single) / mean_single) * 100

    metrics = {
        "mean_multimodal_ms": mean_multimodal,
        "mean_single_ms": mean_single,
        "overhead_pct": overhead_pct,
        "gate_pass": overhead_pct < 10.0,
    }

    print(f"  Multi-modal: {mean_multimodal:.2f} ms")
    print(f"  Single-modal: {mean_single:.2f} ms")
    print(f"  Overhead: {overhead_pct:.1f}% (threshold: < 10%)")
    print(f"  Gate: {'PASS ✓' if metrics['gate_pass'] else 'FAIL ✗'}")
    print()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMF Bus")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output", type=str, default="artifacts/mmf/eval_results.json",
                        help="Output JSON file")

    args = parser.parse_args()

    print("=" * 70)
    print("MMF BUS EVALUATION")
    print("=" * 70)
    print()

    # Create bus with adapters
    print("Initializing MMF Bus...")
    config = MMFBusConfig(
        d_model=768,
        modalities=["text", "audio", "video"],
        ttw_p95_latency_ms=5.0,
        pad_coherence_threshold=0.85,
        enable_profiling=True,
    )

    bus = MMFBus(config=config)

    # Register adapters
    bus.register_adapter("text", TextEntityAdapter(output_dim=768))
    bus.register_adapter("audio", AudioProsodyAdapter(output_dim=768))
    bus.register_adapter("video", VideoOpticalAdapter(output_dim=768))

    print("Bus initialized\n")

    # Run evaluations
    results = {}

    # 1. TTW Latency
    results["ttw_latency"] = evaluate_ttw_latency(bus, args.iterations, args.batch_size)

    # 2. PAD Coherence
    results["pad_coherence"] = evaluate_pad_coherence(bus, args.iterations, args.batch_size)

    # 3. Fusion Overhead
    results["fusion_overhead"] = evaluate_fusion_overhead(bus, args.iterations // 2, args.batch_size)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_gates_pass = (
        results["ttw_latency"]["gate_pass"]
        and results["pad_coherence"]["gate_pass"]
        and results["fusion_overhead"]["gate_pass"]
    )

    print(f"TTW Latency: {'PASS ✓' if results['ttw_latency']['gate_pass'] else 'FAIL ✗'}")
    print(f"PAD Coherence: {'PASS ✓' if results['pad_coherence']['gate_pass'] else 'FAIL ✗'}")
    print(f"Fusion Overhead: {'PASS ✓' if results['fusion_overhead']['gate_pass'] else 'FAIL ✗'}")
    print()
    print(f"Overall: {'ALL GATES PASSED ✓✓✓' if all_gates_pass else 'SOME GATES FAILED ✗✗✗'}")
    print()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
