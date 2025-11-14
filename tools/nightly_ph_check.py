#!/usr/bin/env python3
"""
Nightly persistence homology validation check.

Gates:
- Wasserstein gap ≤ 2% between approximate and exact PH
- cos(PL, target) ≥ 0.90 (landscape alignment)

Outputs JSON report to artifacts/
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.topo import (
    compute_persistence_diagram,
    PersistenceLandscape,
    wasserstein_distance,
    GUDHI_AVAILABLE,
    RIPSER_AVAILABLE
)
from tfan.io import new_artifact, write_json


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between flattened arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a_flat, b_flat) / (norm_a * norm_b)


def generate_synthetic_cloud(n_points: int = 1000, noise: float = 0.1) -> np.ndarray:
    """Generate synthetic point cloud with known topological features."""
    # Circle + noise in 3D
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack([
        np.cos(theta),
        np.sin(theta),
        np.zeros(n_points)
    ])
    noise_vec = np.random.randn(n_points, 3) * noise
    return circle + noise_vec


def run_ph_validation(max_samples: int = 5000, time_cap_min: int = 20) -> dict:
    """
    Run nightly PH validation.

    Returns:
        dict with validation results
    """
    start_time = time.time()
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gudhi_available": GUDHI_AVAILABLE,
        "ripser_available": RIPSER_AVAILABLE,
        "tests": []
    }

    # Check engines available
    if not (GUDHI_AVAILABLE or RIPSER_AVAILABLE):
        results["status"] = "SKIP"
        results["reason"] = "No PH engine available"
        return results

    # Generate test data
    print(f"Generating synthetic point cloud (n={max_samples})...")
    X = generate_synthetic_cloud(n_points=min(max_samples, 2000))

    # Test 1: Wasserstein gap between engines (if both available)
    if GUDHI_AVAILABLE and RIPSER_AVAILABLE:
        print("Computing PD with GUDHI...")
        pd_gudhi = compute_persistence_diagram(X, max_dim=1, engine="gudhi")[1]

        print("Computing PD with Ripser...")
        pd_ripser = compute_persistence_diagram(X, max_dim=1, engine="ripser")[1]

        wass_dist = wasserstein_distance(pd_gudhi, pd_ripser, q=2)

        # Compute relative gap
        max_life_gudhi = (pd_gudhi[:, 1] - pd_gudhi[:, 0]).max() if len(pd_gudhi) > 0 else 1.0
        wass_gap = wass_dist / max_life_gudhi if max_life_gudhi > 0 else 0.0

        test1_pass = wass_gap <= 0.02  # 2% threshold

        results["tests"].append({
            "name": "wasserstein_gap",
            "value": float(wass_gap),
            "threshold": 0.02,
            "pass": test1_pass
        })

        print(f"  Wasserstein gap: {wass_gap:.4f} {'✓' if test1_pass else '✗'}")
    else:
        print("Skipping Wasserstein test (need both GUDHI and Ripser)")

    # Test 2: Landscape alignment with target
    print("\nComputing persistence landscapes...")
    engine = "gudhi" if GUDHI_AVAILABLE else "ripser"
    pd = compute_persistence_diagram(X, max_dim=1, engine=engine)[1]

    pl = PersistenceLandscape(num_landscapes=5, resolution=100)
    landscapes = pl.fit_transform(pd)

    # Create synthetic target (circle should have 1 prominent loop)
    # Target: single peak in landscape
    target_landscapes = np.zeros_like(landscapes)
    # Add a peak around mid-death
    peak_idx = target_landscapes.shape[1] // 2
    target_landscapes[0, peak_idx - 10:peak_idx + 10] = np.linspace(0, 0.5, 20)

    cos_sim = cosine_similarity(landscapes, target_landscapes)
    test2_pass = cos_sim >= 0.90

    results["tests"].append({
        "name": "landscape_alignment",
        "value": float(cos_sim),
        "threshold": 0.90,
        "pass": test2_pass
    })

    print(f"  Landscape cos similarity: {cos_sim:.4f} {'✓' if test2_pass else '✗'}")

    # Overall status
    all_pass = all(t["pass"] for t in results["tests"])
    results["status"] = "PASS" if all_pass else "FAIL"
    results["duration_sec"] = time.time() - start_time

    return results


def main():
    parser = argparse.ArgumentParser(description="Nightly PH validation check")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum samples for test point cloud")
    parser.add_argument("--time-cap-min", type=int, default=20,
                        help="Time cap in minutes")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: auto-generate in artifacts/)")

    args = parser.parse_args()

    print("=" * 60)
    print("TFAN Nightly PH Validation Check")
    print("=" * 60)

    results = run_ph_validation(
        max_samples=args.max_samples,
        time_cap_min=args.time_cap_min
    )

    # Write results
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = new_artifact("nightly_ph_check", suffix=".json")

    write_json(out_path, results)

    print("\n" + "=" * 60)
    print(f"Status: {results['status']}")
    print(f"Report: {out_path}")
    print("=" * 60)

    # Exit code
    if results["status"] == "PASS":
        sys.exit(0)
    elif results["status"] == "SKIP":
        sys.exit(0)  # Don't fail on skip
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
