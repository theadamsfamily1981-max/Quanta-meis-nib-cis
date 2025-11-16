#!/usr/bin/env python
"""
Expected Hypervolume Improvement (EHVI) for Pareto Optimization

Implements:
- Pareto frontier computation
- Hypervolume calculation
- EHVI acquisition function for Bayesian optimization
- Crowding distance for diversity

Usage:
    # Compute Pareto frontier
    frontier = pareto_frontier(
        results,
        minimize=['latency_ms', 'memory_mb'],
        maximize=['accuracy']
    )

    # Compute hypervolume
    hv = hypervolume(frontier, reference_point=[1000, 10000, 0])

    # Compute EHVI for candidate point
    ehvi_score = compute_ehvi(candidate, frontier, reference_point)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ParetoPoint:
    """Single point on Pareto frontier."""
    config_id: str
    params: Dict
    objectives: np.ndarray  # Normalized objective values

    def __repr__(self):
        return f"ParetoPoint({self.config_id}, obj={self.objectives})"


def pareto_frontier(
    results: List,
    minimize: List[str] = None,
    maximize: List[str] = None
) -> List[ParetoPoint]:
    """
    Compute Pareto frontier from results.

    A point is non-dominated if no other point is better in all objectives.

    Args:
        results: List of RunResult objects
        minimize: Objective names to minimize
        maximize: Objective names to maximize

    Returns:
        List of ParetoPoint objects on frontier
    """
    minimize = minimize or []
    maximize = maximize or []

    # Extract successful results
    successful = [r for r in results if r.status == 'success']

    if not successful:
        return []

    # Extract objective values
    points = []
    for result in successful:
        obj_values = []

        # Minimize objectives (no negation needed)
        for obj_name in minimize:
            obj_values.append(result.metrics.get(obj_name, float('inf')))

        # Maximize objectives (negate to convert to minimization)
        for obj_name in maximize:
            obj_values.append(-result.metrics.get(obj_name, float('-inf')))

        obj_array = np.array(obj_values)

        point = ParetoPoint(
            config_id=result.config_id,
            params=result.params,
            objectives=obj_array
        )
        points.append(point)

    # Find non-dominated points
    frontier = []

    for i, point_i in enumerate(points):
        dominated = False

        for j, point_j in enumerate(points):
            if i == j:
                continue

            # Check if point_j dominates point_i
            # (point_j is better or equal in all objectives, strictly better in at least one)
            if _dominates(point_j.objectives, point_i.objectives):
                dominated = True
                break

        if not dominated:
            frontier.append(point_i)

    print(f"✓ Pareto frontier: {len(frontier)}/{len(points)} non-dominated points")

    return frontier


def _dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
    """
    Check if obj1 dominates obj2.

    obj1 dominates obj2 if:
    - obj1[i] <= obj2[i] for all i (better or equal in all objectives)
    - obj1[i] < obj2[i] for at least one i (strictly better in at least one)
    """
    better_or_equal = np.all(obj1 <= obj2)
    strictly_better = np.any(obj1 < obj2)

    return better_or_equal and strictly_better


def hypervolume(
    frontier: List[ParetoPoint],
    reference_point: np.ndarray
) -> float:
    """
    Compute hypervolume of Pareto frontier.

    Hypervolume is the volume of objective space dominated by the frontier.

    Args:
        frontier: List of ParetoPoint objects
        reference_point: Reference point (worst acceptable values)

    Returns:
        Hypervolume value
    """
    if not frontier:
        return 0.0

    # For simplicity, use Monte Carlo approximation
    # (exact hypervolume calculation is complex for >2 objectives)

    objectives_array = np.array([p.objectives for p in frontier])
    num_objectives = objectives_array.shape[1]

    # Monte Carlo sampling
    num_samples = 10000

    # Sample random points in dominated region
    mins = objectives_array.min(axis=0)
    maxs = reference_point

    # Sample uniform points
    samples = np.random.uniform(
        low=mins,
        high=maxs,
        size=(num_samples, num_objectives)
    )

    # Count how many samples are dominated by frontier
    dominated_count = 0

    for sample in samples:
        # Check if any frontier point dominates this sample
        for point in frontier:
            if _dominates(point.objectives, sample):
                dominated_count += 1
                break

    # Estimate hypervolume
    volume_total = np.prod(maxs - mins)
    hv = (dominated_count / num_samples) * volume_total

    return hv


def compute_ehvi(
    candidate: np.ndarray,
    frontier: List[ParetoPoint],
    reference_point: np.ndarray
) -> float:
    """
    Compute Expected Hypervolume Improvement for candidate point.

    EHVI measures the expected increase in hypervolume if we add this candidate.

    Args:
        candidate: Candidate objective values
        frontier: Current Pareto frontier
        reference_point: Reference point

    Returns:
        EHVI score (higher is better)
    """
    # Compute hypervolume with current frontier
    hv_current = hypervolume(frontier, reference_point)

    # Add candidate to frontier
    candidate_point = ParetoPoint(
        config_id='candidate',
        params={},
        objectives=candidate
    )

    # Recompute Pareto frontier with candidate
    frontier_with_candidate = pareto_frontier_from_points(
        frontier + [candidate_point]
    )

    # Compute new hypervolume
    hv_new = hypervolume(frontier_with_candidate, reference_point)

    # EHVI is the improvement
    ehvi = max(0.0, hv_new - hv_current)

    return ehvi


def pareto_frontier_from_points(points: List[ParetoPoint]) -> List[ParetoPoint]:
    """Compute Pareto frontier from ParetoPoint objects."""
    frontier = []

    for i, point_i in enumerate(points):
        dominated = False

        for j, point_j in enumerate(points):
            if i == j:
                continue

            if _dominates(point_j.objectives, point_i.objectives):
                dominated = True
                break

        if not dominated:
            frontier.append(point_i)

    return frontier


def crowding_distance(frontier: List[ParetoPoint], epsilon: float = 0.05) -> np.ndarray:
    """
    Compute crowding distance for each point on frontier.

    Crowding distance measures density of points around each frontier point.
    Used to encourage diversity.

    Args:
        frontier: List of ParetoPoint objects
        epsilon: Minimum crowding distance threshold

    Returns:
        Array of crowding distances
    """
    if len(frontier) <= 2:
        # Boundary points have infinite crowding distance
        return np.full(len(frontier), np.inf)

    objectives_array = np.array([p.objectives for p in frontier])
    num_points, num_objectives = objectives_array.shape

    # Initialize crowding distances
    crowding_dist = np.zeros(num_points)

    # Compute crowding distance for each objective
    for obj_idx in range(num_objectives):
        # Sort points by this objective
        sorted_indices = np.argsort(objectives_array[:, obj_idx])

        # Boundary points get infinite distance
        crowding_dist[sorted_indices[0]] = np.inf
        crowding_dist[sorted_indices[-1]] = np.inf

        # Normalize objective range
        obj_range = (
            objectives_array[sorted_indices[-1], obj_idx] -
            objectives_array[sorted_indices[0], obj_idx]
        )

        if obj_range == 0:
            continue

        # Compute distance for interior points
        for i in range(1, num_points - 1):
            idx = sorted_indices[i]
            idx_prev = sorted_indices[i - 1]
            idx_next = sorted_indices[i + 1]

            distance = (
                objectives_array[idx_next, obj_idx] -
                objectives_array[idx_prev, obj_idx]
            ) / obj_range

            crowding_dist[idx] += distance

    return crowding_dist


def check_crowding_gate(frontier: List[ParetoPoint], epsilon: float = 0.05) -> bool:
    """
    Check if frontier meets minimum crowding distance gate.

    Args:
        frontier: Pareto frontier
        epsilon: Minimum crowding distance threshold

    Returns:
        True if gate passed
    """
    if len(frontier) < 6:
        return False

    distances = crowding_distance(frontier, epsilon)

    # Exclude infinite distances (boundary points)
    finite_distances = distances[np.isfinite(distances)]

    if len(finite_distances) == 0:
        return True

    # Check if minimum distance meets threshold
    min_distance = finite_distances.min()

    return min_distance >= epsilon
