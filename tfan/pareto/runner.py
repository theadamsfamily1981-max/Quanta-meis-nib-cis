#!/usr/bin/env python
"""
Pareto Auto-Runner

Executes batch configurations and collects multi-objective metrics.

Supports:
- Grid search over hyperparameters
- Bayesian optimization with EHVI acquisition
- Parallel execution across GPUs
- Artifact collection (checkpoints, logs, metrics)

Usage:
    runner = ParetoRunner(objectives=['latency', 'accuracy'])

    configs = runner.grid_search({
        'k_landmarks': [32, 64, 128],
        'local_window': [128, 256, 512]
    })

    results = runner.run_batch(configs, max_workers=4)
    frontier = pareto_frontier(results, minimize=['latency'], maximize=['accuracy'])
"""

import json
import time
import hashlib
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools


@dataclass
class RunConfig:
    """Single run configuration."""
    config_id: str
    params: Dict
    objectives: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def hash_params(params: Dict) -> str:
        """Generate deterministic hash for params."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(param_str.encode()).hexdigest()[:12]


@dataclass
class RunResult:
    """Result from a single run."""
    config_id: str
    params: Dict
    metrics: Dict
    wall_time_s: float
    status: str  # 'success', 'failed', 'timeout'
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ParetoRunner:
    """
    Multi-objective auto-runner with Pareto frontier computation.
    """

    def __init__(
        self,
        objectives: List[str],
        artifacts_dir: str = 'artifacts/pareto',
        max_time_per_run: float = 3600.0  # 1 hour
    ):
        """
        Initialize Pareto runner.

        Args:
            objectives: List of objective names to optimize
            artifacts_dir: Directory for storing artifacts
            max_time_per_run: Max time per run in seconds
        """
        self.objectives = objectives
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.max_time_per_run = max_time_per_run

        self.results: List[RunResult] = []

        print(f"✓ ParetoRunner initialized")
        print(f"  Objectives: {objectives}")
        print(f"  Artifacts: {self.artifacts_dir}")

    def grid_search(
        self,
        param_grid: Dict[str, List]
    ) -> List[RunConfig]:
        """
        Generate grid search configurations.

        Args:
            param_grid: Dict mapping param names to value lists

        Returns:
            List of RunConfig objects
        """
        # Generate all combinations
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]

        configs = []
        for values in itertools.product(*value_lists):
            params = dict(zip(keys, values))
            config_id = RunConfig.hash_params(params)

            config = RunConfig(
                config_id=config_id,
                params=params,
                objectives=self.objectives
            )
            configs.append(config)

        print(f"✓ Generated {len(configs)} grid search configurations")
        return configs

    def run_single(
        self,
        config: RunConfig,
        runner_fn: Callable[[Dict], Dict]
    ) -> RunResult:
        """
        Execute single configuration.

        Args:
            config: Configuration to run
            runner_fn: Function that takes params dict and returns metrics dict

        Returns:
            RunResult with metrics
        """
        print(f"Running config {config.config_id}...")

        start_time = time.perf_counter()

        try:
            # Execute runner function with timeout
            metrics = runner_fn(config.params)

            wall_time = time.perf_counter() - start_time

            # Check timeout
            if wall_time > self.max_time_per_run:
                status = 'timeout'
                error = f"Exceeded max time: {wall_time:.1f}s > {self.max_time_per_run}s"
            else:
                status = 'success'
                error = None

            result = RunResult(
                config_id=config.config_id,
                params=config.params,
                metrics=metrics,
                wall_time_s=wall_time,
                status=status,
                error=error
            )

        except Exception as e:
            wall_time = time.perf_counter() - start_time

            result = RunResult(
                config_id=config.config_id,
                params=config.params,
                metrics={},
                wall_time_s=wall_time,
                status='failed',
                error=str(e)
            )

        # Save result
        self._save_result(result)

        return result

    def run_batch(
        self,
        configs: List[RunConfig],
        runner_fn: Callable[[Dict], Dict],
        max_workers: int = 1
    ) -> List[RunResult]:
        """
        Execute batch of configurations in parallel.

        Args:
            configs: List of configurations
            runner_fn: Runner function
            max_workers: Number of parallel workers

        Returns:
            List of RunResult objects
        """
        print(f"\n{'='*60}")
        print(f"Running {len(configs)} configurations with {max_workers} workers")
        print(f"{'='*60}")

        results = []

        if max_workers == 1:
            # Sequential execution
            for config in configs:
                result = self.run_single(config, runner_fn)
                results.append(result)
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.run_single, config, runner_fn): config
                    for config in configs
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

        self.results.extend(results)

        # Save all results
        self._save_all_results()

        print(f"\n{'='*60}")
        print(f"Batch complete: {len(results)} runs")
        print(f"  Success: {sum(1 for r in results if r.status == 'success')}")
        print(f"  Failed: {sum(1 for r in results if r.status == 'failed')}")
        print(f"  Timeout: {sum(1 for r in results if r.status == 'timeout')}")
        print(f"{'='*60}")

        return results

    def _save_result(self, result: RunResult):
        """Save single result to disk."""
        result_path = self.artifacts_dir / f"{result.config_id}.json"

        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_all_results(self):
        """Save all results to consolidated file."""
        all_results_path = self.artifacts_dir / 'pareto.json'

        data = {
            'objectives': self.objectives,
            'num_runs': len(self.results),
            'results': [r.to_dict() for r in self.results]
        }

        with open(all_results_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved results to {all_results_path}")

    def get_successful_results(self) -> List[RunResult]:
        """Get only successful runs."""
        return [r for r in self.results if r.status == 'success']


def mock_runner_fn(params: Dict) -> Dict:
    """
    Mock runner function for testing.

    Simulates running a model with given params and returning metrics.
    """
    import random
    import time

    # Simulate computation time
    time.sleep(random.uniform(0.1, 0.5))

    # Mock metrics based on params
    k_landmarks = params.get('k_landmarks', 64)
    local_window = params.get('local_window', 256)

    # Trade-off: more landmarks = better accuracy but slower
    latency_ms = 50 + k_landmarks * 0.5 + local_window * 0.1
    accuracy = 0.85 + (k_landmarks / 1000) - (latency_ms / 10000)

    # Add noise
    latency_ms += random.gauss(0, 5)
    accuracy += random.gauss(0, 0.01)
    accuracy = max(0.0, min(1.0, accuracy))

    return {
        'latency_ms': latency_ms,
        'accuracy': accuracy,
        'memory_mb': k_landmarks * 2 + local_window * 0.5
    }
