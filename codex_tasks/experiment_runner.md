# Task: Stabilize experiment runner

Goal
- Ensure `experiments/run_experiments.py` discovers and runs configs under `experiments/configs/`.
- Add a summary artifact (JSON) with metrics.

Steps
1) Branch `codex/experiments_runner`.
2) Implement discovery (glob .yaml), run selected experiments, write `artifacts/summary.json`.
3) Add tests: `tests/test_experiment_runner.py` (use dummy config).
4) Commit: "Experiments: runner + summary artifact".
5) PR: "Experiment runner stabilization".
