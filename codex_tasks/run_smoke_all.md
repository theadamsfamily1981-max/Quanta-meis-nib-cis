# Task: Run all smoke experiments and attach summary

Goal
- Execute the experiment runner over all configs and commit artifacts.

Steps
1) Branch `codex/run_smoke_all`.
2) Run: `python experiments/run_experiments.py --all --out artifacts/summary.json`.
3) Commit artifacts/summary.json with message "chore: smoke experiments summary".
4) Open PR titled: "Smoke experiments: summary artifact".
