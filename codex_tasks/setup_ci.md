# Task: Add CI (lint + tests) without secrets

Goal
- Add `.github/workflows/ci.yml` to run ruff + pytest on PRs.

Steps
1) Branch `codex/ci_setup`.
2) Create workflow:
   - Triggers: pull_request
   - Jobs: setup Python 3.10/3.11, install, ruff, pytest
3) Commit: "CI: add lint+tests workflow".
4) PR: "Add CI for lint + tests".

Constraints
- No external secrets or privileged runners.
