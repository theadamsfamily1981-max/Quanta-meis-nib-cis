# Codex Playbook for Quanta-meis-nib-cis

This doc defines how to use OpenAI Codex to work on this repo via CLI/IDE or Codex Cloud, with least-privilege, task-driven workflows.

## 0) One-time setup

```bash
# Install and sign in
npm i -g @openai/codex
codex login

# Connect GitHub (choose only this repo)
codex integrate github
```

Enforce safety in GitHub:

- Enable branch protections on `main` (require PR + review).
- Optional: create an "Owners" CODEOWNERS rule for critical paths.

## 1) Minimal `codex.yaml` (root of repo)

The repository includes a starter `codex.yaml` at the project root that guards what Codex may run/edit and how tasks are executed.

```yaml
project:
  name: quanta-meis-nib-cis
  repo_root: .
  visibility: private

policies:
  # Keep this tight; expand as needed
  allowed_commands:
    - "git status"
    - "git checkout -b codex/*"
    - "git add -A"
    - "git commit -m '{{message}}'"
    - "git push -u origin HEAD"
    - "pytest -q || true"
    - "python -m pip install -U pip || true"
    - "pip install -r requirements.txt || true"
    - "ruff check --fix . || true"
    - "pytest -q --maxfail=1 --disable-warnings || true"
  protected_paths:
    - ".env"
    - "secrets/**"
    - ".github/workflows/release.yml"
    - "data/private/**"
  read_only_paths:
    - "LICENSE"
    - "SECURITY.md"

context:
  default_branch: "main"
  test_command: "pytest -q --maxfail=1 --disable-warnings || true"
  lint_command: "ruff check --fix . || true"

tasks_dir: codex_tasks
```

## 2) How to run tasks

You can run tasks by inline prompt or by file.

```bash
# Inline
codex run --task "Create branch, add tests for NIB EVNI thresholds, run pytest, open PR."

# From file
codex run --task-file codex_tasks/integrate_mam.md
```

## 3) Core tasks (available in `codex_tasks/`)

Each task below is already provided under `codex_tasks/` so Codex can pick it up immediately or you can use it as a template for new work.

### A) `integrate_mam.md`

```
# Task: Integrate MAM (Mycelial-Adaptive Multimodal) modules
Goal:
- Add/organize existing MAM files into:
  - meis/mam/
    - architecture.py
    - routing.py
    - __init__.py
  - tests/test_mam_architecture.py (smoke tests)
- Ensure imports match project layout; run tests; open a PR.

Steps:
1) Create branch `codex/mam_integration`.
2) Move/organize any existing MAM-related files already present in repo into `meis/mam/`.
3) If tests are missing, add a minimal smoke test:
   - import module, call primary entrypoints with dummy inputs, assert shape/types.
4) Run: `{{test_command}}`; fix straightforward issues.
5) Commit with message: "Integrate MAM modules and add smoke tests".
6) Open a PR titled: "MAM integration (initial)".

Constraints:
- Do not modify protected paths.
- Keep public APIs backward-compatible in `meis/__init__.py`.
Acceptance:
- New package structure exists.
- Tests run successfully (even if marked xfail for TODOs).
```

### B) `integrate_snn.md`

```
# Task: Integrate SNN + multimodal glue
Goal:
- Add/organize existing SNN files into `meis/snn/` and multimodal glue into `meis/mm/`.
- Provide adapter shims so SNN encoders can plug into current NIB/MAM.

Steps:
1) Branch: `codex/snn_glue`.
2) Place files:
   - meis/snn/{neurons.py, layers.py, training.py, __init__.py}
   - meis/mm/{adapters.py, fusion.py, __init__.py}
3) Add tests:
   - tests/test_snn_integration.py: construct tiny SNN layer -> adapter -> fusion pipeline; assert pass-through.
4) Run `{{lint_command}}` then `{{test_command}}`; fix trivial issues.
5) Commit: "Add SNN + multimodal adapters with smoke tests".
6) Open PR: "SNN + Multimodal glue (initial)"
```

### C) `nib_refactor.md`

```
# Task: Refactor NIB loops (EVNI, identity updates, sleep SVD)
Goal:
- Extract NIB loop primitives into `meis/nib/core.py`.
- Add tests for:
  - EVNI thresholds/gating
  - identity update dynamics
  - sleep consolidation SVD call path (mocked)

Steps:
1) Branch `codex/nib_core_refactor`.
2) Create/organize:
   - meis/nib/{core.py, __init__.py}
   - tests/test_nib_core.py (cover 3 items above, fine to mock heavy deps)
3) Ensure minimal docs/comments in `core.py`.
4) Run `{{test_command}}`; fix issues.
5) Commit: "Refactor NIB core + tests".
6) PR: "NIB core refactor with EVNI/identity tests"
```

### D) `docs_update.md`

```
# Task: Docs pass (architecture + experiments)
Goal:
- Update docs/architecture_overview.md with MAM+SNN+NIB diagram text.
- Update docs/experiment_design.md describing current test harness and metrics (ECE, Brier, risk-coverage).

Steps:
1) Branch `codex/docs_refresh`.
2) Edit docs with concise sections and TODOs where needed.
3) Commit: "Docs: architecture & experiments refresh".
4) PR: "Docs refresh (architecture + experiments)"
```

### E) `setup_ci.md`

```
# Task: Add CI (lint + tests) without secrets
Goal:
- Add `.github/workflows/ci.yml` to run ruff + pytest on PRs.

Steps:
1) Branch `codex/ci_setup`.
2) Create workflow:
   - Triggers: pull_request
   - Jobs: setup Python 3.10/3.11, install, ruff, pytest
3) Commit: "CI: add lint+tests workflow".
4) PR: "Add CI for lint + tests"

Constraints:
- No external secrets or privileged runners.
```

### F) `experiment_runner.md`

```
# Task: Stabilize experiment runner
Goal:
- Ensure `experiments/run_experiments.py` discovers and runs configs under `experiments/configs/`.
- Add a summary artifact (JSON) with metrics.

Steps:
1) Branch `codex/experiments_runner`.
2) Implement discovery (glob .yaml), run selected experiments, write `artifacts/summary.json`.
3) Add tests: `tests/test_experiment_runner.py` (use dummy config).
4) Commit: "Experiments: runner + summary artifact".
5) PR: "Experiment runner stabilization"
```

## 4) Running the flow (examples)

```bash
# Integrate MAM modules
codex run --task-file codex_tasks/integrate_mam.md

# Integrate SNN glue
codex run --task-file codex_tasks/integrate_snn.md

# Refactor NIB core
codex run --task-file codex_tasks/nib_refactor.md

# Docs, CI, experiments
codex run --task-file codex_tasks/docs_update.md
codex run --task-file codex_tasks/setup_ci.md
codex run --task-file codex_tasks/experiment_runner.md
```

## 5) PR review automation (optional)

If enabled in Codex Cloud/GitHub, you can ask for reviews on a PR:

```text
@codex review
```

Codex will leave a review and checklist. Keep required human review on for main.

## 6) Guardrails & conventions

- Branch names: `codex/<feature>`
- Commits: single-purpose, include short rationale.
- Never store secrets in repo.
- Keep heavy tests xfail or mocked; Codex can still validate plumbing.

## 7) Troubleshooting

- If Codex proposes edits to protected paths, it should skip and continue.
- If tests are red, Codex will attempt minimal fixes; complex refactors should go in a follow-up task.
- Expand `allowed_commands` only when needed.

If you want, I can also generate a tiny `docs/architecture_overview.md` and `experiments/run_experiments.py` skeleton next—so Codex has something concrete to wire into.

## 8) Local validation checklist

- Run the test suite locally with `pytest -q` before opening or updating a PR.
- Capture and share the most relevant command outputs when collaborating asynchronously.
- Re-run targeted modules or linting as needed after addressing review comments.
