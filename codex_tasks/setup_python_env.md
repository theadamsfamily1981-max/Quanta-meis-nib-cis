# Task: Ensure Python environment is usable

Goal
- Create/update virtualenv, install deps, and run a smoke test.

Steps
1) Branch `codex/setup_python_env`.
2) Run:
   - `python -m venv .venv || true`
   - `python -m pip install -U pip || true`
   - `pip install -r requirements.txt || true`
   - `pip install -r dev-requirements.txt || true`
   - `{{lint_command}}`
   - `{{test_command}}`
3) Commit: "chore(py): env setup, lint+test smoke".
4) PR: "Python env setup + smoke".
