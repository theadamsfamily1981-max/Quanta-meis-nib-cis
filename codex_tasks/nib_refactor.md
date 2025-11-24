# Task: Refactor NIB core (EVNI, identity updates, sleep SVD)

Goal
- Extract NIB primitives into `meis/nib/core.py`.
- Add tests for EVNI gating, identity updates, and mocked sleep SVD path.

Steps
1) Branch `codex/nib_core_refactor`.
2) Create/organize:
   - meis/nib/{core.py,__init__.py}
   - tests/test_nib_core.py (3 focused tests)
3) Add docstrings to `core.py` for public functions.
4) Run `{{test_command}}`; fix issues.
5) Commit: "Refactor NIB core + tests".
6) PR: "NIB core refactor with EVNI/identity tests".
