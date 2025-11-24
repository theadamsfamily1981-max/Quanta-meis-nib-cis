# Task: Integrate MAM (Mycelial‑Adaptive Multimodal) modules

Goal
- Organize existing MAM files into:
  - meis/mam/{architecture.py,routing.py,__init__.py}
  - tests/test_mam_architecture.py (smoke tests)
- Ensure imports match project layout; run tests; open a PR.

Steps
1) Create branch `codex/mam_integration`.
2) Move/organize any MAM-related files already present into `meis/mam/`.
3) If tests are missing, add a minimal smoke test:
   - import module, call `route()` with dummy inputs, assert shape/types.
4) Run: `{{lint_command}}` then `{{test_command}}`; fix straightforward issues.
5) Commit: "Integrate MAM modules and add smoke tests".
6) Open PR: "MAM integration (initial)".

Constraints
- Do not modify protected paths.
- Keep public APIs backward-compatible in `meis/__init__.py`.

Acceptance
- New package structure exists.
- Tests run successfully (xfail allowed for heavy TODOs).
