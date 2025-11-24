# Task: Use Makefile for CI

Goal
- Update `.github/workflows/ci.yml` to call `make fmt` and `make test` instead of inline commands.

Steps
1) Branch `codex/make_ci`.
2) Edit CI workflow to run:
   - `make dev`
   - `make fmt`
   - `make test`
3) Commit: "CI: switch to Makefile targets".
4) PR: "Use Makefile in CI".
