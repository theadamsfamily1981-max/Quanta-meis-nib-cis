# Task: Node toolchain setup (optional)

Goal
- Install Node deps and run JS/TS tests if present.

Steps
1) Branch `codex/node_setup`.
2) Detect package manager and install deps: `npm ci || npm i || true`.
3) Run: `{{node_test_command}}` (ignore if script missing).
4) Commit: "chore(node): install deps and test".
5) PR: "Node toolchain setup".
