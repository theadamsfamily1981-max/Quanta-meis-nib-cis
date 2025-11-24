# Task: GPU sanity (read-only)

Goal
- Collect non-sensitive GPU diagnostics for CI notes.

Steps
1) Branch `codex/gpu_sanity`.
2) Run: `nvidia-smi || true` and record output to `artifacts/gpu.txt`.
3) Run Python CUDA probe to `artifacts/cuda.txt`:
   - `python -c "import torch,sys;print(torch.__version__, torch.cuda.is_available())" || true`
4) Commit artifacts: "chore(gpu): add diagnostics".
5) PR: "GPU diagnostics (informational)".

Constraints
- Do not upload large logs or any secrets. Keep artifacts under 64KB.
