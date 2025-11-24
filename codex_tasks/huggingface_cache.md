# Task: Hugging Face cache awareness (read-only)

Goal
- Report HF cache path and free space; do not pull new models.

Steps
1) Branch `codex/hf_cache_info`.
2) Run: `python -c "import os,shutil; p=os.path.expanduser(os.getenv('HF_HOME','~/.cache/huggingface')); import json; print(json.dumps({'hf_home':p, 'exists':__import__('os').path.isdir(p), 'free_gb': round(shutil.disk_usage(p).free/1e9,2)}))" || true`.
3) Save output to `artifacts/hf_cache.json`.
4) Commit: "chore(hf): cache info artifact".
5) PR: "HF cache info".
