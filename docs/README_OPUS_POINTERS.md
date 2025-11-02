# README pointers for Opus + tools

* See **docs/OPUS_RUNBOOK.md** for PR rules and repo layout.
* Environment helpers are in **tools/**:
  - `tools/torch_preflight.py` — CUDA/Torch check + tiny train
  - `tools/gpu_selector.py` — recommends `CUDA_VISIBLE_DEVICES`
  - `tools/repo_smoke_run.py` — quick run of runner + prod suite
* Data env vars: **docs/DATA_DROP_README.md**
* Priority quick bench (no GPU/datasets):
  - `python scripts/run_innov_priority.py`
  - GH Action: add label **bench** to a PR or trigger **Bench (INNOV priority)** via _Run workflow_.
