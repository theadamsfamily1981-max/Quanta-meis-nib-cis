# Tools Overview

This folder contains helper scripts to validate your environment and run basic repo smoke tests.

## torch_preflight.py
Verifies Python/Torch/CUDA/cuDNN, enumerates GPUs, runs a 4k×4k matmul and a 5-step tiny train loop, and writes `torch_preflight_results.json`.

```bash
python tools/torch_preflight.py
```

## gpu_selector.py
Prints a recommended `CUDA_VISIBLE_DEVICES` order based on VRAM and SM capability.

```bash
python tools/gpu_selector.py
```

## repo_smoke_run.py
Runs the experiment runner and production suite. Prefers Torch backend if available.

```bash
python tools/repo_smoke_run.py
```

## Data Drop
See `docs/DATA_DROP_README.md` for dataset env vars and quick-start.
