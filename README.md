# Quanta–MEIS–NIB–CIS (GPU Lab)

Private repository for verified code, reproducible configs, and GPU test builds for QUANTA/MEIS/CIS.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
pytest -q tests/smoke
bash scripts/run_gpu_tests.sh
```
