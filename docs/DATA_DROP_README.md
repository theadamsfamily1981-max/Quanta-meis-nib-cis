# Data Drop README

Set these env vars to point to your local datasets before running:

```bash
export MELD_ROOT=/path/to/MELD
export IEMOCAP_ROOT=/path/to/IEMOCAP
```

Then inside the repo:

```bash
python experiments/runner.py                          # NumPy default
QMNC_BACKEND=torch python experiments/runner.py       # Torch path if installed
python scripts/run_production_suite.py                # Calibration + OOD + selective metrics
```
