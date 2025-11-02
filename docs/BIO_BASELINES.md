# BIO_BASELINES.md

This document shows how to run and compare the **bio-inspired baselines** (ARF, MT‑STC, MEIS) against QUANTA using artifact-first JSON outputs that Opus and CI can ingest.

> The current adapters are CPU-safe stubs. When we vendor the real baselines, the interface and JSON schemas remain the same.

## Quick Start
```bash
python scripts/run_bio_baselines.py
# -> reports/bio_baselines.json

python scripts/run_quanta_bench.py
# -> reports/quanta/{risk_coverage.json,calibration.json,ood.json,latency.json}
```

## JSON Schemas
- `reports/bio_baselines.json`
  ```json
  {
    "ARF": {
      "metrics": {"accuracy": 0.9, "ece": 0.03, "ood_auroc": 0.92, "forgetting": 0.03}
    },
    "MT_STC": {...},
    "MEIS": {...}
  }
  ```

- `reports/quanta/*.json`
  - `risk_coverage.json`: `{ "coverage": [0.6,0.8,0.9], "selective_accuracy": [..] }`
  - `calibration.json`: `{ "ece_10": .., "ece_20": .., "ece_50": .. }`
  - `ood.json`: `{ "energy_auroc": .., "mahalanobis_auroc": .., "fusion_auroc": .. }`
  - `latency.json`: `{ "p50": .., "p95": .., "p99": .. }`

## Interpreting Results
- **Selective@80%**: expect QUANTA ≥ 0.95 on our synthetic suite.
- **ECE**: ≤ 0.10 is acceptable; 0.03–0.05 is excellent.
- **OOD AUROC**: ≥ 0.80 is acceptable; ≥ 0.94 is excellent.
- **Forgetting**: ≤ 0.10 acceptable; ≤ 0.05 is great.

## When we vendor real baselines
- Drop actual code under `packages/bio_baselines/{arf,mt_stc,meis}/` and export a `run()` per module.
- Update `packages/bio_baselines/__init__.py` to call each `run()` instead of stubs.
- Keep JSON shape unchanged so CI/Opus dashboards continue to work.

