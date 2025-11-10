# Quanta-meis-nib-cis

Research for quanta meis nib cis code.

## Validation quickstart

Create and activate a virtual environment, install dependencies, and run both validation suites:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python validation/suites/comprehensive_validation.py --out validation/artifacts/comprehensive_results.json
python validation/suites/impossibility_validation.py --out validation/artifacts/impossibility_results.json
```

Gate the generated metrics against the configured thresholds:

```bash
python validation/validate_thresholds.py \
  --comp validation/artifacts/comprehensive_results.json \
  --imp  validation/artifacts/impossibility_results.json \
  --thresholds validation/thresholds.json
```

Finally, build the plots and markdown report:

```bash
python validation/make_plots.py \
  --comp validation/artifacts/comprehensive_results.json \
  --imp  validation/artifacts/impossibility_results.json \
  --outdir validation/reports
```

Generated artifacts live in `validation/artifacts/` and visual assets in `validation/reports/`.
