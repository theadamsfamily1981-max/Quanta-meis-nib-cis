# Bio Baselines (ARF, MT-STC, MEIS)

This package hosts lightweight adapters for the three bio-inspired baselines from Claude’s drop.

- **Goal:** provide a uniform, CPU-safe entrypoint so Opus/CI can run a smoke bench and produce a single JSON artifact.
- **Status:** stubbed (`stub-0.1`) — swap in the real modules when vendored.

## Run
```bash
python scripts/run_bio_baselines.py
cat reports/bio_baselines.json
```

## Replace with real baselines
- Vendor code into `packages/bio_baselines/{arf,mt_stc,meis}/`
- Export a `run()` in each submodule and wire into `__init__.py`

