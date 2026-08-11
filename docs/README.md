# GRTES Phase III Embodied Integration Launch Kit

This launch kit packages the artefacts required to exercise the Phase III
embodied intelligence stack. It provides a Python entry point for running the
synthetic integration flow, benchmark definitions, and documentation on how to
interpret the generated results.

## Contents

- `run_embodied_integration.py` &ndash; CLI runner that orchestrates the launch
  pipeline.
- `benchmark_suite/` &ndash; Collection of deterministic benchmark stubs used by the
  runner. These modules emulate GLUE and CIFAR style evaluations as well as the
  hardware profile abstraction.
- `data/phase3_results.json` &ndash; Default location for serialised benchmark
  output. The file is overwritten every time the runner executes unless
  `--dry-run` is provided.
- `docs/` &ndash; Documentation hub that expands on configuration, metrics, and
  expected usage.

## Quick start

Run the integrated flow and inspect the results:

```bash
python run_embodied_integration.py --dry-run
```

The command prints the JSON payload that would be written to disk. Remove the
`--dry-run` flag to persist the output to `data/phase3_results.json`.

## Customisation

Pass `--skip-glue` or `--skip-cifar` to evaluate only a subset of the
benchmarks. Additional configuration options for the benchmarks and hardware
profile are available via direct module imports; see the dedicated docs in this
folder for deeper guidance.
