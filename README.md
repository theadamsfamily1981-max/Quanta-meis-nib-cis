# Quanta-meis-nib-cis

This repository provides a lightweight, reproducible harness around the
Topological-FAN (TFAN) research prototypes.  The focus of this snapshot is on
plotting unified objective diagnostics and a dual curriculum illustrating the
No-Free-Lunch (NFL) effect.

## Getting started

Create a virtual environment and install the minimal runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the TFAN demo

The CLI entrypoint mirrors the historic TFAN knobs and adds a `--save-plots`
flag for persisting the diagnostic figures.

```bash
python -m tfan.cli --save-plots plots/
```

The command prints the tracked losses (landmark attention, α-probe, J_T-FAN) and
writes three PNG files into the provided directory:

* `pareto_acc_vs_diss_default.png`
* `throughput_vs_k_default.png`
* `nfl_structured_vs_scrambled.png`

The filenames align with the assets used throughout the associated paper and can
be safely versioned.

## Development

Run the test-suite to exercise the CLI integration and smoke the plotting code
paths:

```bash
pytest
```

The tests execute a CPU-only subset to keep CI fast while covering the critical
imports and argument wiring.
