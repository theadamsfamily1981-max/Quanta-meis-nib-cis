# Quanta-meis-nib-cis

This repository hosts a compact, fully tested snapshot of the "Topology-first
Attention Network" utilities.  The original project is substantially larger and
contains heavy third party dependencies; the version provided here aims to be
lightweight, deterministic and easily testable.

## Layout

```
tfan/                # Core library modules
bench/               # Small benchmark helpers
configs/             # Example experiment configurations
tests/               # Pytest suite exercising the public API
```

## Getting started

Create a virtual environment, install the dependencies and run the test-suite::

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pytest

The modules are pure Python and deliberately avoid heavyweight dependencies, so
they run quickly on modest hardware.
