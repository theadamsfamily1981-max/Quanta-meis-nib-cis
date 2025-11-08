# Geometric Field Theory

This module provides a compact implementation of the Unified Field Equations of
Intelligence (EFE-L).  The code base is intentionally modest in scope and serves
as a reference for experiments that couple information geometry, dissipative
flows, and topological stability diagnostics.

## Components

- **Fisher-Rao metric**: utilities for normalising probability distributions and
evaluating quadratic forms on the simplex.
- **Rayleigh dissipation**: a diagonal quadratic form used to model energy
losses in the system.
- **Einstein field dynamics**: combines the geometric and dissipative models to
produce field updates.
- **Persistent homology**: thin wrappers over ``ripser`` to compute diagrams and
approximate bottleneck distances.
- **Unified field equations**: orchestrates the individual pieces into a
simulation-friendly interface.

## Running tests

```bash
python -m pytest
```
