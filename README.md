# QUANTA–TFAN Unified Framework

This repository hosts the QUANTA–TFAN Unified Framework v1.0, integrating
Temporal Fractal Attention Networks (T-FAN) with the MEIS inference engine,
NIB integration loop, and the antifragility experimentation suite.

## Components

- **MEIS**: Meta-Epistemic Inference System implemented in `core/meis.py`.
- **NIB**: Neural Integration Bridge loop in `core/nib_loop.py`.
- **T-FAN**: Dual implementations for JAX and PyTorch under `tfan/`.
- **Experiments**: Runner utilities for orchestrating the unified workflow
  in `experiments/experiment_runner.py`.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Review `colab_setup.ipynb` for an interactive introduction to the
   framework.
3. Run experiments using your preferred deep learning backend.

## License

This project is released for research and educational use.
