# Topological Field-Adaptive Network (T-FAN)

T-FAN is a research prototype exploring the fusion of persistent homology with
attention-based field reasoning. This repository provides reproducible JAX and
PyTorch implementations, mathematical guarantees, and benchmarking artifacts for
the 567K-parameter Topological Field-Adaptive Network used in the Quanta lab's
2024 experiments.

## Project Highlights

- **Bi-framework parity** – feature-complete implementations in both JAX and
  PyTorch with consistent initialization, forward passes, and validation tools.
- **Topological inductive bias** – persistent homology signatures modulate each
  attention head via differentiable Wasserstein couplings.
- **Field-adaptive routing** – continuous Laplace-Beltrami kernels allow the
  model to interpolate between local and global reasoning regimes.
- **Validated training recipe** – reproducible training loop, optimizer setup,
  logging, and evaluation utilities supplied for both frameworks.
- **Formal backing** – the proofs folder contains six theorems formalizing the
  stability, expressivity, and convergence properties underpinning T-FAN.

## Repository Layout

```
.
├── jax_impl/                     # JAX reference implementation and prototype
│   ├── tfan_jax.py               # Minimal prototype with API documentation
│   └── tfan_jax_working.py       # Full training-ready module (≈567K params)
├── pytorch_impl/
│   └── tfan_pytorch.py           # PyTorch implementation mirroring the JAX one
├── proofs/
│   └── mathematical_proofs.md    # Core theoretical guarantees
├── results/
│   ├── experimental_comparison.md
│   ├── jax_validation.log
│   └── pytorch_validation.log
├── LICENSE                       # MIT license
├── README.md                     # You are here
└── .gitignore
```

## Quick Start

The repository targets Python 3.10+ and assumes JAX 0.4+, Flax 0.7+, PyTorch
2.2+, and standard scientific computing dependencies. You can install the
baseline environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # see below for a minimal list
```

A suggested `requirements.txt` (not included by default) is:

```
jax
jaxlib
flax
torch
torchvision
tqdm
numpy
scipy
``` 

### Running the JAX Reference Model

```bash
python jax_impl/tfan_jax_working.py --epochs 5 --batch-size 32 \
    --learning-rate 3e-4 --seed 7
```

The script loads a synthetic dataset, runs end-to-end training, and prints
validation metrics along with the effective parameter count.

### Running the PyTorch Implementation

```bash
python pytorch_impl/tfan_pytorch.py --epochs 5 --batch-size 32 \
    --learning-rate 3e-4 --seed 7
```

The PyTorch script mirrors the JAX interface, enabling drop-in comparison and
mixed-framework experimentation.

## Reproducing the Logged Results

The `results/` directory stores summarized outputs from the validation runs used
in the accompanying paper draft. To regenerate them, execute the framework-
specific training scripts with the hyperparameters from each log header. The
`experimental_comparison.md` file describes the evaluation protocol and how the
raw metrics were aggregated.

## Theory and Guarantees

The six theorems in `proofs/mathematical_proofs.md` formalize the following:

1. Stability of the differentiable persistence encoder under bounded noise.
2. Approximation guarantees for the Laplace-Beltrami kernelization.
3. Universal approximation of piecewise-smooth fields with adaptive routing.
4. Convergence of the AdamW optimizer under the adaptive scheduling rule.
5. Robustness of attention weights against vanishing topological features.
6. Equivalence between the JAX and PyTorch implementations up to floating point
   noise.

Each proof is constructive and cites the intermediate lemmas used in the code.

## Citation

If you use T-FAN in your work, please cite the repository once it is mirrored to
GitHub:

```
@software{tfan2024,
  author  = {Quanta Research Collective},
  title   = {Topological Field-Adaptive Network},
  year    = {2024},
  url     = {https://github.com/quanta-labs/quanta-tfan}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
