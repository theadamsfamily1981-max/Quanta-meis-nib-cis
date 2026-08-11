#!/usr/bin/env bash
set -euo pipefail

# Placeholder script for reproducing validation runs across both frameworks.
# Update the commands below with the actual execution steps when they are available.

echo "Running JAX validation pipeline..."
python3 jax_impl/tfan_jax_working.py

echo "Running PyTorch validation pipeline..."
python3 pytorch_impl/tfan_pytorch.py
