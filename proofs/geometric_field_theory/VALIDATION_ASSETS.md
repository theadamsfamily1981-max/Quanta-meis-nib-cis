# Validation Assets Placeholder

The user-provided validation assets referenced in `/mnt/user-data/outputs/tfan_complete/` were not present in the working environment.

To complete the validation pipeline, populate the following directories with the expected code and data:

- `tfan_jax/`
- `tfan_torch/`
- `tfan_libtorch/`

Each directory should include the corresponding `validate.py` (or binary for LibTorch) scripts that emit a `results_summary.json` file at the project root.
