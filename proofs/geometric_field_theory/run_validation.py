import json
import subprocess
import sys
from pathlib import Path


def run_step(description, command):
    print(description)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"Warning: command {' '.join(command)} exited with code {result.returncode}")
    return result.returncode


def main():
    print('Running JAX, PyTorch, and LibTorch validation...')
    status = 0
    status |= run_step('Running PyTorch validation...', ['python3', 'tfan_torch/validate.py'])
    status |= run_step('Running JAX validation...', ['python3', 'tfan_jax/validate.py'])
    status |= run_step('Running LibTorch validation...', ['./tfan_libtorch/build/tfan_validation'])

    summary_path = Path('results_summary.json')
    if summary_path.exists():
        try:
            with summary_path.open() as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            print(f'Failed to parse {summary_path}: {exc}')
            return status or 1

        print('\n--- Validation Summary ---')
        for key, value in data.items():
            print(f'{key}: {value}')
    else:
        print(f'Missing expected summary file: {summary_path}')

    return status


if __name__ == '__main__':
    sys.exit(main())
