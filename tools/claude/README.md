# Claude NIB/QUANTA Experiment Suite

This suite programmatically generates a broad set of experiments and asks Claude to flesh each one out into runnable Python, returning a strict JSON payload with a fenced code block.

## Layout
- tools/claude/run_experiments.py (CLI)
- tools/claude/schemas.py (pydantic models)
- tools/claude/prompts.py (system + user templates)
- reports/claude_experiments/ (outputs)

## Install

``bash
pip install pydantic anthropic
``

Optional libs the generated code may use: numpy, pandas, scipy, scikit-learn, torch.

## Usage

Dry-run to just generate specs (no API calls):

``bash
python tools/claude/run_experiments.py --num 40 --dry
``

Run and call Claude (requires env var):

``bash
export ANTHROPIC_API_KEY=***
python tools/claude/run_experiments.py --num 40
``

Outputs go to reports/claude_experiments/. Each JSON contains the code Claude proposes.

## Categories (examples)
- data: mislabels, leakage, imbalance, missingness, duplicate hygiene
- ood: energy/MSP, Mahalanobis, ensembles, corruptions, threshold tuning
- selective: Acc@80, conformal sets, abstention costs, calibration-aware selection
- calibration: T-scaling, isotonic, Dirichlet, reliability diagrams
- optimization: LR/batch/regularization/optimizer sweeps
- cis: PAD->tau monotonicity, tau noise robustness, routing entropy, top-k gating, latency impact
- meta: introspection consistency, identity drift, tail-risk gates

## CI (optional)
A workflow is included to run specs-only if ANTHROPIC_API_KEY is not set, and to execute full generation if provided.
