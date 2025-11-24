# GLUE Multitask Runner (2025-11-07)

This package bundles a reproducible GLUE multitask setup that targets light-weight
continuous-integration execution while remaining faithful to the original
SST-2 + MRPC evaluation described in chat.

## Quickstart

```bash
cd experiments/2025-11-07_glue_multitask
python glue_multitask_runner.py \
  --model bert-base-uncased \
  --tasks sst2,mrpc \
  --epochs 1 \
  --batch 16 \
  --ci \
  --device cpu
```

The runner automatically falls back to synthetic text pairs when Hugging Face
artifacts are unavailable, ensuring deterministic behaviour in offline CI
pipelines. Use `--max-samples` to further throttle dataset size when debugging.

## Result Analysis Utilities

After executing the runner, post-process the metrics with the bundled utilities:

```bash
python analyze_results.py results_glue_multitask.json
python compare_improvements.py results_glue_multitask.json
```

These commands emit `results_glue_multitask_analysis.json` and
`results_glue_multitask_improvements.json`, respectively, capturing aggregate
statistics and per-task deltas relative to the baseline task.

## Output Schema

`results_glue_multitask.json` contains the configuration, per-task metrics, and a
UTC timestamp. The `tasks` dictionary records training loss/accuracy, evaluation
loss/accuracy, and dataset cardinalities for each task.

## Ablation Notes

- **Feature representation:** Instead of transformer hidden states, CI mode
  hashes tokens into a fixed-size bag-of-words vector to avoid heavyweight
  dependencies while retaining relative difficulty differences between tasks.
- **Optimizer and schedule:** A learning-rate-scaled multinomial logistic
  regression updates weights with full-batch gradients. Adjust the `--epochs`
  flag to study convergence sensitivity.
- **Data availability:** When GLUE splits cannot be fetched, deterministic
  synthetic corpora emulate sentiment (SST-2) and paraphrase (MRPC) signals so
  the comparison still yields meaningful deltas.
- **Analysis tooling:** The analysis scripts validate that metrics are present
  and quantify improvements, making it easy to plug the runner into dashboards
  or nightly CI ablations without bespoke parsing logic.
