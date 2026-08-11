# Benchmark Suite Overview

The Phase III launch kit ships with deterministic benchmark implementations to
facilitate integration testing without external dependencies. Each benchmark
produces reproducible metrics which makes the suite suitable for CI pipelines
and documentation examples.

## GLUE baseline

`benchmark_suite.glue_baseline.GLUEBaseline` evaluates synthetic variants of
classic GLUE tasks. The class accepts an optional sequence of
`GlueTaskSpec` entries to customise dataset sizes or baseline scores. The
resulting report contains accuracy and F1 metrics for each task in addition to
metadata describing the configured workloads.

### Customising GLUE

```python
from benchmark_suite.glue_baseline import GLUEBaseline, GlueTaskSpec

tasks = [
    GlueTaskSpec("cola", examples=8551, baseline_accuracy=0.812, baseline_f1=0.794),
]
report = GLUEBaseline(tasks=tasks).run()
print(report.to_dict())
```

## CIFAR adapter

`benchmark_suite.cifar_adapter.CIFARAdapter` simulates deployment scenarios
spanning edge to datacenter environments. Each scenario defines a batch size,
latency budget, and reference accuracy. The adapter yields throughput, accuracy,
and energy metrics derived from deterministic formulae.

### Customising CIFAR

```python
from benchmark_suite.cifar_adapter import CIFARAdapter, DeploymentScenario

scenarios = [
    DeploymentScenario(
        name="edge_debug",
        batch_size=4,
        latency_budget_ms=35.0,
        reference_accuracy=0.78,
    ),
]
report = CIFARAdapter(scenarios=scenarios).run()
```

## Metric registry

The runner aggregates benchmark outputs using
`benchmark_suite.metrics.registry.MetricRegistry`. The registry retains raw
reports for traceability and exposes a `summary()` helper for quick comparisons.
