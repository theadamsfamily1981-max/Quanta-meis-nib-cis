# Quanta / MEIS / NIB – Architecture Overview (Skeleton)

> Purpose: concise map of core modules (MEIS orchestrator, NIB core, MAM, SNN, MM adapters), dataflow, and test/metrics surface.

## 1) System at a glance
- **MEIS Orchestrator**: process graph + event bus; schedules modules, tracks run-state, emits telemetry.
- **NIB Core** (`meis/nib/core.py`): identity + information bottleneck primitives; EVNI gating; sleep-consolidation SVD hooks.
- **MAM** (`meis/mam/*`): Mycelial‑Adaptive Multimodal routing; sparse, asynchronous pathways; load‑adapted fusion.
- **SNN** (`meis/snn/*`): spiking neuron/layer primitives + training stubs; pluggable as encoders.
- **MM Adapters** (`meis/mm/*`): bridges encoders/decoders; late/early fusion interfaces; feature contracts.

## 2) High-level dataflow
```
 Raw Inputs (audio | video | text | sensor)
        │
    [Encoders: SNN/ANN]
        │ features
        ▼
   MM Adapters/Fusion ───► NIB Gating (EVNI)
        │                        │
        │                        ├─► Fast path (LoRA/DoRA style adapters)
        │                        └─► Slow path (consolidate via SVD during sleep)
        ▼
     MEIS Orchestrator ──► Outputs (actions, summaries, embeddings)
        │
        └─► Telemetry & Metrics (ECE, Brier, risk-coverage)
```

## 3) Key interfaces (first pass)
- `meis/nib/core.py`
  - `evni_gate(x, threshold: float) -> x'` (mask/attenuate by novelty/risk)
  - `identity_update(state, obs, η) -> state'`
  - `sleep_consolidate(weights, *, method='svd', k=64) -> weights'`
- `meis/mam/architecture.py`
  - `route(batch, graph_cfg) -> routed_batch`
- `meis/mm/adapters.py`
  - `to_common(feature) -> z`
  - `fuse(z_list, mode='late') -> z*`

## 4) Experiments & metrics
- **Calibration**: ECE, Brier.
- **Selective risk**: risk–coverage curve.
- **Latency**: end‑to‑end (p50/p90).
- **Energy**: rough proxy via FLOPs/ops or device stats (optional).

## 5) Repo conventions
- Branches: `codex/<feature>`; CI: ruff+pytest.
- Tests: prefer fast, mock heavy deps; mark long as `xfail`/`slow`.

## 6) Roadmap / TODO
- [ ] Fill concrete encoder APIs and SNN training loop.
- [ ] Implement real EVNI estimator and thresholds.
- [ ] Wire sleep SVD to persistence layer.
- [ ] Add real multimodal fusion strategies.
