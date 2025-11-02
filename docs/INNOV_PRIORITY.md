# INNOV Priority (001, 016, 036)

This doc defines **design** and **acceptance** for the three high-priority items.

---

## INNOV-001 — CLIP-style Cross-Modal Contrastive Compression
**Goal**: Improve cross-modal alignment (text/audio/vision).
**Stub**: CPU-only synthetic embeddings, InfoNCE, retrieval@10, MMI proxy.
**Accept**: Stub returns metrics dict with `infoNCE_*`, `retrieval10_*`, `MMI_stub`.

## INNOV-016 — Causal Gating Signals (SCM)
**Goal**: Gate on causal evidence, not spurious cues.
**Stub**: Synthetic confounding; backdoor-adjusted score; report selective acc & counterfactual stability.
**Accept**: Metrics dict with `selective_acc_*`, `coverage`, `counterfactual_stability`.

## INNOV-036 — Consolidation Quality Metrics
**Goal**: Track pre/post consolidation (SVD cadence).
**Stub**: Toy low-rank projection; report pre/post acc proxy, ECE proxy, stability index.
**Accept**: Metrics dict with `acc_*`, `delta_acc`, `ece_*_proxy`, `stability_index`.
