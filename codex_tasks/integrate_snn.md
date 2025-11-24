# Task: Integrate SNN + multimodal glue

Goal
- Add/organize SNN files into `meis/snn/` and multimodal glue into `meis/mm/`.
- Provide adapters so SNN encoders can plug into NIB/MAM.

Steps
1) Branch: `codex/snn_glue`.
2) Place files:
   - meis/snn/{neurons.py,layers.py,training.py,__init__.py}
   - meis/mm/{adapters.py,fusion.py,__init__.py}
3) Add tests:
   - tests/test_snn_integration.py: tiny SNN layer -> adapter -> fusion; assert pass-through.
4) Run `{{lint_command}}` then `{{test_command}}`; fix trivial issues.
5) Commit: "Add SNN + multimodal adapters with smoke tests".
6) Open PR: "SNN + Multimodal glue (initial)".

Constraints
- Keep CPU‑only path available in tests (no GPU required).
