# Quanta-meis-nib-cis

Research for quanta meis nib cis code.

## Multimodal Quickstart

1. Install the minimal dependencies:
   ```bash
   pip install torch numpy pytest
   ```
2. Explore the shared projection and attention utilities:
   ```python
   from multimodal import SharedProjection, CrossModalAttention
   ```
3. Run the evaluation helpers to materialise the validation artefacts:
   ```bash
   python scripts/run_unified_embedding_eval.py outputs/unified.json
   python scripts/run_seq_learning_eval.py outputs/seq_learning.json
   python scripts/run_lora_efficiency_eval.py outputs/lora.json
   python scripts/run_landmark_eval.py outputs/landmarks.json
   ```
4. Execute the automated gates:
   ```bash
   pytest
   ```

For required performance thresholds see [VALIDATION.md](VALIDATION.md).
