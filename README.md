# Quanta-meis-nib-cis

Research for quanta meis nib cis code

## Empirical datasets

The `data/` directory now houses the 2025-11-09 Transverse Fan Array sweep referenced in Section D of the LaTeX supplement. Use the parser utilities in `quanta/parser.py` together with the analysis script in `analysis/tfan_section_d_analysis.py` to reproduce the reported stability and spectral noise summaries.

## Getting started

1. Create a virtual environment and install any local dependencies if needed.
2. Run the Section D analysis summary:
   ```bash
   python analysis/tfan_section_d_analysis.py
   ```
