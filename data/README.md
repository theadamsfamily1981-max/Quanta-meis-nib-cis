# 2025-11-09 Transverse Fan Array Dataset

This directory contains the empirical measurements referenced in Section D of the LaTeX supplement for the Quanta Meis Nib Cis study. The measurements originate from the Transverse Fan Array North Lab sweep performed on **2025-11-09** under protocol version `1.2`.

## Files

- `t_fan_results_2025-11-09.json` — canonical export of the sweep, including ambient conditions, stability indices, spectral noise estimates, and phase shift observations for each device under test.

## Usage

The parser utilities in `quanta/parser.py` provide a strongly-typed interface for loading this dataset and extracting derived metrics. See `analysis/tfan_section_d_analysis.py` for an example that reproduces the summary statistics quoted in Section D.

## License & Attribution

Please retain the metadata header when sharing the dataset externally. Cite the LaTeX supplement Section D along with the dataset timestamp when referencing downstream analyses.
