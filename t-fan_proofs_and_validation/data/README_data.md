# T-FAN Empirical Dataset

This directory contains the canonical empirical results collected during the TF-2025A wind-tunnel campaign. The files are organized to make exploratory analysis and reproducibility straightforward.

## Files

- `t_fan_results_2025-11-09.json` — Full dataset recorded on 2025-11-09. Contains metadata about the campaign and a list of experiment records.
- `parse_results.py` — Lightweight loader and plotting utility for generating quick summaries of the dataset.

## JSON Schema Overview

Each experiment entry in `t_fan_results_2025-11-09.json` follows this schema:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique experiment identifier from the TF-2025A campaign. |
| `phase` | integer | Development phase (1: baseline validation, 2: rotor optimization, 3: integrated instrumentation). |
| `configuration` | object | Mechanical setup parameters, including rotor diameter (`rotor_diameter_m`), blade pitch (`blade_pitch_deg`), and instrumented sensors. |
| `operating_conditions` | object | Environmental and inlet conditions measured during the run. |
| `performance` | object | Aggregated aerodynamic performance metrics (efficiency, torque, power). |
| `stability_metrics` | object | Stability indicators derived from vibration and pressure measurements. |
| `notes` | string | Free-form observations from the test engineers. |

## Usage

```bash
python parse_results.py --dataset t_fan_results_2025-11-09.json --metric efficiency_pct
```

This command will print summary statistics and generate a quick-look plot (saved to `figures/efficiency_pct.png` by default).

## Reproducibility Notes

- Units are explicitly encoded in the keys to avoid ambiguity.
- Sensor arrays are ordered by their physical placement along the blade root-to-tip axis.
- Future datasets should append to the `experiments` list with new identifiers; avoid editing historical entries to preserve traceability.
