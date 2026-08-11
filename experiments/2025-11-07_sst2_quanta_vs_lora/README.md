# QUANTA vs LoRA on real SST-2 (2025-11-07)

## What this experiment does
LoRA vs QUANTA on real SST-2, continual task switch with token dropout.

## Repro commands
```
pip install -r requirements.txt
bash run.sh
```

## Logged metrics
Accuracy, forgetting, retention, time, memory KB. Logged in `results_quanta_vs_lora_sst2_real.json`.

## Notes
Frozen encoder, LoRA target modules, QUANTA 3-tier head and EMA consolidation.
