#!/usr/bin/env bash
set -euo pipefail
python quanta_vs_lora_sst2_real.py --epochs_a 1 --epochs_b 1 --batch 32 --model bert-base-uncased
