#!/usr/bin/env bash
set -euo pipefail

python src/training/train_lvit_t_patch.py --config configs/train_lvit_t_patch384.yaml --auto-resume
