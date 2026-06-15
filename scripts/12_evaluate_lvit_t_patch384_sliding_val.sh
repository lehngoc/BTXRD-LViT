#!/usr/bin/env bash
set -euo pipefail

python src/inference/evaluate_lvit_t_sliding_window.py --config configs/train_lvit_t_patch384.yaml --checkpoint experiments/E4_lvit_t_patch384_text_h4/best.pt --split val
