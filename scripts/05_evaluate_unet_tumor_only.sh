#!/bin/bash

python src/training/evaluate_unet.py --config configs/train_unet_tumor_only.yaml --checkpoint experiments/E1a_unet_preprocessed_224_tumor_only/best.pt --split test
