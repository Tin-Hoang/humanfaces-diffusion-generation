#!/bin/bash

# Check diffusion_models/config.py for the parameters

# Training with custom parameters
python scripts/train.py \
    --image-size 64 \
    --train-batch-size 2 \
    --eval-batch-size 16 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --lr-warmup-steps 500 \
    --gradient-accumulation-steps 1 \
    --train-dir "data/CelebA-HQ-split/train_2700" \
    --val-dir "data/CelebA-HQ-split/test_300" \
    --output-dir "outputs/ddpm-celebahq-64-2700train" \
    --val-n-samples 100 \
    --save-image-epochs 2 \
    --seed 42