#!/bin/bash

# Check diffusion_models/config.py for the parameters

# Training with custom parameters
python scripts/train.py \
    --image-size 128 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --lr-warmup-steps 500 \
    --gradient-accumulation-steps 1 \
    --train-dir "data/celeba_hq_256" \
    --output-dir "outputs/ddpm-celebahq-128-2665train" \
    --val-n-samples 100 \
    --save-image-epochs 2 \
    --seed 42