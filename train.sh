#!/bin/bash

# Check diffusion_models/config.py for the parameters

# Training with parameters
python scripts/train.py \
    --model unet_notebook \
    --image-size 128 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --num-epochs 100 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --lr-warmup-steps 500 \
    --save-image-epochs 2 \
    --save-model-epochs 5 \
    --mixed-precision fp16 \
    --output-dir "outputs/ddpm-celebahq-128-2665train" \
    --dataset-name "celeba_hq_128_2665train" \
    --train-dir "data/celeba_hq_256" \
    --val-dir "data/CelebA-HQ-split/test_300" \
    --val-n-samples 100 \
    --num-train-timesteps 1000 \
    --overwrite-output-dir \
    --seed 42 \
    --use-wandb True