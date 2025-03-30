#!/bin/bash

# Basic DDPM generation (slower)
python scripts/generate.py \
    --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
    --pipeline ddpm \
    --num-inference-steps 1000 \
    --num-images 300 \
    --output-dir "outputs/samples/ddpm1000_from_27000train" \
    --batch-size 16 \
    --seed 42

# DDIM generation (faster)
# python scripts/generate.py \
#     --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
#     --pipeline ddim \
#     --num-inference-steps 100 \
#     --num-images 300 \
#     --output-dir "outputs/samples/ddim_fast_from_27000train" \
#     --batch-size 16 \
#     --seed 42
