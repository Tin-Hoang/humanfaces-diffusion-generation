#!/bin/bash

# Example commands for generating images using trained diffusion models

# Basic DDPM generation with default steps (1000)
# python scripts/generate.py \
#     --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
#     --pipeline ddpm \
#     --num-images 16 \
#     --output-dir "outputs/samples/ddpm" \
#     --batch-size 4 \
#     --seed 42

# # DDIM generation with fewer steps (100)
python scripts/generate.py \
    --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
    --pipeline ddim \
    --num-inference-steps 100 \
    --num-images 300 \
    --output-dir "outputs/samples/ddim_fast" \
    --batch-size 8 \
    --seed 42

# # DDPM with more steps (2000)
# python scripts/generate.py \
#     --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
#     --pipeline ddpm \
#     --num-inference-steps 2000 \
#     --num-images 100 \
#     --output-dir "outputs/samples/ddpm_high_quality" \
#     --batch-size 4 \
#     --device cuda \
#     --seed 456

# # Fast CPU generation with few steps (50)
# python scripts/generate.py \
#     --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
#     --pipeline ddim \
#     --num-inference-steps 50 \
#     --num-images 4 \
#     --output-dir "outputs/samples/cpu_fast" \
#     --batch-size 1 \
#     --device cpu \
#     --seed 789

# # High quality DDIM generation (500 steps)
# python scripts/generate.py \
#     --checkpoint "outputs/checkpoint-20000" \
#     --pipeline ddim \
#     --num-inference-steps 500 \
#     --num-images 50 \
#     --output-dir "outputs/samples/ddim_high_quality" \
#     --batch-size 5 \
#     --seed 42 