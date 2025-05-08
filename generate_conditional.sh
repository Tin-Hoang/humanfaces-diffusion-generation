#!/bin/bash

# Example script to run conditional generation based on attributes

# Set parameters
CHECKPOINT="/scratch/group_5/diffusion_checkpoints/attribute_lcunetnbvqvae_celebamask_2700train_infonce_20250417_155653"  # Path to the conditional model checkpoint
INPUT_DIR="data/CelebA-HQ-split/test_300"  # Directory with test images
ATTRIBUTE_FILE="data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt"  # Path to attribute annotation file
OUTPUT_DIR="outputs/attribute_lcunetnbvqvae_celebamask_2700train_infonce_20250417_155653_test_300_generated"  # Directory to save generated images
PIPELINE="ddpm"
STEPS=2000
BATCH_SIZE=128
SEED=42
IMAGE_SIZE=256

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the generation script
python scripts/generate_on_attributes.py \
  --checkpoint ${CHECKPOINT} \
  --input-dir ${INPUT_DIR} \
  --attribute-file ${ATTRIBUTE_FILE} \
  --pipeline ${PIPELINE} \
  --num-inference-steps ${STEPS} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --image-size ${IMAGE_SIZE}

echo "Conditional generation complete. Output saved to ${OUTPUT_DIR}"
