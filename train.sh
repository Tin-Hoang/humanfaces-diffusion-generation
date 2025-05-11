#!/bin/bash

# Check diffusion_models/config.py for the parameters

# If run_name is not provided, it will be set to the default name = dataset_name + timestamp
# If output_dir is not provided, it will be set to the default name = "checkpoints/<run_name>"
# If wandb_project is not provided, it will be set to the default name = "EEEM068_Diffusion_Models"
# If wandb_entity is not provided, it will be set to the default name = "tin-hoang"

# Training with parameters
python3 scripts/train.py \
    --model unet_notebook \
    --run-name "unet2d_128_ddpm_2700train/agmen1" \
    --image-size 128 \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --num-epochs 100 \
    --gradient-accumulation-steps 1 \
    --learning-rate 8e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-steps 500 \
    --save-image-epochs 5 \
    --save-model-epochs 5 \
    --mixed-precision fp16 \
    --dataset-name "celeba_hq_128_2700train" \
    --train-dir "data/celeba_hq_split/train" \
    --val-dir "data/celeba_hq_split/test" \
    --val-n-samples 100 \
    --root-output-dir "/scratch/group_5/diffusion_checkpoints" \
    --num-train-timesteps 1000 \
    --seed 42 \
    --use-wandb True \
    --wandb-project "EEEM068_Diffusion_Models" \
    --wandb-entity "tin-hoang"
   
