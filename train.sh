#!/bin/bash

# Check diffusion_models/config.py for the parameters

# If run_name is not provided, it will be set to the default name = dataset_name + timestamp
# If output_dir is not provided, it will be set to the default name = "checkpoints/<run_name>"
# If wandb_project is not provided, it will be set to the default name = "EEEM068_Diffusion_Models"
# If wandb_entity is not provided, it will be set to the default name = "tin-hoang"

# Training with parameters
# python scripts/train.py \
#     --model unet_notebook \
#     --run-name "unet2d_128_ddpm_2700train" \
#     --image-size 128 \
#     --train-batch-size 16 \
#     --eval-batch-size 16 \
#     --num-epochs 100 \
#     --gradient-accumulation-steps 1 \
#     --learning-rate 1e-4 \
#     --weight-decay 1e-2 \
#     --lr-warmup-steps 500 \
#     --save-image-epochs 5 \
#     --save-model-epochs 5 \
#     --mixed-precision fp16 \
#     --dataset-name "celeba_hq_128_2700train" \
#     --train-dir "data/celeba_hq_split/train" \
#     --val-dir "data/celeba_hq_split/test" \
#     --val-n-samples 100 \
#     --num-train-timesteps 1000 \
#     --overwrite-output-dir \
#     --seed 42 \
#     --use-wandb True \
#     --wandb-project "EEEM068_Diffusion_Models" \
#     --wandb-entity "tin-hoang"
   


   python scripts/train.py \
    --model conditional_unet \
    --run-name "attribute_conditionalunet2d_128_ddpm_2700train" \
    --image-size 128 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --num-epochs 100 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --lr-warmup-steps 500 \
    --save-image-epochs 5 \
    --save-model-epochs 5 \
    --mixed-precision fp16 \
    --dataset-name "celeba_hq_128_2700train" \
    --train-dir "data/celeba_hq_split/train" \
    --val-dir "data/celeba_hq_split/test" \
    --is-conditional True \
    --num-attributes 40 \
    --attribute-file "data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt" \
    --val-n-samples 100 \
    --num-train-timesteps 1000 \
    --overwrite-output-dir \
    --seed 42 \
    --use-wandb False \
    --wandb-project "EEEM068_Diffusion_Models" \
    --wandb-entity "tin-hoang"