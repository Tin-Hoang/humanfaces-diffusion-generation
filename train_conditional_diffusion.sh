#!/bin/bash

# Check diffusion_models/config.py for the parameters

# If run_name is not provided, it will be set to the default name = dataset_name + timestamp
# If output_dir is not provided, it will be set to the default name = "checkpoints/<run_name>"
# If wandb_project is not provided, it will be set to the default name = "EEEM068_Diffusion_Models"
# If wandb_entity is not provided, it will be set to the default name = "tin-hoang" 

python scripts/train.py \
    --model latent_conditional_unet \
    --run-name "attribute_latentconditionalunet2d_256_ddim_27000train" \
    --image-size 256 \
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
    --scheduler-type ddim \
    --dataset-name "celebamask_hq_256_27000train" \
    --train-dir "data/CelebA-HQ-split/train_27000" \
    --is-conditional True \
    --num-attributes 40 \
    --attribute-file "data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt" \
    --grid-attribute-indices 20 \
    --grid-num-samples 16 \
    --grid-sample-random-remaining-indices True \
    --num-train-timesteps 1000 \
    --overwrite-output-dir \
    --seed 42 \
    --use-wandb True \
    --wandb-project "EEEM068_Diffusion_Models" \
    --wandb-entity "tin-hoang"