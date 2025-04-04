#!/bin/bash

## Option 1 | Using fidelity CLI directly
# --input1: Path to the real dataset
# --input2: Path to the generated samples
fidelity --gpu 0 --fid \
    --input1 data/celeba_hq_split/test \
    --input2 outputs/samples/ddpm2000_1000epoch_2700train_o7h0decl


## Option 2 | Using our own pytorch-fid implementation
# uv pip install pytorch-fid
## Run the following command to evaluate the FID score
# python -m pytorch_fid \
#     data/celeba_hq_split/test \
#     outputs/samples/ddpm2000_1000epoch_2700train_o7h0decl


# DEPRECATED | Using our own evaluation script
# python scripts/evaluation.py \
#     --real-dir "data/celeba_hq_split/test" \
#     --fake-dir "outputs/samples/ddpm1000_from_27000train" \
#     --batch-size 16 \
#     --image-size 128
