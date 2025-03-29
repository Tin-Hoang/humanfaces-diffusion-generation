#!/bin/bash

# With optional parameters
python scripts/evaluation.py \
    --real-dir "data/celeba_hq_split/test" \
    --fake-dir "outputs/samples/ddpm1000_from_27000train" \
    --batch-size 16 \
    --image-size 128
