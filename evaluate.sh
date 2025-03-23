#!/bin/bash

# With optional parameters
python scripts/evaluation.py \
    --real-dir "data/CelebA-HQ-split/test_300" \
    --fake-dir "outputs/samples/ddim_fast" \
    --batch-size 16 \
    --image-size 128