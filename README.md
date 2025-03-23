# EEEM068-HumanFaces-Diffusion
*EEEM068 - Coursework - Group 5*

## 1. Project Overview

This repository contains an implementation of diffusion models for image generation, specifically focused on human faces. The project is structured as a Python application with a clear, modular organization.

<p align="center">
  <img src="docs/generated-faces.gif" alt="Generated Human Faces" width="512"/>
</p>

### Table of Contents
1. [Installation](#2-installation)
   - [Setup with UV](#21-setup-with-uv)
2. [Project Structure](#3-project-structure)
3. [Usage](#4-usage)
   - [Training](#41-training-use-script)
   - [Training with Accelerate](#42-training-with-accelerate-distributed-training)
   - [Image Generation](#43-image-generation)
     - [Gradio UI](#431-generation-using-the-gradio-ui)
     - [Command Line](#432-generation-using-the-command-line-script)
   - [Using the Notebook](#44-using-the-notebook)
4. [License](#5-license)

## 2. Installation

### 2.1 Setup with UV

Using UV for dependency management:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv --python 3.11
# Activate the virtual environment
source .venv/bin/activate

# Install required dependencies
uv pip install -e .

# For development (linting, formatting, etc.)
uv pip install -e .[dev]

# For notebook - quick interactive session
uv pip install -e .[notebook]
```

## 3. Project Structure

```
EEEM068-Diffusion-Models/
├── diffusion_models/           # Main package directory
│   ├── models/                 # Model definitions
│   ├── utils/                  # Utility functions
│   ├── datasets/               # Dataset handling
│   └── visualization/          # Visualization tools
├── scripts/                    # Executable scripts
├── data/                       # Data directory
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── outputs/                    # Output files
│   ├── checkpoints/            # Model checkpoints
│   └── samples/                # Generated samples
├── tests/                      # Test directory
├── docs/                       # Documentation
├── notebooks/                  # Jupyter notebooks
├── main.py                     # Application entry point
└── README.md                   # Project documentation
```

## 4. Usage

### 4.1 Training (use script)

To train a model, use the `train.py` script:

```bash
python scripts/train.py \
    --model unet_notebook \
    --run-name "my_experiment" \
    --image-size 128 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --weight-decay 1e-2 \
    --lr-warmup-steps 500 \
    --save-image-epochs 5 \
    --save-model-epochs 5 \
    --mixed-precision fp16 \
    --train-dir "data/celeba_hq_256" \
    --val-dir "data/CelebA-HQ-split/test_300" \
    --val-n-samples 100 \
    --num-train-timesteps 1000 \
    --use-wandb True \
    --wandb-project "EEEM068_Diffusion_Models" \
    --wandb-entity "your_username"
```

Key arguments:
- `--model`: Type of model to use (e.g., "unet_notebook")
- `--run-name`: Name for the run (used for WandB run name and output directory)
- `--image-size`: Target image resolution
- `--train-batch-size`: Training batch size
- `--eval-batch-size`: Evaluation batch size
- `--num-epochs`: Number of training epochs
- `--learning-rate`: Learning rate for optimizer
- `--weight-decay`: Weight decay for optimizer
- `--lr-warmup-steps`: Number of learning rate warmup steps
- `--save-image-epochs`: Save generated images every N epochs
- `--save-model-epochs`: Save model checkpoint every N epochs
- `--mixed-precision`: Mixed precision training type ("no" or "fp16")
- `--train-dir`: Training data directory
- `--val-dir`: Validation data directory
- `--val-n-samples`: Number of samples to generate for FID calculation
- `--num-train-timesteps`: Number of timesteps for DDPM scheduler
- `--use-wandb`: Whether to use WandB logging
- `--wandb-project`: Name of the WandB project
- `--wandb-entity`: Name of the WandB entity

The training script will:
1. Save regular checkpoints every `save_model_epochs` epochs
2. Save the best model (based on FID score) whenever the score improves
3. Generate and save sample images every `save_image_epochs` epochs
4. Log training metrics and generated images to WandB if enabled

The best model will be saved in `{output_dir}/best_model/` while regular checkpoints will be saved in `{output_dir}/`.

### 4.2 Training with Accelerate (distributed training)

The training script uses the Hugging Face 🤗 Accelerate library for distributed training. Follow these steps to train a model:

1. First-time setup: Configure accelerate (one-time setup)
```bash
accelerate config
```

2. Launch training:
```bash
# Training with accelerate
accelerate launch scripts/train.py \
    --train-dir "data/CelebA-HQ-split/train_27000" \
    --val-dir "data/CelebA-HQ-split/test_300" \
    --output-dir "outputs/checkpoints/ddpm-celebahq-128" \
    --image-size 128 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --num-epochs 100 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-4 \
    --lr-warmup-steps 500 \
    --mixed-precision "fp16" \
    --use-wandb \
    --seed 42
```

### 4.3 Image Generation

You can generate images in two ways:

#### 4.3.1 Generation using the Gradio UI:
```bash
python ui/app.py

# Or using UV for dependency management
uv run ui/app.py
```

The UI provides two tabs:
- Single Image Generation: Generate one image at a time with custom noise input
- Batch Generation: Generate multiple images with specified parameters

#### 4.3.2 Generation using the command line script:

Input arguments:
- `--checkpoint`: Path to the model checkpoint directory
- `--pipeline`: Pipeline type (`ddpm` or `ddim`)
- `--num-inference-steps`: Number of denoising steps (default: 100 for DDIM, 1000 for DDPM)
- `--num-images`: Number of images to generate
- `--output-dir`: Directory to save generated images
- `--batch-size`: Number of images to generate in parallel
- `--device`: Device to use (`cuda` or `cpu`)
- `--seed`: Random seed for reproducibility

```bash
# Basic DDIM generation (faster)
python scripts/generate.py \
    --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
    --pipeline ddim \
    --num-inference-steps 100 \
    --num-images 32 \
    --output-dir "outputs/samples/ddim_fast" \
    --batch-size 8 \
    --seed 42

# High quality DDPM generation (slower)
python scripts/generate.py \
    --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
    --pipeline ddpm \
    --num-inference-steps 1000 \
    --num-images 100 \
    --output-dir "outputs/samples/ddpm_high_quality" \
    --batch-size 4 \
    --device cuda \
    --seed 42
```

### 4.4 Using the Notebook

Alternatively, you can use the provided Jupyter notebooks for a more interactive experience:

```bash
# Start Jupyter Lab
jupyter lab
```

Then navigate to `notebooks/` directory and open the relevant notebook.

## 5. License

This project is part of the EEEM068 Diffusion Models coursework at the University of Surrey.
