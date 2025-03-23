# EEEM068-HumanFaces-Diffusion
EEEM068 - Coursework - Group 5 

## Project Overview

This repository contains an implementation of diffusion models for image generation, specifically focused on human faces. The project is structured as a Python application with a clear, modular organization.

## Installation

### Setup with UV

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

## Project Structure

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

## Usage

### Training a Diffusion Model (use script)

Run the training script with the following command:
```bash
python scripts/train.py \
    --train-dir "data/CelebA-HQ-split/train_2700" \
    --val-dir "data/CelebA-HQ-split/test_300" \
    --output-dir "outputs/checkpoints/ddpm-celebahq-128" \
    --image-size 128 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --num-epochs 100 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-4 \
    --mixed-precision "fp16"
```

Key training arguments:
- `--train-dir`: Directory containing training images
- `--val-dir`: Directory containing validation images (if not provided, validation will be ignored during training)
- `--output-dir`: Directory to save model checkpoints and samples
- `--image-size`: Target image resolution (default: 128)
- `--train-batch-size`: Training batch size (default: 16)
- `--eval-batch-size`: Evaluation batch size (default: 16)
- `--num-epochs`: Number of training epochs (default: 100)
- `--gradient-accumulation-steps`: Number of steps for gradient accumulation (default: 1)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--lr-warmup-steps`: Number of warmup steps for learning rate scheduler (default: 500)
- `--mixed-precision`: Mixed precision training mode ("no" for float32, "fp16" for automatic mixed precision)
- `--use-wandb`: Enable Weights & Biases logging (default: True)
- `--seed`: Random seed for reproducibility (default: 42)

The training script will:
1. Save model checkpoints periodically
2. Generate sample images during training
3. Calculate FID scores if validation data is provided
4. Log metrics to Weights & Biases if enabled

### Training a Diffusion Model with Accelerate (distributed training)

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

### Image Generation

You can generate images in two ways:

#### 1. Generation using the Gradio UI:
```bash
python ui/app.py

# Or using UV for dependency management
uv run ui/app.py
```

The UI provides two tabs:
- Single Image Generation: Generate one image at a time with custom noise input
- Batch Generation: Generate multiple images with specified parameters

#### 2. Generation using the command line script:

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

### Using the Notebook

Alternatively, you can use the provided Jupyter notebooks for a more interactive experience:

```bash
# Start Jupyter Lab
jupyter lab
```

Then navigate to `notebooks/` directory and open the relevant notebook.

## License

This project is part of the EEEM068 Diffusion Models coursework at the University of Surrey.
