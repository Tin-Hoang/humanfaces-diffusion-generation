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

### Training a Diffusion Model

TBD

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
