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
uv venv
uv pip install -e .

# For development
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

You can train a diffusion model using the training script:

```bash
# Basic training
python scripts/train.py --data_path data/processed --output_dir outputs/checkpoints

# With additional parameters
python scripts/train.py --data_path data/processed --batch_size 64 --epochs 200 --lr 1e-4
```

Alternatively, you can use the main entry point:

```bash
python main.py train --data_path data/processed --output_dir outputs/checkpoints
```

### Generating Samples

To generate samples from a trained model:

```bash
# Basic sampling
python scripts/sample.py --checkpoint outputs/checkpoints/model_latest.pt --output_dir outputs/samples

# With additional parameters
python scripts/sample.py --checkpoint outputs/checkpoints/model_latest.pt --num_images 16 --image_size 256
```

Or using the main entry point:

```bash
python main.py sample --checkpoint outputs/checkpoints/model_latest.pt
```

### Using the Notebook

Alternatively, you can use the provided Jupyter notebooks for a more interactive experience:

```bash
# Start Jupyter Lab
jupyter lab
```

Then navigate to `notebooks/` directory and open the relevant notebook.

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request
