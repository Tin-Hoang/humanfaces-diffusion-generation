# EEEM068-HumanFaces-Diffusion
EEEM068 - Coursework - Group 5 

## Project Overview

This repository contains implementation of diffusion models for image generation. The project is structured as a proper Python sourcecode and uses UV for dependency management.

## Installation

### Setup with UV

This project uses UV for package management. If you don't have UV installed, you can install it using:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies (editable mode)
uv venv
uv pip install -e .
```

For development:

```bash
uv pip install -e ".[dev]"
```

For notebook development:

```bash
uv pip install -e ".[notebook]"
```

## Project Structure

```
diffusion-models/
├── src/                    # Source code
│   └── diffusion_models/   # Main package
├── tests/                  # Test directory
├── pyproject.toml          # Project configuration
└── README.md               # Documentation
```

## Usage

Example code for training a diffusion model:

<TBD>

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feat/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feat/amazing-feature`
5. Open a Pull Request
