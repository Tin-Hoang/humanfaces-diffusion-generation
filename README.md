# Human Faces Generation with Diffusion Models - A multi-conditioning approach

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square" alt="Python 3.11+">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg?style=flat-square" alt="PyTorch 2.0+">
  </a>
  <a href="https://github.com/Tin-Hoang/EEEM068-Diffusion-Models/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License: MIT">
  </a>
  <a href="https://huggingface.co/docs/diffusers">
    <img src="https://img.shields.io/badge/Hugging%20Face-Diffusers-yellow.svg?style=flat-square" alt="Hugging Face Diffusers">
  </a>
  <a href="https://wandb.ai/tin-hoang/EEEM068_Diffusion_Models?nw=nwusertinhoang">
    <img src="https://img.shields.io/badge/Weights%20%26%20Biases-Supported-green.svg?style=flat-square" alt="Weights & Biases Supported">
  </a>
    <a href="https://github.com/Tin-Hoang/EEEM068-Diffusion-Models/stargazers">
    <img src="https://img.shields.io/github/stars/Tin-Hoang/EEEM068-Diffusion-Models?style=flat-square" alt="GitHub stars">
  </a>
  <a href="https://github.com/Tin-Hoang/EEEM068-Diffusion-Models/forks">
    <img src="https://img.shields.io/github/forks/Tin-Hoang/EEEM068-Diffusion-Models?style=flat-square" alt="GitHub forks">
  </a>
</p>

## 1. Project Overview

This repository contains an implementation of diffusion models for image generation, specifically focused on human faces.

***Abstract:** We present a benchmark of diffusion models for human face generation on a small-scale CelebAMask-HQ dataset, evaluating both unconditional and conditional pipelines. Our study compares UNet and DiT architectures for unconditional generation and explores LoRA-based fine-tuning of pretrained Stable Diffusion models as a separate experiment. Building on the multi-conditioning approach of Giambi and Lisanti, which uses both attribute vectors and segmentation masks, our main contribution is the integration of an InfoNCE loss for attribute embedding and the adoption of a SegFormer-based segmentation encoder. These enhancements improve the semantic alignment and controllability of attribute-guided synthesis. Our results highlight the effectiveness of contrastive embedding learning and advanced segmentation encoding for controlled face generation in limited data settings.*

- Tracked experiments can be found at: [this Weights & Biases project](https://wandb.ai/tin-hoang/EEEM068_Diffusion_Models?nw=nwusertinhoang)

<p align="center">
  <img src="docs/Attributes_Diffusion_Pipeline.png" alt="Attributes Diffusion Pipeline"/>
</p>

<p align="center">
Overview of our proposed attribute-based diffusion pipeline for human faces generation.
</p>


<p align="center">
  <img src="docs/generated-faces.gif" alt="Generated Human Faces" width="512"/>
</p>
<p align="center">
Sample generated human faces through training epochs.
</p>

<p align="center">
  <img src="docs/Sample_InfoNCE_visualisation.png" alt="Ablation Study on InfoNCE" width="512"/>
</p>
<p align="center">
Ablation Study on InfoNCE: the "Without InfoNCE" images (middle) lack clarity, with blurred features, distorted proportions, and unnatural textures. The "With InfoNCE" images (right) are sharper, with better-defined facial features, more accurate attributes generation, and improved realism, though background integration and minor distortions remain issues in both sets.
</p>


### Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation](#2-installation)
   - [2.1 Setup with UV](#21-setup-with-uv)
3. [Project Structure](#3-project-structure)
4. [Usage](#4-usage)
   - [4.1 Training](#41-training)
     - [4.1.1 Training use YAML config (recommended)](#411-training-use-yaml-config-recommended)
     - [4.1.2 Training use script arguments](#412-training-use-script-arguments)
     - [4.1.3 Training with Accelerate (distributed training)](#413-training-with-accelerate-distributed-training)
     - [4.1.4 Using the Notebook (deprecated)](#414-using-the-notebook-old-deprecated-method)
   - [4.2 Image Generation](#42-image-generation)
     - [4.2.1 Generation using the Gradio UI](#421-generation-using-the-gradio-ui)
     - [4.2.2 Generation using the command line script](#422-generation-using-the-command-line-script)
     - [4.2.3 Attribute-Based Generation](#423-attribute-based-generation)
   - [4.3 Evaluation](#43-evaluation)
   - [4.4 LoRA-Based Stable Diffusion Training](#44-lora-based-stable-diffusion-training)
5. [License](#5-license)
6. [Contributors](#6-contributors)

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
├── stable_diffusion_lora/     # Stable Diffusion with LoRA integration
│   ├── v1/                     # Inference scripts
│   │   ├── attribute_stable_diffusion_lora.py
│   │   └── unconditional_stable_diffusion_lora.py
│   ├── v2/                     # Training scripts
│   │   ├── train_lora_conditional.py
│   │   └── train_lora_unconditional.py
│   ├── generate_random_conditional_images.py
│   └── generate_random_unconditional_images.py
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
├── train.sh                    # Base training script
├── train_conditional_diffusion.sh # Conditional training script
├── generate.sh                 # Base generation script
├── generate_conditional.sh    # Conditional generation script
├── evaluate.sh                # Evaluation pipeline
├── pyproject.toml
├── uv.lock
├── LICENSE
├── .gitignore
├── ui/                         # Web UI (if used)
└── README.md                   # Project documentation

```

## 4. Usage

### 4.1 Training

#### 4.1.1 Training use YAML config (recommended)

It is recommended to config your experiment with the YAML config file to train the model.
The sample configs for unconditional and conditional training are located at folder `configs/unconditional` and `configs/conditional` respectively.

```bash
python scripts/train.py --config <path_to_config_file>
```

#### 4.1.2 Training use script arguments

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
- `--model`: Type of model to use (e.g., "unet_notebook", "latent_conditional_unet")
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
- **Conditional diffusion parameters**
  - `--is-conditional`: Whether to use conditional generation
  - `--attribute-file`: Path to the attribute labels file
  - `--num-attributes`: Number of attributes in the dataset
  - `--grid-attribute-indices`: List of attribute indices for grid visualization. Check the [All Attributes for **CelebAMask-HQ** Dataset](#all-attributes-for-celebamask-hq-dataset) for the index of each attribute.
  - `--grid-num-samples`: Number of samples in the visualization grid
  - `--grid-sample-random-remaining-indices`: Whether to randomly sample remaining indices for grid visualization

The training script will:
1. Save regular checkpoints every `save_model_epochs` epochs
2. Save the best model (based on FID score) whenever the score improves
3. Generate and save sample images every `save_image_epochs` epochs
4. Log training metrics and generated images to WandB if enabled

The best model will be saved in `{output_dir}/best_model/` while regular checkpoints will be saved in `{output_dir}/`.

<details>
<summary>All Attributes for **CelebAMask-HQ** Dataset</summary>

Link to the dataset: [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

<p align="center">

| Attribute Index | Attribute Name |
|-----------------|----------------|
| 0 | 5_o_Clock_Shadow |
| 1 | Arched_Eyebrows |
| 2 | Attractive |
| 3 | Bags_Under_Eyes |
| 4 | Bald |
| 5 | Bangs |
| 6 | Big_Lips |
| 7 | Big_Nose |
| 8 | Black_Hair |
| 9 | Blond_Hair |
| 10 | Blurry |
| 11 | Brown_Hair |
| 12 | Bushy_Eyebrows |
| 13 | Chubby |
| 14 | Double_Chin |
| 15 | Eyeglasses |
| 16 | Goatee |
| 17 | Gray_Hair |
| 18 | Heavy_Makeup |
| 19 | High_Cheekbones |
| 20 | Male |
| 21 | Mouth_Slightly_Open |
| 22 | Mustache |
| 23 | Narrow_Eyes |
| 24 | No_Beard |
| 25 | Oval_Face |
| 26 | Pale_Skin |
| 27 | Pointy_Nose |
| 28 | Receding_Hairline |
| 29 | Rosy_Cheeks |
| 30 | Sideburns |
| 31 | Smiling |
| 32 | Straight_Hair |
| 33 | Wavy_Hair |
| 34 | Wearing_Earrings |
| 35 | Wearing_Hat |
| 36 | Wearing_Lipstick |
| 37 | Wearing_Necklace |
| 38 | Wearing_Necktie |
| 39 | Young |

</p>
</details>

#### 4.1.3 Training with Accelerate (distributed training)

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

### 4.1.4 Using the Notebook (old deprecated method)

Alternatively, you can use the provided Jupyter notebooks for a more interactive experience:

**Note:** This method is only for the default unconditional training experiments. It may not be updated with latest models and code.

```bash
# Start Jupyter Lab
jupyter lab
```

Then navigate to `notebooks/` directory and open the relevant notebook.

### 4.2 Image Generation

You can generate images in two ways:

#### 4.2.1 Generation using the Gradio UI:
```bash
python ui/app.py

# Or using UV for dependency management
uv run ui/app.py
```

The UI provides two tabs:
- Single Image Generation: Generate one image at a time with custom noise input
- Batch Generation: Generate multiple images with specified parameters

#### 4.2.2 Generation using the command line script:

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
# Basic DDPM generation (slower)
python scripts/generate.py \
    --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
    --pipeline ddpm \
    --num-inference-steps 1000 \
    --num-images 300 \
    --output-dir "outputs/samples/ddpm1000_from_27000train" \
    --batch-size 16 \
    --seed 42

# DDIM generation (faster)
python scripts/generate.py \
    --checkpoint "checkpoints/ddpm-celebahq-128-27000train-20250316_141247" \
    --pipeline ddim \
    --num-inference-steps 100 \
    --num-images 300 \
    --output-dir "outputs/samples/ddim_fast_from_27000train" \
    --batch-size 16 \
    --seed 42
```

#### 4.2.3 Attribute-Based Generation:

For conditional models, you can generate images based on attributes from existing images using the `generate_on_attributes.py` script. This is useful for creating controlled variations of faces while preserving attribute characteristics.

Input arguments:
- `--checkpoint`: Path to the conditional model checkpoint directory
- `--input-dir`: Directory containing input images with attributes to use
- `--attribute-file`: Path to the attribute label file
- `--pipeline`: Pipeline type (`ddpm` or `ddim`)
- `--num-inference-steps`: Number of denoising steps
- `--output-dir`: Directory to save generated images
- `--batch-size`: Number of images to generate in parallel
- `--device`: Device to use (`cuda` or `cpu`)
- `--seed`: Random seed for reproducibility
- `--image-size`: Size of generated images (default: 256)

```bash
# Generate images based on attributes from test images
python scripts/generate_on_attributes.py \
    --checkpoint "checkpoints/conditional_model" \
    --input-dir "data/CelebA-HQ-split/test_300" \
    --attribute-file "data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt" \
    --pipeline ddim \
    --num-inference-steps 100 \
    --output-dir "outputs/samples/attribute_generated" \
    --batch-size 4 \
    --seed 42

# Or use the convenience script
./generate_conditional.sh
```

The script will:
1. Load the input images and their attributes
2. Generate new images conditioned on those attributes
3. Save the output images with the same IDs as the input files

This allows for direct comparison between input and generated images that share the same facial attributes.


## 4.3 Evaluation

To quantitatively evaluate the performance of your trained diffusion models (e.g., using FID score), follow these steps:

### Step 1: Generate Images

- For **unconditional models**, customize and run the `generate.sh` script to generate images from your trained model.
- For **attribute-conditional models**, customize and run the `generate_conditional.sh` script to generate images conditioned on attributes.

Example (unconditional):
```bash
./generate.sh
```

Example (conditional):
```bash
./generate_conditional.sh
```

These scripts will save generated images to the specified output directory.

### Step 2: Evaluate Generated Images

After generating images, run the `evaluate.sh` script to compute the FID score between the generated images and the reference dataset.

Customize the script as needed, then run:
```bash
./evaluate.sh
```

The script will output the final FID score, which can be used to compare model performance.

> **Note:** Ensure that the paths and parameters in the scripts match your experiment setup and directory structure.


### 4.4 LoRA-Based Stable Diffusion Training

We provide support for fine-tuning **Stable Diffusion v2** using **LoRA (Low-Rank Adaptation)** for both **conditional** and **unconditional** facial image generation.

LoRA enables efficient training by injecting low-rank adapters into transformer layers, significantly reducing the number of trainable parameters.


#### 🏋️ Fine-Tuning the Model

Run the following scripts to fine-tune the LoRA adapters:

```
# 🔧 Conditional Training (with attribute prompts)
python3 stable_diffusion_lora/v2/train_lora_conditional.py

# 🔧 Unconditional Training
python3 stable_diffusion_lora/v2/train_lora_unconditional.py
```

#### 🖼️ Generating Images After Training

Once training is complete, generate images using the LoRA-tuned models:
```
# ✨ Conditional Sampling
python3 stable_diffusion_lora/generate_random_conditional_images.py

# 🌌 Unconditional Sampling
python3 stable_diffusion_lora/generate_random_unconditional_images.py
```
Output images will be saved in the outputs/samples/ directory by default.
You can change save paths, number of samples, and prompts by editing the script files.

## 5. License

This project is part of the Applied Machine Learning coursework at the University of Surrey.
We publish this code under the MIT license for educational purposes.

## 6. Contributors

Project members:
- [Enggen Sherpa​](https://github.com/enggen)
- [Dhruvraj Singh Rawat​](https://github.com/dhruvraj-singh-rawat)
- [Rishikesan Kirupanantha​](https://github.com/rishikesan19)
- [Tin Hoang​](https://github.com/Tin-Hoang)
