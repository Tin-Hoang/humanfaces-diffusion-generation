"""Configuration for training diffusion models."""

import os
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training diffusion models."""
    
    # Model configuration
    model: str = "unet_notebook"  # Type of model to use (e.g., "unet_notebook")
    
    # Training configuration
    image_size: int = 128  # the generated image resolution
    train_batch_size: int = 16
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 5
    save_model_epochs: int = 5
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: Optional[str] = None  # Will be set in parse_args
    dataset_name: str = "celeba_hq_128_2700train"  # Customize the dataset name to note the dataset used
    train_dir: str = "data/CelebA-HQ-split/train_2700"  # Add train directory
    val_dir: str = "data/CelebA-HQ-split/test_300"  # Add validation directory
    val_n_samples: int = 100  # Number of samples to generate for FID calculation
    num_train_timesteps: int = 1000  # num_train_timesteps for DDPM scheduler and pipeline inference

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 42
    use_wandb: bool = True  # Whether to use WandB logging

    def __post_init__(self):
        """Set default output_dir if not provided."""
        if self.output_dir is None:
            self.output_dir = f"checkpoints/ddpm-celebahq-128-27000train-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.train_dir):
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")

        if not os.path.exists(self.val_dir):
            # Warn the user that the validation directory does not exist
            print(f"Warning: Validation directory not found: {self.val_dir}")


def parse_args() -> TrainingConfig:
    """Parse command line arguments and return a TrainingConfig instance."""
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    
    # Get default values from TrainingConfig
    defaults = asdict(TrainingConfig())
    
    # Add arguments for each config field
    parser.add_argument("--image-size", type=int, default=defaults["image_size"],
                      help="The generated image resolution")
    parser.add_argument("--train-batch-size", type=int, default=defaults["train_batch_size"],
                      help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=defaults["eval_batch_size"],
                      help="Evaluation batch size")
    parser.add_argument("--num-epochs", type=int, default=defaults["num_epochs"],
                      help="Number of training epochs")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults["gradient_accumulation_steps"],
                      help="Number of gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"],
                      help="Learning rate")
    parser.add_argument("--lr-warmup-steps", type=int, default=defaults["lr_warmup_steps"],
                      help="Number of learning rate warmup steps")
    parser.add_argument("--save-image-epochs", type=int, default=defaults["save_image_epochs"],
                      help="Save generated images every N epochs")
    parser.add_argument("--save-model-epochs", type=int, default=defaults["save_model_epochs"],
                      help="Save model checkpoint every N epochs")
    parser.add_argument("--mixed-precision", type=str, default=defaults["mixed_precision"],
                      choices=["no", "fp16"], help="Mixed precision training type")
    parser.add_argument("--output-dir", type=str, default=defaults["output_dir"],
                      help="Output directory for model and samples")
    parser.add_argument("--dataset-name", type=str, default=defaults["dataset_name"],
                      help="Name of the dataset")
    parser.add_argument("--train-dir", type=str, default=defaults["train_dir"],
                      help="Training data directory")
    parser.add_argument("--val-dir", type=str, default=defaults["val_dir"],
                      help="Validation data directory")
    parser.add_argument("--val-n-samples", type=int, default=defaults["val_n_samples"],
                      help="Number of samples for FID calculation")
    parser.add_argument("--num-train-timesteps", type=int, default=defaults["num_train_timesteps"],
                      help="Number of training timesteps")
    parser.add_argument("--push-to-hub", action="store_true",
                      help="Push model to Hugging Face Hub")
    parser.add_argument("--hub-private-repo", action="store_true",
                      help="Create private repository on Hugging Face Hub")
    parser.add_argument("--overwrite-output-dir", action="store_true",
                      help="Overwrite output directory if it exists")
    parser.add_argument("--seed", type=int, default=defaults["seed"],
                      help="Random seed")
    parser.add_argument("--use-wandb", type=bool, default=defaults["use_wandb"],
                      help="Use Wandb to track experiments")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert args to dict and handle special cases
    config_dict = vars(args)
    
    # Create and return TrainingConfig instance
    return TrainingConfig(**config_dict)
