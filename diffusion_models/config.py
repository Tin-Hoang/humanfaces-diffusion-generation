"""Configuration for training diffusion models."""

import os
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
from typing import Optional, List
import torch


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@dataclass
class TrainingConfig:
    """Configuration for training diffusion models."""
    
    # Model configuration
    model: str = "unet_notebook"  # Type of model to use (e.g., "unet_notebook")
    
    # Training configuration
    run_name: Optional[str] = None  # Name for the run. To be used for WandB run name and output directory name
    image_size: int = 128  # the generated image resolution
    train_batch_size: int = 16
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 500
    save_image_epochs: int = 5
    save_model_epochs: int = 5
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: Optional[str] = None  # Will be set in parse_args
    dataset_name: str = "celeba_hq_128_2700train"  # Customize the dataset name to note the dataset used
    train_dir: str = "data/celeba_hq_split/train"  # Add train directory
    val_dir: str = None  # Add validation directory
    val_n_samples: int = 100  # Number of samples to generate for FID calculation
    num_train_timesteps: int = 1000  # num_train_timesteps for DDPM scheduler and pipeline inference

    # Conditional generation parameters
    sample_attributes: Optional[torch.Tensor] = None  # Attribute vectors for sample generation
    is_conditional: bool = False  # Whether to use conditional generation
    attribute_file: Optional[str] = None  # Path to the attribute labels file
    num_attributes: int = 40  # Number of attributes (e.g., 40 for CelebA)

    # Grid visualization parameters
    grid_attribute_indices: Optional[List[int]] = None  # Specific attributes for grid visualization
    num_grid_samples: int = 16  # Number of samples in the visualization grid
    grid_sample_attributes: Optional[torch.Tensor] = None  # Attribute vectors for grid visualization

    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 42
    use_wandb: bool = True  # Whether to use WandB logging
    wandb_project: Optional[str] = "EEEM068_Diffusion_Models"
    wandb_entity: Optional[str] = "tin-hoang"

    # Training parameters
    batch_size: int = 128
    num_workers: int = 4
    
    # Model parameters
    channels: int = 3
    time_steps: int = 1000
    
    def __post_init__(self):
        """Set default output_dir if not provided."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Set run_name if not provided
        if not self.run_name:
            self.run_name = f"{self.dataset_name}"
            print(f"No run_name provided, using dataset name: {self.run_name}")
        # Always add timestamp to run_name
        self.run_name += f"_{timestamp}"
    
        # Set output_dir if not provided
        if not self.output_dir:
            self.output_dir = f"checkpoints/{self.run_name}"
            print(f"No output_dir provided, using default: {self.output_dir}")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # train_dir could be from Hugging Face or local directory
        if not self.train_dir or not os.path.exists(self.train_dir):
            raise FileNotFoundError(f"Training directory not found: {self.train_dir}")

        if not self.val_dir or not os.path.exists(self.val_dir):
            # Warn the user that the validation directory does not exist
            print(f"Warning: Validation directory not inputted or not found: {self.val_dir}")
            
        # Set up conditional generation parameters
        if self.is_conditional:
            if not self.attribute_file or not os.path.exists(self.attribute_file):
                raise FileNotFoundError(f"Attribute file not found: {self.attribute_file}")
            if self.grid_attribute_indices is None:
                print("No grid_attribute_indices provided, using default: [17, 20, 24, 35, 39] for Gray_Hair, Male, No_Beard, Wearing_Hat, and Young")
                self.grid_attribute_indices = [17, 20, 24, 35, 39]


def parse_args() -> TrainingConfig:
    """Parse command line arguments and return a TrainingConfig instance."""
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    
    # Get default values from TrainingConfig
    defaults = asdict(TrainingConfig())
    defaults["output_dir"] = None
    defaults["run_name"] = None

    # Add arguments for each config field
    parser.add_argument("--model", type=str, default=defaults["model"],
                      help="Model to use")
    parser.add_argument("--run-name", type=str, default=defaults["run_name"],
                      help="Name for the run. To be used for WandB run name and output directory name")
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
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"],
                      help="Weight decay for optimizer")
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
    
    # Add conditional generation arguments
    parser.add_argument("--is-conditional", type=str2bool, default=defaults["is_conditional"],
                      help="Whether to use conditional generation")
    parser.add_argument("--attribute-file", type=str, default=defaults["attribute_file"],
                      help="Path to the attribute labels file")
    parser.add_argument("--num-attributes", type=int, default=defaults["num_attributes"],
                      help="Number of attributes in the dataset (e.g., 40 for CelebA)")

    # Add grid visualization parameters
    parser.add_argument("--grid-attribute-indices", type=int, nargs="+", default=defaults["grid_attribute_indices"],
                      help="List of attribute indices for grid visualization (e.g., --grid-attribute-indices 0 5 10)")
    parser.add_argument("--num-grid-samples", type=int, default=defaults["num_grid_samples"],
                      help="Number of samples in the visualization grid")

    parser.add_argument("--overwrite-output-dir", action="store_true",
                      help="Overwrite output directory if it exists")
    parser.add_argument("--seed", type=int, default=defaults["seed"],
                      help="Random seed")
    parser.add_argument("--use-wandb", type=str2bool, default=defaults["use_wandb"],
                      help="Use Wandb to track experiments")
    parser.add_argument("--wandb-project", type=str, default=defaults["wandb_project"],
                      help="Name of the WandB project")
    parser.add_argument("--wandb-entity", type=str, default=defaults["wandb_entity"],
                      help="Name of the WandB entity")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert args to dict and handle special cases
    config_dict = vars(args)
    
    # Create and return TrainingConfig instance
    return TrainingConfig(**config_dict)
