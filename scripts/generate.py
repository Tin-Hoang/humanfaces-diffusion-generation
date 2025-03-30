"""Script to generate images from a trained diffusion model."""

import argparse
from pathlib import Path
import torch

from diffusion_models.pipelines.unconditional_pipeline import load_pipeline
from diffusion_models.utils.generation import generate_images_to_dir


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate images from trained diffusion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["ddpm", "ddim"],
        default="ddpm",
        help="Pipeline type to use for generation"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=1000,
        help="Number of denoising steps (default: 1000)"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=16,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load pipeline
    print(f"Loading {args.pipeline} pipeline from {args.checkpoint}")
    pipeline = load_pipeline(
        args.checkpoint,
        args.pipeline
    )
    
    # Generate images
    print(f"Generating {args.num_images} images...")
    generate_images_to_dir(
        pipeline=pipeline,
        num_images=args.num_images,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps
    )
    
    print("Generation complete!")


if __name__ == "__main__":
    main()
