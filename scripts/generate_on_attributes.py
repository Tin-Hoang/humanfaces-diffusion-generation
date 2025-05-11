"""Script to generate images from a trained conditional diffusion model based on input attributes."""

import argparse
from pathlib import Path
import torch
import os
from typing import Dict, Union
from diffusers import DDIMScheduler, DDPMScheduler
from tqdm import tqdm
from PIL import Image

from diffusion_models.datasets.attribute_dataset import AttributeDataset
from diffusion_models.pipelines.attribute_pipeline import AttributeDiffusionPipeline
from diffusion_models.utils.generation import generate_images_from_attributes


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate images from trained conditional diffusion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--attribute-file",
        type=str,
        required=True,
        help="Path to attribute label file"
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
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Size of generated images"
    )
    
    parser.add_argument(
        "--elastic-transform",
        type=float,
        default=250.0,
        action="store_true",
        help="elastic transform for data augmentation",
    )
    parser.add_argument(
        "--color-jitter",
        action="store_true",
        help="randomly change the brightness, contrast and saturation",
    )
    parser.add_argument(
        "--color-aug",
        action="store_true",
        help="randomly alter the intensities of RGB channels",
    )

    return parser.parse_args()


def load_conditional_pipeline(checkpoint_path: str, pipeline_type: str = "ddpm", image_size: int = 256) -> AttributeDiffusionPipeline:
    """Load the conditional pipeline from a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        pipeline_type: Type of pipeline to use ("ddpm" or "ddim")
        image_size: Size of the images

    Returns:
        AttributeDiffusionPipeline object
    """
    # Use the from_pretrained method to load the entire pipeline
    # This will automatically load all components (unet, vae, scheduler, attribute_embedder)
    pipeline = AttributeDiffusionPipeline.from_pretrained(
        checkpoint_path,
        image_size=image_size
    )

    # Update the scheduler if needed
    if pipeline_type.lower() == "ddpm" and not isinstance(pipeline.scheduler, DDPMScheduler):
        pipeline.scheduler = DDPMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler")
    elif pipeline_type.lower() == "ddim" and not isinstance(pipeline.scheduler, DDIMScheduler):
        pipeline.scheduler = DDIMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler")

    return pipeline


def main():
    args = parse_args()

    # Create dataset from input directory and attribute file
    print(f"Loading dataset from {args.input_dir} with attributes from {args.attribute_file}")
    dataset = AttributeDataset(
        image_dir=args.input_dir,
        attribute_label_path=args.attribute_file,
        image_size=args.image_size
    )

    # Load conditional pipeline
    print(f"Loading {args.pipeline} pipeline from {args.checkpoint}")
    pipeline = load_conditional_pipeline(
        checkpoint_path=args.checkpoint,
        pipeline_type=args.pipeline,
        image_size=args.image_size
    )

    # Generate images
    print(f"Generating images based on attributes for {len(dataset)} input images...")
    generate_images_from_attributes(
        pipeline=pipeline,
        dataset=dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps
    )

    print("Generation complete!")


if __name__ == "__main__":
    main()
