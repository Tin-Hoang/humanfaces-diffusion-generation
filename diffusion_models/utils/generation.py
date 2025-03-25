"""Image generation utilities for diffusion models."""

from pathlib import Path
from typing import List, Union
import torch
from PIL import Image
from tqdm import tqdm
import os
from diffusers import DDPMPipeline, DiffusionPipeline

from diffusion_models.config import TrainingConfig


def generate_images(
    pipeline,
    batch_size: int = 4,
    device: str = "cuda",
    seed: int = 42,
    initial_noise: torch.Tensor = None,
    num_inference_steps: int = 1000,
) -> List[Image.Image]:
    """Generate a single batch of images using the pipeline.
    
    Args:
        pipeline: The diffusion pipeline
        batch_size: Batch size for generation
        device: Device to use for generation
        seed: Random seed for reproducibility
        initial_noise: Optional initial noise tensor to use for generation
        num_inference_steps: Number of denoising steps
    
    Returns:
        List of generated PIL Images
    """
    # Move pipeline to device
    pipeline = pipeline.to(device)
    
    # Set number of inference steps
    pipeline.scheduler.set_timesteps(num_inference_steps)
    
    # Generate images
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Use provided initial noise if available, otherwise use random noise
    if initial_noise is not None:
        # Use the scheduler's step method directly with the initial noise
        latents = initial_noise
        for t in pipeline.scheduler.timesteps:
            # Get model prediction
            noise_pred = pipeline.unet(latents, t).sample
            # Get previous sample
            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Convert latents to images
        images = (latents / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = [Image.fromarray(image) for image in images]
    else:
        # Use standard pipeline generation
        images = pipeline(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=num_inference_steps
        ).images
    
    return images


def generate_images_to_dir(
    pipeline,
    num_images: int,
    output_dir: Path,
    batch_size: int = 4,
    device: str = "cuda",
    seed: int = 42,
    num_inference_steps: int = 1000,
):
    """Generate multiple batches of images and save them to directory.
    
    Args:
        pipeline: The diffusion pipeline
        num_images: Number of images to generate
        output_dir: Directory to save generated images
        batch_size: Batch size for generation
        device: Device to use for generation
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images in batches
    remaining_images = num_images
    image_idx = 0
    
    while remaining_images > 0:
        curr_batch_size = min(batch_size, remaining_images)
        
        # Generate images
        images = generate_images(
            pipeline=pipeline,
            batch_size=curr_batch_size,
            device=device,
            seed=seed + image_idx,
            num_inference_steps=num_inference_steps,
        )
        
        # Save images
        for img in images:
            img.save(output_dir / f"generated_{image_idx:04d}.png")
            image_idx += 1
        
        remaining_images -= curr_batch_size
        print(f"Generated {image_idx} of {num_images} images")


def make_grid(images, rows, cols):
    """Create a grid of images."""
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def generate_grid_images(config: TrainingConfig, epoch: int, pipeline: DDPMPipeline):
    """Generate and save a grid of sample images.
    
    Args:
        config: Training configuration
        epoch: Current epoch number
        pipeline: DDPM pipeline for generating images
        
    Returns:
        Tuple of (list of generated images, grid image)
    """
    # Generate sample images
    generator = torch.manual_seed(config.seed)
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=generator,
        num_inference_steps=config.num_train_timesteps
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    return images, image_grid


def generate_grid_images_attributes(
    config: TrainingConfig,
    epoch: int,
    pipeline: DiffusionPipeline,
    attributes: torch.Tensor
) -> tuple:
    """Generate and save a grid of sample images with attribute conditioning.
    
    Args:
        config: Training configuration
        epoch: Current epoch number
        pipeline: Diffusion pipeline for conditional generation
        attributes: Tensor of shape (num_samples, num_attributes) containing
                   the attribute vectors to condition on
        
    Returns:
        Tuple of (list of generated images, grid image)
    """
    # Generate sample images with attribute conditioning
    generator = torch.manual_seed(config.seed)
    output = pipeline(
        batch_size=attributes.shape[0],  # Use number of attribute vectors
        generator=generator,
        num_inference_steps=config.num_train_timesteps,
        class_labels=attributes,  # Pass attributes as class_labels
        output_type="pil"
    )
    images = output.images

    # Calculate grid dimensions based on number of samples
    num_samples = len(images)
    grid_size = int(num_samples ** 0.5)  # Square grid
    rows = grid_size
    cols = (num_samples + grid_size - 1) // grid_size  # Ceiling division

    # Make a grid out of the images
    image_grid = make_grid(images, rows=rows, cols=cols)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    return images, image_grid 