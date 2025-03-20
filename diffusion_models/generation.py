"""Image generation utilities for diffusion models."""

from pathlib import Path
from typing import List
import torch
from PIL import Image
from tqdm import tqdm


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
        latents = 1 / 0.18215 * latents
        images = pipeline.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = [Image.fromarray(image) for image in images]
    else:
        # Use standard pipeline generation
        images = pipeline(
            batch_size=batch_size,
            generator=generator,
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