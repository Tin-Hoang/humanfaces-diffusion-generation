"""Image generation utilities for diffusion models."""

from pathlib import Path
from typing import List, Union, Optional, Dict
import torch
from PIL import Image
from tqdm import tqdm
import os
from diffusers import DDPMPipeline, DiffusionPipeline
from diffusion_models.datasets.data_utils import get_inference_transform
from diffusion_models.pipelines.attribute_pipeline import AttributeDiffusionPipeline

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
    pipeline: AttributeDiffusionPipeline,
    attributes: torch.Tensor
) -> tuple:
    """Generate and save a grid of sample images with attribute conditioning.
    
    Args:
        config: Training configuration
        epoch: Current epoch number
        pipeline: Attribute diffusion pipeline for conditional generation
        attributes: Tensor of shape (num_samples, num_attributes) containing
                   the attribute vectors to condition on
        
    Returns:
        Tuple of (list of generated images, grid image)
    """
    print(f"\nGenerating images for epoch {epoch}")
    print(f"Number of samples: {len(attributes)}")
    print(f"Attributes shape: {attributes.shape}")
    
    # Generate sample images with attribute conditioning
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(config.seed)
    output = pipeline(
        num_inference_steps=config.num_train_timesteps,
        generator=generator,
        attributes=attributes,  # Pass attributes directly
        output_type="pil",
        decode_batch_size=2  # Process 2 image at a time for VAE decoding to save memory
    )
    images = output["sample"]

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


def generate_image2image_with_attributes(
    pipeline: AttributeDiffusionPipeline,
    attributes: torch.Tensor,
    init_images: List[Image.Image],
    num_inference_steps: int = 50,
    strength: float = 0.8,
    generator: Optional[torch.Generator] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    decode_batch_size: int = 2,
    eta: float = 0.0,
) -> Union[Dict[str, Union[List[Image.Image], torch.Tensor]], Union[List[Image.Image], torch.Tensor]]:
    """Generate images conditioned on attributes and initial images.
    
    This function handles the conversion of PIL images to tensors for image-to-image
    generation with attribute conditioning.
    
    Args:
        pipeline: The attribute diffusion pipeline
        attributes: Multi-hot tensor of shape (batch_size, 40)
        init_images: List of PIL images to use as starting point
        num_inference_steps: Number of denoising steps
        strength: How much to transform the init_image (1.0 = completely transform)
        generator: Random number generator for reproducibility
        output_type: "pil" for PIL images, "tensor" for raw tensors
        return_dict: Whether to return a dict with the output
        decode_batch_size: Batch size for VAE decoding to manage memory
        eta: Parameter between 0 and 1, controlling stochasticity (0 = deterministic DDIM)
    
    Returns:
        Dict or List/Tensor: Generated images in the specified format
    """
    # Create transform to resize and normalize images
    transform = get_inference_transform(pipeline.image_size)
    
    # Convert PIL images to tensor
    init_image_tensor = torch.stack([transform(img) for img in init_images])
    
    # Ensure batch size matches
    batch_size = attributes.size(0)
    if init_image_tensor.size(0) != batch_size:
        if init_image_tensor.size(0) == 1:
            # Repeat single image to match batch size
            init_image_tensor = init_image_tensor.repeat(batch_size, 1, 1, 1)
        else:
            raise ValueError(f"Number of init images ({init_image_tensor.size(0)}) does not match batch size ({batch_size})")
    
    # Call the pipeline with the tensor
    return pipeline(
        attributes=attributes,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type=output_type,
        return_dict=return_dict,
        decode_batch_size=decode_batch_size,
        eta=eta,
        init_image=init_image_tensor,
        strength=strength
    ) 