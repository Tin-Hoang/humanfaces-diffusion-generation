"""Image generation utilities for diffusion models."""

from pathlib import Path
from typing import List, Union
from diffusion_models.datasets.attribute_dataset import AttributeDataset
import torch
from PIL import Image
from tqdm import tqdm
import os
from diffusers import DDPMPipeline
from diffusers.models.transformers import DiTTransformer2DModel
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

    # Check if the model is a DiT model
    if isinstance(pipeline.unet, DiTTransformer2DModel):
        # Wrap the UNet forward method to handle timestep and class labels for DiT
        original_forward = pipeline.unet.forward

        def wrapped_forward(sample, timestep, **kwargs):
            # Ensure timestep is a 1D tensor: if it's a scalar, expand it for the batch.
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).repeat(sample.shape[0])
            timestep = timestep.to(sample.device)
            # Inject dummy class labels if they aren’t provided
            if 'class_labels' not in kwargs:
                dummy_class_labels = torch.zeros(sample.shape[0], dtype=torch.long, device=sample.device)
                kwargs['class_labels'] = dummy_class_labels
            return original_forward(sample, timestep, **kwargs)

        # Replace the forward method in the pipeline’s UNet with our wrapped version.
        pipeline.unet.forward = wrapped_forward

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
    # Set the random seed for reproducibility
    generator = torch.manual_seed(config.seed)

    # Check if the model is a DiT model
    if isinstance(pipeline.unet, DiTTransformer2DModel):
        # Wrap the UNet forward method to handle timestep and class labels for DiT
        original_forward = pipeline.unet.forward

        def wrapped_forward(sample, timestep, **kwargs):
            # Ensure timestep is a 1D tensor: if it's a scalar, expand it for the batch.
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).repeat(sample.shape[0])
            timestep = timestep.to(sample.device)
            # Inject dummy class labels if they aren’t provided
            if 'class_labels' not in kwargs:
                dummy_class_labels = torch.zeros(sample.shape[0], dtype=torch.long, device=sample.device)
                kwargs['class_labels'] = dummy_class_labels
            return original_forward(sample, timestep, **kwargs)

        # Replace the forward method in the pipeline’s UNet with our wrapped version.
        pipeline.unet.forward = wrapped_forward

    # Generate images using the pipeline
    output = pipeline(
        batch_size=config.eval_batch_size,
        generator=generator,
        num_inference_steps=config.num_train_timesteps
    )

    # Retrieve generated images from the output
    if hasattr(output, "images"):
        images = output.images
    elif isinstance(output, tuple):
        images = output[0]
    else:
        images = output

    # Create an image grid from the generated images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the grid image to disk
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


def generate_images_from_attributes(
    pipeline: AttributeDiffusionPipeline,
    dataset: AttributeDataset,
    output_dir: Path,
    batch_size: int = 4,
    device: str = "cuda",
    seed: int = 42,
    num_inference_steps: int = 1000
):
    """Generate images from attributes and save them with the same ID as input images.

    Args:
        pipeline: The attribute diffusion pipeline
        dataset: The dataset containing images and their attributes
        output_dir: Directory to save generated images
        batch_size: Batch size for generation
        device: Device to use for generation
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Move pipeline to device
    pipeline = pipeline.to(device)

    # Setup dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Get image_ids from dataset
    image_ids = dataset.attributes_df['image_id'].tolist()

    # Generate images batch by batch
    generated_count = 0

    with torch.no_grad():
        for batch_idx, (_, attributes) in enumerate(tqdm(dataloader, desc="Generating images")):
            # Get current batch image_ids
            batch_image_ids = image_ids[batch_idx*batch_size:batch_idx*batch_size + len(attributes)]

            # Move attributes to device
            attributes = attributes.to(device)

            # Set seed for reproducibility for this batch
            batch_seed = seed + batch_idx
            generator = torch.Generator(device=device).manual_seed(batch_seed)

            # Generate images based on attributes
            output = pipeline(
                attributes=attributes,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil"
            )

            generated_images = output["sample"]

            # Save images with the same ID as input
            for img, img_id in zip(generated_images, batch_image_ids):
                # Generate output filename using the same ID
                output_filename = os.path.splitext(img_id)[0] + ".png"
                img.save(output_dir / output_filename)
                generated_count += 1

    print(f"Generated {generated_count} images in {output_dir}")
