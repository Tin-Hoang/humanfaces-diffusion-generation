"""Image generation utilities for diffusion models."""
import os
from pathlib import Path
from typing import List, Union, Optional, Dict

import torch
from diffusers import DiTTransformer2DModel
from typing import Optional, Dict, Union
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers import DDPMPipeline

from diffusion_models.datasets.data_utils import get_inference_transform
from diffusion_models.datasets.attribute_dataset import AttributeDataset
from diffusers.models.transformers import DiTTransformer2DModel
from diffusion_models.pipelines.attribute_pipeline import AttributeDiffusionPipeline
from diffusion_models.config import TrainingConfig
from diffusers import DDPMPipeline
from torchvision.utils import make_grid, save_image
from diffusion_models.datasets.attribute_dataset import AttributeDataset


def make_pil_grid(images, rows, cols):
    """Create a grid of PIL images."""
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


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
    image_grid = make_pil_grid(images, rows=4, cols=4)

    # Save the grid image to disk
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    return images, image_grid


def generate_grid_images_attributes(config, epoch, pipeline, attributes, segmentation: Optional[torch.Tensor] = None):
    """
    Generate and save a grid of sample images conditioned on attributes (+ segmentation if applicable).

    Args:
        config: Training configuration.
        epoch: Current epoch number.
        pipeline: The diffusion pipeline (e.g., AttributeDiffusionPipeline).
        attributes (torch.Tensor): Attribute tensor [B, 40].
        segmentation (Optional[torch.Tensor]): Optional segmentation tensor [B, 1 or 3, H, W].

    Returns:
        output_path: Path to saved grid image.
        image_grid: The generated image grid as a tensor.
    """
    pipeline = pipeline.to(config.device)
    attributes = attributes.to(config.device)

    if segmentation is not None:
        segmentation = segmentation.to(config.device)

    print(f"[INFO] Generating sample grid at epoch {epoch} with conditioning_type={config.conditioning_type}")
    device_str = "cuda" if "cuda" in str(config.device) else str(config.device)
    generator = torch.Generator(device=device_str).manual_seed(config.seed)

    with torch.no_grad():
        outputs = pipeline(
            attributes=attributes,
            segmentation=segmentation,
            num_inference_steps=config.num_train_timesteps,
            generator=generator,
            output_type="tensor",
            return_dict=True,
        )
        images = outputs["sample"]  # [B, 3, H, W]

    image_grid = make_grid(images, nrow=int(np.sqrt(images.shape[0])), normalize=True, scale_each=True)
    output_path = os.path.join(config.output_dir, f"grid_epoch_{epoch}.png")
    save_image(image_grid, output_path)

    return output_path, image_grid


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
