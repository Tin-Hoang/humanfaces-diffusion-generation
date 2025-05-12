"""Data preprocessing utilities for diffusion models."""

from torchvision import transforms
from typing import Dict, Any
from torchvision.transforms import InterpolationMode
from diffusion_models.config import parse_args, TrainingConfig


def get_preprocess_transform(image_size: int, config: TrainingConfig = None) -> transforms.Compose:
    """Get preprocessing transform for images.

    Args:
        image_size: Target size for the images

    Returns:
        Preprocessing transform
    """
    if config is None:
        config = parse_args()

    base_transform = [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.LANCZOS),  # Resize to target size
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
    ]
    # Additional Data Augmentations
    if config.color_jitter:
        base_transform.append(transforms.ColorJitter())
    if config.elastic_transform:
        base_transform.append(transforms.ElasticTransform(alpha=250.0))

    return transforms.Compose(base_transform + [
        transforms.ToTensor(),  # Convert to tensor in [0, 1] range
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1] range
    ])


def transform(examples: Dict[str, Any], preprocess: transforms.Compose) -> Dict[str, Any]:
    """Apply preprocessing to dataset examples.

    Args:
        examples: Dataset examples
        preprocess: Preprocessing transform

    Returns:
        Processed examples
    """
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


def get_inference_transform(image_size: int) -> transforms.Compose:
    """Get transform for test images / generation process.

    Args:
        image_size: Target size for the images

    Returns:
        Test transform
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform
