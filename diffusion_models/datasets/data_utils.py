"""Data preprocessing utilities for diffusion models."""

from torchvision import transforms
from typing import Dict, Any
from torchvision.transforms import InterpolationMode


def get_preprocess_transform(image_size: int) -> transforms.Compose:
    """Get preprocessing transform for images.
    
    Args:
        image_size: Target size for the images
        
    Returns:
        Preprocessing transform
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size),interpolation=InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
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
