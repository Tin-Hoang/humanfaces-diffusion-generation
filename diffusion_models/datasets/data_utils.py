"""Data preprocessing utilities for diffusion models."""

from torchvision import transforms
from typing import Dict, Any
from torchvision.transforms import InterpolationMode
import argparse

def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Image Augmentation")

    parser.add_argument(
            "--elastic_transform",
            type=float,
            action="store_true",
            help="elastic transform for data augmentation",
        )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="randomly change the brightness, contrast and saturation",
    )
    
    return parser.parse_args()
    

def get_preprocess_transform(image_size: int) -> transforms.Compose:
    """Get preprocessing transform for images.
    
    Args:
        image_size: Target size for the images
        
    Returns:
        Preprocessing transform
    """
    args = parse_args()

    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.LANCZOS),  # Resize to target size
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ToTensor(),  # Convert to tensor in [0, 1] range
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1] range
        transforms.ColorJitter() if args.color_jitter else None,
        transforms.ElasticTransform(alpha=250.0) if args.elastic_tranform else None

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
