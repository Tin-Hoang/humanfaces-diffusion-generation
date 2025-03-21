"""Dataset loading utilities for diffusion models."""

from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import transforms

from diffusion_models.datasets.data_utils import get_preprocess_transform, transform


def setup_dataloader(
    data_dir: str,
    batch_size: int,
    image_size: int,
    shuffle: bool = True,
    split: str = "train"
) -> Tuple[DataLoader, transforms.Compose]:
    """Setup a dataset with preprocessing.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for the dataloader
        image_size: Target size for the images
        shuffle: Whether to shuffle the data
        split: Dataset split to load
        
    Returns:
        Tuple containing:
            - Dataloader
            - Preprocessing transform
    """
    # Load dataset
    dataset = load_dataset("imagefolder", data_dir=data_dir, split=split)
    print(f"Number of images in the dataset: {len(dataset)}")
    
    # Setup preprocessing
    preprocess = get_preprocess_transform(image_size)
    
    # Apply transforms
    dataset.set_transform(lambda examples: transform(examples, preprocess))
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, preprocess
