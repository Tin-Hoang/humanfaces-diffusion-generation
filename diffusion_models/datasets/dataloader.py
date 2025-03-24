"""Dataset loading utilities for diffusion models."""

from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional, Union
from torchvision import transforms
import torch
import os

from diffusion_models.datasets.data_utils import get_preprocess_transform, transform
from .attribute_data_loader import AttributeDataset


def setup_dataloader(
    data_source: str,
    batch_size: int,
    image_size: int,
    shuffle: bool = True,
    split: str = "train",
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> Tuple[DataLoader, transforms.Compose]:
    """Setup a dataset with preprocessing.
    
    Args:
        data_source: Either:
            - A string path to a local image folder, or
            - A string containing the Hugging Face dataset name
        batch_size: Batch size for the dataloader
        image_size: Target size for the images
        shuffle: Whether to shuffle the data
        split: Dataset split to load
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster data transfer to GPU
        drop_last: Whether to drop the last incomplete batch

    Returns:
        Tuple containing:
            - Dataloader
            - Preprocessing transform
    """
    # Setup preprocessing
    preprocess = get_preprocess_transform(image_size)
    
    # Check if data source exists locally
    if os.path.exists(data_source):
        # Local image folder
        dataset = load_dataset("imagefolder", data_dir=data_source, split=split)
        print(f"Loaded local dataset from: {data_source}")
        print(f"Number of images in the dataset: {len(dataset)}")
    else:
        # Try to load from Hugging Face
        try:
            dataset = load_dataset(data_source, split=split)
            print(f"Loaded Hugging Face dataset: {data_source}")
            print(f"Number of images in the dataset: {len(dataset)}")
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from {data_source}. "
                f"Please check if the path exists locally or if the Hugging Face dataset name is correct. "
                f"Error: {str(e)}"
            )
    
    # Apply transforms
    dataset.set_transform(lambda examples: transform(examples, preprocess))
    
    # Create dataloader with all specified settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return dataloader, preprocess


def create_attribute_dataloader(
    image_dir: str,
    attribute_file_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    indices: Optional[torch.Tensor] = None
) -> DataLoader:
    """Create a DataLoader for a specific split of the AttributeDataset.
    
    Args:
        image_dir (str): Directory containing the image files
        attribute_file_path (str): Path to the attribute label file
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the data
        pin_memory (bool): Whether to pin memory for faster data transfer to GPU
        drop_last (bool): Whether to drop the last incomplete batch
        indices (Optional[torch.Tensor]): Specific indices to use for this split.
            If None, the full dataset will be used.
        
    Returns:
        DataLoader: The created DataLoader for the specified split
    """
    # Create the dataset
    dataset = AttributeDataset(
        image_dir=image_dir,
        attribute_file_path=attribute_file_path
    )
    
    # If indices are provided, create a subset
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Create DataLoader with appropriate settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return dataloader


def get_dataset_splits(
    image_dir: str,
    attribute_file_path: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get indices for train/val/test splits of the dataset.
    
    Args:
        image_dir (str): Directory containing the image files
        attribute_file_path (str): Path to the attribute label file
        train_split (float): Proportion of data to use for training
        val_split (float): Proportion of data to use for validation
        test_split (float): Proportion of data to use for testing
        seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            Indices for train, validation, and test splits
    """
    # Create the dataset
    dataset = AttributeDataset(
        image_dir=image_dir,
        attribute_file_path=attribute_file_path
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split the dataset
    train_indices, val_indices, test_indices = random_split(
        range(total_size),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    # Test the dataloader creation
    try:
        # Test local image folder
        print("\nTesting local image folder loader:")
        local_loader, preprocess = setup_dataloader(
            data_source="data/CelebA-HQ-split/test_300",
            batch_size=8,
            image_size=256,
            num_workers=2
        )
        batch = next(iter(local_loader))
        print(f"Local batch shape: {batch['image'].shape}")
        
        # Test Hugging Face dataset
        print("\nTesting Hugging Face dataset loader:")
        hf_loader, preprocess = setup_dataloader(
            data_source="huggan/smithsonian_butterflies_subset",  # Example with Smithsonian butterflies dataset
            batch_size=8,
            image_size=256,
            num_workers=2
        )
        batch = next(iter(hf_loader))
        print(f"HF batch shape: {batch['image'].shape}")
        
        # Test attribute dataloader
        print("\nTesting attribute dataloader:")
        train_indices, val_indices, test_indices = get_dataset_splits(
            image_dir="data/CelebA-HQ-split/test_300",
            attribute_file_path="data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt",
            batch_size=8,
            num_workers=2
        )
        
        train_loader = create_attribute_dataloader(
            image_dir="data/CelebA-HQ-split/test_300",
            attribute_file_path="data/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt",
            indices=train_indices
        )
        batch = next(iter(train_loader))
        images, attributes = batch
        print(f"Attribute batch shapes - Images: {images.shape}, Attributes: {attributes.shape}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise
