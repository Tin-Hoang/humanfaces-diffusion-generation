"""Utilities for handling attributes in conditional diffusion models."""

import torch
from typing import List


def create_sample_attributes(num_samples: int = 4, num_attributes: int = 40) -> torch.Tensor:
    """Create sample attributes for validation image generation.
    
    Args:
        num_samples: Number of different attribute combinations to generate
        num_attributes: Number of attributes in the dataset
        
    Returns:
        Tensor of shape (num_samples, num_attributes) with values 0 (false) or 1 (true)
    """
    # Create some diverse attribute combinations for testing
    sample_attributes = []
    
    # All positive attributes
    sample_attributes.append(torch.ones(num_attributes))
    
    # All negative attributes
    sample_attributes.append(torch.zeros(num_attributes))
    
    # Alternating attributes
    sample_attributes.append(torch.tensor([1 if i % 2 == 0 else 0 for i in range(num_attributes)]))
    
    # Random attributes
    if num_samples > 3:
        for _ in range(num_samples - 3):
            random_attrs = torch.randint(0, 2, (num_attributes,))
            sample_attributes.append(random_attrs)
    
    return torch.stack(sample_attributes).float()


def create_sample_attributes_from_indices(
    attribute_indices: List[int],
    num_attributes: int = 40,
    num_samples: int = None
) -> torch.Tensor:
    """Create sample attributes by manipulating only specific attributes.
    
    This function generates combinations of the specified attributes,
    while keeping other attributes set to 0.
    
    Args:
        attribute_indices: List of attribute indices to manipulate
        num_attributes: Total number of attributes in the dataset
        num_samples: Number of samples to generate. If None, generates all possible
                    combinations (2^n where n is the number of specified attributes).
                    If specified, randomly selects that many unique combinations.
        
    Returns:
        Tensor of shape (num_samples, num_attributes) with values 0 or 1.
        If num_samples is None, shape will be (2^n, num_attributes) where
        n is the number of specified attributes.
    """
    num_specified = len(attribute_indices)
    max_combinations = 2 ** num_specified
    
    # If num_samples not specified or greater than max possible combinations,
    # generate all combinations
    if num_samples is None or num_samples >= max_combinations:
        num_samples = max_combinations
        generate_all = True
    else:
        generate_all = False
    
    # Create base tensor with all zeros
    samples = torch.zeros((num_samples, num_attributes))
    
    if generate_all:
        # Generate all possible combinations for specified attributes
        for i in range(num_samples):
            # Convert number to binary representation
            binary = format(i, f'0{num_specified}b')
            # Set specified attributes according to binary representation
            for j, idx in enumerate(attribute_indices):
                samples[i, idx] = int(binary[j])
    else:
        # Generate random unique combinations
        used_combinations = set()
        for i in range(num_samples):
            while True:
                # Generate random binary number of correct length
                combination = format(torch.randint(0, max_combinations, (1,)).item(), 
                                  f'0{num_specified}b')
                if combination not in used_combinations:
                    used_combinations.add(combination)
                    # Set specified attributes according to binary representation
                    for j, idx in enumerate(attribute_indices):
                        samples[i, idx] = int(combination[j])
                    break
    
    return samples 


def create_multi_hot_attributes(
    attribute_indices: List[int],
    num_attributes: int = 40,
    num_samples: int = 1,
    random_remaining_indices: bool = False
) -> torch.Tensor:
    """Create multi-hot attribute vectors with specific indices set to 1.
    
    This function creates attribute vectors where specified indices are set to 1
    and all other indices are set to 0. Unlike create_sample_attributes_from_indices,
    this function doesn't generate combinations but rather creates the same multi-hot
    vector multiple times if num_samples > 1.
    
    Args:
        attribute_indices: List of attribute indices to set to 1
        num_attributes: Total number of attributes in the dataset
        num_samples: Number of copies of the multi-hot vector to generate
        random_remaining_indices: If True, randomly set remaining indices (not in attribute_indices) 
            to 1 to create more diverse samples (maximum 10 additional random attributes)

    Returns:
        Tensor of shape (num_samples, num_attributes) with values 0 or 1,
        where specified indices are set to 1 and others to 0
    """
    # Create base tensor with all zeros
    samples = torch.zeros((num_samples, num_attributes))
    
    # Set specified indices to 1
    samples[:, attribute_indices] = 1

    if random_remaining_indices:
        # Get all indices not in attribute_indices
        all_indices = set(range(num_attributes))
        remaining_indices = list(all_indices - set(attribute_indices))
        
        # For each sample, randomly set some of the remaining indices to 1
        for i in range(num_samples):
            # Randomly choose how many additional indices to set to 1 (max 10)
            num_additional = min(10, torch.randint(0, len(remaining_indices) + 1, (1,)).item())
            if num_additional > 0:
                # Randomly select indices to set to 1
                selected_indices = torch.randperm(len(remaining_indices))[:num_additional]
                samples[i, [remaining_indices[idx] for idx in selected_indices]] = 1

    return samples
