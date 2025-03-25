"""Conditional UNet model for attribute-based image generation."""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from diffusion_models.config import TrainingConfig


class AttributeEmbedder(nn.Module):
    """Projects attribute vectors to the dimension expected by UNet2DConditionModel."""
    
    def __init__(self, num_attributes: int, hidden_size: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_attributes, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, num_attributes)
        # Output shape: (batch_size, 1, hidden_size) for cross-attention
        x = self.projection(x)  # (batch_size, hidden_size)
        return x.unsqueeze(1)  # Add sequence dimension for cross-attention


def create_model(config: TrainingConfig) -> tuple[UNet2DConditionModel, AttributeEmbedder]:
    """Create and return the Conditional UNet2D model and attribute embedder.
    
    This model is designed for image generation conditioned on attribute vectors.
    The architecture includes cross-attention layers to incorporate the attribute
    information at multiple resolutions.
    
    Args:
        config: Training configuration object
        
    Returns:
        Tuple of (UNet2DConditionModel, AttributeEmbedder)
    """
    # Create the UNet model
    model = UNet2DConditionModel(
        # Image parameters
        sample_size=config.image_size,  # Input image size
        in_channels=3,  # RGB input images
        out_channels=3,  # RGB output images
        
        # Architecture parameters
        cross_attention_dim=64,  # Dimension of cross-attention features
        layers_per_block=2,  # Number of ResNet layers per block
        
        # Channel dimensions for each block
        block_out_channels=(64, 128, 256, 256),  # Output channels for each UNet block
        
        # Downsampling blocks with attention
        down_block_types=(
            "CrossAttnDownBlock2D",    # 128x128 -> 64x64 with cross-attention
            "CrossAttnDownBlock2D",     # 64x64 -> 32x32 with cross-attention
            "DownBlock2D",              # 32x32 -> 16x16 standard downsampling
            "DownBlock2D",              # 16x16 -> 8x8 standard downsampling
        ),
        
        # Upsampling blocks with attention (reverse order)
        up_block_types=(
            "UpBlock2D",               # 8x8 -> 16x16 standard upsampling
            "UpBlock2D",               # 16x16 -> 32x32 standard upsampling
            "CrossAttnUpBlock2D",      # 32x32 -> 64x64 with cross-attention
            "CrossAttnUpBlock2D",      # 64x64 -> 128x128 with cross-attention
        ),
        
        # Additional parameters
        num_class_embeds=None,  # Disable class embeddings since we're using cross-attention
        only_cross_attention=False,  # Enable both self-attention and cross-attention
        
        # Architecture details
        time_embedding_type="positional",  # Type of time step embeddings
        norm_num_groups=32,  # Number of groups for group normalization
        norm_eps=1e-5,  # Epsilon for numerical stability
        cross_attention_norm="layer_norm",  # Normalization for cross-attention
    )
    
    # Create the attribute embedder
    attribute_embedder = AttributeEmbedder(
        num_attributes=config.num_attributes,
        hidden_size=64  # Match cross_attention_dim for compatibility
    )
    
    return model, attribute_embedder 