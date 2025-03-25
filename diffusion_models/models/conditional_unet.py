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
        return x.unsqueeze(1)  # Add sequence dimension


def create_model(config: TrainingConfig) -> tuple[UNet2DConditionModel, AttributeEmbedder]:
    """Create and return the Conditional UNet2D model and attribute embedder.
    
    This model is designed for 128x128 image generation conditioned on
    40-dimensional attribute vectors. The architecture includes cross-attention
    layers to incorporate the attribute information.
    
    Args:
        config: Training configuration object
        
    Returns:
        Tuple of (UNet2DConditionModel, AttributeEmbedder)
    """
    # Create the UNet model with reduced size
    model = UNet2DConditionModel(
        # Image parameters
        sample_size=config.image_size,  # 128x128 images
        in_channels=3,  # RGB images
        out_channels=3,  # RGB output
        
        # Architecture parameters
        cross_attention_dim=32,  # Reduced from 64 to 32
        layers_per_block=1,  # Reduced from 2 to 1
        
        # Channel dimensions for each block (reduced)
        block_out_channels=(64, 128, 192, 256),  # Reduced from (128, 256, 384, 512)
        
        # Downsampling blocks with attention
        down_block_types=(
            "CrossAttnDownBlock2D",    # 128x128 -> 64x64
            "CrossAttnDownBlock2D",     # 64x64 -> 32x32
            "DownBlock2D",              # 32x32 -> 16x16
            "DownBlock2D",              # 16x16 -> 8x8
        ),
        
        # Upsampling blocks with attention (reverse order)
        up_block_types=(
            "UpBlock2D",               # 8x8 -> 16x16
            "UpBlock2D",               # 16x16 -> 32x32
            "CrossAttnUpBlock2D",      # 32x32 -> 64x64
            "CrossAttnUpBlock2D",      # 64x64 -> 128x128
        ),
        
        # Additional parameters
        num_class_embeds=None,  # Disable class embeddings since we're using cross-attention
        only_cross_attention=False,  # Allow both self-attention and cross-attention
        
        # Architecture details
        time_embedding_type="positional",  # Standard positional time embeddings
        norm_num_groups=32,  # Number of groups for group normalization
        norm_eps=1e-5,  # Epsilon for normalization
        cross_attention_norm="layer_norm",  # Use layer normalization for cross attention
        
        # Memory optimizations
        use_linear_projection=True,  # Use linear projection for attention to save memory
        attention_head_dim=16,  # This will result in 2 attention heads (32/16=2)
    )
    
    # Create the attribute embedder with reduced hidden size
    attribute_embedder = AttributeEmbedder(
        num_attributes=config.num_attributes,
        hidden_size=32  # Reduced from 64 to 32 to match cross_attention_dim
    )
    
    return model, attribute_embedder 