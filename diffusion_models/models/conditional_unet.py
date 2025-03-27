"""Conditional UNet model for attribute-based image generation."""

from diffusers import UNet2DConditionModel

from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> UNet2DConditionModel:
    """Create and return the Conditional UNet2D model.
    
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
    
    return model 