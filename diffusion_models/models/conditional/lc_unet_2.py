"""Latent Conditional UNet model for attribute-based latent diffusion."""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> UNet2DConditionModel:
    """Create and return the Conditional UNet2D model.
    
    This model is designed for latent diffusion, operating in the VAE latent space
    and conditioned on attribute vectors. The architecture follows modern diffusion models
    like Stable Diffusion, with extensive use of attention mechanisms and skip connections.
    
    Args:
        config: Training configuration object
        
    Returns:
        UNet2DConditionModel: The conditional UNet model
    """
    # Create the UNet model for latent diffusion
    model = UNet2DConditionModel(
        # Latent space parameters
        sample_size=64,  # 64x64 latents for higher resolution control
        in_channels=3,   # VQ-VAE latent space channels (4 channels)
        out_channels=3,  # Noise prediction in latent space
        
        # Downsampling blocks with extensive attention
        down_block_types=(
            "CrossAttnDownBlock2D",      # 64x64 -> 32x32 with cross-attention
            "CrossAttnDownBlock2D",      # 32x32 -> 16x16 with cross-attention
            "CrossAttnDownBlock2D",      # 16x16 -> 8x8 with cross-attention
            "DownBlock2D",               # 8x8 -> 4x4 standard downsampling
        ),
        
        # Upsampling blocks with symmetric attention
        up_block_types=(
            "UpBlock2D",                 # 4x4 -> 8x8 standard upsampling
            "CrossAttnUpBlock2D",        # 8x8 -> 16x16 with cross-attention
            "CrossAttnUpBlock2D",        # 16x16 -> 32x32 with cross-attention
            "CrossAttnUpBlock2D",        # 32x32 -> 64x64 with cross-attention
        ),
        
        # Architecture parameters (similar to Stable Diffusion)
        block_out_channels=(128, 256, 512, 512),  # Channel dimensions per block
        layers_per_block=2,                       # Two ResNet layers per block for better capacity
        cross_attention_dim=512,                  # Dimension of cross-attention features
        attention_head_dim=32,                     # Size of attention heads
        
        # Model configuration
        use_linear_projection=True,                # Memory-efficient attention
        num_class_embeds=None,                     # No class conditioning
        only_cross_attention=False,                # Enable both self and cross attention
        
        # Architecture details
        act_fn="gelu",                            # GELU activation as requested
        norm_num_groups=32,                        # Group normalization
        norm_eps=1e-5,                            # Numerical stability
        cross_attention_norm="layer_norm",         # Cross-attention normalization
        resnet_time_scale_shift="default",        # Time conditioning in ResNet blocks
        addition_embed_type=None,                 # No additional embeddings
        addition_time_embed_dim=None,             # No additional time embeddings
    )
    
    if hasattr(config, "device"):
        model = model.to(config.device)
        
    print(f"\nCreated UNet2DConditionModel with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model 