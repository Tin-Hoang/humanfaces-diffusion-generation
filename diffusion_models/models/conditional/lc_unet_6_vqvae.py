"""Latent Conditional UNet model for attribute-based latent diffusion."""

from diffusers import UNet2DConditionModel

from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> UNet2DConditionModel:
    """Create and return the Conditional UNet2D model.

    Inspired by the ddpm-ema-celebahq-256 model:
    https://huggingface.co/google/ddpm-ema-celebahq-256/blob/main/config.json

    This model mimicks the topology of the UNet (6 blocks)
    but with cross-attention in the last 2 layers for conditioning.
    It's designed for latent diffusion, operating in the VAE latent space
    and conditioned on attribute vectors.

    Args:
        config: Training configuration object

    Returns:
        UNet2DConditionModel: The conditional UNet model
    """
    # Calculate sample_size based on image_size and VQ-VAE downsampling
    sample_size = config.image_size // 4  # VQ-VAE downsampling factor is 4

    # Create the UNet model for latent diffusion
    model = UNet2DConditionModel(
        # Latent space parameters
        sample_size=sample_size,  # 64x64 for 256x256 images (256/4)
        in_channels=3,   # VQ-VAE latent space channels (3 channels)
        out_channels=3,  # Noise prediction in latent space

        # Downsampling blocks
        down_block_types=(
            "CrossAttnDownBlock2D", # 64x64 -> 32x32 with cross-attention
            "CrossAttnDownBlock2D", # 32x32 -> 16x16 with cross-attention
            "DownBlock2D",          # 16x16 -> 8x8
            "DownBlock2D",          # 8x8 -> 4x4
            "DownBlock2D",          # 4x4 -> 2x2
            "DownBlock2D",          # 2x2 -> 1x1
        ),

        # Upsampling blocks
        up_block_types=(
            "UpBlock2D",            # 1x1 -> 2x2
            "UpBlock2D",            # 2x2 -> 4x4
            "UpBlock2D",            # 4x4 -> 8x8
            "UpBlock2D",            # 8x8 -> 16x16
            "CrossAttnUpBlock2D",   # 16x16 -> 32x32 with cross-attention
            "CrossAttnUpBlock2D",   # 32x32 -> 64x64 with cross-attention
        ),

        # Architecture parameters
        block_out_channels=(128, 128, 256, 256, 512, 512),  # Channel dimensions per block
        layers_per_block=2,                       # Two layers per block for better capacity
        cross_attention_dim=256,                  # Dimension of cross-attention features
        attention_head_dim=8,                     # Size of attention heads

        # Model configuration
        use_linear_projection=True,               # Memory-efficient attention
        num_class_embeds=None,                    # No class conditioning
        only_cross_attention=False,               # Enable both self and cross attention

        # Architecture details
        act_fn="silu",                            # Silu activation function
        norm_num_groups=32,                       # Group normalization
        norm_eps=1e-5,                            # Numerical stability
        cross_attention_norm="layer_norm",        # Cross-attention normalization
    )

    if hasattr(config, "device"):
        model = model.to(config.device)

    # Calculate approximate memory usage
    param_count = sum(p.numel() for p in model.parameters())
    batch_size = 16
    sample_size = config.image_size // 4  # VQ-VAE downsampling factor is 4
    memory_per_sample = param_count * 4  # 4 bytes per float32
    total_memory = memory_per_sample * batch_size

    print(f"\nCreated UNet2DConditionModel:")
    print(f"Parameters: {param_count:,}")
    print(f"Sample size: {sample_size}x{sample_size} (for {config.image_size}x{config.image_size} images)")
    print(f"Approximate memory usage: {total_memory / (1024**3):.2f} GB for batch_size={batch_size}")

    return model
