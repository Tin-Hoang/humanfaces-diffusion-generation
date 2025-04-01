"""Optimized UNet model setup for CelebA-HQ dataset (256x256)."""

from diffusers import UNet2DModel

from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> UNet2DModel:
    """Create and return the optimized UNet2D model for CelebA-HQ."""
    model = UNet2DModel(
        sample_size=config.image_size,  # should be 256 for CelebA-HQ
        in_channels=3,                  # RGB images
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),  # optimal feature sizes
        down_block_types=(
            "DownBlock2D",       # 256→128
            "DownBlock2D",       # 128→64
            "AttnDownBlock2D",   # 64→32 with attention
            "AttnDownBlock2D",   # 32→16 with attention
        ),
        up_block_types=(
            "AttnUpBlock2D",     # 16→32 with attention
            "AttnUpBlock2D",     # 32→64 with attention
            "UpBlock2D",         # 64→128
            "UpBlock2D",         # 128→256
        ),
    )
    return model
