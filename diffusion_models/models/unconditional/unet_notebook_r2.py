from diffusers import UNet2DModel
from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> UNet2DModel:
    """Create and return the optimized UNet2D model for 128x128 images."""
    model = UNet2DModel(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

