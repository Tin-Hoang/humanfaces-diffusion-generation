"""Model creation and setup for diffusion models."""

from diffusers import UNet2DModel
from diffusion_models.config import TrainingConfig
import inspect

def create_model(config: TrainingConfig) -> UNet2DModel:
    """Create and return the UNet2D model based on the given training configuration."""
    kwargs = {}

    # Check for optional arguments based on current diffusers version
    try:
        unet_args = inspect.signature(UNet2DModel.__init__).parameters
        if "use_scale_shift_norm" in unet_args:
            kwargs["use_scale_shift_norm"] = config.use_scale_shift_norm
        if "resnet_time_scale_shift" in unet_args:
            kwargs["resnet_time_scale_shift"] = "scale_shift"
        if "flip_sin_to_cos" in unet_args:
            kwargs["flip_sin_to_cos"] = True
        if "freq_shift" in unet_args:
            kwargs["freq_shift"] = 0
    except Exception as e:
        print(f"[WARNING] Could not inspect UNet2DModel args: {e}")

    # Build the model with deeper blocks and more attention
    model = UNet2DModel(
        sample_size=config.image_size,  # e.g., 128 or 256
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512, 512, 1024),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        **kwargs
    )

    return model

