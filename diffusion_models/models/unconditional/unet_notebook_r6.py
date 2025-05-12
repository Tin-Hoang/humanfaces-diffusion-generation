"""Model creation and setup for diffusion models."""

from diffusers import UNet2DModel
from diffusion_models.config import TrainingConfig
import inspect

def create_model(config: TrainingConfig) -> UNet2DModel:
    """Create and return the improved UNet2D model based on the given training configuration."""
    kwargs = {}

    # Inspect for optional features based on diffusers version
    try:
        unet_args = inspect.signature(UNet2DModel.__init__).parameters
        if "use_scale_shift_norm" in unet_args:
            kwargs["use_scale_shift_norm"] = True  # Enables adaptive group norm (AdaGN)
        if "resnet_time_scale_shift" in unet_args:
            kwargs["resnet_time_scale_shift"] = "scale_shift"
        if "flip_sin_to_cos" in unet_args:
            kwargs["flip_sin_to_cos"] = True
        if "freq_shift" in unet_args:
            kwargs["freq_shift"] = 0
        if "attention_head_dim" in unet_args:
            kwargs["attention_head_dim"] = 64  # Optimal channel/head
    except Exception as e:
        print(f"[WARNING] Could not inspect UNet2DModel args: {e}")

    # Improved UNet architecture from ADM
    model = UNet2DModel(
        sample_size=config.image_size,  # e.g., 256
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(160, 320, 640, 640),  # Wider instead of deeper
        down_block_types=(
            "DownBlock2D",         # 32x32
            "AttnDownBlock2D",     # 16x16 with attention
            "AttnDownBlock2D",     # 8x8 with attention
            "DownBlock2D",         # 4x4
        ),
        up_block_types=(
            "UpBlock2D",           # 4x4
            "AttnUpBlock2D",       # 8x8 with attention
            "AttnUpBlock2D",       # 16x16 with attention
            "UpBlock2D",           # 32x32
        ),
        **kwargs
    )

    return model
