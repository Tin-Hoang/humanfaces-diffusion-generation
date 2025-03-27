"""Model creation and setup for diffusion models."""

from diffusers import UNet2DModel  # ✅ fallback-safe import
from diffusion_models.config import TrainingConfig

def create_model(config: TrainingConfig) -> UNet2DModel:
    kwargs = {}

    # ✅ Only add these if they are supported
    try:
        import inspect
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

    model = UNet2DModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        **kwargs  # ✅ inject supported ADM-style options
    )
    return model

