"""Model creation and setup for diffusion models."""

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch

from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> UNet2DModel:
    """Create and return the UNet2D model."""
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


def create_noise_scheduler(config: TrainingConfig) -> DDPMScheduler:
    """Create and return the DDPMScheduler."""
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
    )
    return noise_scheduler


def setup_optimizer_and_scheduler(model: UNet2DModel, config: TrainingConfig, train_dataloader) -> tuple:
    """Setup optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    return optimizer, lr_scheduler
