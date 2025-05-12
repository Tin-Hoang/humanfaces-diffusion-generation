"""Latent Conditional UNet model for attribute + segmentation-based latent diffusion (VQ-VAE setup)."""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from transformers import SegformerForSemanticSegmentation

from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> UNet2DConditionModel:
    """Create and return the Conditional UNet2D model for VQ-VAE-based latent diffusion."""

    sample_size = config.image_size // 4  # VQ-VAE downsampling factor is 4

    print(f"Using cross_attention_dim = {config.cross_attention_dim}")

    # ✅ Load SegFormer encoder if needed
    segformer = None
    segformer_output_dim = 0
    if config.conditioning_type in ["segmentation", "combined"]:
        print(f"[lc_unet_3_vqvae] Loading SegFormer encoder from: {config.segmentation_encoder_checkpoint}")
        segformer = SegformerForSemanticSegmentation.from_pretrained(
            config.segmentation_encoder_checkpoint
        )
        segformer.eval()
        segformer.requires_grad_(False)
        segformer.to(config.device)

    # ✅ Create the UNet2DConditionModel
    model = UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=3,
        out_channels=3,
        down_block_types=(
            "DownBlock2D",             # 64 -> 32
            "CrossAttnDownBlock2D",    # 32 -> 16
            "DownBlock2D",             # 16 -> 8
            "DownBlock2D",             # 8 -> 4
            "DownBlock2D",             # 4 -> 2
        ),
        up_block_types=(
            "UpBlock2D",               # 2 -> 4
            "UpBlock2D",               # 4 -> 8
            "UpBlock2D",               # 8 -> 16
            "CrossAttnUpBlock2D",      # 16 -> 32
            "UpBlock2D",               # 32 -> 64
        ),
        block_out_channels=(128, 128, 256, 512, 512),
        layers_per_block=2,
        cross_attention_dim=config.cross_attention_dim,
        attention_head_dim=8,
        use_linear_projection=True,
        num_class_embeds=None,
        only_cross_attention=False,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_norm="layer_norm",
    )

    if hasattr(config, "device"):
        model = model.to(config.device)

    # ✅ Attach segmentation encoder and projection layer
    model.segmentation_encoder = segformer

    if segformer is not None:
        segformer_output_dim = segformer.config.hidden_sizes[-1]
        print(f"[lc_unet_3_vqvae] segformer_output_dim = {segformer_output_dim}")
        model.seg_proj = nn.Linear(segformer_output_dim, config.attribute_embed_dim)

    # ❌ Do NOT initialize combined_proj here; it depends on runtime embedding dims.

    # ✅ Debug info
    param_count = sum(p.numel() for p in model.parameters())
    batch_size = 16
    memory_per_sample = param_count * 4
    total_memory = memory_per_sample * batch_size

    print(f"\nCreated UNet2DConditionModel:")
    print(f"Parameters: {param_count:,}")
    print(f"Sample size: {sample_size}x{sample_size} (for {config.image_size}x{config.image_size} images)")
    print(f"Approximate memory usage: {total_memory / (1024**3):.2f} GB for batch_size={batch_size}")

    return model
