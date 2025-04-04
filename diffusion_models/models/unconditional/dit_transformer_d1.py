"""Optimized DiT model setup for CelebA-HQ dataset (256x256)."""

from diffusers import DiTTransformer2DModel
from diffusion_models.config import TrainingConfig

def create_model(config: TrainingConfig) -> DiTTransformer2DModel:
    """Create and return a revised DiT Transformer model for CelebA-HQ with improved generalization."""
    model = DiTTransformer2DModel(
        sample_size=config.image_size,      # e.g., 256 for CelebA-HQ
        in_channels=3,                      # RGB input
        out_channels=3,                     # RGB output
        patch_size=2,                       # 2x2 patches
        num_layers=4,                       # Increased depth for hierarchical feature extraction
        num_attention_heads=8,              # Keeping 8 heads
        attention_head_dim=64,              # Reduced dimension per head -> effective hidden size 512
        dropout=0.2,                        # Dropout to help regularize the model
        norm_num_groups=32,                 # Groups for group normalization (adjustable)
        attention_bias=True,                # Use bias in attention layers
        activation_fn="gelu-approximate",   # Activation function
        num_embeds_ada_norm=1000,           # AdaLayerNorm embeddings (if used)
        upcast_attention=False,             # Keep as is to control memory usage
        norm_type="ada_norm_zero",          # Consider testing with "layer" if needed
        norm_elementwise_affine=False,      # No affine parameters in normalization layers
        norm_eps=1e-5,                      # Epsilon for normalization stability
    )
    return model


# def create_model(config: TrainingConfig) -> DiTTransformer2DModel:
#     """Create and return a lite DiT Transformer model for quick testing.
    
#     This configuration uses:
#       - A smaller image resolution (e.g. 64x64) to speed up training.
#       - Fewer transformer layers and reduced attention dimensions.
#     """
#     # Override sample size for quick testing (if desired)
#     test_sample_size = 64  # Use a smaller resolution for a quick run
    
#     model = DiTTransformer2DModel(
#         sample_size=test_sample_size,      # Reduced resolution for fast training
#         in_channels=3,                     # RGB input
#         out_channels=3,                    # RGB output
#         patch_size=2,                      # 2x2 patches
#         num_layers=1,                      # Reduced depth for quick iteration
#         num_attention_heads=1,             # Fewer attention heads
#         attention_head_dim=4,             # Lower dimension per head (effective hidden size = 4x32 = 128)
#         dropout=0.1,                       # Lower dropout for testing
#         norm_num_groups=8,                 # Fewer groups in normalization
#         attention_bias=True,               # Use bias in attention layers
#         activation_fn="gelu-approximate",  # Activation function
#         num_embeds_ada_norm=100,           # Reduced number of AdaLayerNorm embeddings
#         upcast_attention=False,            # Keep as is to control memory usage
#         norm_type="ada_norm_zero",         # Use supported norm type for patch-based processing
#         norm_elementwise_affine=False,     # No affine parameters in normalization layers
#         norm_eps=1e-5,                     # Epsilon for normalization
#     )
#     return model

