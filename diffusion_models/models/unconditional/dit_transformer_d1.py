"""Optimized DiT model setup for CelebA-HQ dataset (256x256)."""

from diffusers import DiTTransformer2DModel
from diffusion_models.config import TrainingConfig

def create_model(config: TrainingConfig) -> DiTTransformer2DModel:
    """
    DiT-Base variant for diffusion modeling on 128x128 images.

    Configuration:
      - Patch Size: 4 (reduces sequence length to 32x32 = 1024 patches)
      - Number of Layers: 6 (sufficient depth without excessive compute)
      - Attention Heads: 8 with head dimension 64 (total hidden size 512)
      - Dropout: 0.1 (balanced regularization)
      - Normalization: Group normalization with 32 groups and adaptive normalization
    """
    return DiTTransformer2DModel(
        sample_size=config.image_size,      # 128
        in_channels=3,                      # RGB input
        out_channels=3,                     # RGB output
        patch_size=4,                       # Reduces sequence length (fewer patches than patch_size=2)
        num_layers=6,                       # Balanced depth
        num_attention_heads=8,              # Standard for DiT-Base
        attention_head_dim=64,              # Each head with 64 dims; total hidden size = 512
        dropout=0.1,                        # Moderate dropout for regularization
        norm_num_groups=32,                 # Group normalization with 32 groups
        attention_bias=True,                # Use bias in attention layers
        activation_fn="gelu-approximate",   # Efficient GELU approximation
        num_embeds_ada_norm=1000,           # Number of embeddings for adaptive normalization
        upcast_attention=False,             # Optimize memory usage
        norm_type="ada_norm_zero",          # Adaptive normalization variant
        norm_elementwise_affine=False,      # No extra affine parameters in normalization layers
        norm_eps=1e-5,                      # Small epsilon for numerical stability
    )



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

