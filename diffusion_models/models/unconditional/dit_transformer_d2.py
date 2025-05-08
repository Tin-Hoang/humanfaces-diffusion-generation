from diffusers import DiTTransformer2DModel
from diffusion_models.config import TrainingConfig


def create_model(config: TrainingConfig) -> DiTTransformer2DModel:
    """
    DiT-Large variant for high-capacity diffusion modeling on 128x128 images.

    - Patch size = 2 -> very fine patches (4096 patches for 128x128)
    - 12 layers -> deeper model
    - 12 attention heads x 64 dims = 768 total hidden size
    - dropout = 0.2 for stronger regularization
    - Group norm with 32 groups, using adaptive norm
    """
    return DiTTransformer2DModel(
        sample_size=config.image_size,  # 128 for CelebA-HQ
        in_channels=3,
        out_channels=3,
        patch_size=4,                   # Fine patches -> large sequence length
        num_layers=12,                  # Deeper network
        num_attention_heads=12,         # More heads for capturing diverse features
        attention_head_dim=64,          # 64 * 12 = 768 hidden size
        dropout=0.2,                    # Higher dropout for robust regularization
        norm_num_groups=32,
        attention_bias=True,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        upcast_attention=False,
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
        norm_eps=1e-5,
    )
