import torch
import torch.nn as nn
import types  # Needed to create a simple config object

class DiTTransformer2DModel(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, depth=12, num_heads=8, patch_size=16, img_size=256, **kwargs):
        super().__init__()
        self.name = "dit_transformer"
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        # Create a minimal config object that diffusers pipelines expect
        self.config = types.SimpleNamespace()
        self.config.sample_size = img_size         # The input resolution (256)
        self.config.in_channels = in_channels        # e.g., 3 for RGB images
        self.config.out_channels = in_channels       # Typically the same as in_channels
        # If needed, you can add more attributes here (e.g., cross_attention_dim)

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj_out = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

    def forward(self, x, timesteps=None, encoder_hidden_states=None, return_dict=False):
        """
        Forward pass that adapts to the training loop expectations.
        
        Args:
            x (Tensor): Input tensor.
            timesteps: Diffusion timesteps (ignored in this model).
            encoder_hidden_states: Optional conditional embeddings (ignored here).
            return_dict (bool): Whether to return a dict instead of a tuple.
            
        Returns:
            Either a tuple or a dict with the output image.
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch_size"
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.proj_out(x)
        x = x.view(B, self.num_patches, self.in_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.in_channels, H, W)
        
        if return_dict:
            return {"sample": x}
        else:
            return (x,)

def create_model(config):
    return DiTTransformer2DModel(
        in_channels=getattr(config, "in_channels", 3),
        embed_dim=getattr(config, "embed_dim", 512),
        depth=getattr(config, "depth", 12),
        num_heads=getattr(config, "num_heads", 8),
        patch_size=getattr(config, "patch_size", 16),
        img_size=getattr(config, "img_size", 256)
    )
