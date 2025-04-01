import torch
import torch.nn as nn
import types
from diffusers import UNet2DOutput

class DiTTransformer2DModel(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        embed_dim=512, 
        depth=12, 
        num_heads=8, 
        patch_size=16, 
        img_size=256, 
        **kwargs
    ):
        super().__init__()
        
        # Minimal additions for Diffusers compatibility
        self.name = "dit_transformer"
        self.config = types.SimpleNamespace()
        self.config.sample_size = img_size
        self.config.in_channels = in_channels
        self.config.out_channels = in_channels
        self.config.embed_dim = embed_dim
        self.config.depth = depth
        self.config.num_heads = num_heads
        self.config.patch_size = patch_size
        
        # (Optional) Set device attribute
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

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
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch_size"
        
        # Create patch embeddings
        x = self.patch_embed(x)  # Shape: (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, embed_dim)
        x = x + self.pos_embed  # Add positional embeddings
        
        # Prepare transformer input: (sequence_length, batch, embed_dim)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Project back to pixel space
        x = self.proj_out(x)  # Shape: (B, num_patches, patch_size*patch_size*in_channels)
        x = x.view(B, self.num_patches, self.in_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.in_channels, H, W)
        
        # Return a Diffusers-style output (UNet2DOutput) so that downstream code can call .sample
        return UNet2DOutput(sample=x)

def create_model(config):
    return DiTTransformer2DModel(
        in_channels=getattr(config, "in_channels", 3),
        embed_dim=getattr(config, "embed_dim", 512),
        depth=getattr(config, "depth", 12),
        num_heads=getattr(config, "num_heads", 8),
        patch_size=getattr(config, "patch_size", 16),
        img_size=getattr(config, "img_size", 256)
    )
