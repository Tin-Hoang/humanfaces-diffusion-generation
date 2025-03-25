"""Attribute embedder module for diffusion models."""

import torch
from torch import nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention_processor import AttnProcessor


class AttributeEmbedder(ModelMixin, nn.Module):
    """
    Embedder for attribute vectors that projects multi-hot attribute tensors
    to a dimension suitable for cross-attention conditioning.
    
    This class is compatible with diffusers' serialization system.
    """
    config_name = "attribute_embedder_config"
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 512):
        """
        Initialize the attribute embedder.
        
        Args:
            input_dim: Number of attributes (default: 40).
            hidden_dim: Hidden dimension for cross-attention (default: 512).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        # Set default attention processor
        self.set_default_attn_processor()
    
    def set_default_attn_processor(self):
        """Sets the default attention processor."""
        self.processor = AttnProcessor()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project attribute vectors to hidden dimension.
        
        Args:
            x: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, 1, hidden_dim)
        """
        # Add sequence dimension
        output = self.proj(x).unsqueeze(1)
        return output 