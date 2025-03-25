"""Attribute embedder module for diffusion models."""

import torch
from torch import nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class AttributeEmbedder(ModelMixin, ConfigMixin):
    """
    Embedder for attribute vectors that projects multi-hot attribute tensors
    to a dimension suitable for cross-attention conditioning.
    
    This class is compatible with diffusers' serialization system.
    """
    @register_to_config
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