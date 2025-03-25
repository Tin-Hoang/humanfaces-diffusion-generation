"""Attribute embedder module for diffusion models."""

import torch
from torch import nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class AttributeEmbedder(ModelMixin, ConfigMixin):
    """
    Embedder for attribute vectors that projects multi-hot attribute tensors
    to a dimension suitable for cross-attention conditioning.
    
    Uses a multi-layer architecture with gradual dimension increase and 
    non-linearities for better feature extraction from binary attributes.
    
    This class is compatible with diffusers' serialization system.
    """
    @register_to_config
    def __init__(
        self, 
        input_dim: int = 40, 
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        """
        Initialize the attribute embedder.
        
        Args:
            input_dim: Number of attributes (default: 40).
            hidden_dim: Final hidden dimension for cross-attention (default: 256).
            num_layers: Number of layers for gradual scaling (default: 3).
            dropout: Dropout rate between layers (default: 0.2).
        """
        super().__init__()
        
        # Calculate intermediate dimensions for gradual scaling
        dims = []
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = dims[-1]
                
            if i == num_layers - 1:
                out_dim = hidden_dim
            else:
                # More gradual scaling to reach 256
                out_dim = input_dim + (hidden_dim - input_dim) * ((i + 1) / num_layers)
                out_dim = int(out_dim)
                # Cap intermediate dimensions
                out_dim = min(out_dim, hidden_dim)
            dims.append(out_dim)
        
        # Build layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_features = input_dim
            else:
                in_features = dims[i-1]
            
            # Add linear layer
            layers.append(nn.Linear(in_features, dims[i]))
            
            # Add normalization and activation for all but last layer
            if i < num_layers - 1:
                layers.extend([
                    nn.LayerNorm(dims[i]),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
            else:
                # Just normalization for final layer
                layers.append(nn.LayerNorm(dims[i]))
        
        self.net = nn.Sequential(*layers)
        
        print(f"\nInitialized AttributeEmbedder:")
        print(f"Input dim: {input_dim}")
        print(f"Hidden dims: {dims}")
        print(f"Dropout: {dropout}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project attribute vectors to hidden dimension.
        
        Args:
            x: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor of shape (batch_size, 1, hidden_dim)
        """
        # Project through network and add sequence dimension
        output = self.net(x).unsqueeze(1)
        return output 