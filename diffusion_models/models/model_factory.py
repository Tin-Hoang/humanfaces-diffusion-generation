"""Model factory for creating different types of diffusion models."""

import torch
from diffusers import AutoencoderKL, VQModel
from typing import Tuple, Optional, Dict, Any

from diffusion_models.models.conditional.attribute_embedder import AttributeEmbedder


class ModelFactory:
    """Factory class for creating different types of diffusion models."""
    
    @staticmethod
    def create_model(config: Any) -> Tuple[torch.nn.Module, Optional[AttributeEmbedder], Optional[torch.nn.Module]]:
        """Create a model based on the configuration.
        
        Args:
            config: Configuration object containing model parameters
            
        Returns:
            Tuple of (model, attribute_embedder, vae)
        """
        model, attribute_embedder, vae = None, None, None
        
        # Unconditional models
        if config.model == "unet_notebook":
            from diffusion_models.models.unconditional.unet_notebook import create_model
            model = create_model(config)
        elif config.model == "unet_notebook_r1":
            from diffusion_models.models.unconditional.unet_notebook_r1 import create_model
            model = create_model(config)
        elif config.model == "unet_notebook_r2":
            from diffusion_models.models.unconditional.unet_notebook_r2 import create_model
            model = create_model(config)
        elif config.model == "unet_notebook_r3":
            from diffusion_models.models.unconditional.unet_notebook_r3 import create_model
            model = create_model(config)
        elif config.model == "unet_notebook_r4":
            from diffusion_models.models.unconditional.unet_notebook_r4 import create_model
            model = create_model(config)
        elif config.model == "unet_notebook_r5":
            from diffusion_models.models.unconditional.unet_notebook_r5 import create_model
            model = create_model(config)
        elif config.model == "dit_transformer_d1":
            from diffusion_models.models.unconditional.dit_transformer_d1 import create_model
            model = create_model(config)
            model.class_embedder = torch.nn.Identity()
            model.use_class_embedding = False
        elif config.model == "dit_transformer_d2":
            from diffusion_models.models.unconditional.dit_transformer_d2 import create_model
            model = create_model(config)
            model.class_embedder = torch.nn.Identity()
            model.use_class_embedding = False
            #model.gradient_checkpointing_enable()
        # Pixel Conditional models
        elif config.model in ["conditional_unet", "pc_unet_1"]:
            from diffusion_models.models.conditional.pc_unet_1 import create_model
            model = create_model(config)
            attribute_embedder = AttributeEmbedder(
                input_dim=config.num_attributes,  # 40 binary attributes
                hidden_dim=256                    # Match cross_attention_dim
            )
        
        # Latent Conditional models with AutoencoderKL
        elif config.model in ["latent_conditional_unet", "lc_unet_1"]:
            from diffusion_models.models.conditional.lc_unet_1 import create_model
            model = create_model(config)
            vae = AutoencoderKL.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            vae = vae.to(config.device)
            attribute_embedder = AttributeEmbedder(
                input_dim=config.num_attributes,  # 40 binary attributes
                hidden_dim=256                    # Match cross_attention_dim
            )
        
        # Latent Conditional models with VQModel
        elif config.model == "lc_unet_2":
            from diffusion_models.models.conditional.lc_unet_2 import create_model
            model = create_model(config)
            vae = VQModel.from_pretrained(
                "CompVis/ldm-celebahq-256",  # Use CelebA-HQ VQ-VAE
                subfolder="vqvae", 
                torch_dtype=torch.float32
            )
            vae = vae.to(config.device)
            # Freeze VAE
            vae.eval()  # Set to evaluation mode
            vae.requires_grad_(False)  # Disable gradient calculation
            attribute_embedder = AttributeEmbedder(
                input_dim=config.num_attributes,  # 40 binary attributes
                num_layers=3,
                hidden_dim=256                    # Match cross_attention_dim
            )
        
        # Latent Conditional models with VQModel (v3)
        elif config.model == "lc_unet_3_vqvae":
            from diffusion_models.models.conditional.lc_unet_3_vqvae import create_model
            model = create_model(config)
            vae = VQModel.from_pretrained(
                "CompVis/ldm-celebahq-256",  # Use CelebA-HQ VQ-VAE
                subfolder="vqvae", 
                torch_dtype=torch.float32
            )
            vae = vae.to(config.device)
            # Freeze VAE
            vae.eval()  # Set to evaluation mode
            vae.requires_grad_(False)  # Disable gradient calculation
            attribute_embedder = AttributeEmbedder(
                input_dim=config.num_attributes,  # 40 binary attributes
                num_layers=3,
                hidden_dim=256                    # Match cross_attention_dim
            )
        
        # Latent Conditional models with AutoencoderKL (v3)
        elif config.model == "lc_unet_3_vae":
            from diffusion_models.models.conditional.lc_unet_3_vae import create_model
            model = create_model(config)
            vae = AutoencoderKL.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            vae = vae.to(config.device)
            # Freeze VAE
            vae.eval()  # Set to evaluation mode
            vae.requires_grad_(False)  # Disable gradient calculation
            attribute_embedder = AttributeEmbedder(
                input_dim=config.num_attributes,  # 40 binary attributes
                num_layers=3,
                hidden_dim=256                    # Match cross_attention_dim
            )
        
        # Latent Conditional models with VQModel (v4)
        elif config.model == "lc_unet_4_vqvae":
            from diffusion_models.models.conditional.lc_unet_4_vqvae import create_model
            model = create_model(config)
            vae = VQModel.from_pretrained(
                "CompVis/ldm-celebahq-256",  # Use CelebA-HQ VQ-VAE
                subfolder="vqvae", 
                torch_dtype=torch.float32
            )
            vae = vae.to(config.device)
            # Freeze VAE
            vae.eval()  # Set to evaluation mode
            vae.requires_grad_(False)  # Disable gradient calculation
            attribute_embedder = AttributeEmbedder(
                input_dim=config.num_attributes,  # 40 binary attributes
                num_layers=3,
                hidden_dim=256                    # Match cross_attention_dim
            )
        
        else:
            raise ValueError(f"Invalid model type: {config.model}")
        
        return model, attribute_embedder, vae 