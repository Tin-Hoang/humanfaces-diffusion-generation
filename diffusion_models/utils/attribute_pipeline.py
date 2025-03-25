import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from typing import Optional, Dict, Union
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import AttributeEmbedder from models
from diffusion_models.models.attribute_embedder import AttributeEmbedder


class AttributeDiffusionPipeline(DiffusionPipeline):
    """
    A custom diffusion pipeline for generating 128x128 RGB images conditioned on 40 binary attributes.
    Uses DDIM scheduler for faster and higher quality sampling.
    
    Args:
        unet (UNet2DConditionModel): The trained UNet for denoising.
        vae (AutoencoderKL): The pretrained VAE for encoding/decoding latents.
        scheduler (DDIMScheduler): The DDIM scheduler for diffusion steps.
        attribute_embedder (AttributeEmbedder): Projection layer for multi-hot attribute vectors.
    """
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        attribute_embedder: AttributeEmbedder
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            attribute_embedder=attribute_embedder
        )
        self.vae_scale_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)  # 8 for 4 blocks

    @torch.no_grad()
    def __call__(
        self,
        attributes: torch.Tensor,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",  # "pil" or "tensor"
        return_dict: bool = True,
        decode_batch_size: int = 2,  # Process VAE decoding in smaller batches
        eta: float = 0.0,  # Parameter between 0 and 1, controlling the amount of noise to add (0 = deterministic)
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Generate images conditioned on multi-hot attribute vectors using DDIM sampling.
        
        Args:
            attributes (torch.Tensor): Multi-hot tensor of shape (batch_size, 40).
            num_inference_steps (int): Number of denoising steps.
            generator (torch.Generator, optional): Random number generator for reproducibility.
            output_type (str): "pil" for PIL images, "tensor" for raw tensors.
            return_dict (bool): Whether to return a dict with the output.
            decode_batch_size (int): Batch size for VAE decoding to manage memory.
            eta (float): Parameter between 0 and 1, controlling stochasticity (0 = deterministic DDIM).
        
        Returns:
            Dict or Tensor: Generated images in the specified format.
        """
        # Validate input
        batch_size = attributes.size(0)
        if attributes.size(1) != 40:
            raise ValueError("Attributes tensor must have shape (batch_size, 40)")

        # Move to device and dtype
        device = self.unet.device
        dtype = self.unet.dtype
        attributes = attributes.to(device, dtype)

        # Sample initial noise in latent space (64x64 due to VAE downscaling)
        latent_size = self.unet.config.sample_size  # 64
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, latent_size, latent_size),
            device=device,
            dtype=dtype,
            generator=generator
        )

        # Set timesteps for DDIM sampling
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Scale the initial noise (important for DDIM)
        latents = latents * self.scheduler.init_noise_sigma

        # Project attributes to conditioning input
        cond = self.attribute_embedder(attributes)  # (batch_size, 1, 512)
        
        # Print info and setup progress bar
        print(f"\nGenerating {batch_size} images with DDIM sampling | Attributes shape: {attributes.shape}")
        print(f"Using {num_inference_steps} inference steps, eta={eta}")
        print(f"Attribute values: {attributes[0].cpu().numpy()}")  # Print first sample's attributes
        
        # DDIM sampling loop with progress bar
        with tqdm(total=len(self.scheduler.timesteps), desc="DDIM Sampling") as pbar:
            for t in self.scheduler.timesteps:
                # Predict noise residual
                noise_pred = self.unet(latents, t, encoder_hidden_states=cond).sample
                
                # DDIM step with specified eta
                latents = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents,
                    eta=eta,
                    use_clipped_model_output=False,
                    generator=generator,
                ).prev_sample
                
                del noise_pred
                pbar.update(1)

        # Decode latents to images in smaller batches to save memory
        latents = latents / self.vae.config.scaling_factor
        
        # Process in smaller batches
        all_images = []
        for i in tqdm(range(0, batch_size, decode_batch_size), desc="VAE decoding"):
            # Get batch slice
            batch_latents = latents[i:i+decode_batch_size]
            
            # Free memory before decoding
            torch.cuda.empty_cache()
            
            # Decode batch
            batch_images = self.vae.decode(batch_latents).sample
            batch_images = (batch_images / 2 + 0.5).clamp(0, 1)  # Rescale from [-1, 1] to [0, 1]
            
            # Convert to CPU immediately to free GPU memory
            if output_type == "pil":
                for img in batch_images:
                    img_np = img.cpu().float().numpy().transpose(1, 2, 0) * 255
                    all_images.append(Image.fromarray(img_np.astype(np.uint8)))
            else:
                all_images.append(batch_images.cpu())
            
            # Free decoded tensors
            del batch_images, batch_latents
            torch.cuda.empty_cache()
        
        # Combine results if not PIL images
        if output_type != "pil":
            all_images = torch.cat(all_images, dim=0)

        # Return results
        if return_dict:
            return {"sample": all_images}
        return all_images