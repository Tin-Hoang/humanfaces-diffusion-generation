import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torch import nn
from typing import Optional, Dict, Union


class AttributeDiffusionPipeline(DiffusionPipeline):
    """
    A custom diffusion pipeline for generating 128x128 RGB images conditioned on 40 binary attributes.
    
    Args:
        unet (UNet2DConditionModel): The trained UNet for denoising.
        vae (AutoencoderKL): The pretrained VAE for encoding/decoding latents.
        scheduler (DDPMScheduler): The noise scheduler for diffusion steps.
        attribute_proj (nn.Module): Projection layer for multi-hot attribute vectors.
    """
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDPMScheduler,
        attribute_proj: nn.Module
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            attribute_proj=attribute_proj
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
        decode_batch_size: int = 2  # Process VAE decoding in smaller batches
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Generate images conditioned on multi-hot attribute vectors.
        
        Args:
            attributes (torch.Tensor): Multi-hot tensor of shape (batch_size, 40).
            num_inference_steps (int): Number of denoising steps.
            generator (torch.Generator, optional): Random number generator for reproducibility.
            output_type (str): "pil" for PIL images, "tensor" for raw tensors.
            return_dict (bool): Whether to return a dict with the output.
            decode_batch_size (int): Batch size for VAE decoding to manage memory.
        
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

        # Project attributes to conditioning input
        cond = self.attribute_proj(attributes)  # (batch_size, 1, 512)

        # Set timesteps for denoising
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Print info and setup progress bar
        print(f"\nGenerating {batch_size} images | Attributes shape: {attributes.shape}")
        
        # Denoising loop with progress bar
        from tqdm import tqdm
        with tqdm(total=len(self.scheduler.timesteps), desc="Denoising") as pbar:
            for t in self.scheduler.timesteps:
                # Predict noise
                noise_pred = self.unet(latents, t, encoder_hidden_states=cond).sample
                # Update latents
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
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
                from PIL import Image
                import numpy as np
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