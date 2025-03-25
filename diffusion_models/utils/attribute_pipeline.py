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
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Generate images conditioned on multi-hot attribute vectors.
        
        Args:
            attributes (torch.Tensor): Multi-hot tensor of shape (batch_size, 40).
            num_inference_steps (int): Number of denoising steps.
            generator (torch.Generator, optional): Random number generator for reproducibility.
            output_type (str): "pil" for PIL images, "tensor" for raw tensors.
            return_dict (bool): Whether to return a dict with the output.
        
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
                if t % 100 == 0:  # Check every 100 steps
                    print(f"Step {t}, Memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

        # Decode latents to images
        print("Decoding latents to images...")
        latents = latents / self.vae.config.scaling_factor
        print(f"Before VAE decode: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        images = self.vae.decode(latents).sample  # (batch_size, 3, 128, 128)
        print(f"After VAE decode: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        images = (images / 2 + 0.5).clamp(0, 1)  # Rescale from [-1, 1] to [0, 1]

        # Convert to desired output type
        if output_type == "pil":
            from PIL import Image
            import numpy as np
            images_pil = []
            for img in images:
                img_np = img.cpu().float().numpy().transpose(1, 2, 0) * 255
                images_pil.append(Image.fromarray(img_np.astype(np.uint8)))
            images = images_pil

        # Return results
        if return_dict:
            return {"sample": images}
        return images