import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler, DDPMScheduler
from typing import Optional, Dict, Union
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusion_models.models.conditional.attribute_embedder import AttributeEmbedder


class AttributeDiffusionPipeline(DiffusionPipeline):
    """
    A custom diffusion pipeline for generating images conditioned on 40 binary attributes.
    Supports both DDPM and DDIM schedulers for sampling.
    
    Args:
        unet (UNet2DConditionModel): The trained UNet for denoising.
        vae (AutoencoderKL): The pretrained VAE for encoding/decoding latents.
        scheduler (Union[DDIMScheduler, DDPMScheduler]): The scheduler for diffusion steps.
        attribute_embedder (AttributeEmbedder): Projection layer for multi-hot attribute vectors.
        image_size (int, optional): Output image size (both height and width). Defaults to 256.
    """
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
        attribute_embedder: AttributeEmbedder,
        image_size: int = 256
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            attribute_embedder=attribute_embedder
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # 8 for 4 blocks
        self.image_size = image_size
        print(f"vae_scale_factor: {self.vae_scale_factor}")
        
        # Verify that UNet's sample size matches VAE-scaled image size
        expected_sample_size = image_size // self.vae_scale_factor
        if self.unet.config.sample_size != expected_sample_size:
            raise ValueError(
                f"UNet sample_size ({self.unet.config.sample_size}) does not match "
                f"expected size for {image_size}x{image_size} images ({expected_sample_size}). "
                f"The UNet's sample_size should be image_size/{self.vae_scale_factor} due to VAE downsampling."
            )
    
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
        Generate images conditioned on multi-hot attribute vectors using DDPM or DDIM sampling.
        
        Args:
            attributes (torch.Tensor): Multi-hot tensor of shape (batch_size, 40).
            num_inference_steps (int): Number of denoising steps.
            generator (torch.Generator, optional): Random number generator for reproducibility.
            output_type (str): "pil" for PIL images, "tensor" for raw tensors.
            return_dict (bool): Whether to return a dict with the output.
            decode_batch_size (int): Batch size for VAE decoding to manage memory.
            eta (float): Parameter between 0 and 1, controlling stochasticity (0 = deterministic DDIM).
                       Only used with DDIM scheduler.
        
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

        # Sample initial noise in latent space
        latent_size = self.unet.config.sample_size  # Should be image_size/8
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, latent_size, latent_size),
            device=device,
            dtype=dtype,
            generator=generator
        )

        # Set timesteps for sampling
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Scale the initial noise
        latents = latents * self.scheduler.init_noise_sigma

        # Project attributes to conditioning input
        cond = self.attribute_embedder(attributes)  # (batch_size, 1, 256)
        
        # Print info and setup progress bar
        scheduler_name = "DDIM" if isinstance(self.scheduler, DDIMScheduler) else "DDPM"
        print(f"\nGenerating {batch_size} {self.image_size}x{self.image_size} images with {scheduler_name} sampling")
        print(f"Using {num_inference_steps} inference steps")
        if isinstance(self.scheduler, DDIMScheduler):
            print(f"eta={eta} (stochasticity parameter)")
        print(f"Attribute values: {attributes[0].cpu().numpy()}")  # Print first sample's attributes
        
        # Sampling loop with progress bar
        with tqdm(total=len(timesteps), desc=f"{scheduler_name} Sampling") as pbar:
            for t in timesteps:
                # Ensure timestep is on the correct device
                t = t.to(device)
                
                # Predict noise residual
                noise_pred = self.unet(latents, t, encoder_hidden_states=cond).sample
                
                # Step with appropriate scheduler
                if isinstance(self.scheduler, DDIMScheduler):
                    step_output = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                        eta=eta,
                        use_clipped_model_output=False,
                        generator=generator,
                    )
                else:  # DDPM
                    step_output = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                        generator=generator,
                    )
                latents = step_output.prev_sample
                
                # Free memory
                del noise_pred, step_output
                torch.cuda.empty_cache()
                pbar.update(1)

        # Decode latents to images in smaller batches to save memory
        latents = latents / self.vae.config.scaling_factor
        target_size = (self.image_size, self.image_size)
        
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
                    # Convert to PIL and resize if needed
                    pil_image = Image.fromarray(img_np.astype(np.uint8))
                    if pil_image.size != target_size:
                        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
                    all_images.append(pil_image)
            else:
                # For tensor output, use torch interpolate if needed
                if batch_images.shape[-2:] != target_size:
                    batch_images = torch.nn.functional.interpolate(
                        batch_images,
                        size=target_size,
                        mode='bicubic',
                        align_corners=False
                    )
                    batch_images = batch_images.clamp(0, 1)  # Re-clamp after interpolation
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