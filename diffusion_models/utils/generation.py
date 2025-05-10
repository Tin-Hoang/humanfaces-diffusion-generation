import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDIMScheduler, DDPMScheduler,DiTTransformer2DModel
from typing import Optional, Dict, Union
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import os

from diffusion_models.models.conditional.attribute_embedder import AttributeEmbedder
from diffusion_models.config import TrainingConfig
from diffusers import DDPMPipeline
from torchvision.utils import make_grid, save_image
from diffusion_models.datasets.attribute_dataset import AttributeDataset


def make_pil_grid(images, rows, cols):
    """Create a grid of PIL images."""
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


class AttributeDiffusionPipeline(DiffusionPipeline):
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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_size = image_size

        expected_sample_size = image_size // self.vae_scale_factor
        if self.unet.config.sample_size != expected_sample_size:
            raise ValueError(
                f"UNet sample_size ({self.unet.config.sample_size}) does not match expected size "
                f"for {image_size}x{image_size} images ({expected_sample_size})."
            )

    @torch.no_grad()
    def __call__(
        self,
        attributes: torch.Tensor,
        segmentation: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        decode_batch_size: int = 2,
        eta: float = 0.0
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        unet_training = self.unet.training
        vae_training = self.vae.training
        embedder_training = self.attribute_embedder.training

        self.unet.eval()
        self.vae.eval()
        self.attribute_embedder.eval()

        try:
            batch_size = attributes.size(0)
            if attributes.size(1) != 40:
                raise ValueError("Attributes tensor must have shape (batch_size, 40)")

            device = self.unet.device
            dtype = self.unet.dtype
            attributes = attributes.to(device, dtype)

            if generator is None:
                generator = torch.Generator(device=device).manual_seed(42)

            latent_size = self.unet.config.sample_size
            latents = torch.randn(
                (batch_size, self.unet.config.in_channels, latent_size, latent_size),
                device=device,
                dtype=dtype,
                generator=generator
            )

            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            latents = latents * self.scheduler.init_noise_sigma

            cond_attr = self.attribute_embedder(attributes)
            assert cond_attr.shape[-1] == 128, f"[DEBUG] Attribute embedding dim must be 128, got {cond_attr.shape[-1]}"

            if segmentation is not None:
                if segmentation.dim() != 4:
                    raise ValueError("Segmentation tensor must have shape (B, C, H, W)")
                segmentation = segmentation.to(device, dtype)
                if segmentation.shape[1] == 1:
                    segmentation = segmentation.repeat(1, 3, 1, 1)

                seg_input = F.interpolate(segmentation, size=(512, 512), mode='bilinear', align_corners=False)
                base_model = getattr(self.unet.segmentation_encoder, "base_model", None)
                if base_model is None:
                    raise ValueError("UNet is missing segmentation_encoder.base_model")

                encoder_output = base_model(seg_input)
                seg_features = encoder_output.last_hidden_state.mean(dim=1)
                assert seg_features.shape[1] == self.unet.seg_proj.in_features, \
                    f"Expected seg_features shape [B, {self.unet.seg_proj.in_features}], got {seg_features.shape}"
                seg_embedding = self.unet.seg_proj(seg_features).unsqueeze(1)
                assert seg_embedding.shape[-1] == 128, f"[DEBUG] seg_embedding dim must be 128, got {seg_embedding.shape[-1]}"
                cond = torch.cat([cond_attr, seg_embedding], dim=-1)
            else:
                padding = torch.zeros(cond_attr.shape[0], 1, 128, device=device, dtype=dtype)
                cond = torch.cat([cond_attr, padding], dim=-1)

            cond = cond.repeat(1, 4, 1)
            assert cond.shape == (batch_size, 4, 256), f"Final cond shape mismatch: {cond.shape}"

            scheduler_name = "DDIM" if isinstance(self.scheduler, DDIMScheduler) else "DDPM"
            print(f"\nGenerating {batch_size} {self.image_size}x{self.image_size} images with {scheduler_name} sampling")
            print(f"Using {num_inference_steps} inference steps")
            if isinstance(self.scheduler, DDIMScheduler):
                print(f"eta={eta} (stochasticity parameter)")
            print(f"Attribute values: {attributes[0].cpu().numpy()}")

            with tqdm(total=len(timesteps), desc=f"{scheduler_name} Sampling") as pbar:
                for t in timesteps:
                    t = t.to(device)
                    noise_pred = self.unet(latents, t, encoder_hidden_states=cond).sample
                    step_output = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                        eta=eta if isinstance(self.scheduler, DDIMScheduler) else None,
                        generator=generator
                    )
                    latents = step_output.prev_sample
                    del noise_pred, step_output
                    torch.cuda.empty_cache()
                    pbar.update(1)

            latents = latents / self.vae.config.scaling_factor
            target_size = (self.image_size, self.image_size)

            all_images = []
            for i in tqdm(range(0, batch_size, decode_batch_size), desc="VAE decoding"):
                batch_latents = latents[i:i+decode_batch_size]
                batch_images = self.vae.decode(batch_latents).sample
                batch_images = (batch_images / 2 + 0.5).clamp(0, 1)

                if output_type == "pil":
                    for img in batch_images:
                        img_np = img.cpu().float().numpy().transpose(1, 2, 0) * 255
                        pil_image = Image.fromarray(img_np.astype(np.uint8))
                        if pil_image.size != target_size:
                            pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
                        all_images.append(pil_image)
                else:
                    if batch_images.shape[-2:] != target_size:
                        batch_images = torch.nn.functional.interpolate(
                            batch_images,
                            size=target_size,
                            mode='bicubic',
                            align_corners=False
                        )
                    batch_images = batch_images.clamp(0, 1)
                    all_images.append(batch_images.cpu())

            if output_type != "pil":
                all_images = torch.cat(all_images, dim=0)

            if return_dict:
                return {"sample": all_images}
            return all_images

        finally:
            self.unet.train(unet_training)
            self.vae.train(vae_training)
            self.attribute_embedder.train(embedder_training)


def generate_images_to_dir(
    pipeline,
    num_images: int,
    output_dir: Path,
    batch_size: int = 4,
    device: str = "cuda",
    seed: int = 42,
    num_inference_steps: int = 1000,
):
    """Generate multiple batches of images and save them to directory.

    Args:
        pipeline: The diffusion pipeline
        num_images: Number of images to generate
        output_dir: Directory to save generated images
        batch_size: Batch size for generation
        device: Device to use for generation
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate images in batches
    remaining_images = num_images
    image_idx = 0

    while remaining_images > 0:
        curr_batch_size = min(batch_size, remaining_images)

        # Generate images
        images = generate_images(
            pipeline=pipeline,
            batch_size=curr_batch_size,
            device=device,
            seed=seed + image_idx,
            num_inference_steps=num_inference_steps,
        )

        # Save images
        for img in images:
            img.save(output_dir / f"generated_{image_idx:04d}.png")
            image_idx += 1

        remaining_images -= curr_batch_size
        print(f"Generated {image_idx} of {num_images} images")



def generate_grid_images(config: TrainingConfig, epoch: int, pipeline: DDPMPipeline):
    """Generate and save a grid of sample images.

    Args:
        config: Training configuration
        epoch: Current epoch number
        pipeline: DDPM pipeline for generating images

    Returns:
        Tuple of (list of generated images, grid image)
    """
    # Set the random seed for reproducibility
    generator = torch.manual_seed(config.seed)

    # Check if the model is a DiT model
    if isinstance(pipeline.unet, DiTTransformer2DModel):
        # Wrap the UNet forward method to handle timestep and class labels for DiT
        original_forward = pipeline.unet.forward

        def wrapped_forward(sample, timestep, **kwargs):
            # Ensure timestep is a 1D tensor: if it's a scalar, expand it for the batch.
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).repeat(sample.shape[0])
            timestep = timestep.to(sample.device)
            # Inject dummy class labels if they aren’t provided
            if 'class_labels' not in kwargs:
                dummy_class_labels = torch.zeros(sample.shape[0], dtype=torch.long, device=sample.device)
                kwargs['class_labels'] = dummy_class_labels
            return original_forward(sample, timestep, **kwargs)

        # Replace the forward method in the pipeline’s UNet with our wrapped version.
        pipeline.unet.forward = wrapped_forward

    # Generate images using the pipeline
    output = pipeline(
        batch_size=config.eval_batch_size,
        generator=generator,
        num_inference_steps=config.num_train_timesteps
    )

    # Retrieve generated images from the output
    if hasattr(output, "images"):
        images = output.images
    elif isinstance(output, tuple):
        images = output[0]
    else:
        images = output

    # Create an image grid from the generated images
    image_grid = make_pil_grid(images, rows=4, cols=4)

    # Save the grid image to disk
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    return images, image_grid


def generate_grid_images_attributes(config, epoch, pipeline, attributes, segmentation: Optional[torch.Tensor] = None):
    """
    Generate and save a grid of sample images conditioned on attributes (+ segmentation if applicable).
    
    Args:
        config: Training configuration.
        epoch: Current epoch number.
        pipeline: The diffusion pipeline (e.g., AttributeDiffusionPipeline).
        attributes (torch.Tensor): Attribute tensor [B, 40].
        segmentation (Optional[torch.Tensor]): Optional segmentation tensor [B, 1 or 3, H, W].
    
    Returns:
        output_path: Path to saved grid image.
        image_grid: The generated image grid as a tensor.
    """
    pipeline = pipeline.to(config.device)
    attributes = attributes.to(config.device)

    if segmentation is not None:
        segmentation = segmentation.to(config.device)

    print(f"[INFO] Generating sample grid at epoch {epoch} with conditioning_type={config.conditioning_type}")
    device_str = "cuda" if "cuda" in str(config.device) else str(config.device)
    generator = torch.Generator(device=device_str).manual_seed(config.seed)

    with torch.no_grad():
        outputs = pipeline(
            attributes=attributes,
            segmentation=segmentation,
            num_inference_steps=config.num_train_timesteps,
            generator=generator,
            output_type="tensor",
            return_dict=True,
        )
        images = outputs["sample"]  # [B, 3, H, W]

    image_grid = make_grid(images, nrow=int(np.sqrt(images.shape[0])), normalize=True, scale_each=True)
    output_path = os.path.join(config.output_dir, f"grid_epoch_{epoch}.png")
    save_image(image_grid, output_path)

    return output_path, image_grid


def generate_images_from_attributes(
    pipeline: AttributeDiffusionPipeline,
    dataset: AttributeDataset,
    output_dir: Path,
    batch_size: int = 4,
    device: str = "cuda",
    seed: int = 42,
    num_inference_steps: int = 1000
):
    """Generate images from attributes and save them with the same ID as input images.

    Args:
        pipeline: The attribute diffusion pipeline
        dataset: The dataset containing images and their attributes
        output_dir: Directory to save generated images
        batch_size: Batch size for generation
        device: Device to use for generation
        seed: Random seed for reproducibility
        num_inference_steps: Number of denoising steps
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Move pipeline to device
    pipeline = pipeline.to(device)

    # Setup dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Get image_ids from dataset
    image_ids = dataset.attributes_df['image_id'].tolist()

    # Generate images batch by batch
    generated_count = 0

    with torch.no_grad():
        for batch_idx, (_, attributes) in enumerate(tqdm(dataloader, desc="Generating images")):
            # Get current batch image_ids
            batch_image_ids = image_ids[batch_idx*batch_size:batch_idx*batch_size + len(attributes)]

            # Move attributes to device
            attributes = attributes.to(device)

            # Set seed for reproducibility for this batch
            batch_seed = seed + batch_idx
            generator = torch.Generator(device=device).manual_seed(batch_seed)

            # Generate images based on attributes
            output = pipeline(
                attributes=attributes,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil"
            )

            generated_images = output["sample"]

            # Save images with the same ID as input
            for img, img_id in zip(generated_images, batch_image_ids):
                # Generate output filename using the same ID
                output_filename = os.path.splitext(img_id)[0] + ".png"
                img.save(output_dir / output_filename)
                generated_count += 1

    print(f"Generated {generated_count} images in {output_dir}")
