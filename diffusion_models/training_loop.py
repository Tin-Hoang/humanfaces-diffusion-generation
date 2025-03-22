"""Training loop implementation for diffusion models."""

import os
from pathlib import Path
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline
import wandb

from diffusion_models.utils.generation import generate_grid_images
from diffusion_models.utils.metrics import generate_and_calculate_fid


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    """Get the full repository name for Hugging Face Hub."""
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, val_dataloader, preprocess):
    """Main training loop."""
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, "optimizer"), exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    global_step = 0

    # Training loop
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if config.use_wandb:
                wandb.log({
                    "train/batch_loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch
                }, step=global_step)

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            )

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # Generate and save sample images
                _, image_grid = generate_grid_images(config, epoch, pipeline)

                # Log grid images to WandB
                if config.use_wandb:
                    wandb.log({
                        "validation/grid_images": wandb.Image(image_grid), 
                        "validation/epoch": epoch
                    })

                # Calculate FID if validation dataset is available
                if val_dataloader is not None:
                    print(f"Calculating FID score at epoch {epoch + 1}...")
                    fid_score = generate_and_calculate_fid(
                        pipeline=pipeline,
                        val_dataloader=val_dataloader,
                        device=accelerator.device,
                        preprocess=preprocess,
                        num_samples=config.val_n_samples
                    )
                    print(f"FID Score: {fid_score:.2f}")
                    
                    # Log to WandB
                    if config.use_wandb:
                        wandb.log({
                            "validation/fid_score": fid_score
                        })

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
                # Save optimizer and scaler for resuming
                torch.save({
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": accelerator.scaler.state_dict(),
                    "loss": loss.item(),
                }, os.path.join(config.output_dir, "optimizer", "optimizer.pth")) 