"""Training loop implementation for diffusion models."""

import os
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline, VQModel
import wandb
import torch.nn as nn


from diffusion_models.utils.generation import generate_grid_images, generate_grid_images_attributes
from diffusion_models.utils.metrics import generate_and_calculate_fid, generate_and_calculate_fid_attributes, generate_and_calculate_fid_attr_seg
from diffusion_models.pipelines.attribute_pipeline import AttributeDiffusionPipeline
from diffusion_models.losses.info_nce import info_nce

def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler=None,
    val_dataloader=None,
    preprocess=None,
    is_conditional=False,
    grid_attributes=None,
    val_attributes=None,
    attribute_embedder=None,
    vae=None,
    ema=None
):
    """Main training loop."""
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
            os.makedirs(os.path.join(config.output_dir, "best_model"), exist_ok=True)
        accelerator.init_trackers("train_example")

    best_fid_score = float('inf')
    best_epoch = 0

    if ema:
        ema_path = os.path.join(config.output_dir, "ema.pt")
        if os.path.exists(ema_path):
            print(f"[INFO] Loading EMA from {ema_path}")
            ema.load_state_dict(torch.load(ema_path, map_location="cpu"))

    if is_conditional and attribute_embedder is not None:
        model, optimizer, train_dataloader, lr_scheduler, attribute_embedder = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler, attribute_embedder
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
    if vae:
        vae = accelerator.prepare(vae)
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)
    if ema:
        ema.to(accelerator.device)

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        if vae:
            vae.train() if config.finetune_vae else vae.eval()

        for step, batch in enumerate(train_dataloader):
            if config.conditioning_type == "combined":
                clean_images, attributes, segmentation = batch
            elif config.conditioning_type == "attribute":
                clean_images, attributes = batch
            else:
                clean_images = batch["images"]

            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            if vae:
                with torch.set_grad_enabled(config.finetune_vae):
                    latents = vae.encode(clean_images).latent_dist.sample() if not isinstance(vae, VQModel) else vae.encode(clean_images).latents
                    latents *= vae.config.scaling_factor
                latents = latents.to(clean_images.device)
                noise = torch.randn_like(latents)
                noisy_images = noise_scheduler.add_noise(latents, noise, timesteps)
            else:
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                if is_conditional and attribute_embedder is not None:
                    attr_emb = attribute_embedder(attributes)
                    attr_emb = attr_emb.squeeze(1) if attr_emb.dim() == 3 else attr_emb

                    if config.conditioning_type in ["segmentation", "combined"]:
                        if hasattr(model, "segmentation_encoder") and model.segmentation_encoder is not None:
                            with torch.no_grad():
                                seg_input = F.interpolate(segmentation, size=(512, 512), mode='bilinear', align_corners=False)
                                if seg_input.shape[1] == 1:
                                    seg_input = seg_input.repeat(1, 3, 1, 1)

                                base_model = getattr(model.segmentation_encoder, "base_model", None)
                                if base_model is None:
                                    raise ValueError("SegFormer encoder does not have 'base_model'")

                                hidden_states = base_model(seg_input)
                                final_hidden = hidden_states[-1]
                                seg_features = F.adaptive_avg_pool2d(final_hidden, output_size=(1, 1)).squeeze(-1).squeeze(-1)
                                seg_emb = model.seg_proj(seg_features)
                                seg_emb = seg_emb.squeeze(1) if seg_emb.dim() == 3 else seg_emb

                    if config.conditioning_type == "attribute":
                        encoder_hidden_states = attr_emb.unsqueeze(1).repeat(1, 4, 1)

                    elif config.conditioning_type == "segmentation":
                        encoder_hidden_states = seg_emb.unsqueeze(1).repeat(1, 4, 1)

                    elif config.conditioning_type == "combined":
                        combined = torch.cat([attr_emb, seg_emb], dim=1)

                        # Inside the training loop, where combined = torch.cat(...) is used:
                        if not hasattr(model, "combined_proj"):
                            if isinstance(config.cross_attention_dim, list):
                                out_dim = [d for d in config.cross_attention_dim if d is not None][0]
                            else:
                                out_dim = config.cross_attention_dim
                            model.combined_proj = nn.Linear(combined.shape[-1], out_dim).to(combined.device)


                        combined = model.combined_proj(combined)
                        encoder_hidden_states = combined.unsqueeze(1).repeat(1, 4, 1)
                    else:
                        raise ValueError(f"Unsupported conditioning_type: {config.conditioning_type}")

                    # Forward pass
                    noise_pred = model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
                else:
                    # For unconditional model
                    if "dit" in config.model.lower():
                        # DiT model requires class_labels
                        dummy_class_labels = torch.zeros(noisy_images.shape[0], dtype=torch.long, device=noisy_images.device)
                        noise_pred = model(noisy_images, timesteps, class_labels=dummy_class_labels, return_dict=False)[0]
                    else:
                        # Other Unet models
                        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                if is_conditional and config.use_embedding_loss:
                    if encoder_hidden_states.dim() == 3:
                        encoder_hidden_states = encoder_hidden_states.mean(dim=1)
                    embedding_loss = info_nce(encoder_hidden_states, attributes)
                    diffusion_loss = F.mse_loss(noise_pred, noise)
                    loss = diffusion_loss + config.embedding_loss_lambda * embedding_loss
                else:
                    loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema:
                    ema.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if config.use_wandb:
                wandb.log({"train/batch_loss": logs["loss"], "train/lr": logs["lr"], "train/epoch": epoch}, step=global_step)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                if is_conditional:
                    if vae:
                        pipeline = AttributeDiffusionPipeline(
                            unet=accelerator.unwrap_model(model),
                            vae=accelerator.unwrap_model(vae),
                            scheduler=noise_scheduler,
                            attribute_embedder=accelerator.unwrap_model(attribute_embedder),
                            image_size=config.image_size
                        )
                    else:
                        raise NotImplementedError("Pixel-space conditional generation not supported yet")

                    grid_attributes = grid_attributes.to(accelerator.device)
                    _, image_grid = generate_grid_images_attributes(
                        config, epoch, pipeline, attributes=grid_attributes
                    )
                else:
                    if ema:
                        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(ema.ema_model), scheduler=noise_scheduler)
                    else:
                        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                    pipeline = pipeline.to(accelerator.device)
                    _, image_grid = generate_grid_images(config, epoch, pipeline)

                if config.use_wandb:
                    wandb.log({"validation/grid_images": wandb.Image(image_grid), "validation/epoch": epoch})

                if val_dataloader:
                    print(f"Calculating FID score at epoch {epoch + 1}...")

                    if is_conditional:
                        if config.conditioning_type == "attribute" and val_attributes is not None:
                            val_attributes = val_attributes.to(accelerator.device)
                            fid_score = generate_and_calculate_fid_attributes(
                                pipeline=pipeline,
                                val_dataloader=val_dataloader,
                                device=accelerator.device,
                                preprocess=preprocess,
                                num_train_timesteps=config.num_train_timesteps,
                                num_samples=config.val_n_samples,
                                attributes=val_attributes
                            )

                        elif config.conditioning_type == "combined" and val_attributes is not None:
                            fid_score = generate_and_calculate_fid_attr_seg(
                                config=config,
                                model=accelerator.unwrap_model(model),
                                pipeline=pipeline,
                                val_dataloader=val_dataloader,
                                val_attributes=val_attributes,
                                vae=vae,
                                attribute_embedder=attribute_embedder,
                                output_dir=os.path.join(config.output_dir, "val_images"),
                                num_samples=config.val_n_samples
                            )
                        else:
                            raise ValueError(f"[ERROR] Unsupported conditioning type or missing val_attributes: {config.conditioning_type}")
                    else:
                        fid_score = generate_and_calculate_fid(
                            pipeline=pipeline,
                            val_dataloader=val_dataloader,
                            device=accelerator.device,
                            preprocess=preprocess,
                            num_train_timesteps=config.num_train_timesteps,
                            num_samples=config.val_n_samples
                        )

                    print(f"FID Score: {fid_score:.2f}")
                    if config.use_wandb:
                        wandb.log({"validation/fid_score": fid_score})

                    if fid_score < best_fid_score:
                        best_fid_score = fid_score
                        best_epoch = epoch
                        print(f"New best FID score: {best_fid_score:.2f} at epoch {best_epoch}")
                        pipeline.save_pretrained(os.path.join(config.output_dir, "best_model"))
                        os.makedirs(os.path.join(config.output_dir, "best_model", "optimizer"), exist_ok=True)
                        torch.save({
                            "epoch": epoch,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                            "loss": loss.item(),
                            "fid_score": best_fid_score,
                        }, os.path.join(config.output_dir, "best_model", "optimizer", "optimizer.pth"))
                        if ema:
                            torch.save(ema.state_dict(), os.path.join(config.output_dir, "best_model", "ema.pt"))

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
                torch.save({
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                    "loss": loss.item(),
                }, os.path.join(config.output_dir, "optimizer", "optimizer.pth"))
                if ema:
                    torch.save(ema.state_dict(), os.path.join(config.output_dir, "ema.pt"))
