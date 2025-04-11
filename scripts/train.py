"""Training script for diffusion models."""

import wandb
import torch
from ema_pytorch import EMA
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusion_models.utils.attribute_utils import (
    create_sample_attributes,
    create_multi_hot_attributes
)

from diffusion_models.config import parse_args
from diffusion_models.datasets.dataloader import setup_dataloader, create_attribute_dataloader
from diffusion_models.training_loop import train_loop
from diffusion_models.noise_schedulers.ddim_scheduler import create_ddim_scheduler
from diffusion_models.noise_schedulers.ddpm_scheduler import create_ddpm_scheduler
from diffusion_models.models.model_factory import ModelFactory


def main():
    # Parse command line arguments and get config
    config = parse_args()

    # Set device
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {config.device}")

    # Print config
    print("=" * 80)
    print("Training Configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    print("=" * 80)

    # Create model and noise scheduler using the model factory
    model, attribute_embedder, vae = ModelFactory.create_model(config)

    ema = EMA(model, beta=0.9999, update_after_step=0, update_every=1) if config.use_ema else None

    if config.use_wandb:
        wandb.finish()
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.run_name,
            config=config,
        )
        wandb.run.log_code(
            root=".",
            include_fn=lambda path: (
                path.endswith(".py")
                or path.endswith(".ipynb")
                or path.endswith(".sh")
            ),
            exclude_fn=lambda path: (".venv" in path
                                     or ".git" in path
                                     or "checkpoints/" in path
                                     or "outputs/" in path
                                     or "data/" in path)
        )

    # Setup training dataset and preprocessing
    if config.is_conditional:
        # Use attribute dataloader for conditional training
        train_dataloader = create_attribute_dataloader(
            image_dir=config.train_dir,
            attribute_label_path=config.attribute_file,
            batch_size=config.train_batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            image_size=config.image_size
        )
        # Get preprocessing from attribute dataloader
        preprocess = train_dataloader.dataset.transform
    else:
        # Use regular dataloader for unconditional training
        train_dataloader, preprocess = setup_dataloader(
            data_source=config.train_dir,
            batch_size=config.train_batch_size,
            image_size=config.image_size,
            shuffle=True
        )

    # Setup validation dataset if val_dir is provided
    val_dataloader = None
    if config.val_dir:
        if config.is_conditional:
            val_dataloader = create_attribute_dataloader(
                image_dir=config.val_dir,
                attribute_label_path=config.attribute_file,
                batch_size=config.eval_batch_size,
                num_workers=config.num_workers,
                shuffle=False,
                image_size=config.image_size
            )
        else:
            val_dataloader, _ = setup_dataloader(
                data_source=config.val_dir,
                batch_size=config.eval_batch_size,
                image_size=config.image_size,
                shuffle=False
            )
    else:
        print("[Warning] No validation directory provided, skipping validation during training.")

    # Create noise scheduler based on config
    if config.scheduler_type == "ddim":
        noise_scheduler = create_ddim_scheduler(
            num_train_timesteps=config.num_train_timesteps
        )
        print("\nUsing DDIM scheduler for training")
    elif config.scheduler_type == "ddpm":
        noise_scheduler = create_ddpm_scheduler(
            num_train_timesteps=config.num_train_timesteps
        )
        print("\nUsing DDPM scheduler for training")
    else:
        raise ValueError(f"Invalid scheduler type: {config.scheduler_type}")

    # Setup optimizer and learning rate scheduler
    params_to_optimize = []

    # Add model parameters
    params_to_optimize.extend(model.parameters())
    # Add attribute embedder parameters if it exists
    if attribute_embedder:
        params_to_optimize.extend(attribute_embedder.parameters())
    # Add VAE parameters if it exists and finetune_vae is True
    if vae and config.finetune_vae:
        params_to_optimize.extend(vae.parameters())

    optimizer = torch.optim.AdamW(
        params=params_to_optimize,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )

    # Create attribute vectors if using conditional model
    grid_attributes = None
    val_attributes = None
    if config.is_conditional:
        # Create grid visualization attributes
        if config.grid_attribute_indices is not None:
            grid_attributes = create_multi_hot_attributes(
                attribute_indices=config.grid_attribute_indices,
                num_attributes=config.num_attributes,
                num_samples=config.grid_num_samples,
                random_remaining_indices=config.grid_sample_random_remaining_indices
            )
        else:
            # If no specific indices provided, use random combinations
            grid_attributes = create_sample_attributes(
                num_samples=config.grid_num_samples,
                num_attributes=config.num_attributes,
            )

        # Create validation attributes
        if val_dataloader is not None:
            val_attributes = create_sample_attributes(
                num_samples=config.val_n_samples,
                num_attributes=config.num_attributes
            )

    # Move attributes to device
    if val_attributes is not None:
        val_attributes = val_attributes.to(config.device)
        print("val_attributes shape: ", val_attributes.shape)
    if grid_attributes is not None:
        grid_attributes = grid_attributes.to(config.device)
        print("grid_attributes shape: ", grid_attributes.shape)
        # sample first item
        print("grid_attributes first item: ", grid_attributes[0])

    # Run training loop with attribute vectors and embedder
    train_loop(
        config=config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        lr_scheduler=lr_scheduler,
        val_dataloader=val_dataloader,
        preprocess=preprocess,
        is_conditional=config.is_conditional,
        grid_attributes=grid_attributes,
        val_attributes=val_attributes,
        attribute_embedder=attribute_embedder,
        vae=vae,
        ema=ema
    )

    # Close wandb run
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

