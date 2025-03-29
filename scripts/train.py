"""Training script for diffusion models."""

from datetime import datetime
import wandb
import torch
from diffusers import AutoencoderKL

from diffusion_models.config import parse_args
from diffusion_models.datasets.dataloader import setup_dataloader, create_attribute_dataloader
from diffusion_models.training_loop import train_loop
from diffusion_models.noise_schedulers.ddim_scheduler import create_ddim_scheduler
from diffusion_models.noise_schedulers.ddpm_scheduler import create_ddpm_scheduler
from ema_pytorch import EMA

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusion_models.utils.attribute_utils import (
    create_sample_attributes,
    create_multi_hot_attributes
)


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
    
    # Create model and noise scheduler
    model, attribute_embedder, vae = None, None, None
    if config.model == "unet_notebook":
        from diffusion_models.models.unet_notebook import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r1":
        from diffusion_models.models.unet_notebook_r1 import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r2":
        from diffusion_models.models.unet_notebook_r2 import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r4":
        from diffusion_models.models.unet_notebook_r4 import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r5":
        from diffusion_models.models.unet_notebook_r5 import create_model
        model = create_model(config)
    elif config.model == "conditional_unet":
        from diffusion_models.models.conditional_unet import create_model
        from diffusion_models.models.attribute_embedder import AttributeEmbedder
        model = create_model(config)
        attribute_embedder = AttributeEmbedder(
            input_dim=config.num_attributes,              # 40 binary attributes
            hidden_dim=256                                # Match cross_attention_dim
        )
    elif config.model == "latent_conditional_unet":
        from diffusion_models.models.latent_conditional_unet import create_model
        from diffusion_models.models.attribute_embedder import AttributeEmbedder
        # Create model
        model = create_model(config)
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
                "stable-diffusion-v1-5/stable-diffusion-v1-5", 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
        vae = vae.to(config.device)
            
        # Create attribute embedder to project attribute vectors to conditioning dimension
        attribute_embedder = AttributeEmbedder(
            input_dim=config.num_attributes,              # 40 binary attributes
            hidden_dim=256                                # Match cross_attention_dim
        )
    elif config.model == "unet_2":
        raise NotImplementedError("Unet 2 is not implemented yet")
    elif config.model == "unet_3":
        raise NotImplementedError("Unet 3 is not implemented yet")
    else:
        raise ValueError(f"Invalid model type: {config.model}")
    
    ema = EMA(model, beta=0.9999, update_after_step=0, update_every=1) if config.use_ema else None

    if config.use_wandb:
        wandb.finish()
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.run_name,
            config=config,
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
        print("\nNo validation directory provided, skipping validation setup")
    
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
    if config.is_conditional and attribute_embedder is not None:
        # Include attribute embedder parameters in optimization
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(attribute_embedder.parameters()),
            lr=config.learning_rate,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
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

