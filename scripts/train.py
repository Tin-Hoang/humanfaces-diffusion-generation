import wandb
import torch

from diffusion_models.config import parse_args
from diffusion_models.datasets.dataloader import setup_dataloader
from diffusion_models.training_loop import train_loop
from diffusion_models.noise_schedulers.ddpm_scheduler import create_noise_scheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


def main():
    # Parse command line arguments and get config
    config = parse_args()
    
    # Print config
    print("=" * 80)
    print("Training Configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    print("=" * 80)
    
    # Create model and noise scheduler
    if config.model == "unet_notebook":
        from diffusion_models.models.unet_notebook import create_model
        model = create_model(config)
    elif config.model == "unet_2":
        raise NotImplementedError("Unet 2 is not implemented yet")
    elif config.model == "unet_3":
        raise NotImplementedError("Unet 3 is not implemented yet")
    else:
        raise ValueError(f"Invalid model type: {config.model}")
    
    # Initialize WandB if enabled
    if config.use_wandb:
        wandb.finish()  # Finish previous if existed
        run = wandb.init(
            entity="tin-hoang",  # Shared by all members
            project="EEEM068_Diffusion_Models",
            config=config,
        )
    
    # Setup training dataset and preprocessing
    train_dataloader, preprocess = setup_dataloader(
        data_dir=config.train_dir,
        batch_size=config.train_batch_size,
        image_size=config.image_size,
        shuffle=True
    )
    
    # Setup validation dataset if val_dir is provided
    val_dataloader = None
    if config.val_dir:
        val_dataloader, _ = setup_dataloader(
            data_dir=config.val_dir,
            batch_size=config.eval_batch_size,
            image_size=config.image_size,
            shuffle=False
        )
    else:
        print("\nNo validation directory provided, skipping validation setup")
    
    noise_scheduler = create_noise_scheduler(
        num_train_timesteps=config.num_train_timesteps
    )

    # Setup optimizer and learning rate scheduler separately
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )

    # Run training loop directly with the training function
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, val_dataloader, preprocess)

if __name__ == "__main__":
    main()
