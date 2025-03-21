import wandb
from accelerate import notebook_launcher

from diffusion_models.config import parse_args
from diffusion_models.datasets.dataloader import setup_dataloader
from diffusion_models.models.unet_model import create_model, create_noise_scheduler, setup_optimizer_and_scheduler
from diffusion_models.training_loop import train_loop


def main():
    # Parse command line arguments and get config
    config = parse_args()
    
    # Print config
    print("=" * 80)
    print("Training Configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    print("=" * 80)
    
    # Initialize WandB if enabled
    if config.use_wandb:
        wandb.finish()  # Finish previous if existed
        run = wandb.init(
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
    
    # Create model and noise scheduler
    model = create_model(config)
    noise_scheduler = create_noise_scheduler(config)
    
    # Setup optimizer and learning rate scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, config, train_dataloader)
    
    # Prepare arguments for training loop
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, val_dataloader, preprocess)
    
    # Launch training
    notebook_launcher(train_loop, args, num_processes=1)

if __name__ == "__main__":
    main()
