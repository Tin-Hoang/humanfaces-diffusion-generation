from datetime import datetime
import wandb
import torch

from diffusion_models.config import parse_args
from diffusion_models.datasets.dataloader import setup_dataloader
from diffusion_models.training_loop import train_loop
from diffusion_models.noise_schedulers.ddpm_scheduler import create_noise_scheduler
from ema_pytorch import EMA

from diffusers.optimization import get_cosine_schedule_with_warmup

def main():
    config = parse_args()

    print("=" * 80)
    print("Training Configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    print("=" * 80)

    if config.model == "unet_notebook":
        from diffusion_models.models.unet_notebook import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r1":
        from diffusion_models.models.unet_notebook_r1 import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r2":
        from diffusion_models.models.unet_notebook_r2 import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r3":
        from diffusion_models.models.unet_notebook_r3 import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r4":
        from diffusion_models.models.unet_notebook_r4 import create_model
        model = create_model(config)
    elif config.model == "unet_notebook_r5":
        from diffusion_models.models.unet_notebook_r5 import create_model
        model = create_model(config)
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

    train_dataloader, preprocess = setup_dataloader(
        data_dir=config.train_dir,
        batch_size=config.train_batch_size,
        image_size=config.image_size,
        shuffle=True
    )

    val_dataloader = None
    if config.val_dir:
        val_dataloader, _ = setup_dataloader(
            data_dir=config.val_dir,
            batch_size=config.eval_batch_size,
            image_size=config.image_size,
            shuffle=False
        )

    noise_scheduler = create_noise_scheduler(
        num_train_timesteps=config.num_train_timesteps
    )

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

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, val_dataloader, preprocess, ema)

if __name__ == "__main__":
    main()

