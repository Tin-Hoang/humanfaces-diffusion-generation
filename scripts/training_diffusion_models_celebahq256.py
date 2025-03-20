# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


@dataclass
class TrainingConfig:
    image_size: int = 128  # the generated image resolution
    train_batch_size: int = 16
    eval_batch_size: int = 16  # how many images to sample during evaluation
    num_epochs: int = 300
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 5
    save_model_epochs: int = 5
    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str = f"ddpm-celebahq-128-27000train-{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # the model name locally and on the HF Hub
    
    dataset_name: str = "celeba_hq_128_27000train"
    train_dir: str = "data/CelebA-HQ-split/train_27000"  # Add validation directory
    val_dir: str = "data/CelebA-HQ-split/test_300"  # Add validation directory
    val_n_samples: int = 100  # Number of samples to generate for FID calculation
    num_train_timesteps = 1000  # num_train_timesteps for DDPM scheduler and pipeline inference

    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = True  # overwrite the old model when re-running the notebook
    seed: int = 42
    use_wandb: bool = True  # Whether to use WandB logging

config = TrainingConfig()

"""## Load the dataset

You can easily load the [Smithsonian Butterflies](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) dataset with the 🤗 Datasets library:
"""

from datasets import load_dataset


# SurreyLearn Human Faces dataset
dataset = load_dataset("imagefolder", data_dir=config.train_dir, split="train")
val_dataset = load_dataset("imagefolder", data_dir=config.val_dir, split="train")

# Print the number of images in the dataset
print(f"Number of images in the training dataset: {len(dataset)}")
print(f"Number of images in the validation dataset: {len(val_dataset)}")

"""<Tip>

💡 You can find additional datasets from the [HugGan Community Event](https://huggingface.co/huggan) or you can use your own dataset by creating a local [`ImageFolder`](https://huggingface.co/docs/datasets/image_dataset#imagefolder). Set `config.dataset_name` to the repository id of the dataset if it is from the HugGan Community Event, or `imagefolder` if you're using your own images.

</Tip>

🤗 Datasets uses the [Image](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Image) feature to automatically decode the image data and load it as a [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html) which we can visualize:
"""

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for i, image in enumerate(dataset[:4]["image"]):
    axs[i].imshow(image)
    axs[i].set_axis_off()
fig.show()

"""
The images are all different sizes though, so you'll need to preprocess them first:

* `Resize` changes the image size to the one defined in `config.image_size`.
* `RandomHorizontalFlip` augments the dataset by randomly mirroring the images.
* `Normalize` is important to rescale the pixel values into a [-1, 1] range, which is what the model expects."""

from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

"""Use 🤗 Datasets' [set_transform](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.set_transform) method to apply the `preprocess` function on the fly during training:"""

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
val_dataset.set_transform(transform)

"""Feel free to visualize the images again to confirm that they've been resized. Now you're ready to wrap the dataset in a [DataLoader](https://pytorch.org/docs/stable/data#torch.utils.data.DataLoader) for training!"""

import torch

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

"""## Create a UNet2DModel

Pretrained models in 🧨 Diffusers are easily created from their model class with the parameters you want. For example, to create a [UNet2DModel](https://huggingface.co/docs/diffusers/main/en/api/models/unet2d#diffusers.UNet2DModel):
"""

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

"""It is often a good idea to quickly check the sample image shape matches the model output shape:"""

sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)

"""Great! Next, you'll need a scheduler to add some noise to the image.

## Create a scheduler

The scheduler behaves differently depending on whether you're using the model for training or inference. During inference, the scheduler generates image from the noise. During training, the scheduler takes a model output - or a sample - from a specific point in the diffusion process and applies noise to the image according to a *noise schedule* and an *update rule*.

Let's take a look at the [DDPMScheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/ddpm#diffusers.DDPMScheduler) and use the `add_noise` method to add some random noise to the `sample_image` from before:
"""

import torch
from PIL import Image
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(
    num_train_timesteps=config.num_train_timesteps,
    # beta_start=0.0001,
    # beta_end=0.02,
    # beta_schedule="linear"
)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

"""
The training objective of the model is to predict the noise added to the image. The loss at this step can be calculated by:"""

import torch.nn.functional as F

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

"""## Train the model

By now, you have most of the pieces to start training the model and all that's left is putting everything together.

First, you'll need an optimizer and a learning rate scheduler:
"""

from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

"""Then, you'll need a way to evaluate the model. For evaluation, you can use the [DDPMPipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ddpm#diffusers.DDPMPipeline) to generate a batch of sample images and save it as a grid:"""

from diffusers import DDPMPipeline
import math
import os


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    """Evaluate the model and generate sample images.
    
    Args:
        config: Training configuration
        epoch: Current epoch number
        pipeline: DDPM pipeline for generating images
        
    Returns:
        List of generated image tensors
    """
    # Generate sample images
    generator = torch.manual_seed(config.seed)
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=generator,
        num_inference_steps=config.num_train_timesteps  # Use same number of steps for inference
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    # Log evaluate image to Wandb
    if config.use_wandb:
        wandb.log(
            {"validation/grid_images": wandb.Image(image_grid), 
            "validation/epoch": epoch}
        )

    return images

"""Now you can wrap all these components together in a training loop with 🤗 Accelerate for easy TensorBoard logging, gradient accumulation, and mixed precision training. To upload the model to the Hub, write a function to get your repository name and information and then push it to the Hub.

<Tip>

💡 The training loop below may look intimidating and long, but it'll be worth it later when you launch your training in just one line of code! If you can't wait and want to start generating images, feel free to copy and run the code below. You can always come back and examine the training loop more closely later, like when you're waiting for your model to finish training. 🤗

</Tip>
"""

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def calculate_fid(pipeline, val_dataloader, device, num_samples=50):
    """Calculate FID score between generated samples and validation set.
    
    Args:
        pipeline: The DDPM pipeline for generating images
        val_dataloader: DataLoader for validation set
        device: Device to run calculation on
        num_samples: Number of samples to generate (defaults to validation set size)
    
    Returns:
        FID score
    """
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # If num_samples not specified, use validation set size
    if num_samples is None:
        num_samples = len(val_dataloader.dataset)
    
    with torch.no_grad():
        # Generate images using pipeline
        remaining_samples = num_samples
        while remaining_samples > 0:
            batch_size = min(val_dataloader.batch_size, remaining_samples)
            # Generate images using the pipeline
            samples = pipeline(
                batch_size=batch_size,
                generator=torch.manual_seed(42),
                num_inference_steps=config.num_train_timesteps  # Use same number of steps for inference
            ).images
            
            # Convert PIL images to tensors and normalize to [-1, 1]
            sample_tensors = torch.stack([
                preprocess(image) for image in samples
            ]).to(device)
            
            # Convert from [-1, 1] to [0, 1] range for FID
            sample_tensors = (sample_tensors * 0.5 + 0.5).clamp(0, 1)
            fid.update(sample_tensors, real=False)
            
            remaining_samples -= batch_size
        
        # Add validation images to FID
        for batch in val_dataloader:
            real_images = batch["images"].to(device)
            # Convert from [-1, 1] to [0, 1] range for FID
            real_images = (real_images * 0.5 + 0.5).clamp(0, 1)
            fid.update(real_images, real=True)
    
    # Calculate FID score
    fid_score = float(fid.compute())
    return fid_score


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, "optimizer"), exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
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
            # (this is the forward diffusion process)
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
                # Push train logs to wandb
                wandb.log({
                            "train/batch_loss": loss.detach().item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch
                        },
                        step=global_step)

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            )

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                # First run evaluation
                evaluate(config, epoch, pipeline)
                
                print(f"Calculating FID score at epoch {epoch + 1}...")
                fid_score = calculate_fid(
                    pipeline=pipeline,
                    val_dataloader=val_dataloader,
                    device=accelerator.device,
                    num_samples=config.val_n_samples
                )
                print(f"FID Score: {fid_score:.2f}")
                
                # Log to WandB
                if config.use_wandb:
                    wandb.log({
                            "validation/fid_score": fid_score
                        } 
                    )

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)
                # Save optimizer and scaler for resuming
                torch.save({
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": accelerator.scaler.state_dict(),
                    "loss": loss.item(),
                }, os.path.join(config.output_dir, "optimizer", "optimizer.pth"))


"""Phew, that was quite a bit of code! But you're finally ready to launch the training with 🤗 Accelerate's [notebook_launcher](https://huggingface.co/docs/accelerate/main/en/package_reference/launchers#accelerate.notebook_launcher) function. Pass the function the training loop, all the training arguments, and the number of processes (you can change this value to the number of GPUs available to you) to use for training:"""

if config.use_wandb:
    # Init Wandb logger
    import wandb

    wandb.finish()  # Finish previous if existed

    run = wandb.init(
        # Set the project where this run will be logged
        project="EEEM068_Diffusion_Models",
        # Track hyperparameters and run metadata
        config=config,
    )

from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
