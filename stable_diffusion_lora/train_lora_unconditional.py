import os

# Redirect temp storage used by Python & huggingface_hub
os.environ["TMPDIR"] = "/scratch/dr00732/"  # ✅ create this folder if it doesn't exist

# Then configure Hugging Face as before
os.environ["HF_HOME"] = "/scratch/dr00732/"
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
os.environ["DIFFUSERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "diffusers")
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")

# Create folders if they don't exist
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)

# Set environment variables for better performance
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
import random
import wandb
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Enhanced Unconditional dataset class with stronger augmentation
class UnconditionalCelebADataset(Dataset):
    def __init__(self, image_dir: str, image_size: int = 512, augment: bool = True):
        self.image_dir = image_dir
        self.augment = augment
        
        # Base transforms
        base_transforms = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(image_size),
        ]
        
        # Add enhanced augmentation if enabled - stronger for facial data
        if augment:
            # augment_transforms = [
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.02, hue=0.01),
            #     # Add slight rotation for more variety in facial poses
            #     transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            #     # Occasional grayscale to improve robustness
            #     transforms.RandomGrayscale(p=0.02),
            # ]
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                # ↓ light geometric noise only; colour is left intact
                transforms.RandomAffine(degrees=3, translate=(0.01, 0.01), scale=(0.99, 1.01)),
            ]
            base_transforms.extend(augment_transforms)
            
        # Finalize transforms
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.transform = transforms.Compose(base_transforms)
        
        # List of images in directory
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"✅ Found {len(self.image_files)} images in directory")
        
    def __len__(self) -> int:
        return len(self.image_files)
        
    def __getitem__(self, idx: int) -> dict:
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Use empty prompt for unconditional generation
        return {
            "pixel_values": image,
            "prompt": ""  # Empty prompt for unconditional generation
        }
    
# Function to create a unique seed for each image
def make_seed(epoch: int, idx: int, extra: int = 0) -> int:
    # epoch keeps seeds unique across epochs,
    # idx distinguishes images inside the same batch,
    # extra lets you vary by guidance‑scale if you like.
    return epoch * 10_000 + extra * 1_000 + idx

# Enhanced EMA model implementation with better update strategy
class EMAModel:
    def __init__(self, model, decay=0.9999, update_after_step=0, update_every=1):
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.step = 0
        
    def update(self, model):
        self.step += 1
        
        # Only update after specified step and at specified intervals
        if self.step < self.update_after_step:
            return
            
        if self.step % self.update_every != 0:
            return
        
        # Dynamic decay - more aggressive at first, then stabilizing
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                if model_param.requires_grad:
                    ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
                
    def get_model(self):
        return self.model

# Function to create image grid for visualization
def create_image_grid(images, nrow=4):
    """Convert a list of PIL images to a PyTorch tensor grid for visualization"""
    # Convert PIL images to tensors
    tensor_images = []
    for img in images:
        img_tensor = transforms.ToTensor()(img)
        tensor_images.append(img_tensor)
    
    # Create grid
    grid = make_grid(tensor_images, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    # Convert to numpy for wandb (channels first to channels last)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    return grid_np

# Enhanced validation function with better sampling strategy
def run_validation(pipeline, num_images, current_epoch, log_to_wandb=True):
    pipeline.unet.eval()
    val_dir = os.path.join(config["output_dir"], f"validation_epoch_{current_epoch}")
    os.makedirs(val_dir, exist_ok=True)
    
    images = []
    # Add different guidance scales for comparison
    guidance_scales = [1.0, 1.5, 2.0] if current_epoch % 5 == 0 else [config["unconditional_guidance_scale"]]
    
    with torch.no_grad():
        for scale in guidance_scales:
            scale_images = []
            # for j in range(num_images // len(guidance_scales)):
            #     # Set a fixed seed for reproducible validation across epochs
            #     generator = torch.Generator(device="cuda").manual_seed(j + 1000)
            for j in range(num_images // len(guidance_scales)):
                seed = make_seed(current_epoch, j, guidance_scales.index(scale))
                generator = torch.Generator(device="cuda").manual_seed(seed)

                # Generate image with current model
                image = pipeline(
                    "",  # Empty prompt for unconditional generation
                    num_inference_steps=30,
                    guidance_scale=scale,
                    generator=generator,
                ).images[0]
                
                # Save the image
                image_path = os.path.join(val_dir, f"sample_gs{scale}_idx{j}.png")
                image.save(image_path)
                scale_images.append(image)
            images.extend(scale_images)
    
    # Log to wandb
    if log_to_wandb:
        # Create a grid of images
        #image_grid = create_image_grid(images)
        image_grid = create_image_grid(images, nrow=4)
        wandb.log({
            "validation_images": wandb.Image(image_grid, caption=f"Epoch {current_epoch}"),
            "epoch": current_epoch,
        })
    
    pipeline.unet.train()
    print(f"Validation images saved to {val_dir}")
    
    return images

# Improved training function with advanced noise scheduling
def train_one_epoch(dataloader, pipeline, optimizer, lr_scheduler, scaler, epoch, config, ema_model=None):
    pipeline.unet.train()
    epoch_loss = 0
    total_steps = len(dataloader)
    log_interval = max(1, total_steps // 20)  # Log approximately 20 times per epoch
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
    
    # Get tokenizer
    tokenizer = pipeline.tokenizer
    
    for step, batch in enumerate(progress_bar):
        # Zero gradients at the start of each step
        optimizer.zero_grad()
        
        # Get images from the batch
        #images = batch['pixel_values'].to(pipeline.device).to(torch.float16)
        images = batch["pixel_values"].to(pipeline.device, dtype=torch.float32)
        
        # Tokenize empty prompts for unconditional generation
        empty_prompts = [""] * images.shape[0]
        text_inputs = tokenizer(
            empty_prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(pipeline.device)
        
        # Use mixed precision for better memory efficiency
        with autocast(device_type='cuda', dtype=torch.float16):
            # 1. Encode images with VAE
            # with torch.no_grad():
            #     latent_images = pipeline.vae.encode(images).latent_dist.sample() * pipeline.vae.config.scaling_factor
            with torch.no_grad():
                latent_images = (
                    pipeline.vae.encode(images).latent_dist.sample()
                    * pipeline.vae.config.scaling_factor
                ).to(torch.float16)   # now switch to fp16

            # 2. Get noise and timesteps with enhanced noise schedule
            noise = torch.randn_like(latent_images)
            bsz = latent_images.shape[0]
            
            # # Improved weighted timestep sampling strategy
            # # This focuses more on challenging timesteps based on current epoch
            # if epoch < config["num_epochs"] // 3:
            #     # Early epochs: focus more on mid and late timesteps
            #     weights = torch.ones((pipeline.scheduler.config.num_train_timesteps,))
            #     mid_point = pipeline.scheduler.config.num_train_timesteps // 2
            #     weights[mid_point:] = 2.5  # Higher weight for second half of timesteps
            # elif epoch < config["num_epochs"] * 2 // 3:
            #     # Mid epochs: focus more on mid timesteps for detail learning
            #     weights = torch.ones((pipeline.scheduler.config.num_train_timesteps,))
            #     quarter = pipeline.scheduler.config.num_train_timesteps // 4
            #     weights[quarter:3*quarter] = 3.0  # Higher weight for middle half
            # else:
            #     # Late epochs: focus on early timesteps for fine details
            #     weights = torch.ones((pipeline.scheduler.config.num_train_timesteps,))
            #     quarter = pipeline.scheduler.config.num_train_timesteps // 4
            #     weights[:quarter] = 1.5  # Higher weight for early timesteps
            #     weights[quarter:2*quarter] = 2.5  # Even higher for early-mid
            
            # # Sample according to weights
            # timestep_indices = torch.multinomial(
            #     weights.to(pipeline.device), 
            #     bsz, 
            #     replacement=True
            # )
            # timesteps = timestep_indices.long()

            timesteps = torch.randint(
                0,
                pipeline.scheduler.config.num_train_timesteps,
                (bsz,),
                device=pipeline.device,
            ).long()

            
            # 3. Add noise to latents according to diffusion schedule
            noisy_latent_images = pipeline.scheduler.add_noise(latent_images, noise, timesteps)
            
            # 4. Get text encoder hidden states
            with torch.no_grad():
                encoder_hidden_states = pipeline.text_encoder(text_inputs.input_ids)[0]
            
            # 5. Predict noise with UNet
            model_pred = pipeline.unet(
                noisy_latent_images, 
                timesteps, 
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # 6. Calculate loss
            if pipeline.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif pipeline.scheduler.config.prediction_type == "v_prediction":
                target = pipeline.scheduler.get_velocity(latent_images, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {pipeline.scheduler.config.prediction_type}")
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        # Backpropagate with gradient scaling
        scaler.scale(loss).backward()
        
        # Clip gradients to avoid explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pipeline.unet.parameters(), config["grad_clip_value"])
        
        # Update parameters
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA model
        if ema_model:
            ema_model.update(pipeline.unet)
        
        # Update LR scheduler
        lr_scheduler.step()
        
        # Logging
        epoch_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": loss.item(), "lr": current_lr})
        
        # Log to wandb
        if (step + 1) % log_interval == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/step": step + epoch * total_steps,
            })
    
    # Epoch summary
    avg_epoch_loss = epoch_loss / len(dataloader)
    
    # Log epoch metrics to wandb
    wandb.log({
        "train/epoch": epoch + 1,
        "train/epoch_loss": avg_epoch_loss,
    })
    
    return avg_epoch_loss

def main():
    global config
    
    # Enhanced training configuration with optimized hyperparameters for CelebA faces
    config = {
        "model_id": "stabilityai/stable-diffusion-2-base",
        "num_epochs": 25,  # Increased to 25 epochs for better learning
        "train_batch_size": 4,
        "eval_batch_size": 2,
        "learning_rate": 3e-4,  # Increased learning rate for better convergence
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-08,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 500,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "enable_xformers_memory_efficient_attention": True,
        "seed": 42,
        "output_dir": "./sd_lora_unconditional_faces_output",
        "logging_dir": "./unconditional_faces_logs",
        "save_every_n_epochs": 1,
        "num_validation_images": 16,  # Increased validation images
        "unconditional_guidance_scale": 1.5,  # Slightly increased for faces
        "grad_clip_value": 1.0,
        "use_ema": True,
        "ema_decay": 0.9998,  # Slightly lower for quicker EMA adaptation
        "ema_update_after_step": 100,
        "augmentation": True,
        "run_name": "unconditional_stable_diffusion_lora",  # Updated for wandb
        # Enhanced LoRA parameters
        "lora_rank": 24,  # Increased from 16 to 24
        "lora_alpha": 48,   
        "lora_dropout": 0.15,  # Slightly higher dropout for regularization
        "image_dir": "/scratch/dr00732/CelebA-HQ-split/train_27000",  # Your image directory
        # Early stopping parameters
        "use_early_stopping": True,
        "early_stopping_patience": 4,  # Increased patience to allow for learning plateaus
        "early_stopping_threshold": 0.0001,
    }
    
    # Setup directories
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["logging_dir"], exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    
    # Initialize wandb with more metadata
    wandb.init(
        project="EEEM068_Diffusion_Models",
        entity="tin-hoang",
        name=config["run_name"],
        config=config,
        notes="Enhanced LoRA training on CelebA-HQ dataset with facial optimization"
    )
    
    print("Loading pipeline...")
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        config["model_id"], 
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Enable memory optimizations if available
    try:
        if config["enable_xformers_memory_efficient_attention"] and hasattr(pipe.unet, "enable_xformers_memory_efficient_attention"):
            pipe.unet.enable_xformers_memory_efficient_attention()
            print("✅ Using xformers memory efficient attention")
    except ModuleNotFoundError:
        print("⚠️ xformers not available. Continuing without memory efficient attention.")
    
    # Enable gradient checkpointing for memory efficiency
    if config["gradient_checkpointing"]:
        pipe.unet.enable_gradient_checkpointing()
        if hasattr(pipe.text_encoder, "gradient_checkpointing_enable"):
            pipe.text_encoder.gradient_checkpointing_enable()
    
    # Move pipeline to GPU
    pipe.to("cuda")
    
    print("Setting up enhanced LoRA for facial feature learning...")
    # Define LoRA configuration with improved targeting
    from peft import LoraConfig, get_peft_model
    
    # FIXED: Use proper target modules for LoRA that are actually supported by PEFT
    # The error was because you were trying to target module lists and not the actual layers
    target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",  # Attention modules
        "proj_in", "proj_out",              # Projection modules
        "conv"                              # Conv modules
    ]
    
    # Enhanced LoRA configuration specifically for facial features
    peft_config = LoraConfig(
        r=config["lora_rank"],  # Size of LoRA layers (rank) = 24
        lora_alpha=config["lora_alpha"],  # Scaling factor = 72
        target_modules=target_modules,  # Use our identified targets
        lora_dropout=config["lora_dropout"],  # 0.15
        bias="none"
    )
    
    print(f"🎯 Targeting modules: {target_modules}")
    
    # Apply LoRA to the U-Net
    lora_model = get_peft_model(pipe.unet, peft_config)
    pipe.unet = lora_model
    
    # Freeze the base model parameters, only train LoRA parameters
    for name, param in pipe.unet.named_parameters():
        if 'lora' not in name:  # Only train LoRA parameters
            param.requires_grad = False
    
    # Create the EMA model with improved parameters
    ema_model = None
    if config["use_ema"]:
        ema_model = EMAModel(
            pipe.unet, 
            decay=config["ema_decay"],
            update_after_step=config["ema_update_after_step"],
            update_every=1
        )
        print("✅ EMA model initialized")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in pipe.unet.parameters())
    print(f"Training {trainable_params:,} parameters out of {all_params:,} parameters ({trainable_params/all_params:.2%})")
    
    # Log model architecture to wandb
    wandb.watch(pipe.unet, log="all", log_freq=100)
    
    # Initialize the optimizer with better parameters
    optimizer = optim.AdamW(
        [p for p in pipe.unet.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=config["adam_weight_decay"],
        eps=config["adam_epsilon"]
    )
    
    # Create the dataset with enhanced augmentation
    dataset = UnconditionalCelebADataset(
        image_dir=config["image_dir"],
        image_size=512,  # Consistent with SD 2.0
        augment=config["augmentation"]
    )
    
    # Learning rate scheduler with warmup
    num_update_steps_per_epoch = len(dataset) // (config["train_batch_size"] * config["gradient_accumulation_steps"])
    max_train_steps = config["num_epochs"] * num_update_steps_per_epoch
    
    from transformers import get_cosine_schedule_with_warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"],
        num_training_steps=max_train_steps,
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config["train_batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Avoid batch size issues
    )
    
    # Initialize the GradScaler for mixed precision
    scaler = GradScaler()
    
    # Early stopping tracker
    best_loss = float('inf')
    no_improvement_count = 0
    
    # Training loop with improved logging and checkpointing
    global_step = 0
    
    # Run initial validation with random weights
    print("Running initial validation with random weights...")
    run_validation(pipe, config["num_validation_images"], 0)
    
    for epoch in range(config["num_epochs"]):
        epoch_start_time = datetime.now()
        
        # Train for one epoch
        avg_epoch_loss = train_one_epoch(
            dataloader, pipe, optimizer, lr_scheduler, scaler,
            epoch, config, ema_model
        )
        
        # Epoch summary
        epoch_time = datetime.now() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time}. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config["save_every_n_epochs"] == 0:
            checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # If using EMA, save its weights
            if ema_model:
                ema_unet = ema_model.get_model()
                ema_unet.save_pretrained(os.path.join(checkpoint_dir, "unet_ema"))
            
            # Save the regular model
            pipe.unet.save_pretrained(os.path.join(checkpoint_dir, "unet"))
            
            # Save optimizer and scheduler state
            torch.save({
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'scaler': scaler.state_dict(),
            }, os.path.join(checkpoint_dir, "optimizer_state.pt"))
            
            print(f"Checkpoint saved at {checkpoint_dir}")
        
        # Run validation
        # For validation, use EMA model if available
        if ema_model:
            original_unet = pipe.unet
            pipe.unet = ema_model.get_model()
        
        run_validation(
            pipe,
            config["num_validation_images"],
            epoch + 1,
            log_to_wandb=True
        )
        
        # Restore original model after validation
        if ema_model:
            pipe.unet = original_unet
        
        # Early stopping check
        if config["use_early_stopping"]:
            if avg_epoch_loss < best_loss - config["early_stopping_threshold"]:
                best_loss = avg_epoch_loss
                no_improvement_count = 0
                # Save best model
                best_model_dir = os.path.join(config["output_dir"], "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                pipe.unet.save_pretrained(os.path.join(best_model_dir, "unet"))
                if ema_model:
                    ema_model.get_model().save_pretrained(os.path.join(best_model_dir, "unet_ema"))
                print(f"✅ New best model saved! Loss: {best_loss:.6f}")
            else:
                no_improvement_count += 1
                print(f"⚠️ No improvement for {no_improvement_count} epochs. Best loss: {best_loss:.6f}")
                
                if no_improvement_count >= config["early_stopping_patience"]:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    break
    
    # Save the final model
    final_model_dir = os.path.join(config["output_dir"], "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    # Save regular and EMA models
    pipe.unet.save_pretrained(os.path.join(final_model_dir, "unet"))
    if ema_model:
        ema_model.get_model().save_pretrained(os.path.join(final_model_dir, "unet_ema"))
    
    print(f"✅ Training completed! Final model saved at {final_model_dir}")
    
    # Function to generate and save inference examples with the trained model
    def generate_examples(pipeline, output_dir, num_samples=8, guidance_scales=[1.0, 1.5, 2.0]):
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure we're in eval mode
        pipeline.unet.eval() 
        
        print(f"Generating {num_samples} example images with multiple guidance scales...")
        all_images = []
        
        with torch.no_grad():
            for scale in guidance_scales:
                scale_images = []
                for j in range(num_samples):
                    # Set a fixed seed for reproducibility
                    #generator = torch.Generator(device="cuda").manual_seed(j + 2000)
                    seed = make_seed(9999, j, guidance_scales.index(scale))  # 9999 just tags this as "final"
                    generator = torch.Generator(device="cuda").manual_seed(seed)

                    image = pipeline(
                        "",  # Empty prompt for unconditional generation
                        num_inference_steps=50,
                        guidance_scale=scale,
                        generator=generator,
                    ).images[0]
                    
                    # Save the generated image
                    image_path = os.path.join(output_dir, f"example_gs{scale}_idx{j}.png")
                    image.save(image_path)
                    scale_images.append(image)
                all_images.extend(scale_images)
        
        # Create image grid and log to wandb for each guidance scale
        for scale in guidance_scales:
            scale_imgs = [img for i, img in enumerate(all_images) if i % len(guidance_scales) == guidance_scales.index(scale)]
            if scale_imgs:
                image_grid = create_image_grid(scale_imgs, nrow=4)
                wandb.log({f"final_examples_gs{scale}": wandb.Image(image_grid, caption=f"Final Examples (GS={scale})")})
        
        # All images grid
        all_grid = create_image_grid(all_images, nrow=len(guidance_scales))
        wandb.log({"all_final_examples": wandb.Image(all_grid, caption="All Final Generated Examples")})
        
        print(f"Examples generated and saved to {output_dir}")
    
    # Use EMA model for final examples if available
    if ema_model:
        pipe.unet = ema_model.get_model()
    
    generate_examples(
        pipe,
        os.path.join(config["output_dir"], "final_examples"),
        num_samples=4,
        guidance_scales=[1.0, 1.5, 2.0, 3.0]  # Test multiple guidance scales
    )
    
    # Finish wandb run
    wandb.finish()
    
    print("✨ Enhanced LoRA training and inference completed successfully!")

if __name__ == "__main__":    
    main()