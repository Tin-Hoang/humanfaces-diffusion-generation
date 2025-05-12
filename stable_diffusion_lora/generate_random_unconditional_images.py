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

import os
import random
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# === Configuration ===
BASE_MODEL = "stabilityai/stable-diffusion-2-base"
LORA_WEIGHTS_PATH = "./sd_lora_unconditional_faces_output/final_model/unet_ema"  # Path to your LoRA weights
OUTPUT_DIR = "./v2_random_unconditional_images_150_"
NUM_IMAGES = 300
NUM_INFERENCE_STEPS = 150
GUIDANCE_SCALE = 1.0  # Classifier-free guidance weight; 1.0 = no additional guidance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Prepare output directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Stable Diffusion pipeline ===
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
# Inject LoRA weights into UNet
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_PATH)
pipe.to(DEVICE)

# Enable memory-efficient attention if available
if DEVICE == "cuda":
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

# === Generate images unconditionally and record mapping ===
mapping = []
for i in range(NUM_IMAGES):
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(seed)
    image = pipe(
        "",  # empty prompt = unconditional
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]
    filename = f"{i:03d}_{seed}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    mapping.append({"filename": filename, "prompt": ""})
    print(f"Saved {filename}")

# === Save filename-to-prompt mapping ===
mapping_df = pd.DataFrame(mapping)
csv_path = os.path.join(OUTPUT_DIR, "mapping.csv")
mapping_df.to_csv(csv_path, index=False)
print(f"Prompt-image mapping saved to {csv_path}")
