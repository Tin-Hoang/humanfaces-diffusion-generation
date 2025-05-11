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

import random
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# === Configuration ===
BASE_MODEL = "stabilityai/stable-diffusion-2-base"
LORA_WEIGHTS_PATH = "./sd_lora_conditional_faces_output/best_model/unet_ema"
ATTRIBUTE_LABEL_PATH = "/scratch/dr00732/CelebA-HQ-split/CelebAMask-HQ-attribute-anno.txt"
OUTPUT_DIR = "./random_conditional_images_50_inference_steps"
NUM_IMAGES = 300
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Helper: convert attribute row to prompt ===
def attributes_to_prompt(row, attribute_names):
    active = [name.replace('_', ' ').lower() for name in attribute_names if row[name] == 1]
    if active:
        return "A photo of a person with " + ", ".join(active)
    else:
        return "A photo of a person"

# === Load attributes ===
# Read header
with open(ATTRIBUTE_LABEL_PATH, 'r') as f:
    f.readline()  # skip count
    header = f.readline().strip().split()
# Load data
df = pd.read_csv(ATTRIBUTE_LABEL_PATH, skiprows=2, sep=r"\s+", header=None)
df.columns = ['image_id'] + header
# Normalize image_id and attributes
df['image_id'] = df['image_id'].astype(str).apply(lambda x: f"{x}.jpg" if not x.endswith('.jpg') else x)
for col in header:
    df[col] = df[col].map({-1: 0, 1: 1})
# Sample 300 rows
df_sample = df.sample(n=NUM_IMAGES, random_state=42).reset_index(drop=True)

# Pre-generate prompts
prompts = [attributes_to_prompt(row, header) for _, row in df_sample.iterrows()]

# === Prepare output directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load pipeline ===
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None, requires_safety_checker=False
)
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_PATH)
pipe.to(DEVICE)
if DEVICE == "cuda":
    try: pipe.enable_xformers_memory_efficient_attention()
    except: pass

# === Generate images and record mapping ===
mapping = []  # to store dicts {filename, prompt}
for i, prompt in enumerate(prompts):
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(DEVICE).manual_seed(seed)
    image = pipe(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]
    filename = f"{i:03d}_{seed}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    mapping.append({"filename": filename, "prompt": prompt})
    print(f"Saved {filename} → \"{prompt}\"")

# === Save mapping to CSV ===
mapping_df = pd.DataFrame(mapping)
csv_path = os.path.join(OUTPUT_DIR, "mapping.csv")
mapping_df.to_csv(csv_path, index=False)
print(f"Prompt-image mapping saved to {csv_path}")
