"""Single image generation tab for the diffusion model demo."""

import gradio as gr
import numpy as np
import torch
from diffusion_models.pipelines.unconditional_pipeline import load_pipeline
from diffusion_models.utils.generation import generate_images
from ui.constants import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_STEPS,
    NOISE_IMAGE_SIZE
)


def generate_random_noise():
    """Generate a random noise image."""
    noise = np.random.normal(0, 1, (NOISE_IMAGE_SIZE, NOISE_IMAGE_SIZE, 3))
    # Normalize to [0, 1] range for display
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def generate_image(noise_image, pipeline_type="ddim", num_steps=DEFAULT_NUM_STEPS, checkpoint_dir=DEFAULT_CHECKPOINT_DIR):
    """Generate an image from noise using the diffusion model."""
    # Convert normalized display image back to normal distribution
    noise = (noise_image * 2) - 1
    
    # Load pipeline
    pipeline = load_pipeline(checkpoint_dir, pipeline_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    # Generate image using the shared generate_images function
    with torch.no_grad():
        # Convert noise to tensor and add batch dimension
        noise_tensor = torch.from_numpy(noise).permute(2, 0, 1).unsqueeze(0).float()
        noise_tensor = noise_tensor.to(device)
        
        # Generate single image using the shared function
        images = generate_images(
            pipeline=pipeline,
            batch_size=1,
            device=device,
            seed=42,
            initial_noise=noise_tensor,
            num_inference_steps=num_steps
        )
        
        return images[0]  # Return the first (and only) image


def create_single_image_tab():
    """Create the single image generation tab."""
    with gr.Tab("Single Image Generation"):
        with gr.Row():
            # Left side - Input
            with gr.Column():
                noise_image = gr.Image(
                    type="numpy",
                    label="Input Noise",
                    height=NOISE_IMAGE_SIZE,
                    width=NOISE_IMAGE_SIZE
                )
                random_btn = gr.Button("Random Noise")
                
                checkpoint_dir = gr.Textbox(
                    value=DEFAULT_CHECKPOINT_DIR,
                    label="Checkpoint Directory",
                    placeholder="Enter path to checkpoint directory"
                )
                
                with gr.Row():
                    pipeline_type = gr.Radio(
                        choices=["ddpm", "ddim"],
                        value="ddim",
                        label="Pipeline Type"
                    )
                    num_steps = gr.Number(
                        value=DEFAULT_NUM_STEPS,
                        label="Number of Steps",
                        precision=0,
                        minimum=1,
                    )
                
                generate_btn = gr.Button("Generate", variant="primary")
            
            # Right side - Output
            with gr.Column():
                output_image = gr.Image(
                    type="pil",
                    label="Generated Image"
                )
        
        # Set up event handlers
        random_btn.click(
            fn=generate_random_noise,
            outputs=noise_image
        )
        
        generate_btn.click(
            fn=generate_image,
            inputs=[noise_image, pipeline_type, num_steps, checkpoint_dir],
            outputs=output_image
        )
        
        # Examples
        gr.Examples(
            examples=[
                [generate_random_noise(), "ddim", DEFAULT_NUM_STEPS, DEFAULT_CHECKPOINT_DIR],
                [generate_random_noise(), "ddpm", 1000, DEFAULT_CHECKPOINT_DIR],
            ],
            inputs=[noise_image, pipeline_type, num_steps, checkpoint_dir],
            outputs=output_image,
            fn=generate_image,
            cache_examples=True,
        ) 