"""Gradio UI for the diffusion model demo."""
import os
from pathlib import Path

import gradio as gr
import torch
import numpy as np
from PIL import Image
from diffusion_models.pipeline import load_pipeline
from diffusion_models.generation import generate_images, generate_images_to_dir


def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_random_noise():
    """Generate a random noise image."""
    noise = np.random.normal(0, 1, (128, 128, 3))
    # Normalize to [0, 1] range for display
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


def generate_image(noise_image, pipeline_type="ddim", num_steps=100, checkpoint_dir="checkpoints/ddpm-celebahq-128-27000train-20250316_141247"):
    """Generate an image from noise using the diffusion model."""
    # Convert normalized display image back to normal distribution
    noise = (noise_image * 2) - 1
    
    # Load pipeline
    pipeline = load_pipeline(checkpoint_dir, pipeline_type)
    device = get_device()
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


def generate_batch_images(
    num_images: int,
    pipeline_type: str = "ddim",
    num_steps: int = 100,
    checkpoint_dir: str = "checkpoints/ddpm-celebahq-128-27000train-20250316_141247",
    batch_size: int = 4,
):
    """Generate a batch of images using the diffusion model."""
    # Load pipeline
    pipeline = load_pipeline(checkpoint_dir, pipeline_type)
    device = get_device()
    pipeline = pipeline.to(device)
    
    # Create output directory
    output_dir = Path("outputs/batch_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    generate_images_to_dir(
        pipeline=pipeline,
        num_images=num_images,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        num_inference_steps=num_steps,
    )
    
    # Load and return all generated images
    images = []
    for i in range(num_images):
        img_path = output_dir / f"generated_{i:04d}.png"
        if img_path.exists():
            images.append(Image.open(img_path))
    
    return images


def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(title="Diffusion Model Demo") as demo:
        gr.Markdown("# Diffusion Model Demo")
        
        # Add device info
        device = get_device()
        gr.Markdown(f"Running on device: {device}")
        
        with gr.Tabs():
            # Single Image Generation Tab
            with gr.Tab("Single Image Generation"):
                with gr.Row():
                    # Left side - Input
                    with gr.Column():
                        noise_image = gr.Image(
                            type="numpy",
                            label="Input Noise",
                            height=128,
                            width=128
                        )
                        random_btn = gr.Button("Random Noise")
                        
                        checkpoint_dir = gr.Textbox(
                            value="checkpoints/ddpm-celebahq-128-27000train-20250316_141247",
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
                                value=100,
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
                        [generate_random_noise(), "ddim", 100, "checkpoints/ddpm-celebahq-128-27000train-20250316_141247"],
                        [generate_random_noise(), "ddpm", 1000, "checkpoints/ddpm-celebahq-128-27000train-20250316_141247"],
                    ],
                    inputs=[noise_image, pipeline_type, num_steps, checkpoint_dir],
                    outputs=output_image,
                    fn=generate_image,
                    cache_examples=True,
                )
            
            # Batch Generation Tab
            with gr.Tab("Batch Generation"):
                with gr.Row():
                    # Left side - Input
                    with gr.Column():
                        num_images = gr.Number(
                            value=4,
                            label="Number of Images",
                            precision=0,
                            minimum=1,
                        )
                        
                        batch_checkpoint_dir = gr.Textbox(
                            value="checkpoints/ddpm-celebahq-128-27000train-20250316_141247",
                            label="Checkpoint Directory",
                            placeholder="Enter path to checkpoint directory"
                        )
                        
                        with gr.Row():
                            batch_pipeline_type = gr.Radio(
                                choices=["ddpm", "ddim"],
                                value="ddim",
                                label="Pipeline Type"
                            )
                            batch_num_steps = gr.Number(
                                value=100,
                                label="Number of Steps",
                                precision=0,
                                minimum=1,
                            )
                        
                        batch_size = gr.Slider(
                            value=4,
                            label="Batch Size",
                            minimum=1,
                            maximum=64,
                            step=1
                        )
                        
                        batch_generate_btn = gr.Button("Generate Batch", variant="primary")
                    
                    # Right side - Output
                    with gr.Column():
                        output_gallery = gr.Gallery(
                            label="Generated Images",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height=400
                        )
                
                # Set up event handler
                batch_generate_btn.click(
                    fn=generate_batch_images,
                    inputs=[num_images, batch_pipeline_type, batch_num_steps, batch_checkpoint_dir, batch_size],
                    outputs=output_gallery
                )
    
    return demo


def main():
    demo = create_ui()
    demo.launch(share=True)


if __name__ == "__main__":
    main() 