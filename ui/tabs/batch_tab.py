"""Batch image generation tab for the diffusion model demo."""

import gradio as gr
from pathlib import Path
from PIL import Image
import torch
from diffusion_models.pipelines.unconditional_pipeline import load_pipeline
from diffusion_models.utils.generation import generate_images_to_dir
from ui.constants import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_NUM_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_IMAGES,
    GALLERY_COLUMNS,
    GALLERY_ROWS,
    GALLERY_HEIGHT
)


def generate_batch_images(
    num_images: int = DEFAULT_NUM_IMAGES,
    pipeline_type: str = "ddim",
    num_steps: int = DEFAULT_NUM_STEPS,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Generate a batch of images using the diffusion model."""
    # Load pipeline
    pipeline = load_pipeline(checkpoint_dir, pipeline_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def create_batch_tab():
    """Create the batch generation tab."""
    with gr.Tab("Batch Generation"):
        with gr.Row():
            # Left side - Input
            with gr.Column():
                num_images = gr.Number(
                    value=DEFAULT_NUM_IMAGES,
                    label="Number of Images",
                    precision=0,
                    minimum=1,
                )
                
                batch_checkpoint_dir = gr.Textbox(
                    value=DEFAULT_CHECKPOINT_DIR,
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
                        value=DEFAULT_NUM_STEPS,
                        label="Number of Steps",
                        precision=0,
                        minimum=1,
                    )
                
                batch_size = gr.Slider(
                    value=DEFAULT_BATCH_SIZE,
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
                    columns=GALLERY_COLUMNS,
                    rows=GALLERY_ROWS,
                    height=GALLERY_HEIGHT
                )
        
        # Set up event handler
        batch_generate_btn.click(
            fn=generate_batch_images,
            inputs=[num_images, batch_pipeline_type, batch_num_steps, batch_checkpoint_dir, batch_size],
            outputs=output_gallery
        ) 