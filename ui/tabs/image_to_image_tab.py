"""Image-to-image generation with attribute conditioning tab for the diffusion model demo."""

import gradio as gr
import torch
from PIL import Image
from pathlib import Path
from typing import List, Optional

from diffusion_models.pipelines.attribute_pipeline import AttributeDiffusionPipeline
from diffusion_models.utils.generation import generate_image2image_with_attributes
from diffusion_models.utils.attribute_utils import create_multi_hot_attributes
from ui.constants import (
    ATTRIBUTE_NAMES,
    DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
    DEFAULT_NUM_STEPS
)


# Global cache for pipelines
_pipeline_cache = {}


def load_attribute_pipeline(checkpoint_dir: str, pipeline_type: str = "ddim"):
    """Load the attribute-based diffusion pipeline with caching.
    
    Args:
        checkpoint_dir: Directory containing the model checkpoints
        pipeline_type: Type of pipeline to use (ddim or ddpm)
        
    Returns:
        The loaded pipeline
    """
    # Create a cache key based on checkpoint_dir and pipeline_type
    cache_key = f"{checkpoint_dir}_{pipeline_type}"
    
    # Return cached pipeline if available
    if cache_key in _pipeline_cache:
        print(f"Using cached pipeline for {cache_key}")
        return _pipeline_cache[cache_key]
    
    # Load the pipeline if not in cache
    print(f"Loading new pipeline for {cache_key}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pipeline components
    pipeline = AttributeDiffusionPipeline.from_pretrained(checkpoint_dir)
    pipeline = pipeline.to(device)
    
    # Set the scheduler type
    if pipeline_type == "ddim":
        from diffusion_models.noise_schedulers.ddim_scheduler import create_ddim_scheduler
        pipeline.scheduler = create_ddim_scheduler(num_train_timesteps=1000)
    else:
        from diffusion_models.noise_schedulers.ddpm_scheduler import create_ddpm_scheduler
        pipeline.scheduler = create_ddpm_scheduler(num_train_timesteps=1000)
    
    # Cache the pipeline
    _pipeline_cache[cache_key] = pipeline
    
    return pipeline


def create_image_to_image_tab():
    """Create the image-to-image generation tab with attribute conditioning."""
    # Create output directory
    output_dir = Path("output/ui_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_images(
        init_image: Optional[Image.Image],
        selected_attributes: List[str],
        num_inference_steps: int,
        seed: int,
        strength: float,
        checkpoint_dir: str = DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
        pipeline_type: str = "ddim",
    ) -> List[Image.Image]:
        """Generate images conditioned on attributes and initial image.
        
        Args:
            init_image: PIL image to use as starting point
            selected_attributes: List of selected attribute names
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            strength: How much to transform the init_image (1.0 = completely transform)
            checkpoint_dir: Directory containing the model checkpoints
            pipeline_type: Type of pipeline to use (ddim or ddpm)
            
        Returns:
            List of generated images
        """
        if init_image is None:
            return []
        
        # Load pipeline
        pipeline = load_attribute_pipeline(checkpoint_dir, pipeline_type)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get indices of selected attributes
        selected_indices = [ATTRIBUTE_NAMES.index(attr) for attr in selected_attributes]
        
        # Create multi-hot attribute vector
        attributes = create_multi_hot_attributes(
            attribute_indices=selected_indices,
            num_attributes=40,
            num_samples=1
        )
        
        # Set random seed
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Image-to-image generation
        result = generate_image2image_with_attributes(
            pipeline=pipeline,
            attributes=attributes,
            init_images=[init_image],  # Pass as a list with one image
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=generator,
            output_type="pil",
            return_dict=True,
            decode_batch_size=1,
            eta=0.0,
        )
        images = result["sample"]
        
        return images
    
    # Create UI components
    with gr.Tab("Image-to-Image Generation"):
        with gr.Row():
            with gr.Column():
                # Single image upload
                init_image = gr.Image(
                    label="Initial Image (Required)",
                    type="pil",
                    height=256,
                )
                
                # Attribute checkboxes
                attribute_checkboxes = gr.CheckboxGroup(
                    choices=ATTRIBUTE_NAMES,
                    label="Select Attributes",
                    info="Choose the attributes you want to include in the generated image"
                )
                
                # Generation parameters
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="Transformation Strength (0.0 = keep original, 1.0 = completely transform)",
                )
                
                # Pipeline options
                with gr.Column():
                    pipeline_type = gr.Radio(
                        choices=["ddim", "ddpm"],
                        value="ddim",
                        label="Pipeline Type"
                    )
                    checkpoint_dir = gr.Textbox(
                        value=DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
                        label="Checkpoint Directory",
                        placeholder="Enter path to checkpoint directory"
                    )
                    num_inference_steps = gr.Slider(
                        minimum=20,
                        maximum=2000,
                        value=DEFAULT_NUM_STEPS,
                        step=10,
                        label="Number of Inference Steps",
                    )
                    seed = gr.Number(
                        value=42,
                        label="Random Seed",
                        precision=0,
                    )
                
                # Generate button
                generate_btn = gr.Button("Generate Image")
            
            with gr.Column():
                # Output gallery
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height=512,
                )
        
        # Set up event handlers
        generate_btn.click(
            fn=generate_images,
            inputs=[
                init_image,
                attribute_checkboxes,
                num_inference_steps,
                seed,
                strength,
                checkpoint_dir,
                pipeline_type,
            ],
            outputs=[gallery],
        )
    
