"""Attribute-based image generation tab for the diffusion model demo."""

import gradio as gr
import torch
from PIL import Image
from typing import List
from diffusion_models.pipelines.attribute_pipeline import AttributeDiffusionPipeline
from diffusion_models.utils.attribute_utils import create_multi_hot_attributes
from diffusion_models.noise_schedulers.ddim_scheduler import create_ddim_scheduler
from diffusion_models.noise_schedulers.ddpm_scheduler import create_ddpm_scheduler
from ui.constants import (
    DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
    DEFAULT_NUM_STEPS,
    ATTRIBUTE_NAMES
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
        pipeline.scheduler = create_ddim_scheduler(num_train_timesteps=1000)
    else:
        pipeline.scheduler = create_ddpm_scheduler(num_train_timesteps=1000)
    
    # Cache the pipeline
    _pipeline_cache[cache_key] = pipeline
    
    return pipeline


def generate_attribute_images(
    selected_attributes: list,
    num_images: int = 1,
    pipeline_type: str = "ddim",
    num_steps: int = DEFAULT_NUM_STEPS,
    checkpoint_dir: str = DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
) -> List[Image.Image]:
    """Generate images based on selected attributes.
    
    Args:
        selected_attributes: List of selected attribute names
        num_images: Number of images to generate
        pipeline_type: Type of pipeline to use (ddim or ddpm)
        num_steps: Number of denoising steps
        checkpoint_dir: Directory containing the model checkpoints
        
    Returns:
        List of generated images
    """
    # Create multi-hot attribute vector
    attributes = create_multi_hot_attributes(
        attribute_indices=selected_attributes,
        num_attributes=40,
        num_samples=num_images
    )
    
    # Load pipeline
    pipeline = load_attribute_pipeline(checkpoint_dir, pipeline_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate images
    with torch.no_grad():
        attributes = attributes.to(device)
        output = pipeline(
            attributes=attributes,
            num_inference_steps=num_steps,
            output_type="pil"
        )
        
        return output["sample"]


def create_attribute_tab():
    """Create the attribute-based generation tab."""
    with gr.Tab("Attribute-based Generation"):
        with gr.Row():
            # Left side - Input
            with gr.Column():
                # Number of images to generate
                num_images = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=4,
                    step=1,
                    label="Number of Images to Generate",
                )
                
                # Create checkbox group for attributes
                attribute_checkboxes = gr.CheckboxGroup(
                    choices=ATTRIBUTE_NAMES,
                    label="Select Attributes",
                    info="Choose the attributes you want to include in the generated images"
                )

                # Add generation parameters
                with gr.Column():
                    attr_pipeline_type = gr.Radio(
                        choices=["ddpm", "ddim"],
                        value="ddim",
                        label="Pipeline Type"
                    )
                    attr_checkpoint_dir = gr.Textbox(
                        value=DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
                        label="Checkpoint Directory",
                        placeholder="Enter path to checkpoint directory"
                    )
                
                    # Number of steps slider
                    attr_num_steps = gr.Slider(
                        minimum=20,
                        maximum=2000,
                        value=DEFAULT_NUM_STEPS,
                        step=10,
                        label="Number of Inference Steps",
                    )
                
                attr_generate_btn = gr.Button("Generate with Attributes", variant="primary")
            
            # Right side - Output
            with gr.Column():
                attr_output_gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height=512,
                )
        
        # Set up event handler
        def get_selected_indices(selected_attributes):
            return [ATTRIBUTE_NAMES.index(attr) for attr in selected_attributes]
        
        attr_generate_btn.click(
            fn=lambda *args: generate_attribute_images(
                get_selected_indices(args[0]),
                args[1],
                args[2],
                args[3],
                args[4]
            ),
            inputs=[attribute_checkboxes, num_images, attr_pipeline_type, attr_num_steps, attr_checkpoint_dir],
            outputs=attr_output_gallery
        ) 