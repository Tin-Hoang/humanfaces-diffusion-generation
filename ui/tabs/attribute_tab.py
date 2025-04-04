"""Attribute-based image generation tab for the diffusion model demo."""

import gradio as gr
import torch
from diffusion_models.pipelines.attribute_pipeline import AttributeDiffusionPipeline
from diffusion_models.utils.attribute_utils import create_multi_hot_attributes
from diffusion_models.noise_schedulers.ddim_scheduler import create_ddim_scheduler
from diffusion_models.noise_schedulers.ddpm_scheduler import create_ddpm_scheduler
from ui.constants import (
    DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
    DEFAULT_NUM_STEPS,
    ATTRIBUTE_NAMES
)


def load_attribute_pipeline(checkpoint_dir: str, pipeline_type: str = "ddim"):
    """Load the attribute-based diffusion pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pipeline components
    pipeline = AttributeDiffusionPipeline.from_pretrained(checkpoint_dir)
    pipeline = pipeline.to(device)
    
    # Set the scheduler type
    if pipeline_type == "ddim":
        pipeline.scheduler = create_ddim_scheduler(num_train_timesteps=1000)
    else:
        pipeline.scheduler = create_ddpm_scheduler(num_train_timesteps=1000)
    
    return pipeline


def generate_attribute_image(
    selected_attributes: list,
    pipeline_type: str = "ddim",
    num_steps: int = DEFAULT_NUM_STEPS,
    checkpoint_dir: str = DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
):
    """Generate an image based on selected attributes."""
    # Create multi-hot attribute vector
    attributes = create_multi_hot_attributes(
        attribute_indices=selected_attributes,
        num_attributes=40,
        num_samples=1
    )
    
    # Load pipeline
    pipeline = load_attribute_pipeline(checkpoint_dir, pipeline_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate image
    with torch.no_grad():
        attributes = attributes.to(device)
        output = pipeline(
            attributes=attributes,
            num_inference_steps=num_steps,
            output_type="pil"
        )
        
        return output["sample"][0]


def create_attribute_tab():
    """Create the attribute-based generation tab."""
    with gr.Tab("Attribute-based Generation"):
        with gr.Row():
            # Left side - Input
            with gr.Column():
                # Create checkbox group for attributes
                attribute_checkboxes = gr.CheckboxGroup(
                    choices=ATTRIBUTE_NAMES,
                    label="Select Attributes",
                    info="Choose the attributes you want to include in the generated image"
                )
                
                # Add generation parameters
                attr_checkpoint_dir = gr.Textbox(
                    value=DEFAULT_ATTRIBUTE_CHECKPOINT_DIR,
                    label="Checkpoint Directory",
                    placeholder="Enter path to checkpoint directory"
                )
                
                with gr.Row():
                    attr_pipeline_type = gr.Radio(
                        choices=["ddpm", "ddim"],
                        value="ddim",
                        label="Pipeline Type"
                    )
                    attr_num_steps = gr.Number(
                        value=DEFAULT_NUM_STEPS,
                        label="Number of Steps",
                        precision=0,
                        minimum=1,
                    )
                
                attr_generate_btn = gr.Button("Generate with Attributes", variant="primary")
            
            # Right side - Output
            with gr.Column():
                attr_output_image = gr.Image(
                    type="pil",
                    label="Generated Image"
                )
        
        # Set up event handler
        def get_selected_indices(selected_attributes):
            return [ATTRIBUTE_NAMES.index(attr) for attr in selected_attributes]
        
        attr_generate_btn.click(
            fn=lambda *args: generate_attribute_image(
                get_selected_indices(args[0]),
                args[1],
                args[2],
                args[3]
            ),
            inputs=[attribute_checkboxes, attr_pipeline_type, attr_num_steps, attr_checkpoint_dir],
            outputs=attr_output_image
        ) 