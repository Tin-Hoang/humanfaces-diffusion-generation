"""Main application file for the diffusion model demo."""

import gradio as gr
from ui.tabs.single_image_tab import create_single_image_tab
from ui.tabs.batch_tab import create_batch_tab
from ui.tabs.attribute_tab import create_attribute_tab


def create_ui():
    """Create the main UI with all tabs."""
    with gr.Blocks(title="Diffusion Model Demo") as demo:
        gr.Markdown("# Diffusion Model Demo")
        gr.Markdown("Generate images using different diffusion models and configurations.")
        
        # Create tabs
        create_single_image_tab()
        create_batch_tab()
        create_attribute_tab()
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True) 