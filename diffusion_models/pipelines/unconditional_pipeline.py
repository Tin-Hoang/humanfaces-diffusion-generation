"""Pipeline utilities for diffusion models."""
from diffusers import DDPMPipeline, DDIMPipeline


def load_pipeline(checkpoint_path: str, pipeline_type: str = "ddpm"):
    """Load the pipeline from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        pipeline_type: Type of pipeline to use ("ddpm" or "ddim")
    
    Returns:
        Pipeline object
    """
    # Create appropriate pipeline
    if pipeline_type.lower() == "ddpm":
        pipeline = DDPMPipeline.from_pretrained(checkpoint_path)
    elif pipeline_type.lower() == "ddim":
        pipeline = DDIMPipeline.from_pretrained(checkpoint_path)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    return pipeline 