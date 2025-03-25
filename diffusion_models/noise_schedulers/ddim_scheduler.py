"""DDIM scheduler for diffusion models."""

from diffusers import DDIMScheduler


def create_ddim_scheduler(
    num_train_timesteps: int = 100,
    beta_schedule: str = "scaled_linear",
    clip_sample: bool = False,
    set_alpha_to_one: bool = False,
    steps_offset: int = 1,
    prediction_type: str = "epsilon",
    timestep_spacing: str = "leading"
) -> DDIMScheduler:
    """Create a DDIM noise scheduler optimized for high-quality sampling.
    
    Args:
        num_train_timesteps: Number of training timesteps (default: 100)
        beta_schedule: The beta schedule to use (default: "scaled_linear")
        clip_sample: Whether to clip predicted sample between -1 and 1 (default: False)
        set_alpha_to_one: Whether to force the alpha parameter to 1 (default: False)
        steps_offset: Offset added to the inference steps scheduling (default: 1)
        prediction_type: Type of prediction to use ("epsilon" or "v_prediction") (default: "epsilon")
        timestep_spacing: How to space the timesteps ("leading" or "trailing") (default: "leading")
        
    Returns:
        DDIMScheduler: The configured DDIM scheduler
    """
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        clip_sample=clip_sample,
        set_alpha_to_one=set_alpha_to_one,
        steps_offset=steps_offset,
        prediction_type=prediction_type,
        timestep_spacing=timestep_spacing
    ) 