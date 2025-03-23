"""Noise scheduler utilities for diffusion models."""

from diffusers import DDPMScheduler


def create_noise_scheduler(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    beta_schedule: str = "linear"
) -> DDPMScheduler:
    """Create a noise scheduler for the diffusion model.
    
    Args:
        num_train_timesteps: Number of timesteps for the diffusion process
        beta_start: Starting value of beta schedule
        beta_end: Ending value of beta schedule
        beta_schedule: Type of beta schedule to use
        
    Returns:
        DDPMScheduler instance
    """
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule
    ) 