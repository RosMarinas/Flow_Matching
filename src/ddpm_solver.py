"""
DDPM/DDIM Sampling Algorithms

Based on:
- DDPM: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- DDIM: "Denoising Diffusion Implicit Models" (Song et al., 2021)

DDIM enables variable NFE sampling, making it suitable for fair comparison with CFM.
"""

import torch
from typing import Optional


def ddim_sample(
    model,
    num_samples: int,
    input_shape: tuple,
    num_steps: int = 100,
    eta: float = 0.0,
    device: str = "cuda",
    num_timesteps: int = 1000,
    schedule: str = "cosine"  # <--- [FIX 1] 新增参数
) -> torch.Tensor:
    """
    DDIM sampling (Song et al., 2021).

    Advantages over DDPM:
    - Allows arbitrary number of sampling steps (not just 1000)
    - Deterministic when eta=0
    - Enables fair NFE comparison with Flow Matching

    Args:
        model: Trained DDPM model (predicts noise)
        num_samples: Number of samples to generate
        input_shape: Shape of each sample (e.g., (3, 32, 32) or (2,))
        num_steps: Number of DDIM steps (can be << 1000!)
        eta: Stochasticity (0=DDIM deterministic, 1=DDPM stochastic)
        device: torch device
        num_timesteps: Total training timesteps (default: 1000)
        schedule: Beta schedule used during training ("linear" or "cosine")

    Returns:
        samples: Generated samples, shape (num_samples, *input_shape)
    """
    from src.ddpm import DDPM

    # Create temporary DDPM instance to access beta schedule
    ddpm = DDPM(model, num_timesteps=num_timesteps, beta_schedule=schedule)
    
    # Ensure all DDPM buffers are on the correct device
    if ddpm.betas.device != device:
        ddpm.betas = ddpm.betas.to(device)
        ddpm.alphas = ddpm.alphas.to(device)
        ddpm.alphas_cumprod = ddpm.alphas_cumprod.to(device)
        ddpm.sqrt_alphas_cumprod = ddpm.sqrt_alphas_cumprod.to(device)
        ddpm.sqrt_one_minus_alphas_cumprod = ddpm.sqrt_one_minus_alphas_cumprod.to(device)

    # Start from pure noise x_T ~ N(0, I)
    x = torch.randn(num_samples, *input_shape, device=device)

    # Select timesteps (use stride for sub-sampling)
    # We want to go from T-1 to 0 in num_steps
    timesteps = torch.linspace(0, num_timesteps - 1, num_steps).long().flip(0).to(device)

    # Reverse diffusion
    for i, t in enumerate(timesteps):
        # Create batch of timesteps
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Predict noise
        with torch.no_grad():
            epsilon_pred = model(x, t_batch)

        # Get alpha values
        alpha_t = ddpm.alphas_cumprod[t]

        # Next timestep (t_prev)
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_prev = ddpm.alphas_cumprod[t_prev]
        else:
            # Last step: alpha_0 = 1.0
            alpha_prev = torch.tensor(1.0, device=device)

        # Compute sigma (stochasticity)
        # sigma = eta * sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        sigma = (
            eta
            * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        ).item()

        # Predict x_0
        # x_0 = (x_t - sqrt(1 - alpha_t) * ε_θ) / sqrt(alpha_t)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        # Reshape for broadcasting
        if len(x.shape) == 4:  # Images
            sqrt_alpha_t = sqrt_alpha_t.view(1, 1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(1, 1, 1, 1)
        elif len(x.shape) == 2:  # 2D vectors
            sqrt_alpha_t = sqrt_alpha_t.view(1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(1, 1)

        pred_x0 = (x - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t
        
        # Clip x_0 to [-1, 1] for stability (critical for good samples!)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Prepare alpha_prev terms
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        if len(x.shape) == 4:
            sqrt_alpha_prev = sqrt_alpha_prev.view(1, 1, 1, 1)
        elif len(x.shape) == 2:
            sqrt_alpha_prev = sqrt_alpha_prev.view(1, 1)

        # Compute direction pointing to x_t
        # Equation: x_{t-1} = sqrt(alpha_prev) * pred_x0 + sqrt(1 - alpha_prev - sigma^2) * epsilon_pred + sigma * z
        dir_xt_term = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0)) * epsilon_pred
        
        # Random noise for stochasticity
        z = torch.randn_like(x) if sigma > 0 else 0

        # Update x
        x = sqrt_alpha_prev * pred_x0 + dir_xt_term + sigma * z

    return x


def ddim_sample_trajectory(
    model,
    x0: torch.Tensor,
    num_steps: int = 100,
    eta: float = 0.0,
    device: str = "cuda",
    num_timesteps: int = 1000,
    schedule: str = "cosine" # <--- [FIX 1] 新增参数
) -> torch.Tensor:
    """
    DDIM sampling with trajectory return (for visualization).

    Similar to ddim_sample() but returns the full trajectory instead of just final samples.
    Used by plot_ddpm_evolution() to visualize the diffusion process.

    Args:
        model: Trained DDPM model (predicts noise)
        x0: Initial noise, shape (num_samples, *input_shape)
        num_steps: Number of DDIM steps
        eta: Stochasticity (0=DDIM deterministic, 1=DDPM stochastic)
        device: torch device
        num_timesteps: Total training timesteps (default: 1000)

    Returns:
        trajectory: Generated samples at each step, shape (num_steps+1, num_samples, *input_shape)
    """
    from src.ddpm import DDPM

    # Create temporary DDPM instance to access beta schedule
    ddpm = DDPM(model, num_timesteps=num_timesteps, beta_schedule=schedule)
    
    # Ensure all DDPM buffers are on the correct device
    if ddpm.betas.device != device:
        ddpm.betas = ddpm.betas.to(device)
        ddpm.alphas = ddpm.alphas.to(device)
        ddpm.alphas_cumprod = ddpm.alphas_cumprod.to(device)
        ddpm.sqrt_alphas_cumprod = ddpm.sqrt_alphas_cumprod.to(device)
        ddpm.sqrt_one_minus_alphas_cumprod = ddpm.sqrt_one_minus_alphas_cumprod.to(device)

    num_samples = x0.shape[0]
    x = x0.clone()

    # Store trajectory
    trajectory = [x.clone()]

    # Select timesteps (use stride for sub-sampling)
    timesteps = torch.linspace(0, num_timesteps - 1, num_steps).long().flip(0).to(device)

    # Reverse diffusion
    for i, t in enumerate(timesteps):
        # Create batch of timesteps
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Predict noise
        with torch.no_grad():
            epsilon_pred = model(x, t_batch)

        # Get alpha values
        alpha_t = ddpm.alphas_cumprod[t]

        # Next timestep (t_prev)
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_prev = ddpm.alphas_cumprod[t_prev]
        else:
            # Last step: alpha_0 = 1.0
            alpha_prev = torch.tensor(1.0, device=device)

        # Compute sigma (stochasticity)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        ).item()

        # Predict x_0
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        # Reshape for broadcasting
        if len(x.shape) == 4:  # Images
            sqrt_alpha_t = sqrt_alpha_t.view(1, 1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(1, 1, 1, 1)
        elif len(x.shape) == 2:  # 2D vectors
            sqrt_alpha_t = sqrt_alpha_t.view(1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(1, 1)

        pred_x0 = (x - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t
        
        # Clip x_0 to [-1, 1]
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # Prepare alpha_prev terms
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        if len(x.shape) == 4:
            sqrt_alpha_prev = sqrt_alpha_prev.view(1, 1, 1, 1)
        elif len(x.shape) == 2:
            sqrt_alpha_prev = sqrt_alpha_prev.view(1, 1)

        # Compute direction pointing to x_t
        dir_xt_term = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0)) * epsilon_pred

        # Random noise for stochasticity
        z = torch.randn_like(x) if sigma > 0 else 0

        # Update x
        x = sqrt_alpha_prev * pred_x0 + dir_xt_term + sigma * z

        # Store in trajectory
        trajectory.append(x.clone())

    # Stack trajectory: (num_steps+1, num_samples, *input_shape)
    trajectory = torch.stack(trajectory, dim=0)

    return trajectory


def ddpm_sample(
    model,
    num_samples: int,
    input_shape: tuple,
    device: str = "cuda",
    num_timesteps: int = 1000,
    schedule: str = "cosine" # <--- [FIX 1] 新增参数
) -> torch.Tensor:
    """
    Standard DDPM sampling (Ho et al., 2020).

    Uses full reverse diffusion process with all 1000 steps.
    NOTE: This is slow and inflexible. Use ddim_sample() instead for NFE comparison.

    Args:
        model: Trained DDPM model
        num_samples: Number of samples
        input_shape: Shape of each sample
        device: torch device
        num_timesteps: Number of diffusion timesteps

    Returns:
        samples: Generated samples
    """
    from src.ddpm import DDPM

    # Create DDPM instance
    ddpm = DDPM(model, num_timesteps=num_timesteps, beta_schedule=schedule)
    
    # Ensure all DDPM buffers are on the correct device
    if ddpm.betas.device != device:
        ddpm.betas = ddpm.betas.to(device)
        ddpm.alphas = ddpm.alphas.to(device)
        ddpm.alphas_cumprod = ddpm.alphas_cumprod.to(device)
        ddpm.sqrt_alphas_cumprod = ddpm.sqrt_alphas_cumprod.to(device)
        ddpm.sqrt_one_minus_alphas_cumprod = ddpm.sqrt_one_minus_alphas_cumprod.to(device)

    # Start from noise
    x = torch.randn(num_samples, *input_shape, device=device)

    # Reverse diffusion (from T to 0)
    for t in reversed(range(num_timesteps)):
        # Create timestep batch
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Predict noise
        with torch.no_grad():
            epsilon_pred = model(x, t_batch)

        # Get coefficients
        alpha_t = ddpm.alphas[t]
        alpha_bar_t = ddpm.alphas_cumprod[t]
        beta_t = ddpm.betas[t]

        # Compute posterior mean
        # μ_θ(x_t, t) = 1/√α_t (x_t - (β_t/√(1-ᾱ_t)) ε_θ(x_t, t))
        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)

        # Reshape for broadcasting
        if len(x.shape) == 4:
            coeff2 = coeff2.view(1, 1, 1, 1)
        elif len(x.shape) == 2:
            coeff2 = coeff2.view(1, 1)

        mean = coeff1 * (x - coeff2 * epsilon_pred)

        # Compute posterior variance
        if t > 0:
            # Var = σ²_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
            alpha_bar_t_prev = ddpm.alphas_cumprod[t - 1]
            variance = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t
            sigma = torch.sqrt(variance)
        else:
            sigma = 0.0

        # Sample x_{t-1}
        if sigma > 0:
            z = torch.randn_like(x)
            x = mean + sigma * z
        else:
            x = mean

    return x
