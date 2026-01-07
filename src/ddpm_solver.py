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

    Returns:
        samples: Generated samples, shape (num_samples, *input_shape)
    """
    from src.ddpm import DDPM

    # Create temporary DDPM instance to access beta schedule
    ddpm = DDPM(model, num_timesteps=num_timesteps)

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
        # x_0 = sqrt(alpha_prev) * (x_t - sqrt(1 - alpha_t) * ε_θ) / sqrt(alpha_t)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        # Reshape for broadcasting
        if len(x.shape) == 4:  # Images
            sqrt_alpha_t = sqrt_alpha_t.view(1, 1, 1, 1)
            sqrt_alpha_prev = sqrt_alpha_prev.view(1, 1, 1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(1, 1, 1, 1)
        elif len(x.shape) == 2:  # 2D vectors
            sqrt_alpha_t = sqrt_alpha_t.view(1, 1)
            sqrt_alpha_prev = sqrt_alpha_prev.view(1, 1)
            sqrt_one_minus_alpha_t = sqrt_one_minus_alpha_t.view(1, 1)

        x_0_pred = sqrt_alpha_prev * (x - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t

        # Compute direction pointing to x_t
        # dir_xt = sqrt(1 - alpha_prev) * ε_θ
        sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
        if len(x.shape) == 4:
            sqrt_one_minus_alpha_prev = sqrt_one_minus_alpha_prev.view(1, 1, 1, 1)
        elif len(x.shape) == 2:
            sqrt_one_minus_alpha_prev = sqrt_one_minus_alpha_prev.view(1, 1)

        dir_xt = sqrt_one_minus_alpha_prev * epsilon_pred

        # Random noise for stochasticity
        z = torch.randn_like(x) if sigma > 0 else 0

        # Update x
        # x_{t-1} = sqrt(alpha_prev) * ε_θ + sqrt(1 - alpha_prev - σ²) * ε_θ + σ * z
        # Simplified: x_{t-1} = x_0_pred + sqrt(1 - alpha_prev - σ²) * ε_θ + σ * z
        # But more standard: x_{t-1} = sqrt(alpha_prev) * x_0 + sqrt(1 - alpha_prev - σ²) * ε_θ + σ * z

        # Use DDIM formulation:
        # x_{t-1} = sqrt(alpha_prev) * x_0_pred + sqrt(1 - alpha_prev - sigma^2) * epsilon_pred + sigma * z
        x = x_0_pred + torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0)) * epsilon_pred + sigma * z

    return x


def ddpm_sample(
    model,
    num_samples: int,
    input_shape: tuple,
    device: str = "cuda",
    num_timesteps: int = 1000,
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
    ddpm = DDPM(model, num_timesteps=num_timesteps)

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
