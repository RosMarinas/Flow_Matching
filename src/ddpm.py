"""
Denoising Diffusion Probabilistic Model (DDPM) Implementation

Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
and "Flow Matching for Generative Modeling" (ICLR 2023)

DDPM Loss (simplified variational bound):
L = E[t, x1, ε] ||ε - ε_θ(x_t, t)||²

where:
- t ~ Uniform{1, ..., T} (discrete timestep)
- x1 is data sample
- ε ~ N(0, I) is noise
- x_t = √ᾱ_t * x1 + √(1-ᾱ_t) * ε
- ε_θ is the model's noise prediction
- ᾱ_t = ∏_{s=1}^t α_s (cumulative product)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM).

    Training: Model learns to predict noise ε_θ(x_t, t)
    Sampling: Uses reverse diffusion process (via DDIM for variable NFE)

    Args:
        model: Neural network that predicts noise ε_θ(x, t)
        num_timesteps: Number of diffusion timesteps T (default: 1000)
        beta_start: Starting beta value (default: 1e-4)
        beta_end: Ending beta value (default: 2e-2)
        beta_schedule: Type of beta schedule ('linear' or 'cosine')
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: Literal["linear", "cosine"] = "linear",
    ):
        super().__init__()

        self.model = model
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule

        # Register buffer to store beta schedule (moves with model)
        self.register_buffer("betas", self._get_betas(beta_start, beta_end))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

        # For convenience
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - self.alphas_cumprod),
        )

    def _get_betas(self, beta_start: float, beta_end: float) -> torch.Tensor:
        """
        Get beta schedule.

        Args:
            beta_start: Starting beta value
            beta_end: Ending beta value

        Returns:
            betas: Tensor of shape (num_timesteps,)
        """
        if self.beta_schedule == "linear":
            # Linear schedule from Ho et al. (2020)
            return torch.linspace(beta_start, beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal (2021)
            # Improved results for image generation
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def _cosine_beta_schedule(self) -> torch.Tensor:
        """
        Cosine beta schedule from Nichol & Dhariwal (2021).

        Returns:
            betas: Tensor of shape (num_timesteps,)
        """
        s = 0.008  # Small offset to prevent β_t from being too small near t=0

        # Compute α̅_t = cos²((t/T + s) / (1 + s) * π/2)
        steps = self.num_timesteps + 1
        t = torch.linspace(0, self.num_timesteps, steps)

        alphas_cumprod = torch.cos(((t / self.num_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2

        # Compute β_t = 1 - α̅_t / α̅_{t-1}
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        # Clip β_t for numerical stability
        return torch.clip(betas, 0.0001, 0.9999)

    def compute_loss(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Compute DDPM loss (simplified variational bound, Equation 14 in Ho et al., 2020).

        L = E[||ε - ε_θ(x_t, t)||²]

        Args:
            x1: Clean data tensor, shape (B, C, H, W) for images or (B, D) for vectors

        Returns:
            loss: Scalar tensor
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Ensure all buffers are on the correct device
        if self.betas.device != device:
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)

        # 1. Sample random timestep t ~ Uniform{1, ..., T}
        # Note: t=0 is reserved for the data distribution
        t = torch.randint(1, self.num_timesteps, (batch_size,), device=device)

        # 2. Sample noise ε ~ N(0, I)
        epsilon = torch.randn_like(x1)

        # 3. Get alpha values for current timestep
        alpha_bar_t = self.alphas_cumprod[t]

        # 4. Reshape for broadcasting
        if len(x1.shape) == 4:  # Images: (B, C, H, W)
            alpha_bar_t = alpha_bar_t.view(batch_size, 1, 1, 1)
        elif len(x1.shape) == 2:  # 2D vectors: (B, D)
            alpha_bar_t = alpha_bar_t.view(batch_size, 1)
        else:
            raise ValueError(f"Unsupported data shape: {x1.shape}")

        # 5. Compute noisy x_t = √ᾱ_t * x1 + √(1-ᾱ_t) * ε
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        if len(x1.shape) == 4:
            sqrt_alpha_bar = sqrt_alpha_bar.view(batch_size, 1, 1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(batch_size, 1, 1, 1)
        elif len(x1.shape) == 2:
            sqrt_alpha_bar = sqrt_alpha_bar.view(batch_size, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(batch_size, 1)

        x_t = sqrt_alpha_bar * x1 + sqrt_one_minus_alpha_bar * epsilon

        # 6. Predict noise with model
        # Model expects discrete timestep, convert to long tensor
        epsilon_pred = self.model(x_t, t)

        # 7. Compute MSE loss
        loss = F.mse_loss(epsilon_pred, epsilon)

        return loss

    def q_sample(
        self, x1: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x1).

        Sample from q(x_t | x1) = N(x_t | √ᾱ_t * x1, (1-ᾱ_t)I)

        Args:
            x1: Clean data, shape (B, C, H, W) or (B, D)
            t: Timestep tensor, shape (B,)
            noise: Optional pre-sampled noise (for reproducibility)

        Returns:
            x_t: Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x1)

        # Get alpha values
        alpha_bar_t = self.alphas_cumprod[t]

        # Reshape for broadcasting
        batch_size = x1.shape[0]
        if len(x1.shape) == 4:  # Images
            alpha_bar_t = alpha_bar_t.view(batch_size, 1, 1, 1)
        elif len(x1.shape) == 2:  # Vectors
            alpha_bar_t = alpha_bar_t.view(batch_size, 1)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]

        if len(x1.shape) == 4:
            sqrt_alpha_bar = sqrt_alpha_bar.view(batch_size, 1, 1, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(batch_size, 1, 1, 1)
        elif len(x1.shape) == 2:
            sqrt_alpha_bar = sqrt_alpha_bar.view(batch_size, 1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(batch_size, 1)

        x_t = sqrt_alpha_bar * x1 + sqrt_one_minus_alpha_bar * noise

        return x_t

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get cumulative product alpha_bar at timestep t.

        Args:
            t: Timestep tensor, shape (B,)

        Returns:
            alpha_bar: Tensor of shape (B,)
        """
        return self.alphas_cumprod[t]
