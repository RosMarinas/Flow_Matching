"""
Probability Paths for Flow Matching

Implements different probability paths for Conditional Flow Matching:
- OT (Optimal Transport): Straight line paths (Equation 20-22)
- VP (Variance Preserving): DDPM-style diffusion paths (Equation 18)
- VE (Variance Exploding): SDE-style diffusion paths (Equation 16)
"""

import torch
from typing import Callable, Literal, Tuple


def ot_path(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma_min: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimal Transport (OT) conditional path.

    This produces straight-line trajectories from noise to data,
    which is the key innovation of Flow Matching.

    Equation 20-22 from paper:
    - ψ_t(x0) = (1 - (1-σ_min)t)x0 + t*x1
    - u_t = x1 - (1-σ_min)x0

    Args:
        x0: Source noise ~ N(0, I), shape (B, ...)
        x1: Target data, shape (B, ...)
        t: Time tensor in [0, 1], shape (B, 1, ...)
        sigma_min: Minimum noise level (prevents singularity at t=1)

    Returns:
        psi_t: Intermediate state at time t, shape (B, ...)
        target: Target vector field u_t, shape (B, ...)
    """
    # Time-dependent coefficient for noise
    sigma_t = 1 - (1 - sigma_min) * t

    # Conditional path: linear interpolation
    psi_t = sigma_t * x0 + t * x1

    # Target vector field (constant over time for OT path!)
    target = x1 - (1 - sigma_min) * x0

    return psi_t, target


def vp_path(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma_min: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variance Preserving (VP) diffusion path.

    This matches the DDPM diffusion schedule (with cosine schedule),
    producing curved trajectories from noise to data.

    Path (t=0: noise, t=1: data):
    - ψ_t(x0) = sin(t * π/2) * x1 + cos(t * π/2) * x0
    - u_t = d/dt ψ_t = π/2 * (cos(t * π/2) * x1 - sin(t * π/2) * x0)

    Args:
        x0: Source noise ~ N(0, I), shape (B, ...)
        x1: Target data, shape (B, ...)
        t: Time tensor in [0, 1], shape (B, 1, ...)
        sigma_min: Minimum noise level (not used here, for consistency)

    Returns:
        psi_t: Intermediate state at time t
        target: Target vector field u_t
    """
    sin_t = torch.sin(t * torch.pi / 2)
    cos_t = torch.cos(t * torch.pi / 2)

    # Conditional path (Variance Preserving: sin^2 + cos^2 = 1)
    psi_t = sin_t * x1 + cos_t * x0

    # Target vector field (Time derivative)
    target = (torch.pi / 2) * (cos_t * x1 - sin_t * x0)

    return psi_t, target


def ve_path(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma_min: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variance Exploding (VE) diffusion path.

    Goes from large noise at t=0 to data at t=1.

    Path:
    - σ_t = exp((1 - t) * log(sigma_max) + t * log(sigma_min))
    - ψ_t(x0) = x1 + σ_t * x0
    - u_t = d/dt ψ_t = σ_t * (log(sigma_min) - log(sigma_max)) * x0

    Args:
        x0: Source noise ~ N(0, I), shape (B, ...)
        x1: Target data, shape (B, ...)
        t: Time tensor in [0, 1], shape (B, 1, ...)
        sigma_min: Minimum noise level (default 1e-4)

    Returns:
        psi_t: Intermediate state at time t
        target: Target vector field u_t
    """
    sigma_max = 10.0
    # Use the same device and dtype as input
    log_sigma_min = torch.log(torch.tensor(sigma_min, device=x0.device, dtype=x0.dtype))
    log_sigma_max = torch.log(torch.tensor(sigma_max, device=x0.device, dtype=x0.dtype))

    # Time-dependent noise level (from sigma_max to sigma_min)
    log_sigma_t = (1 - t) * log_sigma_max + t * log_sigma_min
    sigma_t = torch.exp(log_sigma_t)

    # Conditional path
    psi_t = x1 + sigma_t * x0

    # Target vector field (Time derivative)
    target = sigma_t * (log_sigma_min - log_sigma_max) * x0

    return psi_t, target


def get_conditional_path(
    path_type: Literal["OT", "VP", "VE"] = "OT", sigma_min: float = 1e-4
) -> Callable:
    """
    Get a conditional path function by type.

    Args:
        path_type: Type of probability path ('OT', 'VP', or 'VE')
        sigma_min: Minimum noise level for OT path

    Returns:
        path_fn: Function that computes (psi_t, target) given (x0, x1, t)
    """
    if path_type == "OT":
        return lambda x0, x1, t: ot_path(x0, x1, t, sigma_min)
    elif path_type == "VP":
        return vp_path
    elif path_type == "VE":
        return ve_path
    else:
        raise ValueError(f"Unknown path type: {path_type}. Must be 'OT', 'VP', or 'VE'")


class ConditionalPath:
    """
    Object-oriented interface for conditional paths.

    Allows for easier state management and configuration.
    """

    def __init__(
        self, path_type: Literal["OT", "VP", "VE"] = "OT", sigma_min: float = 1e-4
    ):
        """
        Initialize conditional path.

        Args:
            path_type: Type of probability path
            sigma_min: Minimum noise level (used for OT path)
        """
        self.path_type = path_type
        self.sigma_min = sigma_min
        self.path_fn = get_conditional_path(path_type, sigma_min)

    def __call__(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute intermediate state and target vector field.

        Args:
            x0: Source noise
            x1: Target data
            t: Time tensor

        Returns:
            psi_t: Intermediate state
            target: Target vector field
        """
        return self.path_fn(x0, x1, t)

    def get_coefficients(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get time-dependent coefficients for the path.

        For different path types, returns (alpha_t, sigma_t, derivative)
        such that psi_t = alpha_t * x1 + sigma_t * x0

        Args:
            t: Time tensor in [0, 1]

        Returns:
            alpha_t: Coefficient for x1
            sigma_t: Coefficient for x0
            d_alpha_t: Time derivative of alpha_t
        """
        if self.path_type == "OT":
            sigma_t = 1 - (1 - self.sigma_min) * t
            alpha_t = t
            d_alpha_t = torch.ones_like(t)
            d_sigma_t = -(1 - self.sigma_min) * torch.ones_like(t)

        elif self.path_type == "VP":
            alpha_t = torch.cos(t * torch.pi / 2).pow(2)
            sigma_t = torch.sin(t * torch.pi / 2)
            d_alpha_t = -torch.pi * torch.sin(t * torch.pi / 2) * torch.cos(t * torch.pi / 2)
            d_sigma_t = (torch.pi / 2) * torch.cos(t * torch.pi / 2)

        elif self.path_type == "VE":
            sigma_min = 1e-4
            sigma_max = 10.0
            log_sigma_min = torch.log(torch.tensor(sigma_min))
            log_sigma_max = torch.log(torch.tensor(sigma_max))
            log_sigma_t = (1 - t) * log_sigma_min + t * log_sigma_max
            sigma_t = torch.exp(log_sigma_t)

            alpha_t = torch.ones_like(t)
            d_alpha_t = torch.zeros_like(t)
            d_sigma_t = sigma_t * (log_sigma_max - log_sigma_min)

        else:
            raise ValueError(f"Unknown path type: {self.path_type}")

        return alpha_t, sigma_t, d_alpha_t
