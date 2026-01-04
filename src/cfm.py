"""
Conditional Flow Matching (CFM) Implementation

Based on "Flow Matching for Generative Modeling" (ICLR 2023)
Equation 2 and 23: L_CFM = E[t, x1, x0] ||v_t(ψ_t(x0)) - u_t||^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class ConditionalFlowMatching:
    """
    Conditional Flow Matching (CFM) for generative modeling.

    Core Loss (Equation 23):
    L_CFM = E[t, x1, x0] ||v_t(ψ_t(x0)) - u_t||^2

    For Optimal Transport (OT) path:
    - ψ_t(x0) = (1 - (1-σ_min)t)x0 + t*x1
    - u_t = x1 - (1-σ_min)x0

    Args:
        model: Neural network that predicts vector field v_t(x, t)
        path_type: Type of probability path ('OT', 'VP', or 'VE')
        sigma_min: Minimum noise for OT path (prevents numerical issues)
    """

    def __init__(
        self,
        model: nn.Module,
        path_type: Literal["OT", "VP", "VE"] = "OT",
        sigma_min: float = 1e-4,
    ):
        self.model = model
        self.path_type = path_type
        self.sigma_min = sigma_min

        # Import path functions
        from paths import get_conditional_path

        self.path_fn = get_conditional_path(path_type, sigma_min)

    def compute_loss(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Compute CFM loss for a batch of data.

        Args:
            x1: Data tensor, shape (B, C, H, W) for images or (B, D) for vectors

        Returns:
            loss: Scalar tensor
        """
        batch_size = x1.shape[0]

        # 1. Sample time t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=x1.device)

        # Reshape t to match data dimensions
        if len(x1.shape) == 4:  # Images: (B, C, H, W)
            t = t.view(batch_size, 1, 1, 1)
        elif len(x1.shape) == 2:  # Vectors: (B, D)
            t = t.view(batch_size, 1)
        else:
            raise ValueError(f"Unsupported data shape: {x1.shape}")

        # 2. Sample noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 3. Compute conditional path ψ_t(x0) and target u_t
        psi_t, target = self.path_fn(x0, x1, t)

        # 4. Predict vector field
        v_pred = self.model(psi_t, t)

        # 5. Compute MSE loss
        loss = F.mse_loss(v_pred, target)

        return loss

    def sample_velocity(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample intermediate state and target velocity for given t.

        Useful for visualization and debugging.

        Args:
            x0: Source noise
            x1: Target data
            t: Time tensor

        Returns:
            psi_t: Intermediate state
            target: Target vector field
        """
        psi_t, target = self.path_fn(x0, x1, t)
        return psi_t, target


class ConditionalFlowMatchingV2:
    """
    Alternative implementation with explicit path definitions.

    This version shows the explicit formulas for OT and VP paths.
    """

    def __init__(
        self,
        model: nn.Module,
        path_type: Literal["OT", "VP"] = "OT",
        sigma_min: float = 1e-4,
    ):
        self.model = model
        self.path_type = path_type
        self.sigma_min = sigma_min

    def compute_loss(self, x1: torch.Tensor) -> torch.Tensor:
        """
        Compute CFM loss with explicit path formulas.
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample time t ~ U[0, 1]
        t = torch.rand(batch_size, device=device)

        # Reshape t to match data dimensions
        if len(x1.shape) == 4:  # Images
            t = t.view(batch_size, 1, 1, 1)
        elif len(x1.shape) == 2:  # Vectors
            t = t.view(batch_size, 1)

        # Sample noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        if self.path_type == "OT":
            # Optimal Transport Path (straight lines)
            # ψ_t(x0) = (1 - (1-σ_min)t)x0 + t*x1
            sigma_t = 1 - (1 - self.sigma_min) * t
            psi_t = sigma_t * x0 + t * x1

            # Target vector field
            # u_t = x1 - (1-σ_min)x0
            target = x1 - (1 - self.sigma_min) * x0

        elif self.path_type == "VP":
            # Variance Preserving Diffusion Path
            # Path: ψ_t = sin(t*π/2)*x1 + cos(t*π/2)*x0
            sin_t = torch.sin(t * torch.pi / 2)
            cos_t = torch.cos(t * torch.pi / 2)

            psi_t = sin_t * x1 + cos_t * x0

            # Target vector field (Time derivative)
            # u_t = π/2 * (cos(t*π/2)*x1 - sin(t*π/2)*x0)
            target = (torch.pi / 2) * (cos_t * x1 - sin_t * x0)

        else:
            raise ValueError(f"Unknown path type: {self.path_type}")

        # Predict vector field
        v_pred = self.model(psi_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, target)

        return loss
