"""
ODE Solvers for Flow Matching

Implements numerical solvers for the ODE:
    dx/dt = v_t(x)

Used for both:
1. Sampling: Generate samples from noise by integrating from t=0 to t=1
2. Visualization: Generate trajectories
"""

import torch
from typing import Literal, Optional, Tuple, Union


def euler_solver(
    model,
    x0: torch.Tensor,
    num_steps: int = 100,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """
    Euler method for ODE integration.

    Simplest first-order solver:
        x_{t+dt} = x_t + dt * v_t(x_t)

    Args:
        model: Vector field network v_t(x, t)
        x0: Initial condition, shape (B, ...)
        num_steps: Number of integration steps
        return_trajectory: If True, return all intermediate states

    Returns:
        x: Final state (or trajectory if return_trajectory=True)
    """
    dt = 1.0 / num_steps
    x = x0.clone()

    if return_trajectory:
        trajectory = [x.clone()]

    for i in range(num_steps):
        # Time ranges from 0 to 1
        t = torch.ones(x.shape[0], device=x.device) * (i / num_steps)

        # Reshape t to match x dimensions for broadcasting if needed
        # (Though most models handle t separately)
        t_model = t
        if len(x.shape) == 4:  # Images (B, C, H, W)
             t_model = t.view(-1, 1, 1, 1)
        elif len(x.shape) == 2:  # 2D vectors (B, 2)
             t_model = t.view(-1, 1)

        # Predict velocity
        v = model(x, t_model)

        # Euler step
        x = x + dt * v

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    else:
        return x


def rk4_solver(
    model,
    x0: torch.Tensor,
    num_steps: int = 100,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """
    Runge-Kutta 4th order method for ODE integration.

    More accurate than Euler:
        k1 = v_t(x_t)
        k2 = v_t(x_t + 0.5*dt*k1)
        k3 = v_t(x_t + 0.5*dt*k2)
        k4 = v_t(x_t + dt*k3)
        x_{t+dt} = x_t + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        model: Vector field network v_t(x, t)
        x0: Initial condition, shape (B, ...)
        num_steps: Number of integration steps
        return_trajectory: If True, return all intermediate states

    Returns:
        x: Final state (or trajectory if return_trajectory=True)
    """
    dt = 1.0 / num_steps
    x = x0.clone()

    if return_trajectory:
        trajectory = [x.clone()]

    for i in range(num_steps):
        t = i / num_steps
        
        def get_t_tensor(t_val):
            t_tensor = torch.ones(x.shape[0], device=x.device) * t_val
            if len(x.shape) == 4:
                return t_tensor.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                return t_tensor.view(-1, 1)
            return t_tensor

        # RK4 steps
        k1 = model(x, get_t_tensor(t))
        k2 = model(x + 0.5 * dt * k1, get_t_tensor(t + 0.5 * dt))
        k3 = model(x + 0.5 * dt * k2, get_t_tensor(t + 0.5 * dt))
        k4 = model(x + dt * k3, get_t_tensor(t + dt))

        # Combine
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    else:
        return x


def sample(
    model,
    num_samples: int,
    input_shape: Tuple[int, ...] = (2,),
    num_steps: int = 100,
    method: Literal["euler", "rk4", "dopri5"] = "euler",
    device: str = "cpu",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """
    Generate samples from trained model.

    Integrates the ODE from t=0 (noise) to t=1 (data):
        x(0) ~ N(0, I)
        dx/dt = v_t(x)
        x(1) ~ p_data

    Args:
        model: Trained vector field network
        num_samples: Number of samples to generate
        input_shape: Shape of a single sample (e.g., (2,) or (3, 32, 32))
        num_steps: Number of integration steps (for fixed-step methods)
        method: Integration method ('euler', 'rk4', 'dopri5')
        device: torch device
        atol: Absolute tolerance (for adaptive methods)
        rtol: Relative tolerance (for adaptive methods)

    Returns:
        samples: Generated samples
    """
    # Initialize from noise
    x0 = torch.randn(num_samples, *input_shape, device=device)

    # Integrate ODE
    if method == "euler":
        samples = euler_solver(model, x0, num_steps=num_steps, return_trajectory=False)
    elif method == "rk4":
        samples = rk4_solver(model, x0, num_steps=num_steps, return_trajectory=False)
    elif method == "dopri5":
        # Use adaptive solver
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError(
                "torchdiffeq is required for dopri5. "
                "Install it with: uv add torchdiffeq"
            )

        def ode_func(t, x):
            t_tensor = torch.ones(x.shape[0], device=x.device) * t
            if len(x.shape) == 4:
                t_tensor = t_tensor.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                t_tensor = t_tensor.view(-1, 1)
            return model(x, t_tensor)

        t = torch.tensor([0.0, 1.0], device=device)
        samples = odeint(ode_func, x0, t, method="dopri5", atol=atol, rtol=rtol)[-1]
    else:
        raise ValueError(f"Unknown method: {method}")

    return samples


@torch.no_grad()
def sample_adaptive(
    model,
    num_samples: int,
    input_shape: Tuple[int, ...] = (2,),
    method: Literal["dopri5", "bosh3"] = "dopri5",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate samples using adaptive step size ODE solver.

    Uses torchdiffeq for high-quality adaptive integration.
    Higher quality but slower than fixed-step methods.

    Args:
        model: Trained vector field network
        num_samples: Number of samples to generate
        input_shape: Shape of a single sample
        method: Adaptive method ('dopri5' or 'bosh3')
        atol: Absolute tolerance
        rtol: Relative tolerance
        device: torch device

    Returns:
        samples: Generated samples
    """
    try:
        from torchdiffeq import odeint
    except ImportError:
        raise ImportError(
            "torchdiffeq is required for adaptive sampling. "
            "Install it with: uv add torchdiffeq"
        )

    # Define ODE function
    def ode_func(t, x):
        # t is scalar, x is (B, ...)
        t_tensor = torch.ones(x.shape[0], device=x.device) * t
        if len(x.shape) == 4:
            t_tensor = t_tensor.view(-1, 1, 1, 1)
        elif len(x.shape) == 2:
            t_tensor = t_tensor.view(-1, 1)
            
        return model(x, t_tensor)

    # Initial condition
    x0 = torch.randn(num_samples, *input_shape, device=device)

    # Time points
    t = torch.tensor([0.0, 1.0], device=device)

    # Solve ODE
    sol = odeint(
        ode_func,
        x0,
        t,
        method=method,
        atol=atol,
        rtol=rtol,
    )

    # Return final state
    return sol[-1]
