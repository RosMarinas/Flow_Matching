"""
Evaluation Metrics for Flow Matching

Implements:
1. NLL (Negative Log Likelihood) calculation using Hutchinson Trace Estimator.
2. FID (Fréchet Inception Distance) wrapper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm
from pathlib import Path

# Try to import torchdiffeq
try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None


def compute_divergence(
    model: nn.Module, 
    x: torch.Tensor, 
    t: torch.Tensor, 
    e: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute divergence of vector field using Hutchinson Trace Estimator.
    
    div(v) = Tr(J) ≈ e^T J e
    where J = dv/dx
    
    Args:
        model: Vector field network v(x, t)
        x: Input tensor (B, ...)
        t: Time tensor (B, 1) or scalar
        e: Random noise for Hutchinson estimator (B, ...)
        
    Returns:
        div: Estimated divergence (B,)
    """
    if e is None:
        e = torch.randn_like(x)
        
    # Enable gradient computation for x
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        
        # Predict velocity
        v = model(x, t)
        
        # Compute vector-Jacobian product: e^T J
        # We compute grad(v^T e, x) = grad(sum(v*e), x)
        v_e = torch.sum(v * e, dim=tuple(range(1, len(x.shape))))
        grad_x = torch.autograd.grad(v_e.sum(), x, create_graph=False)[0]
        
    # Compute e^T J e = e^T * grad_x
    # (Since grad_x is J^T e)
    div = torch.sum(grad_x * e, dim=tuple(range(1, len(x.shape))))
    
    return div


def compute_nll(
    model: nn.Module,
    data: torch.Tensor,
    t_span: Tuple[float, float] = (1.0, 0.0),
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method: str = "dopri5",
) -> Tuple[float, float]:
    """
    Compute Negative Log Likelihood (NLL) in bits/dim.
    
    Integrates the augmented ODE:
        d[x]/dt = v(x, t)
        d[logp]/dt = -div(v(x, t))
        
    Args:
        model: Trained vector field network
        data: Batch of data x1 (B, C, H, W)
        t_span: Integration range (1.0 -> 0.0 for likelihood)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        nll: Average NLL in bits/dim
        std: Standard deviation of NLL
    """
    if odeint is None:
        raise ImportError("torchdiffeq is required for NLL computation.")
    
    # Garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    device = data.device
    batch_size = data.shape[0]
    
    # Store original requires_grad state and set to False to save memory
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad = False
        
    try:
        # Random noise for Hutchinson trace estimator
        e = torch.randn_like(data)
        
        def augmented_ode_func(t, state):
            """
            State is a tuple (x, delta_logp)
            """
            x = state[0]
            # t is scalar from odeint
            t_tensor = torch.ones(batch_size, device=device) * t
            
            # Reshape t for model
            if len(x.shape) == 4:
                t_input = t_tensor.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                t_input = t_tensor.view(-1, 1)
            else:
                t_input = t_tensor
                
            # Compute velocity
            v = model(x, t_input)
            
            # Compute divergence
            div = compute_divergence(model, x, t_input, e)
            
            # d(logp)/dt = -div(v)
            d_logp = -div
            
            return (v, d_logp)
        
        # Initial state at t=1: (x1, 0)
        initial_state = (data, torch.zeros(batch_size, device=device))
        integration_times = torch.tensor(t_span, device=device)
        
        # Solve ODE
        final_state = odeint(
            augmented_ode_func,
            initial_state,
            integration_times,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        x0 = final_state[0][-1]      # Shape (B, ...)
        delta_logp = final_state[1][-1] # Shape (B,)
        
        # log p(x0) (Standard Normal)
        # log N(x0; 0, I) = -d/2 log(2pi) - 1/2 ||x0||^2
        d = np.prod(data.shape[1:])
        log_p_x0 = -0.5 * d * np.log(2 * np.pi) - 0.5 * torch.sum(x0**2, dim=tuple(range(1, len(x0.shape))))
        
        log_p_x1 = log_p_x0 - delta_logp
        
        # Convert to bits/dim
        # NLL = -log p(x1)
        # bits/dim = NLL / (d * log(2))
        nll = -log_p_x1
        bits_per_dim = nll / (d * np.log(2))
        
        return bits_per_dim.mean().item(), bits_per_dim.std().item()

    finally:
        # Restore requires_grad state
        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]
            
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()


def compute_fid_from_files(
    path1: str,
    path2: str,
    batch_size: int = 50,
    device: str = "cuda",
    dims: int = 2048
) -> float:
    """
    Compute FID between two directories of images.
    
    Args:
        path1: Path to directory 1
        path2: Path to directory 2
        batch_size: Batch size for Inception network
        device: Device to run Inception network
        dims: Feature dimension (default 2048)
        
    Returns:
        fid: FID score
    """
    from pytorch_fid import fid_score
    
    fid_value = fid_score.calculate_fid_given_paths(
        [path1, path2],
        batch_size=batch_size,
        device=device,
        dims=dims
    )
    
    return fid_value
