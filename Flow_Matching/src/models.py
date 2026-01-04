import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

# Keep the simple MLP for Toy Data
class VectorFieldNet(nn.Module):
    def __init__(self, hidden_dim=64, context_dim=0, output_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.t_emb_dim = 16
        self.register_buffer('w_freq', torch.randn(1, self.t_emb_dim // 2) * 30.0) 

        input_dim = output_dim + self.t_emb_dim + context_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t, x, context=None):
        t_proj = 2 * np.pi * t.unsqueeze(1) @ self.w_freq 
        t_emb = torch.cat([torch.cos(t_proj), torch.sin(t_proj)], dim=-1)
        
        inputs = [x, t_emb]
        if context is not None:
            inputs.append(context)
            
        net_in = torch.cat(inputs, dim=-1)
        return self.net(net_in)

class ConditionalFlowMatching(ABC):
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    @abstractmethod
    def compute_loss(self, model, x1):
        pass

    def sample_ode(self, model, shape, steps=100, device='cpu', solver='euler'):
        """
        Generate samples by solving the ODE.
        """
        model.eval()
        x0 = torch.randn(shape, device=device)
        
        if solver == 'euler':
            return self._euler_solve(model, x0, steps, device)
        elif solver == 'dopri5':
            return self._dopri5_solve(model, x0, device)
        else:
            raise ValueError(f"Unknown solver: {solver}")

    def _euler_solve(self, model, x0, steps, device):
        dt = 1.0 / steps
        xt = x0.clone()
        with torch.no_grad():
            for i in range(steps):
                t_value = i / steps
                t_batch = torch.full((x0.shape[0],), t_value, device=device)
                vt = model(t_batch, xt)
                xt = xt + vt * dt
        return xt
    
    def _dopri5_solve(self, model, x0, device):
        from torchdiffeq import odeint
        
        def ode_func(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=device)
            return model(t_batch, x)
        
        # Integrate from 0 to 1
        t_span = torch.tensor([0.0, 1.0], device=device)
        traj = odeint(ode_func, x0, t_span, method='dopri5', atol=1e-5, rtol=1e-5)
        return traj[-1]

class OTFlowMatching(ConditionalFlowMatching):
    """Optimal Transport Conditional Flow Matching"""
    def compute_loss(self, model, x1):
        b = x1.shape[0]
        device = x1.device
        
        x0 = torch.randn_like(x1)
        t = torch.rand(b, device=device)
        
        # Reshape t for broadcasting [B, 1, 1, ...]
        t_reshaped = t.view([b] + [1]*(x1.ndim - 1))
        
        # Path: Linear interpolation
        # mu_t = t * x1, sigma_t = 1 - (1 - sigma_min) * t
        mu_t = t_reshaped * x1
        sigma_t = 1 - (1 - self.sigma_min) * t_reshaped
        
        xt = mu_t + sigma_t * x0
        
        # Target Vector Field
        # ut = (x1 - (1 - sigma_min) * xt) / sigma_t
        ut = (x1 - (1 - self.sigma_min) * xt) / sigma_t
        
        vt = model(t, xt)
        loss = torch.mean((vt - ut)**2)
        return loss

class TargetFlowMatching(ConditionalFlowMatching):
    """Target Conditional Flow Matching (Linear mean, const sigma)"""
    def compute_loss(self, model, x1):
        b = x1.shape[0]
        device = x1.device
        x0 = torch.randn_like(x1)
        t = torch.rand(b, device=device)
        t_reshaped = t.view([b] + [1]*(x1.ndim - 1))
        
        # Path: mu_t = t*x1 + (1-t)*x0, sigma_t = sigma_min (fixed small noise)
        # Note: This is one variant. Another is simply interpolation.
        # Let's use the simple interpolation variant which is often robust:
        # xt = t * x1 + (1 - t) * x0
        # The vector field is simply x1 - x0
        
        xt = t_reshaped * x1 + (1 - t_reshaped) * x0
        ut = x1 - x0
        
        vt = model(t, xt)
        loss = torch.mean((vt - ut)**2)
        return loss

class VPFlowMatching(ConditionalFlowMatching):
    """Variance Preserving Flow Matching (Cos/Sin path like DDPM)"""
    def compute_loss(self, model, x1):
        b = x1.shape[0]
        device = x1.device
        x0 = torch.randn_like(x1)
        t = torch.rand(b, device=device)
        
        # We map t [0, 1] to diffusion time.
        # Simple schedule: alpha_t = cos(0.5 * pi * t), sigma_t = sin(0.5 * pi * t)
        # Note: This corresponds to preserving variance x^2 + y^2 = 1
        
        t_reshaped = t.view([b] + [1]*(x1.ndim - 1))
        alpha_t = torch.cos(0.5 * np.pi * t_reshaped)
        sigma_t = torch.sin(0.5 * np.pi * t_reshaped)
        
        xt = alpha_t * x1 + sigma_t * x0
        
        # Target Vector Field for VP path:
        # ut = (dx/dt)
        # d(alpha)/dt = -0.5*pi*sin(...) = -0.5*pi*sigma_t
        # d(sigma)/dt = 0.5*pi*cos(...) = 0.5*pi*alpha_t
        # ut = -0.5*pi*sigma_t * x1 + 0.5*pi*alpha_t * x0
        
        d_alpha = -0.5 * np.pi * sigma_t
        d_sigma = 0.5 * np.pi * alpha_t
        
        ut = d_alpha * x1 + d_sigma * x0
        
        vt = model(t, xt)
        loss = torch.mean((vt - ut)**2)
        return loss
