"""
MLP Vector Field Network for 2D Toy Data

Based on paper's architecture for 2D experiments:
- 5-layer MLP
- 512 hidden units per layer
- SiLU activation
- Time embedding via sinusoidal features

Input: (x, y, t) -> Output: (vx, vy)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal time embedding.

    Projects scalar time t to a higher-dimensional vector using
    sinusoidal functions at different frequencies.

    Args:
        dim: Dimension of embedding
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor, shape (B, 1) or (B,)

        Returns:
            embedding: Shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # Frequencies: 1/10000^(2k/d)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Reshape t for broadcasting
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        # Sinusoidal embedding
        embeddings = t * embeddings  # (B, half_dim)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class MLPVectorField(nn.Module):
    """
    MLP-based vector field network for 2D data.

    Architecture:
    - Input: (x, y) + time embedding(t)
    - Hidden: 5 layers, 512 units each, SiLU activation
    - Output: (vx, vy) - 2D vector field

    Args:
        hidden_dim: Number of hidden units (default: 512)
        num_layers: Number of hidden layers (default: 5)
        time_embed_dim: Dimension of time embedding (default: 256)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 5,
        time_embed_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_mlp = SinusoidalEmbedding(time_embed_dim)

        # Input projection: 2D position + time embedding
        self.input_proj = nn.Linear(2 + time_embed_dim, hidden_dim)

        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            )

        # Output projection: 2D vector
        self.output_proj = nn.Linear(hidden_dim, 2)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Position tensor, shape (B, 2)
            t: Time tensor, shape (B, 1) or (B,)

        Returns:
            v: Predicted vector field, shape (B, 2)
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        elif t.dim() > 2:
            t = t.squeeze()  # Remove extra dimensions
            if t.dim() == 1:
                t = t.unsqueeze(-1)

        # Flatten t if needed
        t = t.squeeze(-1)  # (B,)

        # Time embedding
        t_embed = self.time_mlp(t)  # (B, time_embed_dim)

        # Concatenate position and time
        h = torch.cat([x, t_embed], dim=-1)  # (B, 2 + time_embed_dim)

        # Input projection
        h = self.input_proj(h)  # (B, hidden_dim)
        h = torch.nn.functional.silu(h)

        # Hidden layers
        for layer in self.layers:
            h = layer(h) + h  # Residual connection

        # Output projection
        v = self.output_proj(h)  # (B, 2)

        return v


class MLPVectorFieldSimple(nn.Module):
    """
    Simplified MLP vector field without time embedding.

    Useful for baseline comparisons or debugging.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()

        layers = []
        input_dim = 3  # (x, y, t)

        # Hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Position tensor, shape (B, 2)
            t: Time tensor, shape (B, 1) or (B,)

        Returns:
            v: Predicted vector field, shape (B, 2)
        """
        # Reshape t to match x
        if t.dim() > 2:
            t = t.reshape(x.shape[0], -1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        # Take only first value of t if it has extra dimensions
        t = t[:, :1]  # (B, 1)

        # Concatenate x and t
        h = torch.cat([x, t], dim=-1)  # (B, 3)

        # Forward through network
        v = self.network(h)  # (B, 2)

        return v


class FourierFeatures(nn.Module):
    """
    Fourier feature mapping for better function approximation.

    Based on "Fourier Features Let Networks Learn High Frequency
    Functions in Low Dimensional Domains" (Tancik et al., 2020).

    Args:
        input_dim: Input dimension
        output_dim: Output dimension (must be even)
        scale: Scale of Gaussian frequencies
    """

    def __init__(self, input_dim: int, output_dim: int = 256, scale: float = 1.0):
        super().__init__()

        assert (
            output_dim % 2 == 0
        ), "output_dim must be even for sin/cos pairs"

        # Random Fourier features
        self.register_buffer(
            "W", torch.randn(input_dim, output_dim // 2) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, input_dim)

        Returns:
            features: Fourier features, shape (B, output_dim)
        """
        # Project and apply sin/cos
        h = x @ self.W  # (B, output_dim // 2)
        h = 2 * torch.pi * h
        return torch.cat([torch.sin(h), torch.cos(h)], dim=-1)


class MLPVectorFieldFourier(nn.Module):
    """
    MLP vector field with Fourier features for the spatial dimensions.

    Can better capture high-frequency patterns in the data.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 5,
        fourier_dim: int = 256,
        time_embed_dim: int = 256,
        fourier_scale: float = 1.0,
    ):
        super().__init__()

        # Fourier features for (x, y)
        self.fourier_proj = FourierFeatures(
            input_dim=2, output_dim=fourier_dim, scale=fourier_scale
        )

        # Time embedding
        self.time_mlp = SinusoidalEmbedding(time_embed_dim)

        # Input projection
        self.input_proj = nn.Linear(fourier_dim + time_embed_dim, hidden_dim)

        # Hidden layers
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            )
        self.layers = nn.ModuleList(layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Position tensor, shape (B, 2)
            t: Time tensor, shape (B, 1) or (B,)

        Returns:
            v: Predicted vector field, shape (B, 2)
        """
        # Flatten t if needed
        if t.dim() > 2:
            t = t.squeeze()
        if t.dim() == 1:
            pass  # Already (B,)
        else:
            t = t.squeeze(-1)

        # Fourier features for position
        x_features = self.fourier_proj(x)  # (B, fourier_dim)

        # Time embedding
        t_embed = self.time_mlp(t)  # (B, time_embed_dim)

        # Concatenate
        h = torch.cat([x_features, t_embed], dim=-1)

        # Input projection
        h = self.input_proj(h)
        h = torch.nn.functional.silu(h)

        # Hidden layers
        for layer in self.layers:
            h = layer(h) + h

        # Output
        v = self.output_proj(h)

        return v


# --- U-Net Components for CIFAR-10 ---


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        ch = c // self.num_heads
        q = q.view(b, self.num_heads, ch, h * w)
        k = k.view(b, self.num_heads, ch, h * w)
        v = v.view(b, self.num_heads, ch, h * w)

        # scaled dot-product attention
        # (b, h, ch, hw) @ (b, h, hw, ch) -> (b, h, ch, ch)
        # Wait, the dimensions are slightly different for typical self-attention on pixels
        # Typically: (b, h, hw, ch) @ (b, h, ch, hw) -> (b, h, hw, hw)
        q = q.permute(0, 1, 3, 2)  # (b, h, hw, ch)
        # k is (b, h, ch, hw)
        
        weight = torch.matmul(q, k) * (ch**-0.5)  # (b, h, hw, hw)
        weight = torch.softmax(weight, dim=-1)

        a = torch.matmul(weight, v.permute(0, 1, 3, 2))  # (b, h, hw, ch)
        a = a.permute(0, 1, 3, 2).reshape(b, c, h, w)

        return x + self.proj_out(a)


class UNet(nn.Module):
    """
    U-Net architecture for CIFAR-10 image generation.
    
    Based on the architecture used in Dhariwal & Nichol (2021).
    Input: (B, 3, 32, 32) + time t
    Output: (B, 3, 32, 32)
    """
    def __init__(
        self,
        in_channels=3,
        model_channels=256,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, time_embed_dim, dropout),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        """
        Apply the model to an input batch.

        :param x: an [N x C x H x W] Tensor of inputs.
        :param t: a 1D batch of timesteps (N,) or (N, 1, 1, 1).
        :return: an [N x C x H x W] Tensor of outputs.
        """
        if t.dim() > 1:
            t = t.reshape(x.shape[0])
            
        emb = self.time_embed(t)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
