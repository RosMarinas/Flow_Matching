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


class DiscreteTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding for discrete timesteps (DDPM).

    For DDPM: t ∈ {0, 1, ..., T-1} where T=1000
    We normalize to [0, 1] to match CFM's continuous time embedding.

    Args:
        dim: Dimension of embedding
        max_timesteps: Maximum timestep value (default: 1000)
    """

    def __init__(self, dim: int = 256, max_timesteps: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_timesteps = max_timesteps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Discrete timestep tensor, shape (B,) with values in [0, T-1]

        Returns:
            embedding: Shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # Normalize to [0, 1] for consistency with CFM
        t_normalized = t.float() / self.max_timesteps

        # Use same sinusoidal frequencies as CFM
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Reshape for broadcasting
        if t_normalized.dim() == 1:
            t_normalized = t_normalized.unsqueeze(-1)

        # Sinusoidal embedding
        embeddings = t_normalized * embeddings
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class ClassEmbedding(nn.Module):
    """
    Learnable class embedding for class-labeled generation.

    Projects class indices to embedding vectors that can be
    used as conditioning information in the model.

    Args:
        num_classes: Number of classes (e.g., 10 for CIFAR-10)
        embed_dim: Dimension of embedding (default: 256)
    """
    def __init__(self, num_classes: int = 10, embed_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            labels: Class labels, shape (B,) with values in [0, num_classes-1]

        Returns:
            embedding: Shape (B, embed_dim)
        """
        return self.embedding(labels)


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
        use_discrete_time: Use discrete timestep embedding for DDPM (default: False)
        max_timesteps: Maximum timestep for discrete embedding (default: 1000)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 5,
        time_embed_dim: int = 256,
        dropout: float = 0.0,
        use_discrete_time: bool = False,
        max_timesteps: int = 1000,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.use_discrete_time = use_discrete_time

        # Time embedding
        if use_discrete_time:
            self.time_mlp = DiscreteTimeEmbedding(time_embed_dim, max_timesteps)
        else:
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


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for class label conditioning.

    Performs cross-attention where:
    - Query: Spatial features from image (H x W tokens)
    - Key/Value: Class label embedding (1 token per sample)

    This allows each spatial location to attend to the class information.

    Args:
        channels: Number of channels in spatial features
        context_dim: Dimension of class embedding (default: 1024 to match time embedding)
        num_heads: Number of attention heads (default: 4)
    """
    def __init__(self, channels: int, context_dim: int = 1024, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels)

        # Query projection: from spatial features
        self.q_proj = nn.Conv2d(channels, channels, 1)

        # Key/Value projections: from class embedding
        self.kv_proj = nn.Linear(context_dim, channels * 2)

        # Output projection
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial features, shape (B, C, H, W)
            context: Class label embeddings, shape (B, context_dim)

        Returns:
            out: Label-conditioned features, shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        head_dim = C // self.num_heads

        # Normalize input
        h_in = self.norm(x)  # (B, C, H, W)

        # Query: reshape to (B, num_heads, head_dim, H*W)
        q = self.q_proj(h_in)  # (B, C, H, W)
        q = q.view(B, self.num_heads, head_dim, H * W)  # (B, heads, head_dim, H*W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, H*W, head_dim)

        # Key/Value: from context (B, context_dim) -> (B, 2*C)
        kv = self.kv_proj(context)  # (B, 2*C)
        k, v = kv.chunk(2, dim=1)  # Each (B, C)

        # Reshape for multi-head attention
        k = k.view(B, self.num_heads, head_dim, 1)  # (B, heads, head_dim, 1)
        v = v.view(B, self.num_heads, head_dim, 1)  # (B, heads, head_dim, 1)

        # Attention: (B, heads, H*W, head_dim) @ (B, heads, head_dim, 1)
        # -> (B, heads, H*W, 1)
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q, k) * scale  # (B, heads, H*W, 1)
        attn_weights = torch.softmax(attn_weights, dim=-2)

        # Apply attention to values: (B, heads, H*W, 1) @ (B, heads, 1, head_dim)
        # -> (B, heads, H*W, head_dim)
        attn_out = torch.matmul(attn_weights, v.permute(0, 1, 3, 2))

        # Reshape back to (B, C, H, W)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Output projection and residual
        return x + self.proj_out(attn_out)


class UNet(nn.Module):
    """
    U-Net architecture for CIFAR-10 image generation.

    Based on the architecture used in Dhariwal & Nichol (2021).
    Input: (B, 3, 32, 32) + time t
    Output: (B, 3, 32, 32)

    Args:
        in_channels: Number of input channels (default: 3)
        model_channels: Base channel count (default: 256)
        out_channels: Number of output channels (default: 3)
        num_res_blocks: Number of residual blocks per level (default: 2)
        attention_resolutions: Resolutions at which to use attention (default: (16,))
        dropout: Dropout rate (default: 0.1)
        channel_mult: Channel multiplier for each level (default: (1, 2, 2, 2))
        num_heads: Number of attention heads (default: 4)
        use_discrete_time: Use discrete timestep embedding for DDPM (default: False)
        max_timesteps: Maximum timestep for discrete embedding (default: 1000)
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
        use_discrete_time=False,
        max_timesteps=1000,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_discrete_time = use_discrete_time

        time_embed_dim = model_channels * 4

        # Choose time embedding based on model type
        if use_discrete_time:
            self.time_embed = nn.Sequential(
                DiscreteTimeEmbedding(model_channels, max_timesteps),
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
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


class UNetWithClassLabels(nn.Module):
    """
    U-Net with class label conditioning for CIFAR-10 image generation.

    Architecture:
    - Input: (B, 3, 32, 32) images + time t + class labels
    - Time embedding: Sinusoidal -> MLP
    - Class embedding: Learnable -> MLP (same dim as time)
    - Cross-attention: At resolutions where self-attention exists
    - Output: (B, 3, 32, 32)

    This model uses cross-attention to condition generation on class labels.
    The spatial features attend to the class embedding at each resolution level.

    Args:
        num_classes: Number of classes (default: 10 for CIFAR-10)
        class_emb_dim: Dimension of class embedding before MLP (default: 256)
        in_channels: Number of input channels (default: 3)
        model_channels: Base channel count (default: 128)
        out_channels: Number of output channels (default: 3)
        num_res_blocks: Number of residual blocks per level (default: 2)
        attention_resolutions: Resolutions at which to use attention (default: (16,))
        dropout: Dropout rate (default: 0.1)
        channel_mult: Channel multiplier for each level (default: (1, 2, 2, 2))
        num_heads: Number of attention heads (default: 4)
    """
    def __init__(
        self,
        num_classes: int = 10,
        class_emb_dim: int = 256,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16,),
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        num_heads=4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4

        # Time embedding (same as UNet)
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Class embedding (NEW)
        self.class_embed = nn.Sequential(
            ClassEmbedding(num_classes, class_emb_dim),
            nn.Linear(class_emb_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Encoder blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # Track cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList()
        self.input_cross_attn_indices = []  # Track which input block indices have cross-attention

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels)]
                ch = mult * model_channels
                has_attn = ds in attention_resolutions
                if has_attn:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                    # Add cross-attention block for this resolution
                    self.cross_attn_blocks.append(
                        CrossAttentionBlock(ch, time_embed_dim, num_heads)
                    )
                    # Record the index of this input block
                    self.input_cross_attn_indices.append(len(self.input_blocks))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                input_block_chans.append(ch)
                ds *= 2

        # Middle block with cross-attention
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=num_heads),
            ResBlock(ch, time_embed_dim, dropout),
        )
        self.cross_attn_middle = CrossAttentionBlock(ch, time_embed_dim, num_heads)

        # Decoder blocks
        self.output_blocks = nn.ModuleList([])
        self.output_cross_attn_indices = []  # Track which output block indices have cross-attention

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult)]
                ch = model_channels * mult
                has_attn = ds in attention_resolutions
                if has_attn:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                    # Add cross-attention block for this resolution
                    self.cross_attn_blocks.append(
                        CrossAttentionBlock(ch, time_embed_dim, num_heads)
                    )
                    # Record the index of this output block
                    self.output_cross_attn_indices.append(len(self.output_blocks))
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output layer
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with class label conditioning.

        Args:
            x: Images, shape (B, 3, 32, 32)
            t: Time, shape (B,) or (B, 1, 1, 1)
            labels: Class labels, shape (B,) with values in [0, num_classes-1]

        Returns:
            v: Predicted vector field, shape (B, 3, 32, 32)
        """
        # Ensure correct shapes
        if t.dim() > 1:
            t = t.reshape(x.shape[0])
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        # Compute embeddings
        emb = self.time_embed(t)  # (B, time_embed_dim)
        class_emb = self.class_embed(labels)  # (B, time_embed_dim)

        # Combine embeddings (additive fusion)
        emb = emb + class_emb  # (B, time_embed_dim)

        # Encoder
        hs = []
        h = x
        cross_attn_idx = 0

        for block_idx, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)

            # Apply cross-attention if this block index has cross-attention
            if block_idx in self.input_cross_attn_indices:
                h = self.cross_attn_blocks[cross_attn_idx](h, class_emb)
                cross_attn_idx += 1

        # Middle block with cross-attention
        h = self.middle_block(h, emb)
        h = self.cross_attn_middle(h, class_emb)

        # Decoder
        cross_attn_idx = len(self.input_cross_attn_indices)  # Start after input cross-attention blocks
        for block_idx, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

            # Apply cross-attention if this block index has cross-attention
            if block_idx in self.output_cross_attn_indices:
                h = self.cross_attn_blocks[cross_attn_idx](h, class_emb)
                cross_attn_idx += 1

        return self.out(h)

