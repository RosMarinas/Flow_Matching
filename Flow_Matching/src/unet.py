import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sampled weights during initialization. These are fixed during optimization.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.dense(x)[..., None, None]

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    """
    A residual block with time embeddings.
    """
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.bn1 = nn.GroupNorm(8, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels)
        
        self.time_emb = Dense(time_channels, out_channels)
        
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels)
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        # x: [B, C, H, W]
        # t: [B, time_channels] -> reshaped to [B, out_channels, 1, 1] by Dense
        
        h = self.swish(self.bn1(x))
        h = self.conv1(h)
        
        # Add time embedding
        h += self.time_emb(self.swish(t))
        
        h = self.swish(self.bn2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """
    Self-attention block for U-Net.
    """
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, padding=0)
        self.output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # [B, C, H*W]
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        
        # [B, H*W, H*W]
        w = torch.bmm(q.permute(0, 2, 1), k) * (int(C) ** (-0.5))
        w = F.softmax(w, dim=-1)
        
        # [B, C, H*W]
        h = torch.bmm(v, w.permute(0, 2, 1))
        h = h.view(B, C, H, W)
        
        return x + self.output(h)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, time_emb_dim=128):
        super().__init__()
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(time_emb_dim, scale=30.0),
            nn.Linear(time_emb_dim, time_emb_dim),
            Swish(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Downsampling
        self.conv_in = Conv2d(in_channels, base_channels)
        
        # Block 1
        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.attn1 = AttentionBlock(base_channels)
        self.down_conv1 = Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.down_pool1 = nn.MaxPool2d(2) 
        
        # Block 2
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.attn2 = AttentionBlock(base_channels * 2)
        self.down_conv2 = Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.down_pool2 = nn.MaxPool2d(2)
        
        # Block 3
        self.down3 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.attn3 = AttentionBlock(base_channels * 4)
        self.down_conv3 = Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)
        self.down_pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bot1 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.attn_bot = AttentionBlock(base_channels * 8)
        self.bot2 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        
        # Upsampling
        # Up 1 (From down3)
        self.up_sample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 4, time_emb_dim) # Skip down3 (4ch) + up (4ch) = 8ch? No, concat is on dim 1.
        # Wait, if we concat [B, 4*base, H, W] and [B, 4*base, H, W], we get 8*base input to ResBlock.
        # But ResBlock definition takes (in_channels, out_channels).
        # We need to be careful with in_channels of up blocks.
        # Let's check ResBlock usage: ResidualBlock(in, out)
        # Here: input is concat(up_feat, skip_feat).
        # up_feat is base*4. skip_feat (from down3 output) is base*4.
        # So input is base*8. Target out is base*4.
        
        # Re-check down3 output:
        # down3 outputs base*4.
        # then down_conv3 makes it base*8.
        # so skip connection from down3 is base*4.
        
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 4, time_emb_dim)
        self.attn_up1 = AttentionBlock(base_channels * 4)
        
        # Up 2 (From down2)
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        # Skip down2 is base*2. Up is base*2. Concat = base*4.
        self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.attn_up2 = AttentionBlock(base_channels * 2)

        # Up 3 (From down1)
        self.up_sample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv3 = Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        # Skip down1 is base. Up is base. Concat = base*2.
        self.up3 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        self.attn_up3 = AttentionBlock(base_channels)
        
        # Output
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.swish = Swish()
        self.conv_out = Conv2d(base_channels, out_channels)

    def forward(self, t, x):
        # x: [B, 3, H, W]
        
        t_emb = self.time_mlp(t)
        
        h1 = self.conv_in(x)
        h1 = self.down1(h1, t_emb)
        h1 = self.attn1(h1) # Skip 1: base_channels
        
        h2 = self.down_pool1(self.down_conv1(h1))
        h2 = self.down2(h2, t_emb)
        h2 = self.attn2(h2) # Skip 2: base*2
        
        h3 = self.down_pool2(self.down_conv2(h2))
        h3 = self.down3(h3, t_emb)
        h3 = self.attn3(h3) # Skip 3: base*4
        
        h4 = self.down_pool3(self.down_conv3(h3))
        # Bottleneck
        h4 = self.bot1(h4, t_emb)
        h4 = self.attn_bot(h4)
        h4 = self.bot2(h4, t_emb)
        
        # Up 1
        h_up1 = self.up_sample1(h4)
        h_up1 = self.up_conv1(h_up1)
        h_up1 = torch.cat([h_up1, h3], dim=1) 
        h_up1 = self.up1(h_up1, t_emb)
        h_up1 = self.attn_up1(h_up1)
        
        # Up 2
        h_up2 = self.up_sample2(h_up1)
        h_up2 = self.up_conv2(h_up2)
        h_up2 = torch.cat([h_up2, h2], dim=1) 
        h_up2 = self.up2(h_up2, t_emb)
        h_up2 = self.attn_up2(h_up2)
        
        # Up 3
        h_up3 = self.up_sample3(h_up2)
        h_up3 = self.up_conv3(h_up3)
        h_up3 = torch.cat([h_up3, h1], dim=1)
        h_up3 = self.up3(h_up3, t_emb)
        h_up3 = self.attn_up3(h_up3)
        
        out = self.swish(self.norm_out(h_up3))
        out = self.conv_out(out)
        
        return out
