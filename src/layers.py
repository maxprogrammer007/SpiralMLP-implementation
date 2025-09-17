# src/layers.py
import torch
import torch.nn as nn
import math

class SpiralFC(nn.Module):
    """
    Spiral Fully Connected Layer.
    This layer samples features along a spiral trajectory.
    """
    def __init__(self, in_channels, out_channels, A_max=3, T=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A_max = A_max
        self.T = T
        
        # Trainable weights and bias
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Pre-calculate the spiral offsets
        self.register_buffer('offsets', self._generate_spiral_offsets())

    def _amplitude_fn(self, c):
        """Calculates the amplitude A(c) based on Eq. (6) from the paper."""
        if 0 <= c < self.in_channels / 2:
            return math.floor((2 * self.A_max / self.in_channels) * c)
        else:
            return math.floor(2 * self.A_max - (2 * self.A_max / self.in_channels) * c)

    def _generate_spiral_offsets(self):
        """Pre-calculates the (dh, dw) offsets for each input channel."""
        offsets = []
        for c in range(self.in_channels):
            if self.A_max == 0:
                dh, dw = 0, 0
            else:
                Ac = self._amplitude_fn(c)
                angle = (c * 2 * math.pi) / self.T
                # Using round to get integer offsets
                dh = round(Ac * math.cos(angle))
                dw = round(Ac * math.sin(angle))
            offsets.append((dh, dw))
        return torch.tensor(offsets, dtype=torch.long) # Shape: [C_in, 2]

    def forward(self, x):
        # x shape: [B, C_in, H, W]
        B, C_in, H, W = x.shape
        
        # Create a padded version of the input to handle boundary conditions
        padding = self.A_max
        x_padded = nn.functional.pad(x, (padding, padding, padding, padding))
        
        # Prepare for sampling
        sampled_features = torch.zeros(B, C_in, H, W, device=x.device, dtype=x.dtype)
        
        # Sample features from the padded input based on spiral offsets
        for c in range(C_in):
            dh, dw = self.offsets[c]
            # Adjust indices for padding
            h_start, w_start = padding + dh, padding + dw
            sampled_features[:, c, :, :] = x_padded[:, c, h_start:h_start+H, w_start:w_start+W]
        
        # Perform the fully-connected operation (element-wise multiplication and sum over C_in)
        out = torch.einsum('bchw,co->bohw', sampled_features, self.weight)
        
        # Add bias
        out = out + self.bias.view(1, -1, 1, 1)
        
        return out

class SpiralMixing(nn.Module):
    """
    Combines Self-Spiral FC and Cross-Spiral FC with a Merge Head.
    Based on Figure 2(c).
    """
    def __init__(self, dim, A_max, T):
        super().__init__()
        self.self_spiral = SpiralFC(dim, dim, A_max=0, T=T) # Local features
        self.cross_spiral = SpiralFC(dim, dim, A_max=A_max, T=T) # Spatial features
        
        self.merge_head_pool = nn.AdaptiveAvgPool2d(1)
        self.merge_head_linear = nn.Linear(dim, 2) # 2 for weights a1, a2

    def forward(self, x):
        x_self = self.self_spiral(x)
        x_cross = self.cross_spiral(x)
        
        x_merged = x_self + x_cross
        pooled = self.merge_head_pool(x_merged).flatten(1)
        weights = self.merge_head_linear(pooled).softmax(dim=1)
        
        a1 = weights[:, 0].view(-1, 1, 1, 1)
        a2 = weights[:, 1].view(-1, 1, 1, 1)
        
        out = a1 * x_self + a2 * x_cross
        return out

class ChannelMixing(nn.Module):
    """
    Standard MLP for channel mixing.
    """
    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x