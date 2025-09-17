# src/model.py
import torch
import torch.nn as nn

# Import the fundamental layers from the new layers.py file
from .layers import SpiralMixing, ChannelMixing, PatchEmbed

class SpiralBlock(nn.Module):
    """
    The main building block of SpiralMLP, based on Figure 2(b).
    """
    def __init__(self, dim, mlp_ratio=4., A_max=3, T=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = SpiralMixing(dim, A_max=A_max, T=T)
        
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixer = ChannelMixing(dim, mlp_ratio=mlp_ratio)
        
    def forward(self, x):
        # Input x shape: [B, H, W, C]
        # SpiralMixing expects [B, C, H, W]
        x_token_mixed = self.token_mixer(self.norm1(x).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x + x_token_mixed
        
        x_channel_mixed = self.channel_mixer(self.norm2(x))
        x = x + x_channel_mixed
        return x

class SpiralMLP(nn.Module):
    """
    Main SpiralMLP model, based on PVT-style architecture in Figure 2(a).
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 6, 2],
                 mlp_ratios=[4, 4, 4, 4], A_maxs=[3, 3, 3, 3], Ts=[8, 8, 8, 8]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Stage 1: Patch embedding and first set of blocks
        self.patch_embed1 = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.stage1 = nn.Sequential(*[
            SpiralBlock(dim=embed_dims[0], mlp_ratio=mlp_ratios[0], A_max=A_maxs[0], T=Ts[0])
            for _ in range(depths[0])
        ])

        # Subsequent stages with downsampling
        self.stages = nn.ModuleList()
        for i in range(1, 4):
            patch_embed = PatchEmbed(patch_size=2, in_chans=embed_dims[i-1], embed_dim=embed_dims[i])
            stage_blocks = nn.Sequential(*[
                SpiralBlock(dim=embed_dims[i], mlp_ratio=mlp_ratios[i], A_max=A_maxs[i], T=Ts[i])
                for _ in range(depths[i])
            ])
            self.stages.append(nn.ModuleDict({'patch_embed': patch_embed, 'blocks': stage_blocks}))
        
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x):
        # Stage 1
        x = self.patch_embed1(x) # [B, H/4, W/4, C1]
        x = self.stage1(x)
        
        # Stages 2-4
        for stage in self.stages:
            x = x.permute(0, 3, 1, 2) # [B, C, H, W] for patch embed
            x = stage['patch_embed'](x)
            x = stage['blocks'](x)

        # Classification Head
        x = self.norm(x)
        x = x.mean(dim=[1, 2]) # Global average pooling
        x = self.head(x)
        return x

# --- Sanity Check ---
if __name__ == '__main__':
    # Define the SpiralMLP-B1 configuration for CIFAR-10
    model_b1 = SpiralMLP(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
        A_maxs=[3, 3, 3, 3],
        Ts=[8, 8, 8, 8]
    )
    
    print("SpiralMLP-B1 Model (imports from layers.py):")
    # print(model_b1) # You can uncomment this to see the full structure
    
    dummy_input = torch.randn(4, 3, 32, 32)
    print("\nTesting forward pass...")
    output = model_b1(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 10)
    print("✅ Forward pass successful!")