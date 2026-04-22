"""
Alignment module components adapted from FaithDiff for RM+HYPIR integration.

v4 Architecture:
- AlignmentModule: Dual-side alignment (z_lq, x_hq_t) → aligned latent
- Input: z_lq [B,4,H,W], x_hq_t [B,4,H,W] (both VAE latent space)
- Output: aligned [B,4,H,W] (residual added to x_hq_t)
- Structure: concat(8ch) → encoder(128→256→512) → transformer → proj_out(512→4, zero_init) → residual

Injection point: BEFORE conv_in (aligned latent goes directly into UNet as sample).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class QuickGELU(nn.Module):
    """Fast approximation of GELU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """LayerNorm subclass that preserves input dtype (fp16 safe)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)


class ResidualSwinBlock(nn.Module):
    """
    ResNet-style block for the alignment encoder.

    Structure: GroupNorm → SiLU → Conv3x3 → GroupNorm → SiLU → Conv3x3 + residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class ResidualAttentionBlock(nn.Module):
    """
    Transformer-style block with self-attention and MLP.

    Used in the information transformer layers that fuse the concatenated
    z_lq and x_hq_t features.

    Args:
        d_model: Model dimension (typically 512 for encoder output).
        n_head: Number of attention heads.
        attn_mask: Optional attention mask.
    """

    def __init__(
        self, d_model: int = 512, n_head: int = 8, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AlignmentModule(nn.Module):
    """
    FaithDiff-style dual-side alignment module.

    Input: z_lq [B, 4, H, W], x_hq_t [B, 4, H, W] (both VAE latent space)
    Output: aligned [B, 4, H, W]

    Structure:
        concat(z_lq, x_hq_t) → [B, 8, H, W]
        encoder: 8ch → 128 → 256 → 512 (ResNet blocks, no spatial downsample)
        transformer: ResidualAttentionBlock layers (spatial flatten → attention → reshape)
        proj_out: zero_init Conv2d(512, 4, 1)
        output: x_hq_t + proj_out(transformer_output)

    Args:
        latent_channels: Input latent channels (4 for SD VAE).
        hidden_channels: First encoder block channels.
        encoder_channels: Progressive channel expansion (128, 256, 512).
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_channels: int = 128,
        encoder_channels: tuple = (128, 256, 512),
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder_channels = encoder_channels
        self.hidden_dim = encoder_channels[-1]  # 512

        # Input: concat(z_lq, x_hq_t) → 8 channels
        in_ch = latent_channels * 2  # 8

        # Encoder: progressive channel expansion
        # 8 → 128 → 256 → 512
        self.encoder = nn.ModuleList()

        # First block: 8 → 128
        self.encoder.append(ResidualSwinBlock(in_ch, encoder_channels[0]))

        # Subsequent blocks: 128 → 256 → 512
        for i in range(len(encoder_channels) - 1):
            self.encoder.append(ResidualSwinBlock(encoder_channels[i], encoder_channels[i + 1]))

        # Transformer: ResidualAttentionBlock layers
        self.transformer = nn.Sequential(
            *[ResidualAttentionBlock(self.hidden_dim, num_heads) for _ in range(num_layers)]
        )

        # Output projection: 512 → 4, zero-initialized for residual
        self.proj_out = zero_module(nn.Conv2d(self.hidden_dim, latent_channels, kernel_size=1))

    def forward(self, z_lq: torch.Tensor, x_hq_t: torch.Tensor) -> torch.Tensor:
        """
        Dual-side alignment forward pass.

        Args:
            z_lq: LQ latent [B, 4, H, W].
            x_hq_t: HQ latent (current estimate) [B, 4, H, W].

        Returns:
            aligned: Aligned latent [B, 4, H, W].
        """
        # Concat along channel dimension
        x = torch.cat([z_lq, x_hq_t], dim=1)  # [B, 8, H, W]

        # Encoder: channel expansion, spatial preserved
        for block in self.encoder:
            x = block(x)  # [B, 512, H, W]

        # Transformer: spatial attention
        B, C, H, W = x.shape
        # Flatten spatial dims → [H*W, B, C] for MultiheadAttention
        x = x.view(B, C, H * W).permute(2, 0, 1)  # [H*W, B, C]
        x = self.transformer(x)  # [H*W, B, C]
        # Reshape back → [B, C, H, W]
        x = x.permute(1, 2, 0).view(B, C, H, W)  # [B, 512, H, W]

        # Output projection → [B, 4, H, W]
        delta = self.proj_out(x)

        # Residual: x_hq_t + delta
        return x_hq_t + delta
