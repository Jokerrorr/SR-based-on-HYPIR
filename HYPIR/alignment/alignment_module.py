"""
Alignment module components adapted from FaithDiff for RM+HYPIR integration.

This alignment module takes VAE-encoded features x_en (output of VAE Encoder on RM output)
and aligns them with x_hq features before feeding into UNet.

Pipeline: LQ → RM → VAE Encoder → x_en → Alignment → UNet → VAE Decoder → HQ

Key difference from FaithDiff: operates in latent space (4ch, H/8 × W/8),
not pixel space. No spatial downsampling needed.
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


class ControlNetConditioningEmbedding(nn.Module):
    """
    Preprocess alignment conditioning inputs.

    Takes alignment encoder output and projects to UNet conv_in channel dim.
    Architecture: GroupNorm → Conv → SiLU → zero_module(Conv)

    Args:
        conditioning_embedding_channels: Output channels (320, matching UNet conv_in).
        conditioning_channels: Input channels (alignment encoder output channels).
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_channels: int = 4,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            conditioning_channels, conditioning_channels, kernel_size=3, padding=1
        )
        self.norm_in = nn.GroupNorm(
            num_channels=conditioning_channels,
            num_groups=min(32, conditioning_channels),
            eps=1e-6,
        )
        self.conv_out = zero_module(
            nn.Conv2d(
                conditioning_channels,
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        conditioning = self.norm_in(conditioning)
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class ResidualAttentionBlock(nn.Module):
    """
    Transformer-style block with self-attention and MLP.

    Used in the information transformer layers that fuse UNet conv_in output
    with alignment features.

    Args:
        d_model: Model dimension (typically 640 = 320 * 2 for concatenated features).
        n_head: Number of attention heads.
        attn_mask: Optional attention mask.
    """

    def __init__(
        self, d_model: int = 640, n_head: int = 8, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 2)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 2, d_model)),
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
