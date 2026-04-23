"""
FaithDiff-style alignment module for v5.

Directly ported from FaithDiff's condition_embedding + information_transformer + spatial_ch_proj.
Injection: additive residual after conv_in (320ch feature space).

Reference: FaithDiff/models/unet_2d_condition_vae_extension.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ControlNetConditioningEmbedding(nn.Module):
    """Ported from FaithDiff: 4ch latent → 320ch feature."""

    def __init__(self, conditioning_embedding_channels: int = 320,
                 conditioning_channels: int = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(
            conditioning_channels, conditioning_channels,
            kernel_size=3, padding=1,
        )
        self.norm_in = nn.GroupNorm(
            num_channels=conditioning_channels,
            num_groups=32, eps=1e-6,
        )
        self.conv_out = zero_module(nn.Conv2d(
            conditioning_channels, conditioning_embedding_channels,
            kernel_size=3, padding=1,
        ))

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        conditioning = self.norm_in(conditioning)
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class ResidualAttentionBlock(nn.Module):
    """Ported from FaithDiff: Transformer block with self-attention and MLP."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 2, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class FaithDiffAlignment(nn.Module):
    """FaithDiff-style alignment: additive injection after conv_in.

    Args:
        conditioning_channels: Input z_lq channels (default 4).
        embedding_channels: UNet conv_in output channels (default 320).
        num_trans_channel: Transformer hidden dim (default 640 = 2 * 320).
        num_trans_head: Transformer attention heads (default 8).
        num_trans_layer: Number of transformer layers (default 2).
    """

    def __init__(
        self,
        conditioning_channels: int = 4,
        embedding_channels: int = 320,
        num_trans_channel: int = 640,
        num_trans_head: int = 8,
        num_trans_layer: int = 2,
    ):
        super().__init__()
        self.conditioning_channels = conditioning_channels
        self.embedding_channels = embedding_channels

        self.condition_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=embedding_channels,
            conditioning_channels=conditioning_channels,
        )
        self.information_transformer = nn.Sequential(*[
            ResidualAttentionBlock(num_trans_channel, num_trans_head)
            for _ in range(num_trans_layer)
        ])
        self.spatial_ch_proj = zero_module(
            nn.Linear(num_trans_channel, embedding_channels)
        )

    def forward(self, sample_emb: torch.Tensor, z_lq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample_emb: conv_in output [B, 320, H, W].
            z_lq: LQ latent [B, 4, H, W].
        Returns:
            feat_alpha: [B, 320, H, W] for additive injection.
        """
        # z_lq → input_embedding [B, 320, H, W]
        input_embedding = self.condition_embedding(z_lq)

        # concat sample_emb and input_embedding
        B, C, H, W = sample_emb.shape
        concat_feat = torch.cat([sample_emb, input_embedding], dim=1)  # [B, 640, H, W]
        concat_feat = concat_feat.view(B, 2 * C, H * W).transpose(1, 2)  # [B, HW, 640]

        # transformer
        concat_feat = self.information_transformer(concat_feat)  # [B, HW, 640]

        # spatial projection
        feat_alpha = self.spatial_ch_proj(concat_feat)  # [B, HW, 320]
        feat_alpha = feat_alpha.transpose(1, 2).view(B, C, H, W)  # [B, 320, H, W]

        return feat_alpha
