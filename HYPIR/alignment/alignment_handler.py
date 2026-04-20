"""
Alignment Handler: orchestrates the full alignment module pipeline.

Combines LatentAlignmentEncoder, ControlNetConditioningEmbedding,
ResidualAttentionBlock layers, and spatial projection into a single
module that can be injected into UNet's forward pass.

Operates in VAE latent space:
  Input: x_en [B, 4, H/8, W/8] — VAE Encoder output on RM-restored image
  Flow:
    1. LatentAlignmentEncoder: x_en [B, 4, H/8, W/8] → [B, 512, H/8, W/8]
    2. ConditionEmbedding: [B, 512, H/8, W/8] → [B, 320, H/8, W/8]
    3. Cat with UNet conv_in output → [B, 640, H/8, W/8]
    4. Flatten → Transformer → Project → Reshape → residual add

Args:
    unet_conv_channels: UNet conv_in output channels (320 for SD2).
    latent_channels: VAE latent channels (4 for SD2).
    encoder_block_out_channels: Alignment encoder block channels.
    transformer_layers: Number of ResidualAttentionBlock layers.
    transformer_dim: Transformer hidden dim (= unet_conv_channels * 2).
    transformer_heads: Number of attention heads.
    use_condition_embedding: Whether to apply condition embedding preprocessing.
    add_sample: Whether to add (True) or replace (False) the residual.
"""

import torch
import torch.nn as nn

from .alignment_module import (
    ControlNetConditioningEmbedding,
    ResidualAttentionBlock,
    zero_module,
)
from .alignment_encoder import LatentAlignmentEncoder


class AlignmentHandler(nn.Module):
    def __init__(
        self,
        unet_conv_channels: int = 320,
        latent_channels: int = 4,
        encoder_block_out_channels: tuple = (128, 256, 512, 512),
        transformer_layers: int = 2,
        transformer_dim: int = 640,
        transformer_heads: int = 8,
        use_condition_embedding: bool = True,
        add_sample: bool = True,
    ):
        super().__init__()
        self.unet_conv_channels = unet_conv_channels
        self.transformer_dim = transformer_dim
        self.add_sample = add_sample
        self.use_condition_embedding = use_condition_embedding

        # 1. Latent alignment encoder: VAE latent → feature space (no spatial downsample)
        self.alignment_encoder = LatentAlignmentEncoder(
            in_channels=latent_channels,
            block_out_channels=encoder_block_out_channels,
        )
        encoder_out_channels = encoder_block_out_channels[-1]  # 512

        # 2. Condition embedding: preprocess encoder output
        self.condition_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=unet_conv_channels,
            conditioning_channels=encoder_out_channels,
        )

        # 3. Information transformer: fuse UNet features with alignment features
        self.information_transformer_layers = nn.Sequential(
            *[
                ResidualAttentionBlock(transformer_dim, transformer_heads)
                for _ in range(transformer_layers)
            ]
        )

        # 4. Spatial channel projection: project back to UNet conv_in channels
        self.spatial_ch_projs = zero_module(nn.Linear(transformer_dim, unet_conv_channels))

    def encode_alignment_input(self, x_en: torch.Tensor) -> torch.Tensor:
        """
        Encode VAE latent features into alignment feature maps.

        Args:
            x_en: VAE Encoder output [B, 4, H/8, W/8].
        Returns:
            Alignment features [B, 320, H/8, W/8].
        """
        # Cast input to match handler dtype (params are float32, input may be bf16)
        x_en = x_en.to(dtype=next(self.parameters()).dtype)
        features = self.alignment_encoder(x_en)
        if self.use_condition_embedding:
            features = self.condition_embedding(features)
        return features

    def fuse_with_unet_features(
        self, unet_sample: torch.Tensor, alignment_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse alignment features with UNet conv_in output.

        Args:
            unet_sample: UNet conv_in output [B, 320, H/8, W/8].
            alignment_features: Processed alignment features [B, 320, H/8, W/8].

        Returns:
            Fused sample [B, 320, H/8, W/8].
        """
        # Cast unet_sample to match alignment handler dtype for concatenation
        unet_sample = unet_sample.to(dtype=alignment_features.dtype)
        batch_size, channel, height, width = alignment_features.shape
        # Concat along channel dim → [B, 640, H/8, W/8]
        concat_feat = torch.cat([unet_sample, alignment_features], dim=1)
        # Flatten spatial dims → [B, H*W, 640]
        concat_feat = concat_feat.view(batch_size, self.transformer_dim, height * width).transpose(1, 2)
        # Transformer fusion
        concat_feat = self.information_transformer_layers(concat_feat)
        # Project to UNet channels → [B, H*W, 320]
        feat_alpha = self.spatial_ch_projs(concat_feat)
        # Reshape back → [B, 320, H/8, W/8]
        feat_alpha = feat_alpha.transpose(1, 2).view(batch_size, channel, height, width)
        # Residual connection
        if self.add_sample:
            return unet_sample + feat_alpha
        return feat_alpha

    def forward(
        self, x_en: torch.Tensor, unet_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Full alignment pipeline.

        Args:
            x_en: VAE Encoder output on RM image [B, 4, H/8, W/8].
            unet_sample: UNet conv_in output [B, 320, H/8, W/8].

        Returns:
            Fused UNet sample [B, 320, H/8, W/8].
        """
        alignment_features = self.encode_alignment_input(x_en)
        return self.fuse_with_unet_features(unet_sample, alignment_features)
