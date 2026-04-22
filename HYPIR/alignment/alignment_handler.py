"""
Alignment Handler: wrapper for AlignmentModule (v4).

Simplified wrapper that provides the AlignmentModule interface
for integration with UNetAlignment.

Operates in VAE latent space:
  Input: z_lq [B, 4, H/8, W/8] — VAE Encoder output on LQ image
         x_hq_t [B, 4, H/8, W/8] — current HQ latent estimate (timestep t)
  Output: aligned [B, 4, H/8, W/8] — aligned latent for UNet input

The alignment module is injected BEFORE conv_in (output replaces the
sample that goes into conv_in), not after.

Args:
    latent_channels: VAE latent channels (4 for SD2).
    hidden_channels: First encoder block channels.
    encoder_channels: Progressive channel expansion.
    num_layers: Number of transformer layers.
    num_heads: Number of attention heads.
"""

import torch
import torch.nn as nn

from .alignment_module import AlignmentModule


class AlignmentHandler(nn.Module):
    """
    Wrapper around AlignmentModule for UNet integration.

    Provides a simple forward(z_lq, x_hq_t) → aligned interface.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_channels: int = 128,
        encoder_channels: tuple = (128, 256, 512),
        num_layers: int = 2,
        num_heads: int = 8,
        # Legacy parameters (ignored, kept for config compatibility)
        unet_conv_channels: int = None,
        transformer_dim: int = None,
        transformer_heads: int = None,
        transformer_layers: int = None,
        encoder_block_out_channels: tuple = None,
        use_condition_embedding: bool = None,
        add_sample: bool = None,
    ):
        super().__init__()

        # Map legacy config parameters to new parameter names if provided
        if encoder_block_out_channels is not None:
            encoder_channels = tuple(encoder_block_out_channels)
        if transformer_layers is not None:
            num_layers = transformer_layers
        if transformer_heads is not None:
            num_heads = transformer_heads

        self.alignment_module = AlignmentModule(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            encoder_channels=encoder_channels,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    def forward(self, z_lq: torch.Tensor, x_hq_t: torch.Tensor) -> torch.Tensor:
        """
        Full alignment pipeline.

        Args:
            z_lq: LQ latent [B, 4, H/8, W/8].
            x_hq_t: Current HQ latent estimate [B, 4, H/8, W/8].

        Returns:
            Aligned latent [B, 4, H/8, W/8].
        """
        return self.alignment_module(z_lq, x_hq_t)
