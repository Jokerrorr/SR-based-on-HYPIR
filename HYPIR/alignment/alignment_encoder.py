"""
Latent Alignment Encoder: encodes VAE latent features (x_en) into
higher-dimensional feature maps for fusion with UNet conv_in output.

Unlike FaithDiff's pixel-space encoder (3ch input, 8x downsample),
this encoder operates entirely in latent space:
  - Input: x_en [B, 4, H/8, W/8] (VAE encoded RM output)
  - Output: [B, out_channels, H/8, W/8] (no spatial downsampling)

Architecture: progressive ResNet blocks that expand channels while
preserving spatial resolution.
"""

import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block


class LatentAlignmentEncoder(nn.Module):
    """
    Latent-space encoder for alignment module.

    Takes VAE-encoded features (4ch, H/8×W/8) and produces
    higher-dimensional feature maps at the same spatial resolution.

    Uses DownEncoderBlock2D with add_downsample=False for all blocks,
    so spatial size is preserved while channels expand.

    Architecture (default):
        conv_in (4 → 128)
        block 0 (128 → 128, no downsample)
        block 1 (128 → 256, no downsample)
        block 2 (256 → 512, no downsample)
        block 3 (512 → 512, no downsample)
        mid_block (512)

    Output spatial resolution = input spatial resolution (no downsampling).

    Args:
        in_channels: Input latent channels (4 for VAE latent).
        block_out_channels: Channel sizes for each block.
        layers_per_block: ResNet layers per block.
        norm_num_groups: GroupNorm groups.
        act_fn: Activation function.
    """

    def __init__(
        self,
        in_channels: int = 4,
        block_out_channels: tuple = (128, 256, 512, 512),
        down_block_types: tuple = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])
        self.block_out_channels = block_out_channels

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            # All blocks: no downsampling (input is already at H/8 resolution)
            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=False,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        self.gradient_checkpointing = False

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        return sample

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.encode(sample)
