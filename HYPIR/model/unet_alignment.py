"""
UNet with alignment module integration for RM+HYPIR pipeline.

Extends diffusers' UNet2DConditionModel to inject alignment features
after conv_in, following FaithDiff's approach.

The alignment input is x_en (VAE Encoder output on RM-restored image),
not pixel-space RM output.

  z_lq → conv_in → [B, 320, H/8, W/8]
                        ↓
  x_en → AlignmentHandler → alignment_features [B, 320, H/8, W/8]
                        ↓
              sample = sample + feat_alpha
                        ↓
              down_blocks → mid_block → up_blocks → conv_out
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.utils import BaseOutput
from dataclasses import dataclass

from HYPIR.alignment.alignment_handler import AlignmentHandler


@dataclass
class UNetAlignmentOutput(BaseOutput):
    sample: torch.FloatTensor = None


class UNetAlignment(nn.Module):
    """
    Wrapper around UNet2DConditionModel with alignment module injection.

    The alignment module is inserted between conv_in and down_blocks,
    fusing VAE-encoded RM features (x_en) into the UNet feature stream.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        alignment_handler: AlignmentHandler,
    ):
        super().__init__()
        self.unet = unet
        self.alignment_handler = alignment_handler

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        x_en: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNetAlignmentOutput, Tuple]:
        if x_en is not None and self.alignment_handler is not None:
            return self._forward_with_alignment(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                x_en=x_en,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
            )
        else:
            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
            )
            return out

    def _forward_with_alignment(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        x_en: torch.FloatTensor,
        **kwargs,
    ) -> Union[UNetAlignmentOutput, Tuple]:
        """
        Forward pass with alignment module injection.

        x_en: VAE Encoder output on RM image [B, 4, H/8, W/8].
        """
        original_conv_in = self.unet.conv_in

        # Compute alignment features from VAE-encoded RM output
        with torch.no_grad():
            conv_out = original_conv_in(sample)

        # Apply alignment: fuse x_en features with UNet conv_in output
        aligned = self.alignment_handler.fuse_with_unet_features(
            unet_sample=conv_out,
            alignment_features=self.alignment_handler.encode_alignment_input(x_en),
        )

        # Replace conv_in with identity-like module that returns pre-computed result
        class _InjectedConvIn(nn.Module):
            def __init__(self, precomputed):
                super().__init__()
                self.precomputed = precomputed

            def forward(self, x):
                return self.precomputed

        injected = _InjectedConvIn(aligned)
        self.unet.conv_in = injected

        try:
            out = self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
        finally:
            self.unet.conv_in = original_conv_in

        return out

    # Delegate LoRA adapter methods
    def add_adapter(self, adapter_config):
        return self.unet.add_adapter(adapter_config)

    @property
    def config(self):
        return self.unet.config

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()

    def parameters(self, recurse: bool = True):
        for p in self.unet.parameters(recurse):
            yield p
        if self.alignment_handler is not None:
            for p in self.alignment_handler.parameters(recurse):
                yield p

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, p in self.unet.named_parameters(prefix="unet." + prefix if prefix else "unet.", recurse=recurse):
            yield name, p
        if self.alignment_handler is not None:
            for name, p in self.alignment_handler.named_parameters(prefix="alignment_handler." + prefix if prefix else "alignment_handler.", recurse=recurse):
                yield name, p

    def to(self, *args, **kwargs):
        self.unet = self.unet.to(*args, **kwargs)
        if self.alignment_handler is not None:
            self.alignment_handler = self.alignment_handler.to(*args, **kwargs)
        return self

    def state_dict(self, *args, **kwargs):
        sd = {}
        for k, v in self.unet.state_dict(*args, **kwargs).items():
            sd[f"unet.{k}"] = v
        for k, v in self.alignment_handler.state_dict(*args, **kwargs).items():
            sd[f"alignment_handler.{k}"] = v
        return sd

    def load_state_dict(self, state_dict, strict=True):
        unet_sd = {}
        alignment_sd = {}
        for k, v in state_dict.items():
            if k.startswith("unet."):
                unet_sd[k[5:]] = v
            elif k.startswith("alignment_handler."):
                alignment_sd[k[18:]] = v
            else:
                unet_sd[k] = v

        m_unet, u_unet = self.unet.load_state_dict(unet_sd, strict=False)
        if alignment_sd:
            m_align, u_align = self.alignment_handler.load_state_dict(alignment_sd, strict=strict)
        else:
            m_align, u_align = [], []
        return m_unet + m_align, u_unet + u_align

    def get_alignment_params(self):
        """Return only alignment handler parameters (for separate optimization)."""
        return list(self.alignment_handler.parameters())
