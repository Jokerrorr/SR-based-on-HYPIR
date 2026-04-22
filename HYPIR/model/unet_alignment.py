"""
UNet with alignment module integration for RM+HYPIR pipeline (v4).

Extends diffusers' UNet2DConditionModel with alignment module injection
BEFORE conv_in (FaithDiff v4 approach).

v4 Architecture:
  z_lq, x_hq_t → AlignmentModule → aligned [B, 4, H, W]
  aligned → conv_in → down_blocks → mid_block → up_blocks → conv_out

Key changes from v3:
- Alignment input: dual-side (z_lq, x_hq_t) instead of single-side x_en
- Alignment output: 4ch latent instead of 320ch feature
- Injection point: BEFORE conv_in instead of AFTER conv_in
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

    v4: The alignment module is applied BEFORE conv_in, taking dual-side
    inputs (z_lq, x_hq_t) and producing an aligned 4ch latent.
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
        x_hq_t: Optional[torch.FloatTensor] = None,
        z_lq: Optional[torch.FloatTensor] = None,
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
        # Check if alignment should be applied
        # Need both z_lq and x_hq_t for dual-side alignment
        apply_alignment = (
            z_lq is not None
            and x_hq_t is not None
            and self.alignment_handler is not None
        )

        if apply_alignment:
            return self._forward_with_alignment(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                z_lq=z_lq,
                x_hq_t=x_hq_t,
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
            # Standard UNet forward without alignment
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
        z_lq: torch.FloatTensor,
        x_hq_t: torch.FloatTensor,
        **kwargs,
    ) -> Union[UNetAlignmentOutput, Tuple]:
        """
        Forward pass with alignment module injection BEFORE conv_in.

        v4 approach:
        1. Apply alignment: aligned = AlignmentModule(z_lq, x_hq_t)
        2. Pass aligned through conv_in and rest of UNet

        Args:
            sample: Input sample (typically z_lq * scaling_factor).
            z_lq: LQ latent [B, 4, H/8, W/8].
            x_hq_t: Current HQ latent estimate [B, 4, H/8, W/8].
        """
        # Apply alignment BEFORE conv_in
        # Convert inputs to float32 for alignment handler (params are fp32)
        handler_dtype = next(self.alignment_handler.parameters()).dtype
        aligned = self.alignment_handler(
            z_lq.to(dtype=handler_dtype),
            x_hq_t.to(dtype=handler_dtype)
        )

        # Cast aligned back to UNet's dtype before conv_in
        unet_dtype = self.unet.conv_in.weight.dtype
        aligned = aligned.to(dtype=unet_dtype)

        # Compute conv_in output from aligned latent
        # NO torch.no_grad() here - gradient must flow to alignment handler
        original_conv_in = self.unet.conv_in
        conv_out = original_conv_in(aligned)

        # Replace conv_in with identity-like module that returns pre-computed result
        class _InjectedConvIn(nn.Module):
            def __init__(self, precomputed):
                super().__init__()
                self.precomputed = precomputed

            def forward(self, x):
                return self.precomputed

        injected = _InjectedConvIn(conv_out)
        self.unet.conv_in = injected

        try:
            out = self.unet(
                sample,  # Pass original sample (won't be used by injected conv_in)
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
                # Handle double-dot keys from training: "unet..X" -> "X"
                key = k[len("unet."):]
                key = key.lstrip(".")
                unet_sd[key] = v
            elif k.startswith("alignment_handler."):
                key = k[len("alignment_handler."):]
                key = key.lstrip(".")
                alignment_sd[key] = v
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
