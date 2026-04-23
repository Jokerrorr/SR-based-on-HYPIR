"""
UNet with alignment injection.

Additive injection AFTER conv_in: sample_emb + feat_alpha.
Preserves the original UNet forward flow, adds a learned residual from LQ latent.

Architecture:
  x_hq_t → conv_in → sample_emb [B, 320, H, W] ─┐
                                                    ├→ sample_emb + feat_alpha → Down/Mid/Up → noise_pred
  z_lq → Alignment(sample_emb, z_lq) ─────────────┘
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.utils import BaseOutput
from dataclasses import dataclass

from HYPIR.alignment.alignment import Alignment


@dataclass
class UNetAlignmentOutput(BaseOutput):
    sample: torch.FloatTensor = None


class UNetAlignment(nn.Module):
    """Wrapper around UNet2DConditionModel with FaithDiff-style additive injection.

    The alignment module is applied AFTER conv_in, adding feat_alpha to sample_emb.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        alignment_handler: Alignment,
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
        """FaithDiff-style forward: additive injection after conv_in.

        1. sample_emb = conv_in(x_hq_t)
        2. feat_alpha = Alignment(sample_emb, z_lq)
        3. sample_emb = sample_emb + feat_alpha
        4. Continue UNet forward with injected sample_emb
        """
        # 1. Compute conv_in output from x_hq_t
        unet_dtype = self.unet.conv_in.weight.dtype
        sample_emb = self.unet.conv_in(x_hq_t.to(dtype=unet_dtype))  # [B, 320, H, W]

        # 2. Compute feat_alpha
        handler_dtype = next(self.alignment_handler.parameters()).dtype
        feat_alpha = self.alignment_handler(
            sample_emb.to(dtype=handler_dtype),
            z_lq.to(dtype=handler_dtype),
        )  # [B, 320, H, W]

        # 3. Additive injection
        sample_emb = sample_emb + feat_alpha.to(dtype=unet_dtype)

        # 4. Replace conv_in to return pre-computed result
        original_conv_in = self.unet.conv_in

        class _InjectedConvIn(nn.Module):
            def __init__(self, precomputed):
                super().__init__()
                self.precomputed = precomputed

            def forward(self, x):
                return self.precomputed

        self.unet.conv_in = _InjectedConvIn(sample_emb)

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
        for name, p in self.unet.named_parameters(
            prefix="unet." + prefix if prefix else "unet.", recurse=recurse
        ):
            yield name, p
        if self.alignment_handler is not None:
            for name, p in self.alignment_handler.named_parameters(
                prefix="alignment_handler." + prefix
                if prefix else "alignment_handler.",
                recurse=recurse,
            ):
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
            m_align, u_align = self.alignment_handler.load_state_dict(
                alignment_sd, strict=strict
            )
        else:
            m_align, u_align = [], []
        return m_unet + m_align, u_unet + u_align

    def get_alignment_params(self):
        return list(self.alignment_handler.parameters())
