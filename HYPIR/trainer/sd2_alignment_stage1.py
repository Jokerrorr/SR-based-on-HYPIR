"""
Stage 1 Alignment Pretraining Trainer.

Freezes UNet backbone (no LoRA), only trains Alignment Handler parameters.
Uses the same adversarial training framework as SD2AlignmentTrainer,
but the generator's only learnable components are the alignment module.

Purpose: Let the alignment module learn to extract useful features from
x_en (VAE-encoded RM output) before jointly training with UNet LoRA.

After Stage 1, save alignment weights for Stage 2 initialization.
"""

import os
import csv
import logging
from typing import List, Dict

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import make_grid

from HYPIR.trainer.sd2_alignment import SD2AlignmentTrainer
from HYPIR.alignment.alignment_handler import AlignmentHandler
from HYPIR.model.unet_alignment import UNetAlignment

logger = get_logger(__name__, log_level="INFO")


class SD2AlignmentStage1Trainer(SD2AlignmentTrainer):
    """Stage 1: Freeze UNet, only train alignment handler."""

    def init_generator(self):
        # Load base UNet (frozen, no LoRA)
        unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path, subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        unet.eval().requires_grad_(False)

        if self.config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # No LoRA — UNet is completely frozen
        logger.info("Stage 1: UNet frozen, no LoRA added")

        # Create alignment handler (same as parent)
        alignment_cfg = getattr(self.config, "alignment", None)
        if alignment_cfg is not None:
            handler_kwargs = dict(
                unet_conv_channels=320,
                latent_channels=getattr(alignment_cfg, "latent_channels", 4),
                encoder_block_out_channels=tuple(
                    getattr(alignment_cfg, "encoder_block_out_channels", [128, 256, 512, 512])
                ),
                transformer_layers=getattr(alignment_cfg, "transformer_layers", 2),
                transformer_dim=getattr(alignment_cfg, "transformer_dim", 640),
                transformer_heads=getattr(alignment_cfg, "transformer_heads", 8),
                use_condition_embedding=getattr(alignment_cfg, "use_condition_embedding", True),
                add_sample=getattr(alignment_cfg, "add_sample", True),
            )
        else:
            handler_kwargs = dict(unet_conv_channels=320)

        handler = AlignmentHandler(**handler_kwargs)

        # Wrap UNet with alignment
        self.G = UNetAlignment(unet=unet, alignment_handler=handler)

        # Move alignment handler params to float32 for stable training
        for p in handler.parameters():
            p.data = p.to(torch.float32)

        align_params = sum(p.numel() for p in handler.parameters())
        logger.info(f"Stage 1: Alignment handler params: {align_params/1e6:.2f}M")
        logger.info(f"Stage 1: UNet trainable params: "
                     f"{sum(p.numel() for p in unet.parameters() if p.requires_grad)}")

    def init_optimizers(self):
        """Only alignment handler params in G optimizer."""
        logger.info(f"Stage 1: Creating {self.config.optimizer_type} optimizers (alignment only)")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = None

        # Only alignment handler parameters
        align_lr = getattr(self.config, "alignment_lr", self.config.lr_G)
        self.G_params = list(self.G.alignment_handler.parameters())
        logger.info(f"Stage 1: G optimizer params: {len(self.G_params)} tensors, lr={align_lr}")

        self.G_opt = optimizer_cls(
            self.G_params,
            lr=align_lr,
            **self.config.opt_kwargs,
        )

        self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
        self.D_opt = optimizer_cls(
            self.D_params,
            lr=self.config.lr_D,
            **self.config.opt_kwargs,
        )

    def attach_accelerator_hooks(self):
        """Save only alignment handler weights."""
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                model = models[0]
                weights.pop(0)
                model = self.unwrap_model(model)

                # Only save alignment handler params
                state_dict = {}
                for name, param in model.alignment_handler.named_parameters():
                    state_dict[f"alignment_handler.{name}"] = param.detach().clone().data

                save_path = os.path.join(output_dir, "state_dict.pth")
                torch.save(state_dict, save_path)
                logger.info(f"Stage 1: Saved {len(state_dict)} alignment params to {save_path}")

        def load_model_hook(models, input_dir):
            model = models.pop()
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"))
            m, u = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Stage 1: Loading alignment params, missing: {len(m)}, unexpected: {len(u)}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def log_grads(self):
        """Override: only log alignment handler gradients (no LoRA)."""
        self.unwrap_model(self.D).eval().requires_grad_(False)
        x = self.forward_generator()
        loss_l2 = F.mse_loss(x, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
        loss_lpips = self.net_lpips(x, self.batch_inputs.gt).mean() * self.config.lambda_lpips
        loss_disc = self.D(x, for_G=True).mean() * self.config.lambda_gan
        losses = [("l2", loss_l2), ("lpips", loss_lpips), ("disc", loss_disc)]
        grad_dict = {}
        self.G_opt.zero_grad()
        for idx, (name, loss) in enumerate(losses):
            retain_graph = idx != len(losses) - 1
            loss.backward(retain_graph=retain_graph)

            # Only alignment handler gradients
            for pname, param in self.G.alignment_handler.named_parameters():
                if param.grad is not None:
                    grad_dict[f"align_grad/{pname}_{name}"] = param.grad.norm().item()

            self.G_opt.zero_grad()
        self.accelerator.log(grad_dict, step=self.global_step)
