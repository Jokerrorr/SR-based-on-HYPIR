"""
Stage 1 Alignment Pretraining Trainer (FaithDiff-style).

Loads HYPIR pretrained LoRA (frozen), only trains Alignment Handler parameters.
Uses L1 noise prediction loss — same as FaithDiff Stage 1.

Training flow:
  GT → VAE encode → add_noise(t) → noisy_hq
  RM(LQ) → VAE encode → x_en → AlignmentHandler → features
  UNet(noisy_hq, t, text, features) → noise_pred
  Loss = L1(noise_pred, noise)
"""

import os
import logging

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel
from peft import LoraConfig

from HYPIR.trainer.sd2_alignment import SD2AlignmentTrainer
from HYPIR.alignment.alignment_handler import AlignmentHandler
from HYPIR.model.unet_alignment import UNetAlignment

logger = get_logger(__name__, log_level="INFO")


class SD2AlignmentStage1Trainer(SD2AlignmentTrainer):
    """Stage 1: Load HYPIR pretrained LoRA (frozen), only train alignment handler."""

    def init_generator(self):
        # Load base UNet
        unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path, subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)

        if self.config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Add LoRA to UNet (required to load HYPIR pretrained weights)
        target_modules = self.config.lora_modules
        logger.info(f"Stage 1: Add lora parameters to {target_modules}")
        G_lora_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        unet.add_adapter(G_lora_cfg)

        # Load HYPIR pretrained LoRA weights
        hypir_pretrained = getattr(self.config, "hypir_pretrained_path", None)
        if hypir_pretrained is not None and os.path.exists(hypir_pretrained):
            logger.info(f"Stage 1: Loading HYPIR pretrained LoRA from {hypir_pretrained}")
            pretrained_sd = torch.load(hypir_pretrained, map_location="cpu", weights_only=False)
            m, u = unet.load_state_dict(pretrained_sd, strict=False)
            loaded_lora = sum(1 for k in unet.state_dict() if "lora" in k)
            logger.info(f"Stage 1: Loaded HYPIR LoRA: {len(pretrained_sd)} params, "
                         f"LoRA keys: {loaded_lora}")
        else:
            logger.warning("Stage 1: No HYPIR pretrained LoRA found, using random LoRA init")

        # Freeze UNet entirely (including LoRA)
        unet.eval().requires_grad_(False)
        logger.info("Stage 1: UNet + LoRA frozen, only alignment handler is trainable")

        # Create alignment handler
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
        unet_trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        logger.info(f"Stage 1: Alignment handler params: {align_params/1e6:.2f}M")
        logger.info(f"Stage 1: UNet trainable params: {unet_trainable} (should be 0)")

    def init_optimizers(self):
        """Only alignment handler params in G optimizer."""
        logger.info(f"Stage 1: Creating {self.config.optimizer_type} optimizer (alignment only)")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = torch.optim.AdamW

        # Only alignment handler parameters
        align_lr = getattr(self.config, "alignment_lr", self.config.lr_G)
        self.G_params = list(self.G.alignment_handler.parameters())
        logger.info(f"Stage 1: G optimizer params: {len(self.G_params)} tensors, lr={align_lr}")

        self.G_opt = optimizer_cls(
            self.G_params,
            lr=align_lr,
            **self.config.opt_kwargs,
        )

        # No D optimizer
        self.D_params = []
        self.D_opt = None

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
            state_dict = torch.load(
                os.path.join(input_dir, "state_dict.pth"),
                map_location="cpu", weights_only=False,
            )
            m, u = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Stage 1: Loading alignment params, missing: {len(m)}, unexpected: {len(u)}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def log_grads(self):
        """Override: only log alignment handler gradients."""
        noise_pred = self.forward_generator()
        loss = F.l1_loss(noise_pred.float(), self.batch_inputs.noise.float(), reduction="mean")

        grad_dict = {}
        self.G_opt.zero_grad()
        loss.backward()

        for pname, param in self.G.alignment_handler.named_parameters():
            if param.grad is not None:
                grad_dict[f"align_grad/{pname}"] = param.grad.norm().item()

        self.G_opt.zero_grad()
        self.accelerator.log(grad_dict, step=self.global_step)
