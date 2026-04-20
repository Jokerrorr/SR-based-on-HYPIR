"""
SD2 Trainer with alignment module for RM+HYPIR pipeline.

Extends BaseTrainer to:
1. Initialize UNet with alignment module
2. Optionally load frozen RM model for x_en generation
3. Encode LQ through VAE to get z_lq, and x_en via RM→VAE encode
4. Pass x_en through alignment module to align with x_hq
5. Fuse alignment features into UNet forward pass

Alignment input: x_en = VAE.encode(RM(LQ)) — latent-space representation.
"""

import os
import csv
import json
import logging
from typing import List, Dict
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from peft import LoraConfig
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import make_grid

from HYPIR.trainer.base import BaseTrainer, BatchInput
from HYPIR.alignment.alignment_handler import AlignmentHandler
from HYPIR.model.unet_alignment import UNetAlignment
from HYPIR.rm.restoration_module import RestorationModule

logger = get_logger(__name__, log_level="INFO")


class SD2AlignmentTrainer(BaseTrainer):

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.base_model_path, subfolder="scheduler"
        )

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.base_model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model_path, subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def encode_prompt(self, prompt: List[str]) -> Dict[str, torch.Tensor]:
        txt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.accelerator.device))[0]
        return {"text_embed": text_embed}

    def init_models(self):
        self.init_scheduler()
        self.init_text_models()
        self.init_vae()
        self.init_rm()
        self.init_generator()
        self.init_discriminator()
        self.init_lpips()

    def init_rm(self):
        """Load frozen RM model if configured."""
        rm_cfg = getattr(self.config, "rm", None)
        if rm_cfg is None or not getattr(rm_cfg, "enabled", False):
            logger.info("RM disabled, x_en will use z_lq as fallback")
            self.rm = None
            return

        task = getattr(rm_cfg, "task", "bid")
        weight_path = getattr(rm_cfg, "weight_path", None)
        if weight_path is None:
            raise ValueError("rm.weight_path is required when rm.enabled=true")

        logger.info(f"Loading RM model: task={task}, weights={weight_path}")
        self.rm = RestorationModule(task=task, device=self.device)
        self.rm.load(weight_path=weight_path)

        frozen = getattr(rm_cfg, "frozen", True)
        if frozen:
            self.rm.model.requires_grad_(False).eval()
            logger.info("RM model frozen (no gradient)")

    def init_generator(self):
        # Load base UNet
        unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path, subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        unet.eval().requires_grad_(False)

        if self.config.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        # Add LoRA to UNet
        target_modules = self.config.lora_modules
        logger.info(f"Add lora parameters to {target_modules}")
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
            logger.info(f"Loading HYPIR pretrained LoRA from {hypir_pretrained}")
            pretrained_sd = torch.load(hypir_pretrained, map_location="cpu")
            m, u = unet.load_state_dict(pretrained_sd, strict=False)
            logger.info(f"Loaded HYPIR LoRA: {len(pretrained_sd)} params, "
                        f"missing: {len(m)}, unexpected: {len(u)}")
        else:
            logger.warning("No HYPIR pretrained LoRA found, using random LoRA init")

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

        # Load Stage 1 pretrained alignment weights if provided
        alignment_pretrained = getattr(self.config, "alignment_pretrained_path", None)
        if alignment_pretrained is not None:
            logger.info(f"Loading Stage 1 alignment weights from {alignment_pretrained}")
            pretrained_sd = torch.load(alignment_pretrained, map_location="cpu")
            # Filter to alignment_handler keys only
            align_sd = {}
            for k, v in pretrained_sd.items():
                if k.startswith("alignment_handler."):
                    align_sd[k] = v
            if align_sd:
                m, u = self.G.load_state_dict(align_sd, strict=False)
                logger.info(f"Loaded {len(align_sd)} alignment pretrained params, "
                           f"missing: {len(m)}, unexpected: {len(u)}")
            else:
                logger.warning(f"No alignment_handler keys found in {alignment_pretrained}, "
                              f"loading all keys with strict=False")
                m, u = self.G.load_state_dict(pretrained_sd, strict=False)
                logger.info(f"Loaded pretrained, missing: {len(m)}, unexpected: {len(u)}")

        # Move LoRA params to float32 for stable training
        lora_params = [p for p in unet.parameters() if p.requires_grad]
        assert lora_params, "Failed to find lora parameters"
        for p in lora_params:
            p.data = p.to(torch.float32)

        # Move alignment handler params to float32
        for p in handler.parameters():
            p.data = p.to(torch.float32)

        logger.info(f"Alignment handler params: "
                     f"{sum(p.numel() for p in handler.parameters())/1e6:.2f}M")

    def init_discriminator(self):
        super().init_discriminator()

    def init_optimizers(self):
        """Override to include alignment handler params in G optimizer."""
        logger.info(f"Creating {self.config.optimizer_type} optimizers")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = None

        # Collect LoRA params from UNet + alignment handler params
        self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        align_params = list(self.G.alignment_handler.parameters())
        self.G_params.extend([p for p in align_params if p not in self.G_params])
        logger.info(f"G params: LoRA={sum(1 for p in self.G.unet.parameters() if p.requires_grad)} tensors, "
                     f"Alignment={len(align_params)} tensors")

        self.G_opt = optimizer_cls(
            self.G_params,
            lr=self.config.lr_G,
            **self.config.opt_kwargs,
        )

        self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
        self.D_opt = optimizer_cls(
            self.D_params,
            lr=self.config.lr_D,
            **self.config.opt_kwargs,
        )

    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                model = models[0]
                weights.pop(0)
                model = self.unwrap_model(model)

                state_dict = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        state_dict[name] = param.detach().clone().data

                for name, param in model.alignment_handler.named_parameters():
                    key = f"alignment_handler.{name}"
                    if key not in state_dict:
                        state_dict[key] = param.detach().clone().data

                save_path = os.path.join(output_dir, "state_dict.pth")
                torch.save(state_dict, save_path)
                logger.info(f"Saved {len(state_dict)} params to {save_path}")

        def load_model_hook(models, input_dir):
            model = models.pop()
            state_dict = torch.load(os.path.join(input_dir, "state_dict.pth"))
            m, u = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loading lora+alignment params, missing: {len(m)}, unexpected: {len(u)}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def prepare_batch_inputs(self, batch):
        batch = self.batch_transform(batch)
        gt = (batch["GT"] * 2 - 1).float()
        lq = (batch["LQ"] * 2 - 1).float()
        prompt = batch["txt"]
        bs = len(prompt)

        c_txt = self.encode_prompt(prompt)

        # Encode LQ through VAE → z_lq (used as UNet input)
        z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()

        # x_en = VAE.encode(RM(LQ)): VAE-encoded RM output (latent space)
        if self.rm is not None:
            with torch.no_grad():
                lq_01 = (lq + 1) / 2  # [-1,1] → [0,1] for RM
                x_rm = self.rm.inference(lq_01.to(self.device))  # [0,1] range
                x_rm_normalized = (x_rm * 2 - 1).to(dtype=self.weight_dtype)
                x_en = self.vae.encode(x_rm_normalized).latent_dist.sample()
        else:
            x_en = z_lq  # Fallback when RM is disabled

        timesteps = torch.full(
            (bs,), self.config.model_t, dtype=torch.long, device=self.device
        )

        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_lq=z_lq,
            c_txt=c_txt, timesteps=timesteps,
            prompt=prompt,
        )
        self.batch_inputs.update(x_en=x_en)

    def forward_generator(self) -> torch.Tensor:
        z_in = self.batch_inputs.z_lq * self.vae.config.scaling_factor
        x_en = self.batch_inputs.x_en

        eps = self.G(
            z_in,
            self.batch_inputs.timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["text_embed"],
            x_en=x_en,
        ).sample

        z = self.scheduler.step(eps, self.config.coeff_t, z_in).pred_original_sample
        x = self.vae.decode(
            z.to(self.weight_dtype) / self.vae.config.scaling_factor
        ).sample.float()
        return x

    # ------------------------------------------------------------------
    # Phase 6: Training loss persistence, visualization, logging
    # ------------------------------------------------------------------

    def on_training_start(self):
        """Initialize EMA (from base) and loss CSV logger."""
        super().on_training_start()
        self.loss_csv_path = os.path.join(self.config.output_dir, "loss_log.csv")
        if self.accelerator.is_main_process:
            with open(self.loss_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "G_total", "G_mse", "G_lpips", "G_disc", "D", "D_logits_real", "D_logits_fake"])

    def log_loss_to_csv(self, step: int, train_loss: dict):
        """Append loss values to CSV file."""
        if not self.accelerator.is_main_process:
            return
        row = [
            step,
            train_loss.get("G_total", ""),
            train_loss.get("G_mse", ""),
            train_loss.get("G_lpips", ""),
            train_loss.get("G_disc", ""),
            train_loss.get("D", ""),
            train_loss.get("D_logits_real", ""),
            train_loss.get("D_logits_fake", ""),
        ]
        with open(self.loss_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_images(self):
        """Override to add RM output and alignment feature heatmaps."""
        N = 4
        image_logs = dict(
            lq=(self.batch_inputs.lq[:N] + 1) / 2,
            gt=(self.batch_inputs.gt[:N] + 1) / 2,
            G=(self.G_pred[:N] + 1) / 2,
            prompt=(self._log_txt_as_img(self.batch_inputs.prompt[:N]) + 1) / 2,
        )

        # RM intermediate output
        if self.rm is not None:
            with torch.no_grad():
                lq_01 = (self.batch_inputs.lq[:N] + 1) / 2
                x_rm = self.rm.inference(lq_01.to(self.device))
                image_logs["x_rm"] = x_rm[:N].clamp(0, 1)

        # EMA output
        if self.config.use_ema:
            self.ema_handler.activate_ema_weights()
            with torch.no_grad():
                ema_x = self.forward_generator()
                image_logs["G_ema"] = (ema_x[:N] + 1) / 2
            self.ema_handler.deactivate_ema_weights()

        # Alignment feature heatmaps
        with torch.no_grad():
            handler = self.G.alignment_handler
            x_en = self.batch_inputs.x_en[:N]
            handler_dtype = next(handler.parameters()).dtype
            enc_feat = handler.alignment_encoder(x_en.to(dtype=handler_dtype))  # [B, 512, H/8, W/8]
            emb_feat = handler.condition_embedding(enc_feat)  # [B, 320, H/8, W/8]
            unet_sample = self.G.unet.conv_in(
                self.batch_inputs.z_lq[:N] * self.vae.config.scaling_factor
            )
            fused = handler.fuse_with_unet_features(unet_sample, emb_feat)

            # Take first channel and normalize to [0,1] for visualization
            for tag, feat in [("align_encoder", enc_feat), ("align_embedding", emb_feat), ("align_fused", fused)]:
                heatmap = feat[:, 0:1]  # [B, 1, H, W]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                heatmap = F.interpolate(heatmap, size=(512, 512), mode="bilinear", align_corners=False)
                image_logs[tag] = heatmap.expand(-1, 3, -1, -1)  # grayscale → 3ch

        if not self.accelerator.is_main_process:
            return

        # Tensorboard
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                for tag, images in image_logs.items():
                    tracker.writer.add_image(
                        f"image/{tag}",
                        make_grid(images.float(), nrow=4),
                        self.global_step,
                    )

        # Save to disk
        for key, images in image_logs.items():
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8) \
                .permute(0, 2, 3, 1).contiguous().cpu().numpy()
            save_dir = os.path.join(
                self.config.output_dir, self.config.logging_dir, "log_images",
                f"{self.global_step:07}", key)
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(image_arrs):
                from PIL import Image
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))

    @staticmethod
    def _log_txt_as_img(texts):
        """Import helper for log_txt_as_img."""
        from HYPIR.utils.common import log_txt_as_img
        return log_txt_as_img((256, 256), texts)

    def run(self):
        """Override to add loss CSV logging during training."""
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        while self.global_step < self.config.max_train_steps:
            train_loss = {}
            for batch in self.dataloader:
                self.prepare_batch_inputs(batch)
                bs = len(self.batch_inputs.lq)
                generator_step = ((self.batch_count // self.config.gradient_accumulation_steps) % 2) == 0
                if generator_step:
                    loss_dict = self.optimize_generator()
                else:
                    loss_dict = self.optimize_discriminator()

                for k, v in loss_dict.items():
                    avg_loss = self.accelerator.gather(v.repeat(bs)).mean()
                    if k not in train_loss:
                        train_loss[k] = 0
                    train_loss[k] += avg_loss.item() / self.config.gradient_accumulation_steps

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    if generator_step:
                        self.ema_handler.update()
                    from HYPIR.utils.common import print_vram_state
                    state = "Generator     Step" if not generator_step else "Discriminator Step"
                    _, _, peak = print_vram_state(None)
                    self.pbar.set_description(f"{state}, VRAM peak: {peak:.2f} GB")

                if self.accelerator.sync_gradients and not generator_step:
                    self.global_step += 1
                    self.pbar.update(1)
                    # Log to CSV before clearing
                    self.log_loss_to_csv(self.global_step, train_loss)
                    log_dict = {}
                    for k in train_loss.keys():
                        log_dict[f"loss/{k}"] = train_loss[k]
                    train_loss = {}
                    self.accelerator.log(log_dict, step=self.global_step)
                    if self.global_step % self.config.log_image_steps == 0 or self.global_step == 1:
                        self.log_images()
                    if self.global_step % self.config.log_grad_steps == 0 or self.global_step == 1:
                        self.log_grads()
                    if self.global_step % self.config.checkpointing_steps == 0 or self.global_step == 1:
                        self.save_checkpoint()

                if self.global_step >= self.config.max_train_steps:
                    break
        self.accelerator.end_training()

    def log_grads(self):
        """Override to also log alignment handler gradients."""
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
            # LoRA module gradients
            lora_module_grads = {}
            for module_name, module in self.unwrap_model(self.G).named_modules():
                for suffix in self.config.log_grad_modules:
                    if module_name.endswith(suffix):
                        flat_grad = torch.cat([
                            p.grad.flatten() for p in module.parameters() if p.requires_grad
                        ])
                        lora_module_grads.setdefault(suffix, []).append(flat_grad)
                        break
            for k, v in lora_module_grads.items():
                grad_dict[f"grad_norm/{k}_{name}"] = torch.norm(torch.cat(v)).item()

            # Alignment handler gradients
            for pname, param in self.G.alignment_handler.named_parameters():
                if param.grad is not None:
                    grad_dict[f"align_grad/{pname}_{name}"] = param.grad.norm().item()

            self.G_opt.zero_grad()
        self.accelerator.log(grad_dict, step=self.global_step)
