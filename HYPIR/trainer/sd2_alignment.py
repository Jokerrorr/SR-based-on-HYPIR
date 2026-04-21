"""
SD2 Alignment Trainer — FaithDiff-style noise prediction training.

Training flow (follows FaithDiff, fixed t=200 for single-step inference):
  1. GT → VAE encode → z_hq → add_noise(t=200) → noisy_hq (UNet input)
  2. LQ → RM → VAE encode → x_en → AlignmentHandler → features
  3. UNet(noisy_hq, t=200, text, features) → noise_pred
  4. Loss = L1(noise_pred, noise)

No discriminator needed. Pure diffusion noise prediction.
Fixed timestep ensures training/inference consistency for single-step mode.
"""

import os
import csv
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
from HYPIR.utils.common import print_vram_state

logger = get_logger(__name__, log_level="INFO")


class SD2AlignmentTrainer(BaseTrainer):

    def init_scheduler(self):
        self.noise_scheduler = DDPMScheduler.from_pretrained(
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
        # Skip discriminator — not needed for noise prediction
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
            pretrained_sd = torch.load(hypir_pretrained, map_location="cpu", weights_only=False)
            # Handle double-dot keys
            cleaned_sd = {}
            for k, v in pretrained_sd.items():
                cleaned_sd[k] = v
            m, u = unet.load_state_dict(cleaned_sd, strict=False)
            loaded_lora = sum(1 for k in unet.state_dict() if "lora" in k)
            logger.info(f"Loaded HYPIR LoRA: {len(cleaned_sd)} keys, "
                        f"LoRA keys in model: {loaded_lora}")
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
            pretrained_sd = torch.load(alignment_pretrained, map_location="cpu", weights_only=False)
            align_sd = {}
            for k, v in pretrained_sd.items():
                if k.startswith("alignment_handler."):
                    key = k[len("alignment_handler."):].lstrip(".")
                    align_sd[key] = v
                else:
                    align_sd[k] = v
            if align_sd:
                m, u = self.G.alignment_handler.load_state_dict(align_sd, strict=False)
                logger.info(f"Loaded alignment pretrained params, "
                            f"missing: {len(m)}, unexpected: {len(u)}")

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
        """Skip discriminator — not needed for noise prediction training."""
        self.D = None
        logger.info("Skipping discriminator (noise prediction training)")

    def init_lpips(self):
        """Skip LPIPS — not needed for noise prediction training."""
        self.net_lpips = None

    def init_optimizers(self):
        """Override: only G optimizer with LoRA + alignment params."""
        logger.info(f"Creating {self.config.optimizer_type} optimizer (LoRA + alignment)")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = torch.optim.AdamW

        # Collect LoRA params from UNet + alignment handler params
        self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        align_params = list(self.G.alignment_handler.parameters())
        existing_ids = {id(p) for p in self.G_params}
        self.G_params.extend([p for p in align_params if id(p) not in existing_ids])
        logger.info(f"G params: {len(self.G_params)} tensors")

        self.G_opt = optimizer_cls(
            self.G_params,
            lr=self.config.lr_G,
            **self.config.opt_kwargs,
        )

        # No D optimizer
        self.D_params = []
        self.D_opt = None

    def prepare_all(self):
        """Override: only prepare G, G_opt, dataloader (no D)."""
        logger.info("Wrapping models, optimizers and dataloaders")
        attrs = ["G", "G_opt", "dataloader"]
        prepared_objs = self.accelerator.prepare(
            self.G, self.G_opt, self.dataloader
        )
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)
        print_vram_state("After accelerator.prepare", logger=logger)

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
            state_dict = torch.load(
                os.path.join(input_dir, "state_dict.pth"),
                map_location="cpu", weights_only=False,
            )
            m, u = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loading lora+alignment params, missing: {len(m)}, unexpected: {len(u)}")

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def prepare_batch_inputs(self, batch):
        """FaithDiff-style: encode GT + add noise, encode RM(LQ) for alignment."""
        batch = self.batch_transform(batch)
        gt = (batch["GT"] * 2 - 1).float()
        lq = (batch["LQ"] * 2 - 1).float()
        prompt = batch["txt"]
        bs = len(prompt)

        c_txt = self.encode_prompt(prompt)

        # 1. GT → VAE encode → z_hq → add noise
        with torch.no_grad():
            z_hq = self.vae.encode(gt.to(self.weight_dtype)).latent_dist.sample()
            z_hq = z_hq * self.vae.config.scaling_factor

        noise = torch.randn_like(z_hq)
        noise_offset = getattr(self.config, "noise_offset", None)
        if noise_offset:
            noise += noise_offset * torch.randn(
                (z_hq.shape[0], z_hq.shape[1], 1, 1),
                device=z_hq.device, dtype=z_hq.dtype,
            )

        # Fixed timestep t=200 for single-step inference compatibility
        model_t = getattr(self.config, "model_t", 200)
        timesteps = torch.full((bs,), model_t, device=z_hq.device, dtype=torch.long)
        noisy_hq = self.noise_scheduler.add_noise(z_hq, noise, timesteps)

        # 2. RM(LQ) → VAE encode → x_en
        if self.rm is not None:
            with torch.no_grad():
                lq_01 = (lq + 1) / 2  # [-1,1] → [0,1] for RM
                x_rm = self.rm.inference(lq_01.to(self.device))
                x_rm_normalized = (x_rm * 2 - 1).to(dtype=self.weight_dtype)
                x_en = self.vae.encode(x_rm_normalized).latent_dist.sample()
        else:
            with torch.no_grad():
                x_en = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()

        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_hq=z_hq, noise=noise,
            noisy_hq=noisy_hq,
            timesteps=timesteps,
            c_txt=c_txt, prompt=prompt,
        )
        self.batch_inputs.update(x_en=x_en)

    def forward_generator(self) -> torch.Tensor:
        """Predict noise from noisy_hq using UNet + alignment features."""
        x_en = self.batch_inputs.x_en
        noise_pred = self.G(
            self.batch_inputs.noisy_hq,
            self.batch_inputs.timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["text_embed"],
            x_en=x_en,
        ).sample
        return noise_pred

    def optimize_generator(self):
        """Single optimization step: L1 noise prediction loss."""
        with self.accelerator.accumulate(self.G):
            noise_pred = self.forward_generator()
            noise = self.batch_inputs.noise

            loss = F.l1_loss(noise_pred.float(), noise.float(), reduction="mean")

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
            self.G_opt.step()
            self.G_opt.zero_grad()

        loss_dict = dict(noise_loss=loss)
        return loss_dict

    def optimize_discriminator(self):
        """No-op: discriminator not used in noise prediction training."""
        return {}

    def log_images(self):
        """Log LQ, GT, and noise prediction quality."""
        N = 4
        image_logs = dict(
            lq=(self.batch_inputs.lq[:N] + 1) / 2,
            gt=(self.batch_inputs.gt[:N] + 1) / 2,
            prompt=(self._log_txt_as_img(self.batch_inputs.prompt[:N]) + 1) / 2,
        )

        # RM intermediate output
        if self.rm is not None:
            with torch.no_grad():
                lq_01 = (self.batch_inputs.lq[:N] + 1) / 2
                x_rm = self.rm.inference(lq_01.to(self.device))
                image_logs["x_rm"] = x_rm[:N].clamp(0, 1)

        # Alignment feature heatmaps
        with torch.no_grad():
            handler = self.G.alignment_handler
            x_en = self.batch_inputs.x_en[:N]
            handler_dtype = next(handler.parameters()).dtype
            enc_feat = handler.alignment_encoder(x_en.to(dtype=handler_dtype))
            emb_feat = handler.condition_embedding(enc_feat)
            unet_sample = self.G.unet.conv_in(
                self.batch_inputs.noisy_hq[:N]
            )
            fused = handler.fuse_with_unet_features(unet_sample, emb_feat)

            for tag, feat in [("align_encoder", enc_feat), ("align_embedding", emb_feat), ("align_fused", fused)]:
                heatmap = feat[:, 0:1]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                heatmap = F.interpolate(heatmap, size=(512, 512), mode="bilinear", align_corners=False)
                image_logs[tag] = heatmap.expand(-1, 3, -1, -1)

        if not self.accelerator.is_main_process:
            return

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
        from HYPIR.utils.common import log_txt_as_img
        return log_txt_as_img((256, 256), texts)

    def log_grads(self):
        """Log alignment handler and LoRA gradient norms."""
        self.unwrap_model(self.G)
        noise_pred = self.forward_generator()
        loss = F.l1_loss(noise_pred.float(), self.batch_inputs.noise.float(), reduction="mean")

        grad_dict = {}
        self.G_opt.zero_grad()
        loss.backward()

        # Alignment handler gradients
        for pname, param in self.G.alignment_handler.named_parameters():
            if param.grad is not None:
                grad_dict[f"align_grad/{pname}"] = param.grad.norm().item()

        self.G_opt.zero_grad()
        self.accelerator.log(grad_dict, step=self.global_step)

    def on_training_start(self):
        """Initialize EMA and CSV logger."""
        super().on_training_start()
        self.loss_csv_path = os.path.join(self.config.output_dir, "loss_log.csv")
        if self.accelerator.is_main_process:
            with open(self.loss_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "noise_loss"])

    def log_loss_to_csv(self, step: int, train_loss: dict):
        if not self.accelerator.is_main_process:
            return
        row = [step, train_loss.get("noise_loss", "")]
        with open(self.loss_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def run(self):
        """Simplified training loop: no G/D alternation, just noise prediction."""
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        while self.global_step < self.config.max_train_steps:
            train_loss = {}
            for batch in self.dataloader:
                self.prepare_batch_inputs(batch)

                # Single optimization step (no G/D alternation)
                loss_dict = self.optimize_generator()

                bs = len(self.batch_inputs.lq)
                for k, v in loss_dict.items():
                    avg_loss = self.accelerator.gather(v.repeat(bs)).mean()
                    if k not in train_loss:
                        train_loss[k] = 0
                    train_loss[k] += avg_loss.item()

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    self.ema_handler.update()
                    _, _, peak = print_vram_state(None)
                    self.pbar.set_description(f"Step, VRAM peak: {peak:.2f} GB")

                    self.global_step += 1
                    self.pbar.update(1)
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
