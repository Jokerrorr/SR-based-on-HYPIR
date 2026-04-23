"""
SD2 Alignment Trainer — HYPIR adversarial fine-tuning with alignment.

Stage 2: Joint training of LoRA + Alignment with adversarial loss.

Training flow:
  GT → VAE.encode → z_hq → add_noise(t=200) → x_hq_t → conv_in → sample_emb ─┐
                                                                                ├→ sample_emb + feat_alpha → UNet → noise_pred
  RM(LQ) → VAE.encode → z_lq → Alignment(sample_emb, z_lq) ──────────────────┘

Loss:
  loss_G = lambda_l2    * MSE(x_pred, gt)         (pixel reconstruction)
         + lambda_lpips * LPIPS(x_pred, gt)       (perceptual)
         + lambda_gan   * D(x_pred, for_G=True)   (adversarial)
         + lambda_align * L1(noise_pred, noise)    (alignment noise prediction)

  x_pred: noise_pred → scheduler.step → z_pred → VAE.decode → pixel space
  G/D alternating training (same as original HYPIR).
"""

import os
import csv
import logging
from typing import List, Dict

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from peft import LoraConfig
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import make_grid

from HYPIR.trainer.base import BaseTrainer, BatchInput
from HYPIR.alignment.alignment import Alignment
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
        self.init_discriminator()  # HYPIR adversarial training
        self.init_lpips()

    def init_rm(self):
        """Load frozen RM model if configured."""
        rm_cfg = getattr(self.config, "rm", None)
        if rm_cfg is None or not getattr(rm_cfg, "enabled", False):
            logger.info("RM disabled, z_lq will use VAE.encode(LQ) as fallback")
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
            m, u = unet.load_state_dict(pretrained_sd, strict=False)
            loaded_lora = sum(1 for k in unet.state_dict() if "lora" in k)
            logger.info(f"Loaded HYPIR LoRA: {len(pretrained_sd)} keys, "
                        f"LoRA keys in model: {loaded_lora}")
        else:
            logger.warning("No HYPIR pretrained LoRA found, using random LoRA init")

        # Create FaithDiff-style alignment handler
        alignment_cfg = getattr(self.config, "alignment", None)
        if alignment_cfg is not None:
            handler = Alignment(
                conditioning_channels=getattr(alignment_cfg, "conditioning_channels", 4),
                embedding_channels=getattr(alignment_cfg, "embedding_channels", 320),
                num_trans_channel=getattr(alignment_cfg, "num_trans_channel", 640),
                num_trans_head=getattr(alignment_cfg, "num_trans_head", 8),
                num_trans_layer=getattr(alignment_cfg, "num_trans_layer", 2),
            )
        else:
            handler = Alignment()

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

    def init_optimizers(self):
        """G optimizer (LoRA + alignment) and D optimizer."""
        logger.info(f"Creating {self.config.optimizer_type} optimizers")
        if self.config.optimizer_type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.optimizer_type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            optimizer_cls = torch.optim.AdamW

        # G params: LoRA + alignment handler
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

        # D params
        self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
        self.D_opt = optimizer_cls(
            self.D_params,
            lr=self.config.lr_D,
            **self.config.opt_kwargs,
        )

    def prepare_all(self):
        """Prepare G, D, optimizers, and dataloader for accelerator."""
        logger.info("Wrapping models, optimizers and dataloaders")
        attrs = ["G", "D", "G_opt", "D_opt", "dataloader"]
        prepared_objs = self.accelerator.prepare(
            self.G, self.D, self.G_opt, self.D_opt, self.dataloader
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
        """v4: FaithDiff-style dual-side alignment data preparation.

        GT → VAE.encode → z_hq → add_noise(t=200) → x_hq_t
        RM(LQ) → VAE.encode → z_lq
        """
        batch = self.batch_transform(batch)
        gt = (batch["GT"] * 2 - 1).float()
        lq = (batch["LQ"] * 2 - 1).float()
        prompt = batch["txt"]
        bs = len(prompt)

        c_txt = self.encode_prompt(prompt)

        # 1. GT → VAE encode → z_hq → add noise → x_hq_t
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

        # Fixed timestep t=200
        model_t = getattr(self.config, "model_t", 200)
        timesteps = torch.full((bs,), model_t, device=z_hq.device, dtype=torch.long)
        x_hq_t = self.noise_scheduler.add_noise(z_hq, noise, timesteps)

        # 2. RM(LQ) → VAE encode → z_lq
        if self.rm is not None:
            with torch.no_grad():
                lq_01 = (lq + 1) / 2  # [-1,1] → [0,1] for RM
                x_rm = self.rm.inference(lq_01.to(self.device))
                self._x_rm_cache = x_rm  # cache for log_images reuse
                x_rm_normalized = (x_rm * 2 - 1).to(dtype=self.weight_dtype)
                z_lq = self.vae.encode(x_rm_normalized).latent_dist.sample()
                z_lq = z_lq * self.vae.config.scaling_factor
        else:
            with torch.no_grad():
                self._x_rm_cache = None
                z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
                z_lq = z_lq * self.vae.config.scaling_factor

        self.batch_inputs = BatchInput(
            gt=gt, lq=lq,
            z_hq=z_hq, noise=noise,
            x_hq_t=x_hq_t, z_lq=z_lq,
            timesteps=timesteps,
            c_txt=c_txt, prompt=prompt,
        )

    def forward_generator(self) -> torch.Tensor:
        """v4: AlignmentModule(z_lq, x_hq_t) → aligned → UNet → noise_pred → scheduler.step → VAE.decode → x_pred.

        Returns pixel-space prediction [B, 3, H, W].
        Caches noise_pred in self._noise_pred for alignment loss.
        """
        noise_pred = self.G(
            self.batch_inputs.x_hq_t,
            self.batch_inputs.timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["text_embed"],
            z_lq=self.batch_inputs.z_lq,
            x_hq_t=self.batch_inputs.x_hq_t,
        ).sample
        self._noise_pred = noise_pred  # cache for alignment loss

        coeff_t = getattr(self.config, "coeff_t", self.config.model_t)
        z_pred = self.noise_scheduler.step(
            noise_pred, coeff_t, self.batch_inputs.x_hq_t
        ).pred_original_sample
        x_pred = self.vae.decode(
            z_pred.to(self.weight_dtype) / self.vae.config.scaling_factor
        ).sample.float()
        return x_pred

    def optimize_generator(self):
        """HYPIR adversarial loss + alignment L1 noise prediction loss."""
        with self.accelerator.accumulate(self.G):
            self.unwrap_model(self.D).eval().requires_grad_(False)
            x_pred = self.forward_generator()
            self.G_pred = x_pred

            # HYPIR original losses (pixel space)
            loss_l2 = F.mse_loss(x_pred, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
            loss_lpips = self.net_lpips(x_pred, self.batch_inputs.gt).mean() * self.config.lambda_lpips
            loss_disc = self.D(x_pred, for_G=True).mean() * self.config.lambda_gan

            # Alignment noise prediction loss (latent space, using cached noise_pred)
            loss_align = F.l1_loss(
                self._noise_pred.float(), self.batch_inputs.noise.float(), reduction="mean"
            ) * self.config.lambda_align

            loss_G = loss_l2 + loss_lpips + loss_disc + loss_align
            self.accelerator.backward(loss_G)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.G_params, self.config.max_grad_norm)
            self.G_opt.step()
            self.G_opt.zero_grad()

        loss_dict = dict(
            G_total=loss_G, G_mse=loss_l2, G_lpips=loss_lpips,
            G_disc=loss_disc, G_align=loss_align,
        )
        return loss_dict

    def on_training_start(self):
        """Initialize EMA and CSV logger."""
        super().on_training_start()
        self.loss_csv_path = os.path.join(self.config.output_dir, "loss_log.csv")
        if self.accelerator.is_main_process:
            with open(self.loss_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "G_total", "G_mse", "G_lpips", "G_disc", "G_align", "D"])

    def log_loss_to_csv(self, step: int, train_loss: dict):
        if not self.accelerator.is_main_process:
            return
        row = [step,
               train_loss.get("G_total", ""),
               train_loss.get("G_mse", ""),
               train_loss.get("G_lpips", ""),
               train_loss.get("G_disc", ""),
               train_loss.get("G_align", ""),
               train_loss.get("D", "")]
        with open(self.loss_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_images(self):
        """Log LQ, GT, G_pred, alignment features, and EMA results."""
        N = 4
        image_logs = dict(
            lq=(self.batch_inputs.lq[:N] + 1) / 2,
            gt=(self.batch_inputs.gt[:N] + 1) / 2,
            G=(self.G_pred[:N] + 1) / 2,
            prompt=(self._log_txt_as_img(self.batch_inputs.prompt[:N]) + 1) / 2,
        )

        if self.config.use_ema:
            self.ema_handler.activate_ema_weights()
            with torch.no_grad():
                ema_x = self.forward_generator()
                image_logs["G_ema"] = (ema_x[:N] + 1) / 2
            self.ema_handler.deactivate_ema_weights()

        # RM intermediate output (reuse cached result from prepare_batch_inputs)
        if self.rm is not None and self._x_rm_cache is not None:
            image_logs["x_rm"] = self._x_rm_cache[:N].clamp(0, 1)

        # Alignment feature heatmaps
        with torch.no_grad():
            handler = self.G.alignment_handler
            z_lq = self.batch_inputs.z_lq[:N]
            x_hq_t = self.batch_inputs.x_hq_t[:N]
            handler_dtype = next(handler.parameters()).dtype
            unet_dtype = self.G.unet.conv_in.weight.dtype

            sample_emb = self.G.unet.conv_in(x_hq_t.to(dtype=unet_dtype))
            feat_alpha = handler(sample_emb.to(dtype=handler_dtype), z_lq.to(dtype=handler_dtype))

            feat = feat_alpha[:, :1]
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
            feat = F.interpolate(feat, size=(512, 512), mode="bilinear", align_corners=False)
            image_logs["feat_alpha"] = feat.expand(-1, 3, -1, -1)

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
        """Log gradients for LoRA modules and alignment handler."""
        self.unwrap_model(self.D).eval().requires_grad_(False)
        x_pred = self.forward_generator()
        self.G_pred = x_pred

        loss_l2 = F.mse_loss(x_pred, self.batch_inputs.gt, reduction="mean") * self.config.lambda_l2
        loss_lpips = self.net_lpips(x_pred, self.batch_inputs.gt).mean() * self.config.lambda_lpips
        loss_disc = self.D(x_pred, for_G=True).mean() * self.config.lambda_gan
        loss_align = F.l1_loss(
            self._noise_pred.float(), self.batch_inputs.noise.float(), reduction="mean"
        ) * self.config.lambda_align

        losses = [("l2", loss_l2), ("lpips", loss_lpips),
                  ("disc", loss_disc), ("align", loss_align)]

        grad_dict = {}
        self.G_opt.zero_grad()
        for idx, (name, loss) in enumerate(losses):
            retain_graph = idx != len(losses) - 1
            loss.backward(retain_graph=retain_graph)

            # LoRA module gradients
            lora_module_grads = {}
            for module_name, module in self.unwrap_model(self.G).named_modules():
                for suffix in getattr(self.config, "log_grad_modules", ["conv_out"]):
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
