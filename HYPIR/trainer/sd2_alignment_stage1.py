"""
Stage 1 Alignment Pretraining Trainer (FaithDiff-style).

Loads HYPIR pretrained LoRA (frozen), only trains Alignment Handler parameters.
Uses L1 noise prediction loss with FaithDiff-style additive injection after conv_in.

Training flow:
  GT → VAE.encode → z_hq → add_noise(t=200) → x_hq_t → conv_in → sample_emb ─┐
                                                                              ├→ sample_emb + feat_alpha → UNet → noise_pred
  RM(LQ) → VAE.encode → z_lq → FaithDiffAlignment(sample_emb, z_lq) ─────────┘
  Loss = L1(noise_pred, noise)
"""

import os
import csv
import logging

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel
from torchvision.utils import make_grid

from HYPIR.trainer.sd2_alignment import SD2AlignmentTrainer
from HYPIR.alignment.faithdiff_alignment import FaithDiffAlignment
from HYPIR.model.unet_alignment import UNetAlignment
from HYPIR.utils.common import print_vram_state

logger = get_logger(__name__, log_level="INFO")


class SD2AlignmentStage1Trainer(SD2AlignmentTrainer):
    """Stage 1: Load HYPIR pretrained LoRA (frozen), only train alignment handler.
    FaithDiff-style additive injection after conv_in.
    No discriminator, no LPIPS, no G/D alternation.
    """

    def init_models(self):
        """Override: skip D and LPIPS for Stage 1."""
        self.init_scheduler()
        self.init_text_models()
        self.init_vae()
        self.init_rm()
        self.init_generator()
        # No discriminator or LPIPS for Stage 1
        self.D = None
        self.net_lpips = None

    def init_generator(self):
        from peft import LoraConfig

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

        # Create FaithDiff-style alignment handler
        alignment_cfg = getattr(self.config, "alignment", None)
        if alignment_cfg is not None:
            handler = FaithDiffAlignment(
                conditioning_channels=getattr(alignment_cfg, "conditioning_channels", 4),
                embedding_channels=getattr(alignment_cfg, "embedding_channels", 320),
                num_trans_channel=getattr(alignment_cfg, "num_trans_channel", 640),
                num_trans_head=getattr(alignment_cfg, "num_trans_head", 8),
                num_trans_layer=getattr(alignment_cfg, "num_trans_layer", 2),
            )
        else:
            handler = FaithDiffAlignment()

        # Wrap UNet with alignment
        self.G = UNetAlignment(unet=unet, alignment_handler=handler)
        logger.info("Stage 1: FaithDiff-style additive injection after conv_in enabled")

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

    def prepare_all(self):
        """Override: only prepare G, G_opt, dataloader (no D)."""
        logger.info("Stage 1: Wrapping models, optimizers and dataloaders (no D)")
        attrs = ["G", "G_opt", "dataloader"]
        prepared_objs = self.accelerator.prepare(
            self.G, self.G_opt, self.dataloader
        )
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)
        print_vram_state("After accelerator.prepare", logger=logger)

    def forward_generator(self) -> torch.Tensor:
        """Stage 1: Return noise_pred directly (no VAE decode needed)."""
        noise_pred = self.G(
            self.batch_inputs.x_hq_t,
            self.batch_inputs.timesteps,
            encoder_hidden_states=self.batch_inputs.c_txt["text_embed"],
            z_lq=self.batch_inputs.z_lq,
            x_hq_t=self.batch_inputs.x_hq_t,
        ).sample
        return noise_pred

    def optimize_generator(self):
        """Stage 1: L1 noise prediction loss only."""
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
        """No-op: no discriminator for Stage 1."""
        return {}

    def on_training_start(self):
        """Initialize EMA and CSV logger for Stage 1."""
        super().on_training_start()
        self.loss_csv_path = os.path.join(self.config.output_dir, "loss_log.csv")
        if self.accelerator.is_main_process:
            # Only write header if not resuming (file doesn't exist or starting fresh)
            if self.global_step == 0 or not os.path.exists(self.loss_csv_path):
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

    def run(self):
        """Stage 1: Simplified loop without G/D alternation."""
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

    def log_images(self):
        """Stage 1: Log LQ, GT, RM output, feat_alpha heatmap."""
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

        # feat_alpha heatmap visualization
        with torch.no_grad():
            handler = self.G.alignment_handler
            z_lq = self.batch_inputs.z_lq[:N]
            x_hq_t = self.batch_inputs.x_hq_t[:N]
            handler_dtype = next(handler.parameters()).dtype
            unet_dtype = self.G.unet.conv_in.weight.dtype

            # Compute feat_alpha
            sample_emb = self.G.unet.conv_in(x_hq_t.to(dtype=unet_dtype))
            feat_alpha = handler(sample_emb.to(dtype=handler_dtype), z_lq.to(dtype=handler_dtype))

            # Take first channel as heatmap
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
