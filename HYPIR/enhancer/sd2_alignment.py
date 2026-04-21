"""
SD2 Alignment Enhancer — single-step inference with FaithDiff-style training.

Training: GT → VAE encode → add_noise(t=200) → noisy_hq → UNet → noise_pred
          LQ → RM → VAE encode → x_en → AlignmentHandler → features
          Loss = L1(noise_pred, noise)

Inference (single-step):
  z_lq = VAE.encode(LQ)
  x_en = VAE.encode(RM(LQ))
  z_in = add_noise(z_lq, noise, t=200)  # match training noise level
  noise_pred = UNet(z_in, t=200, text, alignment_features)
  z_pred = scheduler.step(noise_pred, 200, z_in).pred_original_sample
  HQ = VAE.decode(z_pred)
"""

import numpy as np
import torch
from torch.nn import functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from PIL import Image
from typing import Literal, List

from HYPIR.enhancer.base import BaseEnhancer
from HYPIR.alignment.alignment_handler import AlignmentHandler
from HYPIR.model.unet_alignment import UNetAlignment
from HYPIR.utils.common import wavelet_reconstruction


class SD2AlignmentEnhancer(BaseEnhancer):

    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.base_model_path, subfolder="scheduler"
        )

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path, subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def init_generator(self):
        # Load base UNet
        unet = UNet2DConditionModel.from_pretrained(
            self.base_model_path, subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)

        # Add LoRA
        target_modules = self.lora_modules
        G_lora_cfg = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_rank,
            init_lora_weights="gaussian", target_modules=target_modules,
        )
        unet.add_adapter(G_lora_cfg)

        # Create alignment handler (latent-space input)
        handler = AlignmentHandler(unet_conv_channels=320, latent_channels=4)

        # Wrap into UNetAlignment
        self.G = UNetAlignment(unet=unet, alignment_handler=handler)

        # Load weights
        print(f"Load model weights from {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)

        missing, unexpected = self.G.load_state_dict(state_dict, strict=False)

        # Report loading results
        lora_loaded = sum(1 for k in self.G.unet.state_dict() if "lora" in k)
        print(f"LoRA keys present in model: {lora_loaded}")
        align_keys = [k for k in state_dict if k.startswith("alignment_handler")]
        if align_keys:
            print(f"Loaded {len(align_keys)} alignment_handler keys")
        else:
            print("No alignment_handler keys in weight file (using random init)")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys")

        self.G.to(device=self.device, dtype=self.weight_dtype)
        self.G.eval().requires_grad_(False)

    def prepare_inputs(self, batch_size, prompt):
        bs = batch_size
        txt_ids = self.tokenizer(
            [prompt] * bs,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.device))[0]
        c_txt = {"text_embed": text_embed}
        self.inputs = dict(c_txt=c_txt)

    def set_rm_output(self, rm_output: torch.Tensor):
        """Store pixel-space RM output [B, 3, H, W] in [0, 1]."""
        self._rm_pixel = rm_output

    @torch.no_grad()
    def enhance(
        self,
        lq: torch.Tensor,
        prompt: str,
        scale_by: Literal["factor", "longest_side"] = "factor",
        upscale: int = 1,
        target_longest_side: int | None = None,
        patch_size: int = 512,
        stride: int = 256,
        model_t: int = 200,
        return_type: Literal["pt", "np", "pil"] = "pt",
    ) -> torch.Tensor | np.ndarray | List[Image.Image]:
        """Single-step inference with alignment features (HYPIR-style speed)."""
        if stride <= 0:
            raise ValueError("Stride must be greater than 0.")
        if patch_size <= 0:
            raise ValueError("Patch size must be greater than 0.")

        bs = len(lq)

        # --- Scale LQ ---
        if scale_by == "factor":
            lq = F.interpolate(lq, scale_factor=upscale, mode="bicubic")
        elif scale_by == "longest_side":
            if target_longest_side is None:
                raise ValueError("target_longest_side must be specified.")
            h, w = lq.shape[2:]
            if h >= w:
                new_h, new_w = target_longest_side, int(w * (target_longest_side / h))
            else:
                new_w, new_h = target_longest_side, int(h * (target_longest_side / w))
            lq = F.interpolate(lq, size=(new_h, new_w), mode="bicubic")

        ref = lq
        h0, w0 = lq.shape[2:]

        # Ensure minimum size for processing
        vae_scale_factor = 8
        if min(h0, w0) < patch_size:
            lq = F.interpolate(lq, size=(
                max(h0, patch_size), max(w0, patch_size)
            ), mode="bicubic")

        h1, w1 = lq.shape[2:]

        # Pad to VAE multiple
        ph = (vae_scale_factor - h1 % vae_scale_factor) % vae_scale_factor
        pw = (vae_scale_factor - w1 % vae_scale_factor) % vae_scale_factor
        lq_norm = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
        lq_norm = F.pad(lq_norm, (0, pw, 0, ph), mode="constant", value=0)

        # --- Encode LQ latent ---
        z_lq = self.vae.encode(lq_norm).latent_dist.sample()

        # --- Encode RM output for alignment features ---
        rm_pixel = getattr(self, "_rm_pixel", None)
        if rm_pixel is not None:
            rm_scaled = F.interpolate(
                rm_pixel.to(device=self.device), size=(h1, w1),
                mode="bilinear", align_corners=False,
            )
            rm_norm = (rm_scaled * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
            rm_norm = F.pad(rm_norm, (0, pw, 0, ph), mode="constant", value=0)
            x_en = self.vae.encode(rm_norm).latent_dist.sample()
        else:
            # Fallback: use LQ latent as alignment input
            x_en = z_lq

        # --- Single-step inference ---
        self.prepare_inputs(batch_size=bs, prompt=prompt)

        # Add noise to z_lq at t=model_t (match training noise level)
        noise = torch.randn_like(z_lq)
        t_tensor = torch.tensor([model_t], device=self.device, dtype=torch.long)
        z_in = self.scheduler.add_noise(z_lq, noise, t_tensor)

        # UNet predicts noise
        noise_pred = self.G(
            z_in, t_tensor,
            encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
            x_en=x_en,
        ).sample

        # Single-step: get predicted original sample
        z_pred = self.scheduler.step(noise_pred, model_t, z_in).pred_original_sample

        # Decode
        z_pred = z_pred / self.vae.config.scaling_factor
        x = self.vae.decode(z_pred.to(self.weight_dtype)).sample.float()

        # Crop padding and resize back
        x = x[..., :h1, :w1]
        x = (x + 1) / 2
        x = F.interpolate(input=x, size=(h0, w0), mode="bicubic", antialias=True)
        x = wavelet_reconstruction(x, ref.to(device=self.device))

        if return_type == "pt":
            return x.clamp(0, 1).cpu()
        elif return_type == "np":
            return self.tensor2image(x)
        else:
            return [Image.fromarray(img) for img in self.tensor2image(x)]
