"""
SD2 Alignment Enhancer — FaithDiff-style additive injection inference.

Training: FaithDiffAlignment(sample_emb, z_lq) → feat_alpha → sample_emb + feat_alpha → UNet → noise_pred
Inference: add_noise(z_lq, t=200) → conv_in → sample_emb + feat_alpha → UNet → scheduler.step → z_pred → VAE.decode → HQ

Additive injection after conv_in (320ch feature space).
No GT at inference — use add_noise(VAE.encode(RM(LQ)), t=200) as x_hq_t approximation.
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
from HYPIR.alignment.faithdiff_alignment import FaithDiffAlignment
from HYPIR.model.unet_alignment import UNetAlignment
from HYPIR.utils.common import wavelet_reconstruction, make_tiled_fn


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

        # Create FaithDiff-style alignment handler
        handler = FaithDiffAlignment(
            conditioning_channels=4,
            embedding_channels=320,
            num_trans_channel=640,
            num_trans_head=8,
            num_trans_layer=2,
        )

        # Wrap into UNetAlignment
        self.G = UNetAlignment(unet=unet, alignment_handler=handler)

        # Load weights
        print(f"Load model weights from {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)

        # UNetAlignment.load_state_dict auto-separates unet.* / alignment_handler.* keys
        missing, unexpected = self.G.load_state_dict(state_dict, strict=False)

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
        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = dict(c_txt=c_txt, timesteps=timesteps)

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
        return_type: Literal["pt", "np", "pil"] = "pt",
    ) -> torch.Tensor | np.ndarray | List[Image.Image]:
        """Inference: FaithDiff-style additive injection after conv_in."""
        if stride <= 0:
            raise ValueError("Stride must be greater than 0.")
        if patch_size <= 0:
            raise ValueError("Patch size must be greater than 0.")
        if patch_size < stride:
            raise ValueError("Patch size must be greater than or equal to stride.")

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

        # Ensure minimum size
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

        # --- VAE encoding with tiled processing ---
        rm_pixel = getattr(self, "_rm_pixel", None)
        if rm_pixel is not None:
            rm_scaled = F.interpolate(
                rm_pixel.to(device=self.device), size=(h1, w1),
                mode="bilinear", align_corners=False,
            )
            rm_norm = (rm_scaled * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
            rm_norm = F.pad(rm_norm, (0, pw, 0, ph), mode="constant", value=0)
            encode_input = rm_norm
        else:
            encode_input = lq_norm

        z_lq_full = make_tiled_fn(
            fn=lambda tile: self.vae.encode(tile).latent_dist.sample(),
            size=patch_size,
            stride=stride,
            scale_type="down",
            scale=vae_scale_factor,
            progress=True,
            channel=self.vae.config.latent_channels,
            desc="VAE encoding",
        )(encode_input.to(self.weight_dtype))

        self._z_lq_precomputed = z_lq_full

        # --- Tiled generator forward ---
        self.prepare_inputs(batch_size=bs, prompt=prompt)
        latent_patch = patch_size // vae_scale_factor
        latent_stride = stride // vae_scale_factor

        z = make_tiled_fn(
            fn=self._tiled_generator_forward,
            size=latent_patch,
            stride=latent_stride,
            progress=True,
            desc="Generator Forward",
        )(z_lq_full.to(self.weight_dtype))

        # Decode
        x = make_tiled_fn(
            fn=lambda tile: self.vae.decode(tile).sample.float(),
            size=latent_patch,
            stride=latent_stride,
            scale_type="up",
            scale=vae_scale_factor,
            progress=True,
            channel=3,
            desc="VAE decoding",
        )(z.to(self.weight_dtype))

        x = x[..., :h1, :w1]
        x = (x + 1) / 2
        x = F.interpolate(input=x, size=(h0, w0), mode="bicubic", antialias=True)
        x = wavelet_reconstruction(x, ref.to(device=self.device))

        # Cleanup
        self._z_lq_precomputed = None

        if return_type == "pt":
            return x.clamp(0, 1).cpu()
        elif return_type == "np":
            return self.tensor2image(x)
        else:
            return [Image.fromarray(img) for img in self.tensor2image(x)]

    def _tiled_generator_forward(self, z_lq_tile, **kwargs):
        """Tiled forward: conv_in → sample_emb + feat_alpha → UNet → scheduler.step."""
        z_lq_tile = z_lq_tile * self.vae.config.scaling_factor

        # Approximate x_hq_t: add noise to z_lq at t=model_t
        noise = torch.randn_like(z_lq_tile)
        x_hq_t = self.scheduler.add_noise(z_lq_tile, noise, self.inputs["timesteps"])

        # FaithDiff-style: additive injection after conv_in
        eps = self.G(
            z_lq_tile,
            self.inputs["timesteps"],
            encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
            z_lq=z_lq_tile,
            x_hq_t=x_hq_t,
        ).sample

        z = self.scheduler.step(eps, self.model_t, z_lq_tile).pred_original_sample
        return z / self.vae.config.scaling_factor
