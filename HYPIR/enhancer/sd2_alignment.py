"""
SD2 Enhancer with alignment module for RM+HYPIR inference.

Extends SD2Enhancer to inject alignment features from x_en
(VAE Encoder output on RM-restored image) into UNet forward pass.

Alignment input: pixel-space RM output. x_en is computed inside enhance()
at the same scale as z_lq so spatial dimensions always match.

Pipeline flow:
  LQ → VAE Enc → z_lq
  RM(pixel) → scale/pad same as LQ → VAE Enc → x_en
  z_lq, x_en → UNet(conv_in(z_lq) + alignment(x_en)) → z_out → VAE Dec
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

        # Create alignment handler (latent-space input)
        handler = AlignmentHandler(unet_conv_channels=320, latent_channels=4)

        # Wrap into UNetAlignment
        self.G = UNetAlignment(unet=unet, alignment_handler=handler)

        # Load weights — supports both:
        #   1. HYPIR original LoRA-only weights (514 keys, no alignment_handler prefix)
        #   2. Stage 2 trained weights (LoRA + alignment_handler keys)
        print(f"Load model weights from {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)

        # UNetAlignment.load_state_dict auto-separates unet.* / alignment_handler.* keys
        missing, unexpected = self.G.load_state_dict(state_dict, strict=False)

        # Report loading results
        missing_unet_base = [k for k in missing if "lora" not in k and "alignment" not in k]
        missing_lora = [k for k in missing if "lora" in k]
        if missing_lora:
            print(f"Warning: {len(missing_lora)} LoRA keys failed to load")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys in weight file")
        lora_loaded = sum(1 for k in self.G.unet.state_dict() if "lora" in k)
        print(f"LoRA keys present in model: {lora_loaded}")

        # Report alignment keys
        align_keys = [k for k in state_dict if k.startswith("alignment_handler.")]
        if align_keys:
            print(f"Loaded {len(align_keys)} alignment_handler keys")
        else:
            print("No alignment_handler keys in weight file (using random init)")

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
        self.inputs = dict(
            c_txt=c_txt,
            timesteps=timesteps,
        )

    def set_rm_output(self, rm_output: torch.Tensor):
        """Store pixel-space RM output [B, 3, H, W] in [0, 1].

        x_en will be computed inside enhance() at the same scale as z_lq.
        """
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
        """Override enhance() to compute x_en at the same scale as z_lq."""
        if stride <= 0:
            raise ValueError("Stride must be greater than 0.")
        if patch_size <= 0:
            raise ValueError("Patch size must be greater than 0.")
        if patch_size < stride:
            raise ValueError("Patch size must be greater than or equal to stride.")

        bs = len(lq)

        # --- Scale LQ (same as BaseEnhancer) ---
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
        if min(h0, w0) <= patch_size:
            lq = self.resize_at_least(lq, size=patch_size)

        # Pad to VAE multiple
        vae_scale_factor = 8
        lq_norm = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
        h1, w1 = lq_norm.shape[2:]
        ph = (h1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - h1
        pw = (w1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - w1
        lq_norm = F.pad(lq_norm, (0, pw, 0, ph), mode="constant", value=0)

        # --- Encode x_en from RM pixel output at the SAME size as padded LQ ---
        rm_pixel = getattr(self, "_rm_pixel", None)
        if rm_pixel is not None:
            rm_scaled = F.interpolate(rm_pixel.to(device=self.device), size=(h1, w1),
                                       mode="bilinear", align_corners=False)
            rm_norm = (rm_scaled * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
            rm_norm = F.pad(rm_norm, (0, pw, 0, ph), mode="constant", value=0)
            x_en_full = self.vae.encode(rm_norm).latent_dist.sample()
            self._x_en_precomputed = x_en_full
        else:
            self._x_en_precomputed = None

        # VAE encode LQ
        z_lq = make_tiled_fn(
            fn=lambda lq_tile: self.vae.encode(lq_tile).latent_dist.sample(),
            size=patch_size,
            stride=stride,
            scale_type="down",
            scale=vae_scale_factor,
            progress=True,
            channel=self.vae.config.latent_channels,
            desc="VAE encoding",
        )(lq_norm.to(self.weight_dtype))

        # Generator forward with x_en tile cropping
        self.prepare_inputs(batch_size=bs, prompt=prompt)
        latent_patch = patch_size // vae_scale_factor
        latent_stride = stride // vae_scale_factor
        z = make_tiled_fn(
            fn=self._tiled_generator_forward,
            size=latent_patch,
            stride=latent_stride,
            progress=True,
            desc="Generator Forward",
        )(z_lq.to(self.weight_dtype), _align=True)

        # Decode
        x = make_tiled_fn(
            fn=lambda lq_tile: self.vae.decode(lq_tile).sample.float(),
            size=patch_size // vae_scale_factor,
            stride=stride // vae_scale_factor,
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
        self._x_en_precomputed = None

        if return_type == "pt":
            return x.clamp(0, 1).cpu()
        elif return_type == "np":
            return self.tensor2image(x)
        else:
            return [Image.fromarray(img) for img in self.tensor2image(x)]

    def _tiled_generator_forward(self, z_lq_tile, **kwargs):
        """Forward generator for a single tile, cropping x_en to match."""
        x_en = getattr(self, "_x_en_precomputed", None)
        if x_en is not None and x_en.shape != z_lq_tile.shape:
            tile_idx = kwargs.get("index")
            if tile_idx is not None:
                x_en = x_en[..., tile_idx.hi:tile_idx.hi_end, tile_idx.wi:tile_idx.wi_end]
            else:
                x_en = x_en[..., :z_lq_tile.shape[2], :z_lq_tile.shape[3]]

        z_in = z_lq_tile * self.vae.config.scaling_factor
        eps = self.G(
            z_in, self.inputs["timesteps"],
            encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
            x_en=x_en,
        ).sample
        z = self.scheduler.step(eps, self.coeff_t, z_in).pred_original_sample
        return z / self.vae.config.scaling_factor
