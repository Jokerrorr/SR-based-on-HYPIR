"""
RM+HYPIR Two-Stage Pipeline.

Combines DiffBIR Restoration Module (RM) for degradation removal
with HYPIR for detail generation.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import numpy as np
from PIL import Image
from pathlib import Path

# Import RM module
from ..rm.restoration_module import RestorationModule

# Try to import HYPIR enhancer
try:
    from ..enhancer.sd2 import SD2Enhancer
    HYPIR_AVAILABLE = True
except ImportError:
    print("Warning: HYPIR enhancer not available. Make sure HYPIR is installed.")
    HYPIR_AVAILABLE = False


class RMHYPIRPipeline:
    """
    Two-stage pipeline: RM (degradation removal) + HYPIR (detail generation).

    This pipeline combines the strengths of both models:
    1. RM (Restoration Module): Removes noise, artifacts, and degradations
    2. HYPIR: Generates realistic details and textures
    """

    def __init__(self,
                 rm_model: Optional[RestorationModule] = None,
                 hypir_model: Optional['SD2Enhancer'] = None,
                 device: str = 'cuda'):
        """
        Initialize pipeline.

        Args:
            rm_model: Pre-initialized RestorationModule
            hypir_model: Pre-initialized SD2Enhancer
            device: Device to run on
        """
        self.device = device
        self.rm_model = rm_model
        self.hypir_model = hypir_model
        self._initialized = False

    def load_rm(self,
                task: str = 'bid',
                weight_path: str = None,
                config_path: Optional[str] = None) -> None:
        """
        Load RM model.

        Args:
            task: RM task type ('bid', 'bfr', 'bsr')
            weight_path: Path to RM weight file
            config_path: Optional path to RM config file
        """
        if weight_path is None:
            raise ValueError("weight_path is required for RM model")

        self.rm_model = RestorationModule(task=task, device=self.device)
        self.rm_model.load(weight_path=weight_path, config_path=config_path)

        print(f"RM model loaded: task={task}, weights={weight_path}")

    def load_hypir(self,
                   base_model_path: str,
                   weight_path: str,
                   lora_modules: str = "attn1,attn2,ff",
                   lora_rank: int = 256,
                   model_t: int = 1000,
                   coeff_t: int = 400) -> None:
        """
        Load HYPIR model.

        Args:
            base_model_path: Path to base Stable Diffusion model
            weight_path: Path to HYPIR LoRA weights
            lora_modules: Comma-separated LoRA module names
            lora_rank: LoRA rank
            model_t: Model timestep
            coeff_t: Coefficient timestep
        """
        if not HYPIR_AVAILABLE:
            raise ImportError("HYPIR not available. Check installation.")

        self.hypir_model = SD2Enhancer(
            base_model_path=base_model_path,
            weight_path=weight_path,
            lora_modules=lora_modules.split(","),
            lora_rank=lora_rank,
            model_t=model_t,
            coeff_t=coeff_t,
            device=self.device,
        )

        # Initialize HYPIR models
        print("Initializing HYPIR models...")
        self.hypir_model.init_models()
        print("HYPIR model loaded")

    def _ensure_initialized(self) -> None:
        """Check if both models are loaded."""
        if self.rm_model is None:
            raise RuntimeError("RM model not loaded. Call load_rm() first.")
        if self.hypir_model is None:
            raise RuntimeError("HYPIR model not loaded. Call load_hypir() first.")
        self._initialized = True

    def _ensure_min_size(self, image: torch.Tensor, min_size: int = 512) -> torch.Tensor:
        """
        Ensure image meets minimum size requirement.

        Args:
            image: Input tensor (1, 3, H, W)
            min_size: Minimum size

        Returns:
            Resized tensor if needed
        """
        _, _, h, w = image.shape

        if h >= min_size and w >= min_size:
            return image

        # Calculate scale factor
        scale_h = max(1.0, min_size / h)
        scale_w = max(1.0, min_size / w)
        scale = max(scale_h, scale_w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        print(f"Resizing from ({h}, {w}) to ({new_h}, {new_w}) "
              f"to meet minimum size {min_size}")

        # Use bicubic interpolation
        image_resized = torch.nn.functional.interpolate(
            image, size=(new_h, new_w), mode='bicubic', align_corners=False
        )

        return image_resized

    def run_rm_stage(self,
                     lq_image: Union[torch.Tensor, Image.Image, np.ndarray],
                     tile_size: int = 512,
                     tile_stride: int = 256,
                     upscale: float = 1.0,
                     save_intermediate: Optional[str] = None,
                     ensure_min_size: bool = False) -> torch.Tensor:
        """
        Run RM stage only.

        Args:
            lq_image: Low-quality input image
            tile_size: RM tile size
            tile_stride: RM tile stride
            upscale: RM upscale factor (default: 1.0)
            save_intermediate: Optional path to save RM output

        Returns:
            RM output tensor
        """
        self._ensure_initialized()

        print("=== Running RM Stage ===")
        print(f"Input shape: {getattr(lq_image, 'shape', 'unknown')}")

        # Process with RM
        rm_output = self.rm_model.process_image(
            image=lq_image,
            tile_size=tile_size,
            tile_stride=tile_stride,
            ensure_min_size=ensure_min_size,  # Keep original size by default
            upscale=upscale
        )

        print(f"RM output shape: {rm_output.shape}")

        # Save intermediate result if requested
        if save_intermediate:
            self.rm_model.save_output(rm_output, save_intermediate)
            print(f"Saved RM output to {save_intermediate}")

        return rm_output

    def run_hypir_stage(self,
                        rm_output: torch.Tensor,
                        prompt: str = "",
                        upscale: float = 1.0,
                        patch_size: int = 512,
                        stride: int = 256,
                        captioner: str = "empty",
                        fixed_caption: Optional[str] = None,
                        scale_by: str = "factor",
                        target_longest_side: Optional[int] = None) -> torch.Tensor:
        """
        Run HYPIR stage only.

        Args:
            rm_output: RM output tensor (1, 3, H, W) in range [0, 1]
            prompt: Text prompt for enhancement
            scale_by: Scaling method ('factor' or 'longest_side')
            upscale: Upscale factor
            target_longest_side: Target longest side when scale_by='longest_side'
            patch_size: HYPIR patch size
            stride: HYPIR stride
            captioner: Captioner type ('empty' or 'fixed')
            fixed_caption: Fixed caption if captioner='fixed'

        Returns:
            HYPIR output tensor
        """
        self._ensure_initialized()

        print("=== Running HYPIR Stage ===")
        print(f"Input shape: {rm_output.shape}")
        print(f"Prompt: '{prompt}'")
        print(f"HYPIR Stage Upscale: {upscale}")

        # Ensure RM output is on correct device
        rm_output = rm_output.to(self.device)

        # Run HYPIR enhancement
        with torch.no_grad():
            hypir_output = self.hypir_model.enhance(
                lq=rm_output,
                prompt=prompt,
                scale_by=scale_by,
                upscale=upscale,
                target_longest_side=target_longest_side,
                patch_size=patch_size,
                stride=stride
            )

        print(f"HYPIR output shape: {hypir_output.shape}")

        return hypir_output

    def run(self,
            lq_image: Union[torch.Tensor, Image.Image, np.ndarray],
            prompt: str = "",
            rm_tile_size: int = 512,
            rm_tile_stride: int = 256,
            rm_upscale: float = 1.0,
            hypir_patch_size: int = 512,
            hypir_stride: int = 256,
            hypir_upscale: float = 1.0,
            save_intermediate: Optional[str] = None,
            captioner: str = "empty",
            fixed_caption: Optional[str] = None,
            scale_by: str = "factor",
            target_longest_side: Optional[int] = None,
            ensure_min_size: bool = False) -> torch.Tensor:
        """
        Run full two-stage pipeline.

        Args:
            lq_image: Low-quality input image
            prompt: Text prompt for HYPIR
            rm_tile_size: RM tile size
            rm_tile_stride: RM tile stride
            rm_upscale: RM upscale factor (default: 1.0)
            hypir_patch_size: HYPIR patch size
            hypir_stride: HYPIR stride
            hypir_upscale: HYPIR upscale factor
            save_intermediate: Optional path to save RM output
            captioner: Captioner type
            fixed_caption: Fixed caption
            scale_by: Scaling method for HYPIR
            target_longest_side: Target longest side when scale_by='longest_side'
            ensure_min_size: Whether to resize RM output to meet minimum size (default: False)

        Returns:
            Final enhanced tensor
        """
        self._ensure_initialized()

        print("=" * 50)
        print("Running RM+HYPIR Two-Stage Pipeline")
        print("=" * 50)

        # Stage 1: RM degradation removal
        rm_output = self.run_rm_stage(
            lq_image=lq_image,
            tile_size=rm_tile_size,
            tile_stride=rm_tile_stride,
            upscale=rm_upscale,
            save_intermediate=save_intermediate,
            ensure_min_size=ensure_min_size
        )

        # Stage 2: HYPIR detail generation
        final_output = self.run_hypir_stage(
            rm_output=rm_output,
            prompt=prompt,
            upscale=hypir_upscale,
            patch_size=hypir_patch_size,
            stride=hypir_stride,
            captioner=captioner,
            fixed_caption=fixed_caption,
            scale_by=scale_by,
            target_longest_side=target_longest_side
        )

        print("=" * 50)
        print("Pipeline completed successfully!")
        print(f"Final output shape: {final_output.shape}")
        print("=" * 50)

        return final_output

    def save_output(self,
                    output: torch.Tensor,
                    save_path: Union[str, Path],
                    format: str = 'png') -> None:
        """
        Save output tensor as image.

        Args:
            output: Output tensor in range [0, 1], shape (1, 3, H, W)
            save_path: Path to save image
            format: Image format
        """
        # Convert tensor to PIL Image
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)

        # Save image
        img = Image.fromarray(output_np)
        img.save(save_path, format=format)
        print(f"Saved final output to {save_path}")

    def get_pipeline_info(self) -> dict:
        """Get information about pipeline configuration."""
        rm_info = self.rm_model.get_model_info() if self.rm_model else None

        return {
            'rm_model': rm_info,
            'hypir_available': HYPIR_AVAILABLE,
            'device': self.device,
            'initialized': self._initialized
        }