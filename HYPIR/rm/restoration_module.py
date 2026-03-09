"""
Unified Restoration Module (RM) interface for HYPIR integration.

Provides a consistent interface for different DiffBIR restoration modules
with support for tiled inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
from pathlib import Path

from .model_loader import RMModelLoader, load_rm_model
from .tiled_inference import tiled_inference


class RestorationModule:
    """
    Unified interface for DiffBIR Restoration Modules.

    This class provides a consistent interface for different restoration modules
    (SwinIR for BFR, SCUNet for BID, BSRNet for BSR) with support for tiled inference.
    """

    def __init__(self, task: str = 'bid', device: str = 'cuda'):
        """
        Initialize restoration module.

        Args:
            task: Task type - 'bid' (Blind Image Denoising),
                             'bfr' (Blind Face Restoration),
                             'bsr' (Blind Super-Resolution)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.task = task
        self.device = device
        self.model = None
        self.model_loader = None
        self._initialized = False

    def load(self, weight_path: str, config_path: Optional[str] = None) -> None:
        """
        Load RM model with weights.

        Args:
            weight_path: Path to weight file (.pth, .ckpt, .pt)
            config_path: Optional path to config file (YAML)
        """
        self.model_loader = RMModelLoader(task=self.task, device=self.device)

        # Load configuration
        self.model_loader.load_config(config_path)

        # Load weights
        self.model_loader.load_weights(weight_path)

        # Get the model
        self.model = self.model_loader.get_model()
        self._initialized = True

        print(f"Loaded {self.task.upper()} model ({self.model_loader.model_type}) "
              f"from {weight_path}")

    def _ensure_initialized(self) -> None:
        """Check if model is loaded."""
        if not self._initialized or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

    def prepare_input(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Prepare input for RM model.

        Args:
            image: Input image as torch.Tensor, PIL Image, or numpy array.
                   Tensor should be in range [0, 1], shape (C, H, W) or (1, C, H, W).
                   PIL/numpy should be in range [0, 255], shape (H, W, C).

        Returns:
            Preprocessed tensor in range [0, 1], shape (1, 3, H, W)
        """
        self._ensure_initialized()

        # Convert to tensor if needed
        if isinstance(image, Image.Image):
            # PIL Image to numpy
            image = np.array(image).astype(np.float32) / 255.0
            # (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            # numpy array to tensor
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image)
            if image.ndim == 3:
                # (H, W, C) -> (C, H, W)
                if image.shape[2] == 3:
                    image = image.permute(2, 0, 1)

        # Ensure tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(image)}")

        # Ensure proper shape: (1, C, H, W)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Ensure 3 channels
        if image.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[1]}")

        # Ensure range [0, 1]
        if image.min() < 0 or image.max() > 1:
            print(f"Warning: Input tensor range [{image.min():.3f}, {image.max():.3f}], "
                  f"clamping to [0, 1]")
            image = torch.clamp(image, 0, 1)

        # Move to device
        image = image.to(self.device)

        return image

    def _ensure_min_size(self, image: torch.Tensor, min_size: int = 512) -> torch.Tensor:
        """
        Ensure image meets minimum size requirement for HYPIR.

        Args:
            image: Input tensor (1, 3, H, W)
            min_size: Minimum size (height and width)

        Returns:
            Resized tensor if needed
        """
        _, _, h, w = image.shape

        if h >= min_size and w >= min_size:
            return image

        # Calculate scale factor to reach min_size
        scale_h = max(1.0, min_size / h)
        scale_w = max(1.0, min_size / w)
        scale = max(scale_h, scale_w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        print(f"Resizing RM output from ({h}, {w}) to ({new_h}, {w}) "
              f"to meet HYPIR minimum size {min_size}")

        # Use bicubic interpolation
        image_resized = torch.nn.functional.interpolate(
            image, size=(new_h, new_w), mode='bicubic', align_corners=False
        )

        return image_resized

    def inference(self,
                  lq_tensor: torch.Tensor,
                  tile_size: int = 512,
                  tile_stride: int = 256,
                  ensure_min_size: bool = True) -> torch.Tensor:
        """
        Perform inference with RM model.

        Args:
            lq_tensor: Low-quality input tensor in range [0, 1], shape (1, 3, H, W)
            tile_size: Tile size for tiled inference
            tile_stride: Stride for tiled inference
            ensure_min_size: Whether to ensure output meets HYPIR minimum size (512)

        Returns:
            Restored tensor in range [0, 1], shape (1, 3, H', W')
        """
        self._ensure_initialized()

        # Store original size
        _, _, h, w = lq_tensor.shape
        original_size = (h, w)

        # Perform inference
        with torch.no_grad():
            if h <= tile_size and w <= tile_size:
                # Single forward pass
                output = self.model(lq_tensor)
            else:
                # Tiled inference
                print(f"Using tiled inference: tile_size={tile_size}, stride={tile_stride}")
                output = tiled_inference(
                    model=self.model,
                    x=lq_tensor,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                    device=self.device
                )

        # Ensure output meets HYPIR minimum size if requested
        if ensure_min_size:
            output = self._ensure_min_size(output, min_size=512)

        print(f"RM inference completed: {original_size} -> {output.shape[2:]}")

        return output

    def process_image(self,
                      image: Union[torch.Tensor, Image.Image, np.ndarray],
                      tile_size: int = 512,
                      tile_stride: int = 256,
                      ensure_min_size: bool = True,
                      upscale: float = 1.0) -> torch.Tensor:
        """
        Complete processing pipeline: prepare input -> inference.

        Args:
            image: Input image (Tensor, PIL, or numpy)
            tile_size: Tile size for tiled inference
            tile_stride: Stride for tiled inference
            ensure_min_size: Whether to ensure output meets minimum size
            upscale: Upscale factor (default: 1.0). Note: RM models may have built-in upsampling.

        Returns:
            Restored tensor in range [0, 1], shape (1, 3, H', W')
        """
        # Check upscale parameter
        if upscale != 1.0:
            print(f"Note: RM upscale={upscale}, but RM models may have built-in upsampling.")
            print(f"      Actual upscale behavior depends on the specific RM model ({self.task}).")

        # Prepare input
        lq_tensor = self.prepare_input(image)

        # Perform inference
        restored = self.inference(
            lq_tensor,
            tile_size=tile_size,
            tile_stride=tile_stride,
            ensure_min_size=ensure_min_size
        )

        return restored

    def save_output(self,
                    output: torch.Tensor,
                    save_path: Union[str, Path],
                    format: str = 'png') -> None:
        """
        Save output tensor as image.

        Args:
            output: Output tensor in range [0, 1], shape (1, 3, H, W)
            save_path: Path to save image
            format: Image format ('png', 'jpg', etc.)
        """
        self._ensure_initialized()

        # Convert tensor to PIL Image
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)

        # Save image
        img = Image.fromarray(output_np)
        img.save(save_path, format=format)
        print(f"Saved output to {save_path}")

    def get_model_info(self) -> dict:
        """Get information about loaded model."""
        self._ensure_initialized()

        return {
            'task': self.task,
            'model_type': self.model_loader.model_type,
            'device': self.device,
            'initialized': self._initialized
        }