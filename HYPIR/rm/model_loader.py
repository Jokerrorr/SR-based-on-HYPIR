"""
RM Model Loader for HYPIR integration.

Loads DiffBIR restoration modules (RM) for integration with HYPIR.
Supports BID (SCUNet), BFR (SwinIR), and BSR (BSRNet) tasks.
"""

import os
import yaml
import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Tuple
from pathlib import Path

# Import from our local DiffBIR copy
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "DiffBIR"))

try:
    from DiffBIR.models.swinir import SwinIR
    from DiffBIR.models.scunet import SCUNet
    from DiffBIR.models.bsrnet import RRDBNet
    DIFFBIR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DiffBIR models: {e}")
    print("Make sure you have copied the model files to HYPIR/DiffBIR/models/")
    DIFFBIR_AVAILABLE = False


class RMModelLoader:
    """Loader for DiffBIR Restoration Modules (RM)."""

    SUPPORTED_TASKS = {
        'bid': 'SCUNet',      # Blind Image Denoising
        'bfr': 'SwinIR',      # Blind Face Restoration
        'bsr': 'BSRNet',      # Blind Super-Resolution
    }

    def __init__(self, task: str, device: str = 'cuda'):
        """
        Initialize RM model loader.

        Args:
            task: Task type - 'bid', 'bfr', or 'bsr'
            device: Device to load model on ('cuda' or 'cpu')
        """
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported: {list(self.SUPPORTED_TASKS.keys())}")

        self.task = task
        self.device = device
        self.model_type = self.SUPPORTED_TASKS[task]
        self.model = None
        self.config = None

    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """
        Load configuration for the RM model.

        Args:
            config_path: Path to YAML config file. If None, uses default config.

        Returns:
            Config dictionary
        """
        if config_path is None:
            # Use default config based on task
            config_dir = Path(__file__).parent.parent.parent / "DiffBIR" / "configs" / "inference"
            if self.task == 'bid':
                config_path = config_dir / "scunet.yaml"
            elif self.task == 'bfr':
                config_path = config_dir / "swinir.yaml"
            elif self.task == 'bsr':
                config_path = config_dir / "bsrnet.yaml"
            else:
                raise ValueError(f"No default config for task: {self.task}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        return self.config

    def create_model(self) -> nn.Module:
        """
        Create RM model instance from config.

        Returns:
            Instantiated model
        """
        if not DIFFBIR_AVAILABLE:
            raise ImportError("DiffBIR models not available. Check installation.")

        if self.config is None:
            self.load_config()

        # Extract model parameters from config
        # DiffBIR configs have direct 'params' key, not nested under 'model'
        model_config = self.config.get('params', {})

        if self.model_type == 'SwinIR':
            # SwinIR model for BFR task
            model = SwinIR(
                img_size=model_config.get('img_size', 64),
                patch_size=model_config.get('patch_size', 1),
                in_chans=model_config.get('in_chans', 3),
                embed_dim=model_config.get('embed_dim', 96),
                depths=model_config.get('depths', [6, 6, 6, 6]),
                num_heads=model_config.get('num_heads', [6, 6, 6, 6]),
                window_size=model_config.get('window_size', 7),
                mlp_ratio=model_config.get('mlp_ratio', 4.0),
                qkv_bias=model_config.get('qkv_bias', True),
                qk_scale=model_config.get('qk_scale', None),
                drop_rate=model_config.get('drop_rate', 0.0),
                attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
                drop_path_rate=model_config.get('drop_path_rate', 0.1),
                ape=model_config.get('ape', False),
                patch_norm=model_config.get('patch_norm', True),
                use_checkpoint=model_config.get('use_checkpoint', False),
                sf=model_config.get('sf', 1),  # scale factor
                img_range=model_config.get('img_range', 1.0),
                upsampler=model_config.get('upsampler', ''),
                resi_connection=model_config.get('resi_connection', '1conv'),
                unshuffle=model_config.get('unshuffle', False),
                unshuffle_scale=model_config.get('unshuffle_scale', None),
            )

        elif self.model_type == 'SCUNet':
            # SCUNet model for BID task
            model = SCUNet(
                in_nc=model_config.get('in_nc', 3),
                config=model_config.get('config', [4, 4, 4, 4, 4, 4, 4]),
                dim=model_config.get('dim', 64),
                drop_path_rate=model_config.get('drop_path_rate', 0.0),
                input_resolution=model_config.get('input_resolution', 256),
            )

        elif self.model_type == 'BSRNet':
            # BSRNet (RRDBNet) for BSR task
            model = RRDBNet(
                in_nc=model_config.get('in_nc', 3),
                out_nc=model_config.get('out_nc', 3),
                nf=model_config.get('nf', 64),
                nb=model_config.get('nb', 23),
                gc=model_config.get('gc', 32),
                sf=model_config.get('sf', 4),
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model = model
        return model

    def load_weights(self, weight_path: str, strict: bool = True) -> nn.Module:
        """
        Load weights from checkpoint file.

        Args:
            weight_path: Path to weight file (.pth, .ckpt, .pt)
            strict: Whether to strictly enforce key matching

        Returns:
            Model with loaded weights
        """
        if self.model is None:
            self.create_model()

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        # Load weights
        sd = torch.load(weight_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in sd:
            sd = sd['state_dict']

        # Remove 'module.' prefix if present
        if list(sd.keys())[0].startswith('module.'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(sd, strict=strict)

        if missing_keys:
            print(f"Warning: Missing keys in weight file: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in weight file: {unexpected_keys}")

        # Move to device and set to eval mode
        self.model = self.model.to(self.device).eval()

        return self.model

    def get_model(self) -> nn.Module:
        """Get loaded model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_weights() first.")
        return self.model

    def prepare_input(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Prepare input tensor for RM model.

        Args:
            image_tensor: Input tensor in range [0, 1], shape (1, 3, H, W)

        Returns:
            Prepared tensor
        """
        # RM models typically expect input in [0, 1] range
        # Some models might have different preprocessing
        return image_tensor


def load_rm_model(task: str, weight_path: str, config_path: Optional[str] = None,
                  device: str = 'cuda') -> nn.Module:
    """
    Convenience function to load RM model.

    Args:
        task: Task type ('bid', 'bfr', 'bsr')
        weight_path: Path to weight file
        config_path: Optional path to config file
        device: Device to load on

    Returns:
        Loaded model
    """
    loader = RMModelLoader(task=task, device=device)
    loader.load_config(config_path)
    loader.load_weights(weight_path)
    return loader.get_model()