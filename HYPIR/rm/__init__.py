"""
Restoration Module (RM) integration for HYPIR.
Provides unified interface for DiffBIR restoration modules.
"""

from .model_loader import RMModelLoader, load_rm_model
from .restoration_module import RestorationModule
from .tiled_inference import tiled_inference

__all__ = [
    'RMModelLoader',
    'load_rm_model',
    'RestorationModule',
    'tiled_inference'
]