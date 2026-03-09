"""
Tiled inference for large images.

Implements sliding window inference with Gaussian weighting for seamless tile blending.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Callable
from tqdm import tqdm


def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> List[Tuple[int, int, int, int]]:
    """
    Generate sliding window coordinates.

    Args:
        h: Image height
        w: Image width
        tile_size: Tile size
        tile_stride: Tile stride

    Returns:
        List of (top, bottom, left, right) coordinates
    """
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))

    return coords


def gaussian_weights(tile_width: int, tile_height: int, sigma: float = 0.01) -> np.ndarray:
    """
    Generate Gaussian weight matrix for tile blending.

    Args:
        tile_width: Tile width
        tile_height: Tile height
        sigma: Gaussian sigma parameter

    Returns:
        Weight matrix of shape (tile_height, tile_width)
    """
    midpoint_w = (tile_width - 1) / 2
    midpoint_h = (tile_height - 1) / 2

    # Generate 1D Gaussian profiles
    x_probs = [
        np.exp(-(x - midpoint_w) * (x - midpoint_w) / (tile_width * tile_width) / (2 * sigma))
        / np.sqrt(2 * np.pi * sigma)
        for x in range(tile_width)
    ]

    y_probs = [
        np.exp(-(y - midpoint_h) * (y - midpoint_h) / (tile_height * tile_height) / (2 * sigma))
        / np.sqrt(2 * np.pi * sigma)
        for y in range(tile_height)
    ]

    # Create 2D Gaussian weights
    weights = np.outer(y_probs, x_probs)

    # Normalize
    weights = weights / weights.max()

    return weights


def tiled_inference(
    model: nn.Module,
    x: torch.Tensor,
    tile_size: int = 512,
    tile_stride: int = 256,
    device: str = 'cuda',
    weight_type: str = 'gaussian',
    progress: bool = True
) -> torch.Tensor:
    """
    Perform tiled inference on large images.

    Args:
        model: PyTorch model
        x: Input tensor of shape (1, C, H, W)
        tile_size: Tile size
        tile_stride: Tile stride
        device: Device to run on
        weight_type: Weight type for blending ('gaussian' or 'uniform')
        progress: Show progress bar

    Returns:
        Output tensor
    """
    # Ensure model is in eval mode
    model.eval()

    # Get input dimensions
    b, c, h, w = x.shape
    if b != 1:
        raise ValueError(f"Batch size must be 1 for tiled inference, got {b}")

    # Create output tensor
    output = torch.zeros_like(x, device=device)
    count = torch.zeros_like(x, dtype=torch.float32, device=device)

    # Generate weights
    if weight_type == 'gaussian':
        weights = gaussian_weights(tile_size, tile_size)
        weights = torch.tensor(weights, device=device).view(1, 1, tile_size, tile_size)
    else:  # uniform
        weights = torch.ones(1, 1, tile_size, tile_size, device=device)

    # Generate window coordinates
    windows = sliding_windows(h, w, tile_size, tile_stride)

    # Process tiles
    pbar = tqdm(windows, desc="Tiled inference", disable=not progress)
    for hi, hi_end, wi, wi_end in pbar:
        # Extract tile
        tile = x[:, :, hi:hi_end, wi:wi_end].to(device)

        # Process tile
        with torch.no_grad():
            output_tile = model(tile)

        # Add weighted output to result
        output[:, :, hi:hi_end, wi:wi_end] += output_tile * weights
        count[:, :, hi:hi_end, wi:wi_end] += weights

        # Update progress description
        pbar.set_description(f"Tile [{hi}:{hi_end}, {wi}:{wi_end}]")

    # Average overlapping regions
    output = output / count

    return output


def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    tile_size: int = 512,
    tile_stride: int = 256,
    weight_type: str = 'gaussian',
    progress: bool = True
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a tiled version of a function.

    Args:
        fn: Function to tile (takes tensor, returns tensor)
        tile_size: Tile size
        tile_stride: Tile stride
        weight_type: Weight type for blending
        progress: Show progress bar

    Returns:
        Tiled function
    """
    def tiled_function(x: torch.Tensor) -> torch.Tensor:
        return tiled_inference(
            model=lambda x: fn(x),
            x=x,
            tile_size=tile_size,
            tile_stride=tile_stride,
            device=x.device,
            weight_type=weight_type,
            progress=progress
        )

    return tiled_function