# SPDX-License-Identifier: GPL-3.0-or-later
#
# OpenIMC – Interactive analysis toolkit for IMC data
#
# Copyright (C) 2025 University of Southern California
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy import stats

# Optional sklearn for PCA
_HAVE_SKLEARN = False
try:
    from sklearn.decomposition import PCA  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def robust_percentile_scale(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    if not (0.0 <= low < high <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    a = arr.astype(np.float32, copy=False)
    lo = np.percentile(a, low)
    hi = np.percentile(a, high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def arcsinh_normalize(arr: np.ndarray, cofactor: float = 1.0) -> np.ndarray:
    """Apply arcsinh transformation (without normalizing to 0-1).
    
    Note: For segmentation, normalization to 0-1 is handled separately
    in the preprocessing pipeline after this transformation.
    """
    if not np.isfinite(cofactor) or cofactor <= 0:
        raise ValueError("cofactor must be a positive finite value")
    a = arr.astype(np.float32, copy=False)
    return np.arcsinh(a / cofactor)


def percentile_clip_normalize(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= p_low < p_high <= 100")
    a = arr.astype(np.float32, copy=False)
    vmin = np.percentile(a, p_low)
    vmax = np.percentile(a, p_high)
    clipped = np.clip(a, vmin, vmax)
    if vmax > vmin:
        normalized = (clipped - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(clipped)
    return normalized


def benjamini_hochberg_adjust(p_values) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values, preserving NaNs."""
    values = np.asarray(p_values, dtype=float).reshape(-1)
    adjusted = np.full(values.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return adjusted

    finite_values = np.clip(values[finite_mask], 0.0, 1.0)
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n_values = ranked.size

    ranked_adjusted = ranked * n_values / np.arange(1, n_values + 1, dtype=float)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)

    finite_adjusted = np.empty_like(ranked_adjusted)
    finite_adjusted[order] = ranked_adjusted
    adjusted[finite_mask] = finite_adjusted
    return adjusted


def benjamini_hochberg_adjust_matrix(p_values) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values with the original array shape."""
    values = np.asarray(p_values, dtype=float)
    return benjamini_hochberg_adjust(values.reshape(-1)).reshape(values.shape)


def combine_pvalues_fisher(p_values) -> float:
    """Combine finite p-values with Fisher's method."""
    values = np.asarray(p_values, dtype=float).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    if finite_values.size == 1:
        return float(np.clip(finite_values[0], 0.0, 1.0))

    clipped = np.clip(finite_values, np.finfo(float).tiny, 1.0)
    try:
        _, combined = stats.combine_pvalues(clipped, method='fisher')
    except Exception:
        return float(np.nanmean(clipped))
    return float(np.clip(combined, 0.0, 1.0))


def channelwise_minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize each channel to 0-1 range using min-max scaling.
    
    This function scales each channel image independently to [0, 1] using
    min-max scaling before combining channels. This is useful for segmenting
    when channels have different intensity ranges.
    
    Args:
        arr: Input image array
        
    Returns:
        Normalized array with values in [0, 1] range
    """
    a = arr.astype(np.float32, copy=False)
    vmin = np.min(a)
    vmax = np.max(a)
    if vmax > vmin:
        normalized = (a - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(a)
    return normalized


def combine_channels(images: List[np.ndarray], method: str, weights: List[float] = None) -> np.ndarray:
    if not images:
        raise ValueError("No images provided")

    if method == "single":
        return images[0]

    if method == "mean":
        return np.mean(np.stack(images, axis=0), axis=0)

    if method == "max":
        return np.max(np.stack(images, axis=0), axis=0)

    if method == "weighted":
        if weights is None or len(weights) != len(images):
            raise ValueError("Weights must be provided and match number of images")
        w = np.asarray(weights, dtype=np.float32)
        if not np.all(np.isfinite(w)) or np.any(w < 0) or float(np.sum(w)) <= 0:
            raise ValueError("Weights must be finite, non-negative, and have a positive sum")
        w = w / np.sum(w)
        stack = np.stack(images, axis=0)
        return np.tensordot(w, stack, axes=(0, 0))

    if method == "pca1":
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn required for PCA method")
        flattened = np.array([img.flatten() for img in images]).T
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(flattened)
        return pca_result.reshape(images[0].shape)

    raise ValueError(f"Unknown combination method: {method}")


class PreprocessingCache:
    """Cache for preprocessing statistics to ensure identical batch runs."""

    def __init__(self):
        self.cache: Dict[str, Dict] = {}

    def get_key(self, acq_id: str, channel: str, method: str, **params) -> str:
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        return f"{acq_id}_{channel}_{method}_{param_str}"

    def get_stats(self, acq_id: str, channel: str, method: str, **params) -> dict:
        key = self.get_key(acq_id, channel, method, **params)
        return self.cache.get(key, {})

    def set_stats(self, acq_id: str, channel: str, method: str, stats: dict, **params):
        key = self.get_key(acq_id, channel, method, **params)
        self.cache[key] = stats

    def clear(self):
        self.cache.clear()


# Color definitions for multi-channel blending
COLOR_DEFINITIONS = {
    'Blue': (0.0, 0.0, 1.0),
    'Teal': (0.0, 0.7, 0.7),  # Teal is safer than green for multiplexed images
    'Green': (0.0, 1.0, 0.0),
    'Yellow': (1.0, 1.0, 0.0),
    'Magenta': (1.0, 0.0, 1.0),
    'Red': (1.0, 0.0, 0.0),
    'White': (1.0, 1.0, 1.0),
}


def additive_blend_channels(
    channel_images: List[np.ndarray],
    channel_colors: List[str],
    alpha: float = 1.0,
    normalize_per_channel: bool = True
) -> np.ndarray:
    """Blend multiple channels using additive blending with color assignment.
    
    This implements napari-style additive blending where each channel is assigned
    a color and blended additively with a given alpha value.
    
    Args:
        channel_images: List of 2D channel images to blend
        channel_colors: List of color names for each channel (must match length of channel_images)
        alpha: Alpha value for blending (default 1.0, hidden from users)
        normalize_per_channel: If True, normalize each channel to [0, 1] before blending
        
    Returns:
        RGB image array of shape (H, W, 3) with values in [0, 1]
    """
    if not channel_images:
        raise ValueError("No channel images provided")
    
    if len(channel_images) != len(channel_colors):
        raise ValueError(f"Number of channel images ({len(channel_images)}) must match "
                        f"number of colors ({len(channel_colors)})")
    
    # Get image shape from first channel
    H, W = channel_images[0].shape
    
    # Initialize RGB output
    rgb_output = np.zeros((H, W, 3), dtype=np.float32)
    
    # Process each channel
    for img, color_name in zip(channel_images, channel_colors):
        if color_name not in COLOR_DEFINITIONS:
            raise ValueError(f"Unknown color: {color_name}. Must be one of {list(COLOR_DEFINITIONS.keys())}")
        
        # Get RGB color components
        color_rgb = COLOR_DEFINITIONS[color_name]
        
        # Normalize channel to [0, 1] if requested
        if normalize_per_channel:
            img_min = np.min(img)
            img_max = np.max(img)
            if img_max > img_min:
                img_normalized = (img.astype(np.float32) - img_min) / (img_max - img_min)
            else:
                img_normalized = np.zeros_like(img, dtype=np.float32)
        else:
            # Just ensure it's float32, but don't normalize
            img_normalized = img.astype(np.float32)
        
        # Additively blend: add channel intensity * color * alpha to each RGB component
        for c in range(3):
            rgb_output[..., c] += img_normalized * color_rgb[c] * alpha
    
    # Normalize final output to [0, 1] range to prevent overflow
    rgb_max = np.max(rgb_output)
    if rgb_max > 1.0:
        rgb_output = rgb_output / rgb_max
    
    # Clip to ensure values are in [0, 1]
    rgb_output = np.clip(rgb_output, 0.0, 1.0)
    
    return rgb_output


def stack_to_rgb(stack: np.ndarray) -> np.ndarray:
    H, W, C = stack.shape
    if C == 1:
        g = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        return np.dstack([g, g, g])
    elif C == 2:
        r = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        g = (stack[..., 1] - np.min(stack[..., 1])) / (np.max(stack[..., 1]) - np.min(stack[..., 1]) + 1e-8)
        return np.dstack([r, g, g])
    else:
        r = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        g = (stack[..., 1] - np.min(stack[..., 1])) / (np.max(stack[..., 1]) - np.min(stack[..., 1]) + 1e-8)
        b = (stack[..., 2] - np.min(stack[..., 2])) / (np.max(stack[..., 2]) - np.min(stack[..., 2]) + 1e-8)
        return np.dstack([r, g, b])
