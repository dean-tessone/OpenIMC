"""
Worker functions for OME-TIFF export with multiprocessing support.
"""

from typing import Dict, Tuple
import numpy as np

# Optional scikit-image for denoising
try:
    from skimage import morphology
    from skimage.filters import gaussian, median
    from skimage.morphology import disk, footprint_rectangle
    from skimage.restoration import denoise_nl_means, estimate_sigma
    from scipy import ndimage as ndi
    try:
        from skimage.restoration import rolling_ball as _sk_rolling_ball  # type: ignore
        _HAVE_ROLLING_BALL = True
    except Exception:
        _HAVE_ROLLING_BALL = False
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False
    _HAVE_ROLLING_BALL = False

from openimc.ui.utils import arcsinh_normalize, percentile_clip_normalize


def _apply_custom_denoise_to_channel(channel_img: np.ndarray, channel_name: str, 
                                     custom_denoise_settings: Dict) -> np.ndarray:
    """Apply custom denoise steps for a channel in raw domain.
    
    This is a module-level function that can be pickled for multiprocessing.
    """
    if not _HAVE_SCIKIT_IMAGE:
        return channel_img
    
    cfg = custom_denoise_settings.get(channel_name)
    if not cfg:
        return channel_img
    
    out = channel_img.astype(np.float32, copy=False)

    # Apply denoising steps in order: hot pixel -> speckle -> background
    # Hot pixel removal
    hot = cfg.get("hot")
    if hot:
        method = hot.get("method")
        if method == "median3":
            try:
                out = median(out, footprint=footprint_rectangle(3, 3).astype(bool))
            except Exception:
                out = ndi.median_filter(out, size=3)
        elif method == "n_sd_local_median":
            n_sd = float(hot.get("n_sd", 5.0))
            try:
                local_median = median(out, footprint=footprint_rectangle(3, 3).astype(bool))
            except Exception:
                local_median = ndi.median_filter(out, size=3)
            diff = out - local_median
            local_var = ndi.uniform_filter(diff * diff, size=3)
            local_std = np.sqrt(np.maximum(local_var, 1e-8))
            mask_hot = diff > (n_sd * local_std)
            out = np.where(mask_hot, local_median, out)
    
    # Speckle smoothing
    speckle = cfg.get("speckle")
    if speckle:
        method = speckle.get("method")
        if method == "gaussian":
            sigma = float(speckle.get("sigma", 0.8))
            out = gaussian(out, sigma=sigma, preserve_range=True)
        elif method == "nl_means":
            mn, mx = float(np.min(out)), float(np.max(out))
            scale = mx - mn
            scaled = (out - mn) / scale if scale > 0 else out
            sigma_est = np.mean(estimate_sigma(scaled, channel_axis=None))
            out = denoise_nl_means(
                scaled,
                h=1.15 * sigma_est,
                fast_mode=True,
                patch_size=5,
                patch_distance=6,
                channel_axis=None,
            )
            out = out * scale + mn
    
    # Background subtraction
    bg = cfg.get("background")
    if bg:
        method = bg.get("method")
        radius = int(bg.get("radius", 15))
        if method == "white_tophat":
            se = disk(radius)
            try:
                out = morphology.white_tophat(out, selem=se)
            except TypeError:
                out = morphology.white_tophat(out, footprint=se)
        elif method == "black_tophat":
            se = disk(radius)
            try:
                out = morphology.black_tophat(out, selem=se)
            except TypeError:
                out = morphology.black_tophat(out, footprint=se)
        elif method == "rolling_ball":
            if _HAVE_ROLLING_BALL:
                background = _sk_rolling_ball(out, radius=radius)
                out = out - background
                out = np.clip(out, 0, None)
            else:
                se = disk(radius)
                try:
                    opened = morphology.opening(out, selem=se)
                except TypeError:
                    opened = morphology.opening(out, footprint=se)
                out = out - opened
                out = np.clip(out, 0, None)
    
    # Rescale to preserve original max intensity of this channel
    try:
        orig_max = float(np.max(channel_img))
        new_max = float(np.max(out))
        if new_max > 0 and orig_max > 0:
            out = out * (orig_max / new_max)
    except Exception:
        pass
    
    # Clip to dtype range if integer
    if np.issubdtype(channel_img.dtype, np.integer):
        info = np.iinfo(channel_img.dtype)
        out = np.clip(out, info.min, info.max)
    else:
        out = np.clip(out, 0, None)
    return out.astype(channel_img.dtype, copy=False)


def process_channel_for_export(
    channel_img: np.ndarray,
    channel_name: str,
    denoise_source: str,
    custom_denoise_settings: Dict,
    normalization_method: str,
    arcsinh_cofactor: float,
    percentile_params: Tuple[float, float],
    viewer_denoise_func=None
) -> np.ndarray:
    """Process a single channel for export with denoising and normalization.
    
    This is a module-level function that can be pickled for multiprocessing.
    
    Args:
        channel_img: Raw channel image
        channel_name: Name of the channel
        denoise_source: "none", "viewer", or "custom"
        custom_denoise_settings: Dictionary of custom denoising settings per channel
        normalization_method: "None", "arcsinh", or "percentile_clip"
        arcsinh_cofactor: Cofactor for arcsinh normalization
        percentile_params: (low, high) percentiles for percentile clipping
        viewer_denoise_func: Function to apply viewer denoising (must be None for multiprocessing)
    
    Returns:
        Processed channel image
    """
    result = channel_img.copy()
    
    # Apply denoising
    if denoise_source == "viewer":
        # Note: viewer_denoise_func cannot be pickled, so this should be handled
        # in the main process before calling this function
        if viewer_denoise_func is not None:
            result = viewer_denoise_func(channel_name, result)
    elif denoise_source == "custom":
        result = _apply_custom_denoise_to_channel(result, channel_name, custom_denoise_settings)
    
    # Apply normalization
    if normalization_method == "arcsinh":
        result = arcsinh_normalize(result, cofactor=arcsinh_cofactor)
    elif normalization_method == "percentile_clip":
        p_low, p_high = percentile_params
        result = percentile_clip_normalize(result, p_low=p_low, p_high=p_high)
    
    return result

