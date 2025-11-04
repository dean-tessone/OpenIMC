"""
Watershed segmentation worker for OpenIMC.

Implements marker-controlled watershed segmentation with nucleus-seeded,
membrane-guided cell segmentation as an alternative to Cellpose.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

# Optional scikit-image imports
try:
    from skimage import morphology, filters, segmentation
    from skimage.filters import gaussian, median, sobel, scharr
    from skimage.morphology import disk, remove_small_objects, label
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.restoration import denoise_nl_means, estimate_sigma
    from skimage.filters import meijering, sato
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False

from openimc.ui.utils import combine_channels, arcsinh_normalize


def _apply_preprocessing_pipeline(
    img: np.ndarray,
    denoise_settings: Optional[Dict] = None,
    enable_tophat: bool = False,
    tophat_radius: int = 15,
    normalization_method: str = "arcsinh",
    arcsinh_cofactor: float = 5.0,
    percentile_params: Tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """
    Apply preprocessing pipeline: hot-pixel removal → light denoising → 
    optional top-hat → arcsinh/percentile scaling.
    
    Args:
        img: Input image
        denoise_settings: Denoising configuration
        enable_tophat: Whether to apply top-hat filtering
        tophat_radius: Radius for top-hat filtering
        normalization_method: "arcsinh" or "percentile_clip"
        arcsinh_cofactor: Cofactor for arcsinh normalization
        percentile_params: (low, high) percentiles for clipping
        
    Returns:
        Preprocessed image
    """
    if not _HAVE_SCIKIT_IMAGE:
        return img
    
    result = img.copy().astype(np.float32)
    
    # Apply denoising if settings provided
    if denoise_settings:
        # Hot pixel removal
        hot_config = denoise_settings.get("hot")
        if hot_config:
            method = hot_config.get("method", "median3")
            n_sd = float(hot_config.get("n_sd", 5.0))
            if method == "median3":
                result = median(result, disk(1))
            elif method == "n_sd_local_median":
                local_median = median(result, disk(1))
                diff = result - local_median
                local_var = ndi.uniform_filter(diff * diff, size=3)
                local_std = np.sqrt(np.maximum(local_var, 1e-8))
                mask_hot = diff > (n_sd * local_std)
                result = np.where(mask_hot, local_median, result)
        
        # Light denoising
        speckle_config = denoise_settings.get("speckle")
        if speckle_config:
            method = speckle_config.get("method", "gaussian")
            sigma = float(speckle_config.get("sigma", 0.8))
            if method == "gaussian":
                result = gaussian(result, sigma=sigma)
            elif method == "nl_means":
                est = estimate_sigma(result)
                result = denoise_nl_means(result, h=est * sigma)
    
    # Optional top-hat filtering
    if enable_tophat:
        selem = disk(tophat_radius)
        result = morphology.white_tophat(result, selem)
    
    # Normalization for display only
    if normalization_method == "arcsinh":
        result = arcsinh_normalize(result, cofactor=arcsinh_cofactor)
    elif normalization_method == "percentile_clip":
        p_low, p_high = percentile_params
        low_val = np.percentile(result, p_low)
        high_val = np.percentile(result, p_high)
        result = np.clip(result, low_val, high_val)
        result = (result - low_val) / (high_val - low_val + 1e-8)
    
    return result


def _generate_seed_map(
    nuclear_channels: List[str],
    img_stack: np.ndarray,
    channel_names: List[str],
    fusion_method: str = "mean",
    channel_weights: Optional[Dict[str, float]] = None,
    threshold_method: str = "otsu",
    min_seed_area: int = 10,
    min_distance_peaks: int = 5,
    denoise_settings: Optional[Dict] = None
) -> np.ndarray:
    """
    Generate seed map from nuclear channels.
    
    Pipeline: fuse nuclear channels → normalize → threshold → clean → 
    distance transform → local maxima → label as seeds.
    
    Args:
        nuclear_channels: List of nuclear channel names
        img_stack: Image stack (H, W, C)
        channel_names: List of all channel names
        fusion_method: "mean", "weighted", or "pca1"
        channel_weights: Weights for weighted fusion
        threshold_method: "otsu" or "percentile"
        min_seed_area: Minimum seed area in pixels
        min_distance_peaks: Minimum distance between peaks
        denoise_settings: Denoising configuration
        
    Returns:
        Seed map as labeled image
    """
    if not _HAVE_SCIKIT_IMAGE:
        raise ImportError("scikit-image is required for watershed segmentation")
    
    # Get nuclear channel indices
    nuclear_indices = [i for i, ch in enumerate(channel_names) if ch in nuclear_channels]
    if not nuclear_indices:
        raise ValueError("No nuclear channels found")
    
    # Extract nuclear channels
    nuclear_stack = img_stack[..., nuclear_indices]
    
    # Apply preprocessing to nuclear channels
    preprocessed_nuclear = np.zeros_like(nuclear_stack)
    for i, idx in enumerate(nuclear_indices):
        ch_name = channel_names[idx]
        ch_denoise = denoise_settings.get(ch_name) if denoise_settings else None
        preprocessed_nuclear[..., i] = _apply_preprocessing_pipeline(
            nuclear_stack[..., i], ch_denoise
        )
    
    # Fuse nuclear channels
    if fusion_method == "mean":
        fused_nuclear = np.mean(preprocessed_nuclear, axis=-1)
    elif fusion_method == "weighted":
        if channel_weights:
            weights = np.array([channel_weights.get(ch, 1.0) for ch in nuclear_channels])
            weights = weights / np.sum(weights)  # Normalize
            fused_nuclear = np.average(preprocessed_nuclear, axis=-1, weights=weights)
        else:
            fused_nuclear = np.mean(preprocessed_nuclear, axis=-1)
    elif fusion_method == "pca1":
        # Use first principal component
        reshaped = preprocessed_nuclear.reshape(-1, preprocessed_nuclear.shape[-1])
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(reshaped)
        fused_nuclear = pca_result.reshape(preprocessed_nuclear.shape[:2])
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Normalize fused image
    fused_nuclear = (fused_nuclear - np.min(fused_nuclear)) / (np.max(fused_nuclear) - np.min(fused_nuclear) + 1e-8)
    
    # Threshold
    if threshold_method == "otsu":
        threshold = filters.threshold_otsu(fused_nuclear)
        binary = fused_nuclear > threshold
    elif threshold_method == "percentile":
        # Use a lower percentile for small cells to be more sensitive
        threshold = np.percentile(fused_nuclear, 75)  # Lower percentile for small cells
        binary = fused_nuclear > threshold
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Clean binary image
    binary = morphology.binary_opening(binary, disk(1))
    binary = ndi.binary_fill_holes(binary)
    
    # Remove small objects
    binary = remove_small_objects(binary, min_size=min_seed_area)
    
    # Distance transform
    distance = distance_transform_edt(binary)
    
    # Find local maxima
    # Use a lower threshold for small cells to detect more peaks
    threshold_abs = max(np.max(distance) * 0.05, 1.0)  # Lower threshold, minimum 1.0
    local_maxima = peak_local_max(
        distance, 
        min_distance=min_distance_peaks,
        threshold_abs=threshold_abs
    )
    
    # Create seed markers
    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (y, x) in enumerate(local_maxima):
        markers[y, x] = i + 1
    
    # Label connected components
    markers = label(markers)
    
    return markers


def _generate_boundary_map(
    membrane_channels: List[str],
    img_stack: np.ndarray,
    channel_names: List[str],
    fusion_method: str = "mean",
    channel_weights: Optional[Dict[str, float]] = None,
    boundary_method: str = "sobel",
    boundary_sigma: float = 1.0,
    denoise_settings: Optional[Dict] = None
) -> np.ndarray:
    """
    Generate boundary map from membrane/edge channels.
    
    Pipeline: fuse membrane channels → normalize → smooth → 
    compute gradient magnitude or use membrane channels directly.
    
    Args:
        membrane_channels: List of membrane channel names
        img_stack: Image stack (H, W, C)
        channel_names: List of all channel names
        fusion_method: "mean", "weighted", or "pca1"
        channel_weights: Weights for weighted fusion
        boundary_method: "sobel", "scharr", or "membrane_channels"
        boundary_sigma: Smoothing sigma for boundary detection
        denoise_settings: Denoising configuration
        
    Returns:
        Boundary map (higher values = stronger boundaries)
    """
    if not _HAVE_SCIKIT_IMAGE:
        raise ImportError("scikit-image is required for watershed segmentation")
    
    # Get membrane channel indices
    membrane_indices = [i for i, ch in enumerate(channel_names) if ch in membrane_channels]
    if not membrane_indices:
        raise ValueError("No membrane channels found")
    
    # Extract membrane channels
    membrane_stack = img_stack[..., membrane_indices]
    
    # Apply preprocessing to membrane channels
    preprocessed_membrane = np.zeros_like(membrane_stack)
    for i, idx in enumerate(membrane_indices):
        ch_name = channel_names[idx]
        ch_denoise = denoise_settings.get(ch_name) if denoise_settings else None
        preprocessed_membrane[..., i] = _apply_preprocessing_pipeline(
            membrane_stack[..., i], ch_denoise
        )
    
    # Fuse membrane channels
    if fusion_method == "mean":
        fused_membrane = np.mean(preprocessed_membrane, axis=-1)
    elif fusion_method == "weighted":
        if channel_weights:
            weights = np.array([channel_weights.get(ch, 1.0) for ch in membrane_channels])
            weights = weights / np.sum(weights)  # Normalize
            fused_membrane = np.average(preprocessed_membrane, axis=-1, weights=weights)
        else:
            fused_membrane = np.mean(preprocessed_membrane, axis=-1)
    elif fusion_method == "pca1":
        # Use first principal component
        reshaped = preprocessed_membrane.reshape(-1, preprocessed_membrane.shape[-1])
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(reshaped)
        fused_membrane = pca_result.reshape(preprocessed_membrane.shape[:2])
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    # Normalize fused image
    fused_membrane = (fused_membrane - np.min(fused_membrane)) / (np.max(fused_membrane) - np.min(fused_membrane) + 1e-8)
    
    # Smooth
    fused_membrane = gaussian(fused_membrane, sigma=boundary_sigma)
    
    # Generate boundary map
    if boundary_method == "sobel":
        boundary_map = sobel(fused_membrane)
    elif boundary_method == "scharr":
        boundary_map = scharr(fused_membrane)
    elif boundary_method == "membrane_channels":
        # Use membrane channels directly (inverted probability)
        boundary_map = 1.0 - fused_membrane
    else:
        raise ValueError(f"Unknown boundary method: {boundary_method}")
    
    # Normalize boundary map
    boundary_map = (boundary_map - np.min(boundary_map)) / (np.max(boundary_map) - np.min(boundary_map) + 1e-8)
    
    return boundary_map


def _postprocess_watershed(
    labels: np.ndarray,
    min_cell_area: int = 100,
    max_cell_area: int = 10000,
    boundary_smoothing: bool = True
) -> np.ndarray:
    """
    Postprocess watershed segmentation results.
    
    Args:
        labels: Watershed segmentation labels
        min_cell_area: Minimum cell area in pixels
        max_cell_area: Maximum cell area in pixels
        boundary_smoothing: Whether to apply boundary smoothing
        
    Returns:
        Postprocessed labels
    """
    if not _HAVE_SCIKIT_IMAGE:
        return labels
    
    result = labels.copy()
    
    # Remove objects that are too small or too large
    props = regionprops(result)
    for prop in props:
        if prop.area < min_cell_area or prop.area > max_cell_area:
            result[result == prop.label] = 0
    
    # Relabel contiguous regions
    result = label(result > 0)
    
    # Optional boundary smoothing
    if boundary_smoothing:
        # Apply small morphological opening to smooth boundaries
        result = morphology.binary_opening(result > 0, disk(1))
        result = label(result)
    
    return result


def watershed_segmentation(
    img_stack: np.ndarray,
    channel_names: List[str],
    nuclear_channels: List[str],
    membrane_channels: List[str],
    # Preprocessing parameters
    denoise_settings: Optional[Dict] = None,
    enable_tophat: bool = False,
    tophat_radius: int = 15,
    normalization_method: str = "arcsinh",
    arcsinh_cofactor: float = 5.0,
    percentile_params: Tuple[float, float] = (1.0, 99.0),
    # Nuclear fusion parameters
    nuclear_fusion_method: str = "mean",
    nuclear_weights: Optional[Dict[str, float]] = None,
    # Seed generation parameters
    seed_threshold_method: str = "otsu",
    min_seed_area: int = 10,
    min_distance_peaks: int = 5,
    # Membrane fusion parameters
    membrane_fusion_method: str = "mean",
    membrane_weights: Optional[Dict[str, float]] = None,
    # Boundary detection parameters
    boundary_method: str = "sobel",
    boundary_sigma: float = 1.0,
    # Watershed parameters
    compactness: float = 0.01,
    min_cell_area: int = 100,
    max_cell_area: int = 10000,
    # Tiling parameters
    tile_size: int = 512,
    tile_overlap: int = 64,
    rng_seed: int = 42
) -> np.ndarray:
    """
    Perform marker-controlled watershed segmentation.
    
    Args:
        img_stack: Image stack (H, W, C)
        channel_names: List of channel names
        nuclear_channels: List of nuclear channel names
        membrane_channels: List of membrane channel names
        denoise_settings: Denoising configuration per channel
        enable_tophat: Whether to apply top-hat filtering
        tophat_radius: Radius for top-hat filtering
        normalization_method: Normalization method
        arcsinh_cofactor: Cofactor for arcsinh normalization
        percentile_params: Percentile parameters for clipping
        nuclear_fusion_method: Method for fusing nuclear channels
        nuclear_weights: Weights for nuclear channel fusion
        seed_threshold_method: Method for seed thresholding
        min_seed_area: Minimum seed area in pixels
        min_distance_peaks: Minimum distance between peaks
        membrane_fusion_method: Method for fusing membrane channels
        membrane_weights: Weights for membrane channel fusion
        boundary_method: Method for boundary detection
        boundary_sigma: Smoothing sigma for boundary detection
        compactness: Watershed compactness parameter
        min_cell_area: Minimum cell area in pixels
        max_cell_area: Maximum cell area in pixels
        tile_size: Tile size for processing large images
        tile_overlap: Overlap between tiles
        rng_seed: Random seed for deterministic results
        
    Returns:
        Segmentation labels
    """
    if not _HAVE_SCIKIT_IMAGE:
        raise ImportError("scikit-image is required for watershed segmentation")
    
    # Set random seed for deterministic results
    np.random.seed(rng_seed)
    
    height, width = img_stack.shape[:2]
    
    # Check if tiling is needed
    if height <= tile_size and width <= tile_size:
        # Process entire image at once
        return _watershed_single_tile(
            img_stack, channel_names, nuclear_channels, membrane_channels,
            denoise_settings, enable_tophat, tophat_radius,
            normalization_method, arcsinh_cofactor, percentile_params,
            nuclear_fusion_method, nuclear_weights,
            seed_threshold_method, min_seed_area, min_distance_peaks,
            membrane_fusion_method, membrane_weights,
            boundary_method, boundary_sigma,
            compactness, min_cell_area, max_cell_area
        )
    else:
        # Process in tiles
        return _watershed_tiled(
            img_stack, channel_names, nuclear_channels, membrane_channels,
            denoise_settings, enable_tophat, tophat_radius,
            normalization_method, arcsinh_cofactor, percentile_params,
            nuclear_fusion_method, nuclear_weights,
            seed_threshold_method, min_seed_area, min_distance_peaks,
            membrane_fusion_method, membrane_weights,
            boundary_method, boundary_sigma,
            compactness, min_cell_area, max_cell_area,
            tile_size, tile_overlap, rng_seed
        )


def _watershed_single_tile(
    img_stack: np.ndarray,
    channel_names: List[str],
    nuclear_channels: List[str],
    membrane_channels: List[str],
    denoise_settings: Optional[Dict],
    enable_tophat: bool,
    tophat_radius: int,
    normalization_method: str,
    arcsinh_cofactor: float,
    percentile_params: Tuple[float, float],
    nuclear_fusion_method: str,
    nuclear_weights: Optional[Dict[str, float]],
    seed_threshold_method: str,
    min_seed_area: int,
    min_distance_peaks: int,
    membrane_fusion_method: str,
    membrane_weights: Optional[Dict[str, float]],
    boundary_method: str,
    boundary_sigma: float,
    compactness: float,
    min_cell_area: int,
    max_cell_area: int
) -> np.ndarray:
    """Process a single tile with watershed segmentation."""
    
    # Generate seed map
    seeds = _generate_seed_map(
        nuclear_channels, img_stack, channel_names,
        nuclear_fusion_method, nuclear_weights,
        seed_threshold_method, min_seed_area, min_distance_peaks,
        denoise_settings
    )
    
    # Generate boundary map
    boundaries = _generate_boundary_map(
        membrane_channels, img_stack, channel_names,
        membrane_fusion_method, membrane_weights,
        boundary_method, boundary_sigma,
        denoise_settings
    )
    
    # Run watershed
    labels = segmentation.watershed(
        boundaries, seeds, compactness=compactness, connectivity=1
    )
    
    # Postprocess
    labels = _postprocess_watershed(labels, min_cell_area, max_cell_area)
    
    return labels


def _watershed_tiled(
    img_stack: np.ndarray,
    channel_names: List[str],
    nuclear_channels: List[str],
    membrane_channels: List[str],
    denoise_settings: Optional[Dict],
    enable_tophat: bool,
    tophat_radius: int,
    normalization_method: str,
    arcsinh_cofactor: float,
    percentile_params: Tuple[float, float],
    nuclear_fusion_method: str,
    nuclear_weights: Optional[Dict[str, float]],
    seed_threshold_method: str,
    min_seed_area: int,
    min_distance_peaks: int,
    membrane_fusion_method: str,
    membrane_weights: Optional[Dict[str, float]],
    boundary_method: str,
    boundary_sigma: float,
    compactness: float,
    min_cell_area: int,
    max_cell_area: int,
    tile_size: int,
    tile_overlap: int,
    rng_seed: int
) -> np.ndarray:
    """Process image in tiles and stitch results."""
    
    height, width = img_stack.shape[:2]
    result_labels = np.zeros((height, width), dtype=np.int32)
    
    # Calculate tile positions
    step_size = tile_size - tile_overlap
    tile_positions = []
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile_positions.append((y, x, y_end, x_end))
    
    next_label = 1
    
    for i, (y_start, x_start, y_end, x_end) in enumerate(tile_positions):
        # Extract tile
        tile = img_stack[y_start:y_end, x_start:x_end]
        
        # Process tile
        tile_labels = _watershed_single_tile(
            tile, channel_names, nuclear_channels, membrane_channels,
            denoise_settings, enable_tophat, tophat_radius,
            normalization_method, arcsinh_cofactor, percentile_params,
            nuclear_fusion_method, nuclear_weights,
            seed_threshold_method, min_seed_area, min_distance_peaks,
            membrane_fusion_method, membrane_weights,
            boundary_method, boundary_sigma,
            compactness, min_cell_area, max_cell_area
        )
        
        # Relabel to ensure unique IDs across tiles
        tile_labels[tile_labels > 0] += next_label - 1
        next_label = np.max(tile_labels) + 1
        
        # Place tile in result
        result_labels[y_start:y_end, x_start:x_end] = tile_labels
    
    # Handle overlapping regions by keeping the label with the highest confidence
    # (simplified approach - could be improved with more sophisticated stitching)
    
    return result_labels
