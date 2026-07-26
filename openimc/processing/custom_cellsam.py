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

"""
Custom CellSAM implementation with memory leak fixes.

This module provides a memory-efficient implementation of cellsam_pipeline
that properly caches the model and prevents memory leaks.
"""

import functools
import gc
import operator
import os
import sys
import time
import warnings
from typing import Optional


class CUDAMemoryError(Exception):
    """Custom exception for CUDA out-of-memory errors during segmentation.
    
    This exception is raised when CUDA runs out of memory during segmentation.
    It includes a helpful message recommending batch size reduction.
    """
    def __init__(self, message: str, batch_size: int = None):
        super().__init__(message)
        self.batch_size = batch_size
        self.recommended_batch_size = max(1, (batch_size // 2) if batch_size else 1)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.batch_size is not None:
            return (
                f"{base_msg}\n\n"
                f"CUDA out of memory error occurred during segmentation.\n"
                f"Current batch size: {self.batch_size}\n"
                f"Recommendation: Reduce the batch size to {self.recommended_batch_size} or lower.\n"
                f"You can adjust the batch size in the segmentation dialog."
            )
        return (
            f"{base_msg}\n\n"
            f"CUDA out of memory error occurred during segmentation.\n"
            f"Recommendation: Reduce the batch size in the segmentation dialog."
        )

# CRITICAL: Configure dask BEFORE any dask imports
# This ensures compatibility with squidpy/spatialdata
os.environ.setdefault('DASK_DATAFRAME__QUERY_PLANNING', 'True')
try:
    import dask
    dask.config.set({'dataframe.query-planning': True})
except (ImportError, AttributeError):
    pass

import numpy as np
import torch
from tqdm import tqdm

# Try to import psutil for better memory tracking
try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

# Memory debugging removed - no longer used

try:
    import dask.array as da
    _HAVE_DASK = True
except ImportError:
    _HAVE_DASK = False

# Import necessary functions from cellSAM
try:
    from cellSAM import get_model, get_local_model, segment_cellular_image
    from cellSAM.utils import (
        format_image_shape,
        fill_holes_and_remove_small_masks,
        subtract_boundaries,
    )
    _HAVE_CELLSAM = True
except (ImportError, OSError):
    _HAVE_CELLSAM = False
    get_model = None
    get_local_model = None
    segment_cellular_image = None
    format_image_shape = None
    fill_holes_and_remove_small_masks = None
    subtract_boundaries = None

# Import dask_image and skimage helpers
try:
    import dask
    from dask_image.ndmeasure import label
    from dask_image.ndmeasure._utils import _label
    from skimage.exposure import equalize_adapthist, adjust_gamma
    from sklearn.metrics import confusion_matrix as sk_metrics_confusion_matrix
    _HAVE_IMAGE_HELPERS = True
except (ImportError, OSError):
    _HAVE_IMAGE_HELPERS = False
    dask = None
    label = None
    _label = None
    equalize_adapthist = None
    adjust_gamma = None
    sk_metrics_confusion_matrix = None


# Global model cache - ensures only one model instance exists
_CACHED_MODEL = None
_CACHED_MODEL_DEVICE = None
_CACHED_MODEL_BBOX_THRESHOLD = None


def get_median_size(labels: np.ndarray):
    """Get median cell size from segmentation labels.
    
    Args:
        labels: Segmentation mask array
        
    Returns:
        Tuple of (median_size, sizes, sizes_abs)
    """
    sizes = []
    sizes_abs = []
    for mask in np.unique(labels):
        if mask == 0:
            continue
        area = (labels == mask).sum().item()
        # normalizing by area
        sizes.append(area / (labels.shape[0] * labels.shape[1]))
        sizes_abs.append(area)
    sizes = np.array(sizes)
    sizes_abs = np.array(sizes_abs)
    # median size
    median_size = np.median(sizes)
    return median_size, sizes, sizes_abs


def segment_chunk(chunk, model=None, **kwargs):
    """Segments an individual chunk using a specific GPU.
    
    Args:
        chunk: Image data to segment.
        model: Model instance to use for segmentation.
        **kwargs: Additional arguments passed to segment_cellular_image.
        
    Returns:
        Tuple of (mask, max_label) where mask is int32 and max_label is the maximum label value.
    """
    import logging
    if not _HAVE_CELLSAM or segment_cellular_image is None:
        raise ImportError("CellSAM not installed. segment_cellular_image not available.")
    
    try:
        mask = segment_cellular_image(chunk, model, **kwargs)[0]
    except Exception as e:
        logging.error(f"Error segmenting chunk: {e}")
        mask = np.zeros(chunk.shape[:-1], dtype=np.int64)
    
    return (mask.astype(np.int32), mask.max())


def _postprocess_predictions(all_masks: np.ndarray):
    """Postprocess segmentation predictions using morphological operations.
    
    This is a reimplementation of the postprocess_predictions function from CellSAM.
    """
    from skimage.morphology import (
        disk,
        binary_opening,
        binary_closing,
        binary_erosion,
        binary_dilation,
    )
    from scipy.ndimage import gaussian_filter
    from segment_anything.utils.amg import remove_small_regions
    
    mask_values = np.unique(all_masks)
    new_masks = []
    selem = disk(2)
    for mask_value in mask_values[1:]:  # Skip background (0)
        mask = all_masks == mask_value
        mask, _ = remove_small_regions(mask, 20, mode="holes")
        mask, _ = remove_small_regions(mask, 20, mode="islands")
        opened_mask = binary_opening(mask, selem)
        closed_mask = binary_closing(opened_mask, selem)
        mask = closed_mask
        selem = disk(10)
        mask = binary_dilation(mask, selem)
        mask = binary_erosion(mask, selem)
        mask = gaussian_filter(mask.astype(np.float32), sigma=3)
        mask = mask > 0.5
        mask = mask.astype(np.uint8) * mask_value
        new_masks.append(mask)
    
    if len(new_masks) == 0:
        return np.zeros_like(all_masks, dtype=np.uint8)
    
    return np.max(new_masks, axis=0)


def is_low_contrast_clahe(image, lower_threshold=0.04, upper_threshold=0.05, kernel_size=256):
    """Check if image is low contrast using CLAHE.
    
    Args:
        image: Input image array
        lower_threshold: Lower threshold for contrast classification
        upper_threshold: Upper threshold for contrast classification
        kernel_size: Kernel size for adaptive histogram equalization
        
    Returns:
        List of [is_low_contrast, mean_diff, mean_std]
    """
    if not _HAVE_IMAGE_HELPERS or equalize_adapthist is None:
        raise ImportError("skimage.exposure.equalize_adapthist not available")
    
    cp = equalize_adapthist(image, kernel_size=kernel_size)
    diff = np.abs(image - cp)
    diff = diff[diff > 0]
    mean_diff = np.median(diff)
    mean_std = np.std(diff)
    islowcontrast = lower_threshold < mean_diff < upper_threshold
    return [islowcontrast, mean_diff, mean_std]


def get_slices_and_axes(chunks, shape, depth):
    """Get slices and axes for processing block boundaries.
    
    Args:
        chunks: Chunk structure from dask array
        shape: Shape of the array
        depth: Depth for overlap
        
    Returns:
        List of (slice, axis) tuples
    """
    ndim = len(shape)
    depth = da.overlap.coerce_depth(ndim, depth)
    slices = da.core.slices_from_chunks(chunks)
    slices_and_axes = []
    for ax in range(ndim):
        for sl in slices:
            if sl[ax].stop == shape[ax]:
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(
                sl[ax].stop - 2 * depth[ax], sl[ax].stop + 2 * depth[ax]
            )
            slices_and_axes.append((tuple(slice_to_append), ax))
    return slices_and_axes


def _across_block_label_iou(face, axis, iou_threshold):
    """Compute IOU between labels across block boundaries.
    
    Args:
        face: Face of the block boundary
        axis: Axis along which to split
        iou_threshold: IOU threshold for grouping
        
    Returns:
        Array of grouped labels (2, N) where N is number of valid mappings
    """
    if not _HAVE_IMAGE_HELPERS or sk_metrics_confusion_matrix is None:
        raise ImportError("sklearn.metrics.confusion_matrix not available")
    
    unique = np.unique(face)
    face0, face1 = np.split(face, 2, axis)
    
    intersection = sk_metrics_confusion_matrix(face0.reshape(-1), face1.reshape(-1))
    sum0 = intersection.sum(axis=0, keepdims=True)
    sum1 = intersection.sum(axis=1, keepdims=True)
    
    # Note that sum0 and sum1 broadcast to square matrix size.
    union = sum0 + sum1 - intersection
    
    # Ignore errors with divide by zero, which the np.where sets to zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(intersection > 0, intersection / union, 0)
    
    labels0, labels1 = np.nonzero(iou >= iou_threshold)
    
    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    grouped = np.stack([labels0_orig, labels1_orig])
    
    valid = np.all(grouped != 0, axis=0)  # Discard any mappings with bg pixels
    return grouped[:, valid]


def _across_block_iou_delayed(face, axis, iou_threshold):
    """Delayed version of _across_block_label_iou.
    
    Args:
        face: Face of the block boundary
        axis: Axis along which to split
        iou_threshold: IOU threshold for grouping
        
    Returns:
        Dask array of grouped labels
    """
    if not _HAVE_IMAGE_HELPERS or dask is None:
        raise ImportError("dask not available")
    
    _across_block_label_grouping_ = dask.delayed(_across_block_label_iou)
    grouped = _across_block_label_grouping_(face, axis, iou_threshold)
    return da.from_delayed(grouped, shape=(2, np.nan), dtype=np.int32)


def label_adjacency_graph(labels, nlabels, depth, iou_threshold):
    """Build a label adjacency graph for linking labels across blocks.
    
    Args:
        labels: Dask array of labeled blocks
        nlabels: Total number of labels
        depth: Depth for label adjacency
        iou_threshold: IOU threshold for linking
        
    Returns:
        CSR matrix representing label adjacency graph
    """
    if not _HAVE_IMAGE_HELPERS or _label is None:
        raise ImportError("dask_image.ndmeasure._utils._label not available")
    
    all_mappings = [da.empty((2, 0), dtype=np.int32, chunks=1)]
    
    slices_and_axes = get_slices_and_axes(labels.chunks, labels.shape, depth)
    for face_slice, axis in tqdm(slices_and_axes):
        face = labels[face_slice]
        mapped = _across_block_iou_delayed(face.compute(), axis, iou_threshold)
        # TODO: double check this
        if (isinstance(mapped, np.ndarray) and mapped.size == 0):
            continue
        all_mappings.append(mapped)  # len is > 0
    
    i, j = da.concatenate(all_mappings, axis=1)
    result = _label._to_csr_matrix(i, j, nlabels + 1)
    return result


def link_labels(block_labeled, total, depth, iou_threshold=1):
    """Build a label connectivity graph that groups labels across blocks,
    use this graph to find connected components, and then relabel each
    block according to those.
    
    Args:
        block_labeled: Dask array of labeled blocks
        total: Total number of labels
        depth: Depth for label adjacency
        iou_threshold: IOU threshold for linking
        
    Returns:
        Relabeled dask array
    """
    if not _HAVE_IMAGE_HELPERS or label is None or _label is None:
        raise ImportError("dask_image.ndmeasure.label or dask_image.ndmeasure._utils._label not available")
    
    label_groups = label_adjacency_graph(block_labeled, total, depth, iou_threshold)
    new_labeling = _label.connected_components_delayed(label_groups)
    return _label.relabel_blocks(block_labeled, new_labeling)


def enhance_low_contrast(
    img,
    model=None,
    lower_contrast_threshold=0.05,
    upper_contrast_threshold=0.1,
    max_green_channel_value=0,
    clip_limit_default=0.01,
    kernel_size_default=256,
    gamma_default=2,
    clip_limit_high_diff=0.02,
    kernel_size_high_diff=384,
    gamma_high_diff=1.2,
    bbox_threshold_high_diff=0.15,
    clip_limit_very_high_diff=0.05,
    bbox_threshold_very_high_diff=0.15,
    clip_limit_adjusted=0.01,
    std_range=(0.035, 0.04),
    mean_diff_threshold=0.065,
    mean_std_threshold=0.05
):
    """Enhance low contrast images using CLAHE and gamma adjustment.
    
    Args:
        img: Input image.
        model: Model instance (may modify model.bbox_threshold).
        lower_contrast_threshold: Lower threshold for contrast classification.
        upper_contrast_threshold: Upper threshold for contrast classification.
        max_green_channel_value: Maximum allowed value in the green channel.
        clip_limit_default: Default CLAHE clip limit.
        kernel_size_default: Default CLAHE kernel size.
        gamma_default: Default gamma adjustment value.
        clip_limit_high_diff: CLAHE clip limit for high mean_diff.
        kernel_size_high_diff: CLAHE kernel size for high mean_diff.
        gamma_high_diff: Gamma adjustment value for high mean_diff.
        bbox_threshold_high_diff: Bbox threshold for high mean_diff.
        clip_limit_very_high_diff: CLAHE clip limit for very high mean_diff.
        bbox_threshold_very_high_diff: Bbox threshold for very high mean_diff.
        clip_limit_adjusted: Adjusted CLAHE clip limit for specific std range.
        std_range: Range of mean_std for specific adjustments.
        mean_diff_threshold: Threshold for very high mean_diff.
        mean_std_threshold: Threshold for mean_std classification.
        
    Returns:
        Enhanced image.
    """
    if not _HAVE_IMAGE_HELPERS or equalize_adapthist is None or adjust_gamma is None:
        # If helpers not available, return image unchanged
        return img
    
    result = is_low_contrast_clahe(
        img,
        lower_threshold=lower_contrast_threshold,
        upper_threshold=upper_contrast_threshold
    )
    
    low_contrast, mean_diff, mean_std = result
    
    low_contrast = (low_contrast and img[..., 1].max() == max_green_channel_value) \
        if mean_diff < mean_std_threshold else low_contrast
    
    if low_contrast:
        clip_limit = clip_limit_default
        kernel_size = kernel_size_default
        gamma = gamma_default
        
        if mean_diff > lower_contrast_threshold and mean_std < mean_std_threshold:
            clip_limit = clip_limit_high_diff
            kernel_size = kernel_size_high_diff
            gamma = gamma_high_diff
            if model is not None:
                model.bbox_threshold = bbox_threshold_high_diff
        
        if mean_diff > mean_diff_threshold and mean_std < mean_std_threshold:
            clip_limit = clip_limit_very_high_diff
            if model is not None:
                model.bbox_threshold = bbox_threshold_very_high_diff
        
        if mean_diff > mean_diff_threshold and (std_range[0] < mean_std < std_range[1]):
            clip_limit = clip_limit_adjusted
        
        img = equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit)
        img = adjust_gamma(img, gamma=gamma)
    
    return img


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range channelwise.
    
    Args:
        img: Image array with shape (W, H) or (W, H, C)
        
    Returns:
        Normalized image array
    """
    img = img.astype(np.float32)
    
    # Handle 2D images
    if img.ndim == 2:
        img = img[..., None]
    
    # Normalize each channel
    for i in range(img.shape[-1]):
        channel = img[..., i]
        min_val = np.min(channel)
        max_val = np.max(channel)
        if (max_val - min_val) != 0:
            img[..., i] = (channel - min_val) / (max_val - min_val)
        else:
            img[..., i] = channel
    
    return img


def _get_cached_model(model_path: Optional[str] = None, bbox_threshold: float = 0.4):
    """Get or create a cached model instance.
    
    This ensures only one model instance exists in memory, preventing memory leaks.
    
    Args:
        model_path: Optional path to model weights. If None, uses default model.
        bbox_threshold: Bbox threshold for the model.
        
    Returns:
        Cached model instance
    """
    global _CACHED_MODEL, _CACHED_MODEL_DEVICE, _CACHED_MODEL_BBOX_THRESHOLD
    
    if not _HAVE_CELLSAM or get_model is None:
        raise ImportError("CellSAM not installed. Install with: pip install git+https://github.com/vanvalenlab/cellSAM.git")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if we need to create or update the cached model
    if _CACHED_MODEL is None:
        # Create new model
        if model_path is not None:
            if get_local_model is None:
                raise ImportError("get_local_model not available from CellSAM")
            _CACHED_MODEL = get_local_model(model_path)
            _CACHED_MODEL.bbox_threshold = bbox_threshold
        else:
            _CACHED_MODEL = get_model()
            _CACHED_MODEL.bbox_threshold = bbox_threshold
        
        _CACHED_MODEL = _CACHED_MODEL.to(device)
        _CACHED_MODEL.eval()
        _CACHED_MODEL_DEVICE = device
        _CACHED_MODEL_BBOX_THRESHOLD = bbox_threshold
    else:
        # Reuse existing model, but update bbox_threshold if needed
        if _CACHED_MODEL_BBOX_THRESHOLD != bbox_threshold:
            _CACHED_MODEL.bbox_threshold = bbox_threshold
            _CACHED_MODEL_BBOX_THRESHOLD = bbox_threshold
        
        # Ensure model is on the correct device
        if _CACHED_MODEL_DEVICE != device:
            _CACHED_MODEL = _CACHED_MODEL.to(device)
            _CACHED_MODEL_DEVICE = device
    
    return _CACHED_MODEL


def segment_wsi_custom(
    image: da.Array,
    model,
    block_size: int,
    overlap: int,
    iou_depth: int,
    iou_threshold: float,
    bbox_threshold: float,
    normalize: bool = True
) -> np.ndarray:
    """Segment a whole-slide image using tiling.
    
    This is a custom implementation that uses the cached model.
    
    Args:
        image: Dask array of the image
        model: Cached model instance
        block_size: Size of tiles
        overlap: Tile overlap region
        iou_depth: IOU depth for label merging
        iou_threshold: IOU threshold for label merging
        bbox_threshold: Bbox threshold for segmentation
        normalize: Whether to normalize the image
        
    Returns:
        Segmentation mask as numpy array
    """
    if image.ndim == 2:
        image = image[..., None]
    
    image = da.asarray(image)
    # Rechunk to block_size
    image = image.rechunk({0: block_size, 1: block_size, -1: -1})
    
    depth = (overlap, overlap)
    boundary = "periodic"
    image = da.overlap.overlap(image, depth + (0,), boundary)
    
    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )
    
    labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
    total = None
    
    total_blocks = np.prod(image.numblocks)
    
    for index, input_block in tqdm(block_iter, total=total_blocks):
        # segment_chunk signature: segment_chunk(chunk, model=None, **kwargs)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        labeled_block, n = segment_chunk(
            input_block,
            model=model,
            normalize=normalize,
            device=device,
            bbox_threshold=bbox_threshold
        )
        
        shape = input_block.shape[:-1]
        total = np.int32(n) if total is None else np.int32(total + n)
        block_label_offset = np.where(labeled_block > 0, total, np.int32(0))
        labeled_block += block_label_offset
        
        labeled_blocks[index[:-1]] = labeled_block
        total += np.int32(n)
    
    block_labeled = da.block(labeled_blocks.tolist())
    
    depth = da.overlap.coerce_depth(len(depth), depth)
    
    if np.prod(block_labeled.numblocks) > 1:
        iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)
        
        if any(iou_depth[ax] > depth[ax] for ax in depth.keys()):
            raise ValueError("iou_depth (%s) > depth (%s)" % (iou_depth, depth))
        
        trim_depth = {k: depth[k] - iou_depth[k] for k in depth.keys()}
        block_labeled = da.overlap.trim_internal(
            block_labeled, trim_depth, boundary=boundary
        )
        block_labeled = link_labels(
            block_labeled,
            total,
            iou_depth,
            iou_threshold=iou_threshold,
        )
    else:
        iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)
    
    block_labeled = da.overlap.trim_internal(
        block_labeled, iou_depth, boundary=boundary
    )
    result = block_labeled.compute()
    return result


def use_cellsize_gaging_custom(
    inp: da.Array,
    model,
    device: str,
    block_size: int = 400,
    overlap: int = 200,
    iou_depth: int = 200,
    iou_threshold: float = 0.5,
    bbox_threshold: float = 0.4,
    medium_cell_threshold: float = 0.002,
    tile_size: int = 256,
) -> np.ndarray:
    """Use cell size gaging to determine segmentation strategy.
    
    Args:
        inp: Dask array of the image
        model: Cached model instance
        device: Device to use ('cuda' or 'cpu')
        block_size: Size of tiles
        overlap: Tile overlap region
        iou_depth: IOU depth for label merging
        iou_threshold: IOU threshold for label merging
        bbox_threshold: Bbox threshold for segmentation
        medium_cell_threshold: Threshold for determining cell size
        tile_size: Tile size for rechunking
        
    Returns:
        Segmentation mask as numpy array
    """
    labels = segment_wsi_custom(
        inp, model, block_size, overlap, iou_depth, iou_threshold,
        bbox_threshold, normalize=False
    )
    
    median_size, sizes, sizes_abs = get_median_size(labels)
    
    # Only if cells are small we do WSI inference
    if median_size < medium_cell_threshold:
        # Cells are medium or small -> do WSI
        labels = segment_wsi_custom(
            inp, model, block_size, overlap, iou_depth, iou_threshold,
            bbox_threshold, normalize=False
        )
    else:
        # Large cells -> use direct segmentation
        if not _HAVE_CELLSAM or segment_cellular_image is None:
            raise ImportError("CellSAM not installed. segment_cellular_image not available.")
        labels = segment_cellular_image(inp, model=model, normalize=False, device=device)[0]
    
    # Handle case where CellSAM fails to detect any cells
    if labels is None:
        raise ValueError(
            "CellSAM failed to detect any cells during cell size gauging. "
            "Try lowering --bbox-threshold (e.g., 0.1) or using --low-contrast-enhancement"
        )
    
    return labels


def cellsam_pipeline_custom(
    img: np.ndarray,
    chunks: int = 256,
    model_path: Optional[str] = None,
    bbox_threshold: float = 0.4,
    low_contrast_enhancement: bool = False,
    swap_channels: bool = False,
    use_wsi: bool = True,
    gauge_cell_size: bool = False,
    block_size: int = 400,
    overlap: int = 56,
    iou_depth: int = 56,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Custom CellSAM pipeline with proper model caching.
    
    This is a memory-efficient implementation that caches the model
    and prevents memory leaks by reusing the same model instance.
    
    Args:
        img: Image array with shape (W, H) or (W, H, C)
        chunks: Chunk size for dask arrays
        model_path: Optional path to model weights
        bbox_threshold: Threshold for cell detection
        low_contrast_enhancement: Whether to enhance low contrast images
        swap_channels: Whether to swap channels (deprecated)
        use_wsi: Whether to use tiling for large images
        gauge_cell_size: Whether to gauge cell size first
        block_size: Size of tiles when use_wsi is True
        overlap: Tile overlap region
        iou_depth: IOU depth for label merging
        iou_threshold: IOU threshold for label merging
        
    Returns:
        Segmentation mask as numpy array (uint32)
    """
    if not _HAVE_CELLSAM:
        raise ImportError("CellSAM not installed. Install with: pip install git+https://github.com/vanvalenlab/cellSAM.git")
    
    if not _HAVE_DASK:
        raise ImportError("dask not installed. Install with: pip install dask")

    # Avoid noisy progress bars and repetitive warnings from the upstream package.
    os.environ.setdefault("TQDM_DISABLE", "1")
    
    # Get cached model (this ensures only one instance exists)
    model = _get_cached_model(model_path, bbox_threshold)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Normalize image
    img = img.astype(np.float32)
    img = normalize_image(img)
    
    if low_contrast_enhancement:
        # enhance_low_contrast may modify model.bbox_threshold
        # Since we're using a cached model, we need to restore the original bbox_threshold afterward
        original_bbox_threshold = model.bbox_threshold
        img = enhance_low_contrast(img, model=model)
        # Restore original bbox_threshold if it was modified
        if hasattr(model, 'bbox_threshold') and model.bbox_threshold != original_bbox_threshold:
            model.bbox_threshold = original_bbox_threshold
    
    inp = da.from_array(img, chunks=chunks)
    
    # gauge_cell_size can be used with or without use_wsi
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Low IOU threshold, ignoring mask\\.",
            category=UserWarning,
        )
        if gauge_cell_size:
            # Gauge cell size first, then decide on best segmentation approach
            labels = use_cellsize_gaging_custom(
                inp, model, device, block_size=block_size, overlap=overlap,
                iou_depth=iou_depth, iou_threshold=iou_threshold,
                bbox_threshold=bbox_threshold
            )
        elif use_wsi:
            labels = segment_wsi_custom(
                inp, model, block_size, overlap, iou_depth, iou_threshold,
                bbox_threshold, normalize=True
            )
        else:
            if not _HAVE_CELLSAM or segment_cellular_image is None:
                raise ImportError("CellSAM not installed. segment_cellular_image not available.")
            labels = segment_cellular_image(inp, model=model, normalize=True, device=device)[0]
    
    # Handle case where CellSAM fails to detect any cells (returns None)
    if labels is None:
        raise ValueError(
            "CellSAM failed to detect any cells. This can happen if:\n"
            "  1. The image has very low contrast - try --low-contrast-enhancement\n"
            "  2. The bbox_threshold is too high - try lowering --bbox-threshold (e.g., 0.1)\n"
            "  3. The nuclear/cytoplasm channels don't show clear cell boundaries\n"
            "  4. For images with many cells (>500), try adding --use-wsi"
        )
    
    return labels


def run_cellsam_pipeline_subprocess(
    img: np.ndarray,
    bbox_threshold: float = 0.4,
    use_wsi: bool = True,
    low_contrast_enhancement: bool = False,
    gauge_cell_size: bool = False,
    chunks: int = 256,
    block_size: int = 400,
    overlap: int = 56,
    iou_depth: int = 56,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Process-safe entry point for CellSAM inference.

    This function is designed to be executed in a subprocess to avoid
    Qt UI stalling from long-running Python/GIL-heavy sections.
    """
    return cellsam_pipeline_custom(
        img=img,
        chunks=chunks,
        model_path=None,
        bbox_threshold=bbox_threshold,
        low_contrast_enhancement=low_contrast_enhancement,
        swap_channels=False,
        use_wsi=use_wsi,
        gauge_cell_size=gauge_cell_size,
        block_size=block_size,
        overlap=overlap,
        iou_depth=iou_depth,
        iou_threshold=iou_threshold,
    )


def clear_model_cache():
    """Clear the cached model to free memory.
    
    This can be called when you're done with segmentation to free GPU/CPU memory.
    """
    global _CACHED_MODEL, _CACHED_MODEL_DEVICE, _CACHED_MODEL_BBOX_THRESHOLD
    
    if _CACHED_MODEL is not None:
        # Move model to CPU and clear CUDA cache
        try:
            if torch.cuda.is_available():
                _CACHED_MODEL = _CACHED_MODEL.cpu()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except:
            pass
        
        # Delete model
        del _CACHED_MODEL
        _CACHED_MODEL = None
        _CACHED_MODEL_DEVICE = None
        _CACHED_MODEL_BBOX_THRESHOLD = None
        
        # Force garbage collection
        import gc
        gc.collect()
        gc.collect()
