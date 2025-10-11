"""
Instance segmentation from probability maps using watershed.

This module implements the conversion from pixel-level probability maps
to instance masks using local maxima detection and watershed segmentation.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, label
from skimage import morphology, segmentation
# Use scipy alternative for peak detection
from scipy.ndimage import maximum_filter
from skimage.morphology import disk, remove_small_objects
from skimage.measure import regionprops

# Optional dependencies
try:
    from skimage import filters, morphology, segmentation
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False


class InteractivePixelInstanceSegmenter:
    """
    Converts probability maps to instance masks using watershed segmentation.
    
    The process involves:
    1. Extract seeds from nucleus probability map (local maxima)
    2. Create boundary map from cytoplasm probability map
    3. Apply watershed segmentation
    """
    
    def __init__(self,
                 min_distance: int = 10,
                 threshold_abs: float = 0.3,
                 threshold_rel: float = 0.2,
                 min_seed_area: int = 5,
                 min_cell_area: int = 50,
                 max_cell_area: int = 2000,
                 compactness: float = 0.1,
                 nucleus_expansion: int = 5):
        """
        Initialize the instance segmenter.
        
        Args:
            min_distance: Minimum distance between seed points
            threshold_abs: Absolute threshold for seed detection
            threshold_rel: Relative threshold for seed detection
            min_seed_area: Minimum area for seed regions
            min_cell_area: Minimum area for final cell regions
            max_cell_area: Maximum area for final cell regions
            compactness: Compactness parameter for watershed
            nucleus_expansion: Expansion radius for nucleus-only segmentation
        """
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.min_seed_area = min_seed_area
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        self.compactness = compactness
        self.nucleus_expansion = nucleus_expansion
    
    def segment_instances(self, 
                         probability_maps: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert probability maps to instance segmentation.
        
        Args:
            probability_maps: Dictionary with class names as keys and 
                            probability maps as values
            
        Returns:
            Instance labels array of shape (H, W)
        """
        if not _HAVE_SCIKIT_IMAGE:
            raise ImportError("scikit-image is required for instance segmentation")
        
        # Extract probability maps
        p_nucleus = probability_maps.get('nucleus', np.zeros_like(list(probability_maps.values())[0]))
        p_cytoplasm = probability_maps.get('cytoplasm', np.zeros_like(p_nucleus))
        p_background = probability_maps.get('background', np.zeros_like(p_nucleus))
        
        # Generate seeds from nucleus probability map
        seeds = self._generate_seeds(p_nucleus)
        
        # Generate boundary map from cytoplasm probability
        boundaries = self._generate_boundaries(p_cytoplasm, p_nucleus)
        
        # Apply watershed segmentation
        labels = segmentation.watershed(
            boundaries, 
            seeds, 
            compactness=self.compactness,
            connectivity=1
        )
        
        # Post-process to remove small objects and merge oversegmented regions
        labels = self._postprocess_labels(labels)
        
        return labels
    
    def _generate_seeds(self, p_nucleus: np.ndarray) -> np.ndarray:
        """
        Generate seed points from nucleus probability map.
        
        Args:
            p_nucleus: Nucleus probability map of shape (H, W)
            
        Returns:
            Seed labels array of shape (H, W)
        """
        # Find local maxima using maximum filter
        local_maxima = self._find_local_maxima(
            p_nucleus,
            min_distance=self.min_distance,
            threshold_abs=self.threshold_abs,
            threshold_rel=self.threshold_rel
        )
        
        # Create seed image
        seeds = np.zeros_like(p_nucleus, dtype=int)
        for i, (row, col) in enumerate(local_maxima):
            seeds[row, col] = i + 1
        
        # Dilate seeds to create seed regions
        if self.min_seed_area > 1:
            # Use distance transform to create seed regions
            distance = distance_transform_edt(seeds == 0)
            seed_regions = distance < (self.min_seed_area / 2)
            seeds = label(seed_regions)[0]
        
        return seeds
    
    def _find_local_maxima(self, image: np.ndarray, min_distance: int, 
                          threshold_abs: float, threshold_rel: float) -> List[Tuple[int, int]]:
        """Find local maxima in an image."""
        # Apply thresholds
        if threshold_abs is not None:
            image = np.where(image >= threshold_abs, image, 0)
        
        if threshold_rel is not None:
            threshold = np.max(image) * threshold_rel
            image = np.where(image >= threshold, image, 0)
        
        # Find local maxima using maximum filter
        footprint = np.ones((min_distance*2+1, min_distance*2+1))
        local_maxima = maximum_filter(image, footprint=footprint) == image
        
        # Only keep pixels that are actually local maxima (not flat regions)
        # A true local maximum should be strictly greater than its neighbors
        local_maxima = local_maxima & (image > 0)  # Must be above threshold
        
        # Remove edge effects
        local_maxima[:min_distance] = False
        local_maxima[-min_distance:] = False
        local_maxima[:, :min_distance] = False
        local_maxima[:, -min_distance:] = False
        
        # Get coordinates
        coords = np.where(local_maxima)
        return list(zip(coords[0], coords[1]))
    
    def _generate_boundaries(self, 
                           p_cytoplasm: np.ndarray, 
                           p_nucleus: np.ndarray) -> np.ndarray:
        """
        Generate boundary map for watershed.
        
        Args:
            p_cytoplasm: Cytoplasm probability map
            p_nucleus: Nucleus probability map
            
        Returns:
            Boundary map where high values indicate boundaries
        """
        # Check if we have cytoplasm information
        has_cytoplasm = np.any(p_cytoplasm > 0.1)
        
        if has_cytoplasm:
            # Standard 3-class approach: use both nucleus and cytoplasm
            max_prob = np.maximum(p_cytoplasm, p_nucleus)
            boundaries = 1.0 - max_prob
        else:
            # 2-class approach: nucleus vs background only
            # Use nucleus probability gradients and distance transforms
            boundaries = self._generate_nucleus_boundaries(p_nucleus)
        
        # Smooth boundaries slightly
        boundaries = ndimage.gaussian_filter(boundaries, sigma=0.5)
        
        return boundaries
    
    def _generate_nucleus_boundaries(self, p_nucleus: np.ndarray) -> np.ndarray:
        """
        Generate boundaries for nucleus-only segmentation.
        
        Args:
            p_nucleus: Nucleus probability map
            
        Returns:
            Boundary map optimized for nucleus-only segmentation
        """
        # Method 1: Use nucleus probability gradients
        # Compute gradients to find edges
        grad_x = ndimage.sobel(p_nucleus, axis=1)
        grad_y = ndimage.sobel(p_nucleus, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Method 2: Use distance transform from high-confidence nuclei
        # Create binary mask of high-confidence nuclei
        nucleus_mask = p_nucleus > 0.5
        if np.any(nucleus_mask):
            # Distance transform from nuclei
            distance = distance_transform_edt(~nucleus_mask)
            # Normalize distance
            distance = distance / (np.max(distance) + 1e-8)
        else:
            distance = np.ones_like(p_nucleus)
        
        # Method 3: Use morphological operations
        # Create a mask of nucleus regions
        nucleus_regions = p_nucleus > 0.3
        if np.any(nucleus_regions):
            # Dilate nucleus regions to create cell boundaries
            dilated = ndimage.binary_dilation(nucleus_regions, structure=disk(self.nucleus_expansion))
            # Create boundaries as edges of dilated regions
            boundaries_morph = ndimage.binary_dilation(dilated) ^ dilated
            boundaries_morph = boundaries_morph.astype(float)
        else:
            boundaries_morph = np.ones_like(p_nucleus)
        
        # Combine methods with weights
        # Gradient-based boundaries (good for sharp edges)
        gradient_boundaries = 1.0 - (gradient_magnitude / (np.max(gradient_magnitude) + 1e-8))
        
        # Distance-based boundaries (good for cell shape)
        distance_boundaries = 1.0 - distance
        
        # Morphological boundaries (good for overall structure)
        morph_boundaries = 1.0 - boundaries_morph
        
        # Weighted combination
        boundaries = (0.4 * gradient_boundaries + 
                     0.4 * distance_boundaries + 
                     0.2 * morph_boundaries)
        
        # Ensure boundaries are in [0, 1] range
        boundaries = np.clip(boundaries, 0, 1)
        
        return boundaries
    
    def _postprocess_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Post-process segmentation labels.
        
        Args:
            labels: Initial watershed labels
            
        Returns:
            Post-processed labels
        """
        # Remove small objects
        labels = remove_small_objects(labels, min_size=self.min_cell_area)
        
        # Remove objects that are too large
        if self.max_cell_area < np.inf:
            labels = self._remove_large_objects(labels, self.max_cell_area)
        
        # Fill holes in objects
        labels = self._fill_holes(labels)
        
        # Relabel to ensure consecutive labels
        labels, _ = label(labels > 0)
        
        return labels
    
    def _remove_large_objects(self, labels: np.ndarray, max_area: int) -> np.ndarray:
        """Remove objects larger than max_area."""
        result = labels.copy()
        
        # Get region properties
        props = regionprops(labels)
        
        for prop in props:
            if prop.area > max_area:
                # Remove this object
                result[labels == prop.label] = 0
        
        return result
    
    def _fill_holes(self, labels: np.ndarray) -> np.ndarray:
        """Fill holes in labeled objects."""
        result = labels.copy()
        
        # Fill holes for each label
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        for label_id in unique_labels:
            mask = (labels == label_id)
            filled_mask = ndimage.binary_fill_holes(mask)
            result[filled_mask] = label_id
        
        return result
    
    def get_seed_statistics(self, seeds: np.ndarray) -> Dict[str, int]:
        """Get statistics about detected seeds."""
        unique_seeds = np.unique(seeds)
        unique_seeds = unique_seeds[unique_seeds > 0]  # Remove background
        
        return {
            "n_seeds": len(unique_seeds),
            "min_seed_area": self.min_seed_area,
            "min_distance": self.min_distance
        }
    
    def get_instance_statistics(self, labels: np.ndarray) -> Dict[str, any]:
        """Get statistics about final instances."""
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        if len(unique_labels) == 0:
            return {"n_instances": 0}
        
        # Get region properties
        props = regionprops(labels)
        areas = [prop.area for prop in props]
        
        return {
            "n_instances": len(unique_labels),
            "min_area": min(areas) if areas else 0,
            "max_area": max(areas) if areas else 0,
            "mean_area": np.mean(areas) if areas else 0,
            "median_area": np.median(areas) if areas else 0
        }


def segment_instances_from_probabilities(probability_maps: Dict[str, np.ndarray],
                                       min_distance: int = 10,
                                       threshold_abs: float = 0.3,
                                       min_cell_area: int = 50) -> np.ndarray:
    """
    Convenience function to segment instances from probability maps.
    
    Args:
        probability_maps: Dictionary with class names as keys and 
                        probability maps as values
        min_distance: Minimum distance between seed points
        threshold_abs: Absolute threshold for seed detection
        min_cell_area: Minimum area for final cell regions
        
    Returns:
        Instance labels array of shape (H, W)
    """
    segmenter = InteractivePixelInstanceSegmenter(
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        min_cell_area=min_cell_area
    )
    
    return segmenter.segment_instances(probability_maps)


def create_probability_overlay(probability_maps: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create a colored overlay from probability maps for visualization.
    
    Args:
        probability_maps: Dictionary with class names as keys and 
                        probability maps as values
        
    Returns:
        RGB overlay array of shape (H, W, 3)
    """
    height, width = list(probability_maps.values())[0].shape
    overlay = np.zeros((height, width, 3), dtype=np.float32)
    
    # Color mapping: background=black, nucleus=red, cytoplasm=green
    colors = {
        'background': [0.0, 0.0, 0.0],
        'nucleus': [1.0, 0.0, 0.0],
        'cytoplasm': [0.0, 1.0, 0.0]
    }
    
    for class_name, prob_map in probability_maps.items():
        if class_name in colors:
            color = colors[class_name]
            for i in range(3):
                overlay[:, :, i] += prob_map * color[i]
    
    # Clamp values to [0, 1]
    overlay = np.clip(overlay, 0, 1)
    
    return overlay
