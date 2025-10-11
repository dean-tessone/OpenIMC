"""
Tiled inference pipeline for Interactive pixel level classification segmentation.

This module implements efficient tiled processing for large images,
computing features and running inference in overlapping tiles to avoid
boundary artifacts.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

from .interactive_pixel_features import InteractivePixelFeatureComputer
from .interactive_pixel_classifier import InteractivePixelClassifier
from .interactive_pixel_instances import InteractivePixelInstanceSegmenter


class InteractivePixelInferencePipeline:
    """
    Tiled inference pipeline for Interactive pixel level classification segmentation.
    
    Processes large images by:
    1. Computing features in overlapping tiles
    2. Running pixel classification in tiles
    3. Stitching results together
    4. Converting to instance segmentation
    """
    
    def __init__(self,
                 classifier: InteractivePixelClassifier,
                 tile_size: int = 512,
                 tile_overlap: int = 64,
                 n_workers: int = None,
                 use_multiprocessing: bool = True):
        """
        Initialize the inference pipeline.
        
        Args:
            classifier: Trained InteractivePixelClassifier
            tile_size: Size of tiles for processing
            tile_overlap: Overlap between tiles
            n_workers: Number of worker processes/threads
            use_multiprocessing: Whether to use multiprocessing vs threading
        """
        self.classifier = classifier
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.n_workers = n_workers or min(mp.cpu_count(), 4)
        self.use_multiprocessing = use_multiprocessing
        
        # Initialize feature computer
        self.feature_computer = InteractivePixelFeatureComputer(
            tile_size=tile_size,
            tile_overlap=tile_overlap
        )
        
        # Initialize instance segmenter
        self.instance_segmenter = InteractivePixelInstanceSegmenter()
    
    def process_image(self, 
                     img_stack: np.ndarray,
                     channel_names: List[str],
                     return_probabilities: bool = True,
                     return_instances: bool = True) -> Dict[str, np.ndarray]:
        """
        Process an entire image through the pipeline.
        
        Args:
            img_stack: Image stack of shape (H, W, C)
            channel_names: List of channel names
            return_probabilities: Whether to return probability maps
            return_instances: Whether to return instance segmentation
            
        Returns:
            Dictionary with results
        """
        height, width, n_channels = img_stack.shape
        
        # Compute features in tiles
        print(f"Computing features for image of size {height}x{width}")
        features = self._compute_features_tiled(img_stack, channel_names)
        
        # Run inference in tiles
        print("Running pixel classification")
        probability_maps = self._run_inference_tiled(features)
        
        results = {}
        
        if return_probabilities:
            results['probability_maps'] = probability_maps
        
        if return_instances:
            print("Converting to instance segmentation")
            instance_labels = self.instance_segmenter.segment_instances(probability_maps)
            results['instance_labels'] = instance_labels
            
            # Get statistics
            stats = self.instance_segmenter.get_instance_statistics(instance_labels)
            results['instance_statistics'] = stats
        
        return results
    
    def _compute_features_tiled(self, 
                              img_stack: np.ndarray, 
                              channel_names: List[str]) -> np.ndarray:
        """Compute features using tiled processing."""
        height, width, n_channels = img_stack.shape
        
        # Calculate number of features per channel
        n_features_per_channel = self.feature_computer._get_n_features_per_channel()
        total_features = n_features_per_channel * n_channels
        
        # Initialize output array
        features = np.zeros((height, width, total_features), dtype=np.float32)
        
        # Generate tile coordinates
        tile_coords = self._generate_tile_coordinates(height, width)
        
        # Process tiles
        if self.use_multiprocessing and len(tile_coords) > 1:
            # Use multiprocessing for CPU-intensive feature computation
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                process_func = partial(
                    self._process_tile_features,
                    img_stack=img_stack,
                    channel_names=channel_names
                )
                tile_results = list(executor.map(process_func, tile_coords))
        else:
            # Use threading for I/O-bound operations
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                process_func = partial(
                    self._process_tile_features,
                    img_stack=img_stack,
                    channel_names=channel_names
                )
                tile_results = list(executor.map(process_func, tile_coords))
        
        # Stitch results together
        for tile_coord, tile_features in zip(tile_coords, tile_results):
            y_start, y_end, x_start, x_end = tile_coord
            features[y_start:y_end, x_start:x_end] = tile_features
        
        return features
    
    def _run_inference_tiled(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Run pixel classification using tiled processing."""
        height, width, n_features = features.shape
        
        # Initialize probability maps
        n_classes = self.classifier.n_classes
        probability_maps = {
            class_name: np.zeros((height, width), dtype=np.float32)
            for class_name in self.classifier.class_names
        }
        
        # Generate tile coordinates
        tile_coords = self._generate_tile_coordinates(height, width)
        
        # Process tiles
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            process_func = partial(self._process_tile_inference, features=features)
            tile_results = list(executor.map(process_func, tile_coords))
        
        # Stitch results together
        for tile_coord, tile_proba in zip(tile_coords, tile_results):
            y_start, y_end, x_start, x_end = tile_coord
            
            for i, class_name in enumerate(self.classifier.class_names):
                probability_maps[class_name][y_start:y_end, x_start:x_end] = tile_proba[:, :, i]
        
        return probability_maps
    
    def _generate_tile_coordinates(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """Generate tile coordinates with overlap."""
        tile_coords = []
        
        step_size = self.tile_size - self.tile_overlap
        
        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                y_start = y
                y_end = min(y + self.tile_size, height)
                x_start = x
                x_end = min(x + self.tile_size, width)
                
                tile_coords.append((y_start, y_end, x_start, x_end))
        
        return tile_coords
    
    def _process_tile_features(self, 
                             tile_coord: Tuple[int, int, int, int],
                             img_stack: np.ndarray,
                             channel_names: List[str]) -> np.ndarray:
        """Process features for a single tile."""
        y_start, y_end, x_start, x_end = tile_coord
        
        # Extract tile
        tile_img = img_stack[y_start:y_end, x_start:x_end]
        
        # Compute features for this tile
        tile_features = self.feature_computer.compute_features_tiled(tile_img, channel_names)
        
        return tile_features
    
    def _process_tile_inference(self, 
                              tile_coord: Tuple[int, int, int, int],
                              features: np.ndarray) -> np.ndarray:
        """Process inference for a single tile."""
        y_start, y_end, x_start, x_end = tile_coord
        
        # Extract tile
        tile_features = features[y_start:y_end, x_start:x_end]
        
        # Run inference
        tile_proba = self.classifier.predict_image_proba(tile_features)
        
        return tile_proba
    
    def set_instance_parameters(self, **kwargs):
        """Update instance segmentation parameters."""
        for key, value in kwargs.items():
            if hasattr(self.instance_segmenter, key):
                setattr(self.instance_segmenter, key, value)
    
    def get_pipeline_info(self) -> Dict[str, any]:
        """Get information about the pipeline configuration."""
        return {
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "n_workers": self.n_workers,
            "use_multiprocessing": self.use_multiprocessing,
            "classifier_trained": self.classifier.is_trained,
            "n_classes": self.classifier.n_classes,
            "class_names": self.classifier.class_names
        }


def create_inference_pipeline(classifier: InteractivePixelClassifier,
                            tile_size: int = 512,
                            tile_overlap: int = 64,
                            n_workers: int = None) -> InteractivePixelInferencePipeline:
    """
    Convenience function to create an inference pipeline.
    
    Args:
        classifier: Trained InteractivePixelClassifier
        tile_size: Size of tiles for processing
        tile_overlap: Overlap between tiles
        n_workers: Number of worker processes/threads
        
    Returns:
        Initialized InteractivePixelInferencePipeline instance
    """
    return InteractivePixelInferencePipeline(
        classifier=classifier,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        n_workers=n_workers
    )
