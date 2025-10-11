"""
Interactive pixel level classification per-pixel feature computation for segmentation.

This module implements the feature computation pipeline for interactive pixel classification,
including Gaussian blur at multiple scales, Laplacian of Gaussian, gradient
magnitude, structure tensor eigenvalues, Hessian eigenvalues, and entropy.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, gaussian_laplace, sobel, generic_filter
from skimage.filters import gaussian, laplace, sobel as sk_sobel
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.filters.rank import entropy
from skimage.morphology import disk

# Optional dependencies
try:
    from skimage import filters
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False


class InteractivePixelFeatureComputer:
    """
    Computes per-pixel features for Interactive pixel level classification segmentation.
    
    Features include:
    - Gaussian blur at multiple σ values
    - Laplacian of Gaussian (LoG)
    - Gradient magnitude
    - Structure tensor eigenvalues
    - Hessian eigenvalues
    - Entropy at multiple radii
    """
    
    def __init__(self, 
                 sigma_values: List[float] = None,
                 entropy_radii: List[int] = None,
                 tile_size: int = 512,
                 tile_overlap: int = 64):
        """
        Initialize feature computer.
        
        Args:
            sigma_values: List of sigma values for Gaussian blur and LoG
            entropy_radii: List of radii for entropy computation
            tile_size: Size of tiles for processing large images
            tile_overlap: Overlap between tiles
        """
        self.sigma_values = sigma_values or [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0]
        self.entropy_radii = entropy_radii or [3, 5, 7, 9, 11]
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
    def compute_features_tiled(self, 
                             img_stack: np.ndarray, 
                             channel_names: List[str]) -> np.ndarray:
        """
        Compute features for a multi-channel image stack using tiling.
        
        Args:
            img_stack: Image stack of shape (H, W, C)
            channel_names: List of channel names
            
        Returns:
            Feature array of shape (H, W, N_features)
        """
        height, width, n_channels = img_stack.shape
        
        # Calculate number of features per channel
        n_features_per_channel = self._get_n_features_per_channel()
        total_features = n_features_per_channel * n_channels
        
        # Initialize output array
        features = np.zeros((height, width, total_features), dtype=np.float32)
        
        # Process each channel
        feature_idx = 0
        for ch_idx, channel_name in enumerate(channel_names):
            channel_img = img_stack[:, :, ch_idx]
            
            # Compute features for this channel
            channel_features = self._compute_channel_features(channel_img)
            
            # Store in output array
            n_channel_features = channel_features.shape[2]
            features[:, :, feature_idx:feature_idx + n_channel_features] = channel_features
            feature_idx += n_channel_features
            
        return features
    
    def _compute_channel_features(self, img: np.ndarray) -> np.ndarray:
        """
        Compute all features for a single channel image.
        
        Args:
            img: Single channel image of shape (H, W)
            
        Returns:
            Feature array of shape (H, W, N_features)
        """
        height, width = img.shape
        features = []
        
        # 1. Gaussian blur at multiple scales
        for sigma in self.sigma_values:
            blurred = gaussian_filter(img, sigma=sigma)
            features.append(blurred)
        
        # 2. Laplacian of Gaussian at multiple scales
        for sigma in self.sigma_values:
            log_img = gaussian_laplace(img, sigma=sigma)
            features.append(log_img)
        
        # 3. Gradient magnitude
        grad_mag = self._compute_gradient_magnitude(img)
        features.append(grad_mag)
        
        # 4. Structure tensor eigenvalues
        if _HAVE_SCIKIT_IMAGE:
            st_eigvals = self._compute_structure_tensor_eigenvalues(img)
            features.extend(st_eigvals)
        
        # 5. Hessian eigenvalues
        hess_eigvals = self._compute_hessian_eigenvalues(img)
        features.extend(hess_eigvals)
        
        # 6. Entropy at multiple radii
        for radius in self.entropy_radii:
            entropy_img = self._compute_entropy(img, radius)
            features.append(entropy_img)
        
        # Ensure all features have the same shape
        for i, feature in enumerate(features):
            if feature.shape != (height, width):
                print(f"Warning: Feature {i} has shape {feature.shape}, expected ({height}, {width})")
                # Handle different cases
                if feature.ndim == 2 and feature.shape[0] == height:
                    # Feature has wrong width, pad or crop
                    if feature.shape[1] < width:
                        # Pad with zeros
                        padded = np.zeros((height, width))
                        padded[:, :feature.shape[1]] = feature
                        features[i] = padded
                    else:
                        # Crop to correct width
                        features[i] = feature[:, :width]
                elif feature.ndim == 2 and feature.shape[1] == width:
                    # Feature has wrong height, pad or crop
                    if feature.shape[0] < height:
                        # Pad with zeros
                        padded = np.zeros((height, width))
                        padded[:feature.shape[0], :] = feature
                        features[i] = padded
                    else:
                        # Crop to correct height
                        features[i] = feature[:height, :]
                else:
                    # Completely wrong shape, create zeros
                    features[i] = np.zeros((height, width))
        
        # Stack all features
        feature_array = np.stack(features, axis=2)
        return feature_array
    
    def _compute_gradient_magnitude(self, img: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude using Sobel operators."""
        if _HAVE_SCIKIT_IMAGE:
            # Use scikit-image for better performance
            grad_x = sk_sobel(img, axis=1)
            grad_y = sk_sobel(img, axis=0)
        else:
            # Fallback to scipy
            grad_x = sobel(img, axis=1)
            grad_y = sobel(img, axis=0)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return grad_mag
    
    def _compute_structure_tensor_eigenvalues(self, img: np.ndarray) -> List[np.ndarray]:
        """Compute structure tensor eigenvalues."""
        if not _HAVE_SCIKIT_IMAGE:
            return []
        
        try:
            # Compute structure tensor
            st = structure_tensor(img, sigma=1.0)
            
            # Compute eigenvalues
            eigvals = structure_tensor_eigenvalues(st)
            
            # Handle different output shapes
            height, width = img.shape
            
            if eigvals.ndim == 3:
                # Check if it's (2, H, W) or (H, W, 2)
                if eigvals.shape[0] == 2:
                    # Eigenvalues first: (2, H, W)
                    return [eigvals[0, :, :], eigvals[1, :, :]]
                else:
                    # Spatial first: (H, W, 2)
                    return [eigvals[:, :, 0], eigvals[:, :, 1]]
            elif eigvals.ndim == 2:
                # Single eigenvalue case: (H, W)
                return [eigvals, np.zeros((height, width))]
            elif eigvals.ndim == 4:
                # Multiple eigenvalues case: (H, W, 1, 2) or similar
                if eigvals.shape[2] == 1:
                    return [eigvals[:, :, 0, 0], eigvals[:, :, 0, 1]]
                else:
                    return [eigvals[:, :, 0], eigvals[:, :, 1]]
            else:
                # Fallback: return zeros if shape is unexpected
                print(f"Warning: Unexpected structure tensor eigenvalues shape: {eigvals.shape}")
                return [np.zeros((height, width)), np.zeros((height, width))]
        except Exception as e:
            print(f"Warning: Structure tensor computation failed: {e}")
            height, width = img.shape
            return [np.zeros((height, width)), np.zeros((height, width))]
    
    def _compute_hessian_eigenvalues(self, img: np.ndarray) -> List[np.ndarray]:
        """Compute Hessian eigenvalues."""
        # Compute second derivatives
        hxx = gaussian_filter(img, sigma=1.0, order=(0, 2))
        hyy = gaussian_filter(img, sigma=1.0, order=(2, 0))
        hxy = gaussian_filter(img, sigma=1.0, order=(1, 1))
        
        # Compute eigenvalues of Hessian matrix
        # For 2D: λ = (hxx + hyy ± sqrt((hxx - hyy)² + 4*hxy²)) / 2
        trace = hxx + hyy
        det = hxx * hyy - hxy**2
        discriminant = trace**2 - 4 * det
        
        # Avoid negative discriminant (shouldn't happen for real eigenvalues)
        discriminant = np.maximum(discriminant, 0)
        
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2
        
        return [lambda1, lambda2]
    
    def _compute_entropy(self, img: np.ndarray, radius: int) -> np.ndarray:
        """Compute local entropy using a circular neighborhood."""
        if not _HAVE_SCIKIT_IMAGE:
            # Fallback implementation
            return self._compute_entropy_fallback(img, radius)
        
        # Normalize image to 0-255 for entropy computation
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        
        # Compute entropy
        footprint = disk(radius)
        entropy_img = entropy(img_norm, footprint)
        
        return entropy_img.astype(np.float32)
    
    def _compute_entropy_fallback(self, img: np.ndarray, radius: int) -> np.ndarray:
        """Fallback entropy computation without scikit-image."""
        height, width = img.shape
        entropy_img = np.zeros_like(img)
        
        # Normalize image
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        
        # Compute entropy for each pixel
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                # Extract neighborhood
                neighborhood = img_norm[i-radius:i+radius+1, j-radius:j+radius+1]
                
                # Compute histogram
                hist, _ = np.histogram(neighborhood, bins=256, range=(0, 256))
                hist = hist / hist.sum()  # Normalize to probabilities
                
                # Compute entropy
                hist = hist[hist > 0]  # Remove zero probabilities
                entropy_val = -np.sum(hist * np.log2(hist))
                entropy_img[i, j] = entropy_val
        
        return entropy_img
    
    def _get_n_features_per_channel(self) -> int:
        """Get the number of features computed per channel."""
        n_gaussian = len(self.sigma_values)
        n_log = len(self.sigma_values)
        n_gradient = 1
        n_structure = 2 if _HAVE_SCIKIT_IMAGE else 0
        n_hessian = 2
        n_entropy = len(self.entropy_radii)
        
        return n_gaussian + n_log + n_gradient + n_structure + n_hessian + n_entropy
    
    def get_feature_names(self, channel_names: List[str]) -> List[str]:
        """Get descriptive names for all computed features."""
        feature_names = []
        
        for ch_name in channel_names:
            # Gaussian blur features
            for sigma in self.sigma_values:
                feature_names.append(f"{ch_name}_gaussian_sigma_{sigma:.1f}")
            
            # LoG features
            for sigma in self.sigma_values:
                feature_names.append(f"{ch_name}_log_sigma_{sigma:.1f}")
            
            # Gradient magnitude
            feature_names.append(f"{ch_name}_gradient_magnitude")
            
            # Structure tensor eigenvalues
            if _HAVE_SCIKIT_IMAGE:
                feature_names.append(f"{ch_name}_structure_tensor_eigenval_1")
                feature_names.append(f"{ch_name}_structure_tensor_eigenval_2")
            
            # Hessian eigenvalues
            feature_names.append(f"{ch_name}_hessian_eigenval_1")
            feature_names.append(f"{ch_name}_hessian_eigenval_2")
            
            # Entropy features
            for radius in self.entropy_radii:
                feature_names.append(f"{ch_name}_entropy_radius_{radius}")
        
        return feature_names


def compute_interactive_pixel_features(img_stack: np.ndarray, 
                           channel_names: List[str],
                           sigma_values: List[float] = None,
                           entropy_radii: List[int] = None,
                           tile_size: int = 512) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to compute Interactive pixel level classification features.
    
    Args:
        img_stack: Image stack of shape (H, W, C)
        channel_names: List of channel names
        sigma_values: List of sigma values for Gaussian blur and LoG
        entropy_radii: List of radii for entropy computation
        tile_size: Size of tiles for processing large images
        
    Returns:
        Tuple of (features, feature_names) where:
        - features: Feature array of shape (H, W, N_features)
        - feature_names: List of feature names
    """
    computer = InteractivePixelFeatureComputer(
        sigma_values=sigma_values,
        entropy_radii=entropy_radii,
        tile_size=tile_size
    )
    
    features = computer.compute_features_tiled(img_stack, channel_names)
    feature_names = computer.get_feature_names(channel_names)
    
    return features, feature_names
