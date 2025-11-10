"""
Unit tests for denoising functions.
"""
import numpy as np
import pytest

from openimc.processing.feature_worker import _apply_denoise_to_channel


@pytest.mark.unit
class TestDenoising:
    """Tests for denoising functions."""
    
    def test_denoise_no_settings(self, sample_image_2d):
        """Test that denoising returns original image when no settings provided."""
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", {})
        
        assert np.array_equal(result, sample_image_2d)
    
    def test_denoise_hot_pixel_median3(self, sample_image_2d):
        """Test hot pixel removal with median3 method."""
        denoise_settings = {
            'hot': {
                'method': 'median3',
                'n_sd': 5.0
            }
        }
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", denoise_settings)
        
        assert result.shape == sample_image_2d.shape
        assert result.dtype == sample_image_2d.dtype
    
    def test_denoise_speckle_gaussian(self, sample_image_2d):
        """Test speckle noise reduction with gaussian method."""
        denoise_settings = {
            'speckle': {
                'method': 'gaussian',
                'sigma': 0.8
            }
        }
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", denoise_settings)
        
        assert result.shape == sample_image_2d.shape
    
    def test_denoise_background_white_tophat(self, sample_image_2d):
        """Test background subtraction with white tophat."""
        denoise_settings = {
            'background': {
                'method': 'white_tophat',
                'radius': 15
            }
        }
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", denoise_settings)
        
        assert result.shape == sample_image_2d.shape
    
    def test_denoise_full_pipeline(self, sample_image_2d, sample_denoise_settings):
        """Test full denoising pipeline with all steps."""
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", sample_denoise_settings)
        
        assert result.shape == sample_image_2d.shape
    
    def test_denoise_partial_settings(self, sample_image_2d):
        """Test denoising with partial settings."""
        denoise_settings = {
            'hot': {
                'method': 'median3'
            }
        }
        result = _apply_denoise_to_channel(sample_image_2d, "test_channel", denoise_settings)
        
        assert result.shape == sample_image_2d.shape

