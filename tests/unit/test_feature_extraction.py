"""
Unit tests for feature extraction functions.
"""
import numpy as np
import pandas as pd
import pytest

from openimc.processing.feature_worker import extract_features_for_acquisition


@pytest.mark.unit
class TestFeatureExtraction:
    """Tests for feature extraction."""
    
    def test_extract_features_basic(self, sample_segmentation_mask, sample_image_stack_chw, sample_acquisition_info):
        """Test basic feature extraction."""
        # Convert CHW to HWC for feature extraction
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)
        
        selected_features = {
            'area_um2': True,
            'perimeter_um': True,
            'mean': True,
            'median': True
        }
        
        result = extract_features_for_acquisition(
            acq_id='test_1',
            mask=sample_segmentation_mask,
            selected_features=selected_features,
            acq_info=sample_acquisition_info,
            acq_label='Test',
            img_stack=img_stack_hwc,
            arcsinh_enabled=False,
            cofactor=10.0,
            denoise_source='None',
            custom_denoise_settings=None,
            spillover_config=None,
            source_file='test.mcd',
            excluded_channels=None
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'label' in result.columns
        assert 'area_um2' in result.columns or 'area' in result.columns
    
    def test_extract_features_with_arcsinh(self, sample_segmentation_mask, sample_image_stack_chw, sample_acquisition_info):
        """Test feature extraction with arcsinh transformation."""
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)
        
        selected_features = {
            'mean': True,
            'median': True
        }
        
        result = extract_features_for_acquisition(
            acq_id='test_1',
            mask=sample_segmentation_mask,
            selected_features=selected_features,
            acq_info=sample_acquisition_info,
            acq_label='Test',
            img_stack=img_stack_hwc,
            arcsinh_enabled=True,
            cofactor=10.0,
            denoise_source='None',
            custom_denoise_settings=None,
            spillover_config=None,
            source_file='test.mcd',
            excluded_channels=None
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_extract_features_morphological_only(self, sample_segmentation_mask, sample_image_stack_chw, sample_acquisition_info):
        """Test feature extraction with only morphological features."""
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)
        
        selected_features = {
            'area_um2': True,
            'perimeter_um': True,
            'eccentricity': True,
            'circularity': True
        }
        
        result = extract_features_for_acquisition(
            acq_id='test_1',
            mask=sample_segmentation_mask,
            selected_features=selected_features,
            acq_info=sample_acquisition_info,
            acq_label='Test',
            img_stack=img_stack_hwc,
            arcsinh_enabled=False,
            cofactor=10.0,
            denoise_source='None',
            custom_denoise_settings=None,
            spillover_config=None,
            source_file='test.mcd',
            excluded_channels=None
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_extract_features_intensity_only(self, sample_segmentation_mask, sample_image_stack_chw, sample_acquisition_info):
        """Test feature extraction with only intensity features."""
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)
        
        selected_features = {
            'mean': True,
            'median': True,
            'std': True,
            'integrated': True
        }
        
        result = extract_features_for_acquisition(
            acq_id='test_1',
            mask=sample_segmentation_mask,
            selected_features=selected_features,
            acq_info=sample_acquisition_info,
            acq_label='Test',
            img_stack=img_stack_hwc,
            arcsinh_enabled=False,
            cofactor=10.0,
            denoise_source='None',
            custom_denoise_settings=None,
            spillover_config=None,
            source_file='test.mcd',
            excluded_channels=None
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_extract_features_empty_mask(self, sample_image_stack_chw, sample_acquisition_info):
        """Test feature extraction with empty mask."""
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)
        empty_mask = np.zeros((100, 100), dtype=np.uint32)
        
        selected_features = {
            'mean': True
        }
        
        result = extract_features_for_acquisition(
            acq_id='test_1',
            mask=empty_mask,
            selected_features=selected_features,
            acq_info=sample_acquisition_info,
            acq_label='Test',
            img_stack=img_stack_hwc,
            arcsinh_enabled=False,
            cofactor=10.0,
            denoise_source='None',
            custom_denoise_settings=None,
            spillover_config=None,
            source_file='test.mcd',
            excluded_channels=None
        )
        
        # Should return empty dataframe or handle gracefully
        assert isinstance(result, pd.DataFrame)

