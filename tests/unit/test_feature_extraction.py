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
Unit tests for feature extraction functions.
"""
import numpy as np
import pandas as pd
import pytest

from openimc.processing.feature_worker import (
    extract_features_for_acquisition,
    drop_excluded_channel_feature_columns
)


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
        # Note: feature_worker renames 'label' to 'cell_id' (see feature_worker.py line 363)
        assert 'cell_id' in result.columns, "Features should have 'cell_id' column"
        assert 'area_um2' in result.columns or 'area' in result.columns
    
    def test_extract_features_with_arcsinh(self, sample_segmentation_mask, sample_image_stack_chw, sample_acquisition_info):
        """Test feature extraction with arcsinh transformation using core.extract_features."""
        import tempfile
        import tifffile
        from pathlib import Path
        from openimc.core import extract_features
        from openimc.data.mcd_loader import AcquisitionInfo, MCDLoader
        
        # Create a mock loader with the sample data
        class MockLoader:
            def __init__(self, img_stack_chw, channels):
                self.img_stack_chw = img_stack_chw
                self.channels = channels
            
            def get_channels(self, acq_id):
                return self.channels
            
            def get_all_channels(self, acq_id):
                # Return HWC format
                return np.moveaxis(self.img_stack_chw, 0, -1)
        
        # Create mock acquisition info
        channels = sample_acquisition_info['channels']
        acq_info = AcquisitionInfo(
            id='test_1',
            name='Test Acquisition',
            well=None,
            size=(100, 100),
            channels=channels,
            channel_metals=[],
            channel_labels=[],
            metadata={},
            source_file='test.mcd'
        )
        
        # Write mask to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = Path(tmpdir) / 'test_mask.tif'
            tifffile.imwrite(mask_path, sample_segmentation_mask.astype(np.uint32))
            
            # Create mock loader
            loader = MockLoader(sample_image_stack_chw, channels)
            
            # Extract features WITHOUT arcsinh
            result_no_arcsinh = extract_features(
                loader=loader,
                acquisitions=[acq_info],
                mask_path=mask_path,
                output_path=None,
                morphological=False,
                intensity=True,
                arcsinh=False,
                arcsinh_cofactor=5.0
            )
            
            # Extract features WITH arcsinh
            result_with_arcsinh = extract_features(
                loader=loader,
                acquisitions=[acq_info],
                mask_path=mask_path,
                output_path=None,
                morphological=False,
                intensity=True,
                arcsinh=True,
                arcsinh_cofactor=5.0
            )
        
        # Verify results
        assert isinstance(result_no_arcsinh, pd.DataFrame)
        assert isinstance(result_with_arcsinh, pd.DataFrame)
        assert len(result_no_arcsinh) > 0
        assert len(result_with_arcsinh) > 0
        
        # Find intensity columns
        intensity_cols = [col for col in result_no_arcsinh.columns 
                         if any(col.endswith(f'_{ft}') for ft in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated'])]
        
        assert len(intensity_cols) > 0, "Should have intensity columns"
        
        # Verify arcsinh transformation was applied
        # arcsinh(x/cofactor) should be different from x for positive values
        from openimc.ui.utils import arcsinh_normalize
        for col in intensity_cols[:3]:  # Check first 3 intensity columns
            original_vals = result_no_arcsinh[col].values
            arcsinh_vals = result_with_arcsinh[col].values
            expected_vals = arcsinh_normalize(original_vals, cofactor=5.0)
            
            # Values should be different (unless all zeros)
            if np.any(original_vals > 0):
                assert not np.allclose(original_vals, arcsinh_vals, rtol=0.01), \
                    f"Arcsinh should transform values in column {col}"
                # Verify transformation is correct
                assert np.allclose(arcsinh_vals, expected_vals, rtol=1e-5), \
                    f"Arcsinh values should match expected transformation for column {col}"
    
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

    def test_extract_features_respects_intensity_feature_selection(
        self,
        sample_segmentation_mask,
        sample_image_stack_chw,
        sample_acquisition_info,
    ):
        """Unselected intensity statistics should not appear in the output table."""
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)

        selected_features = {
            'area_um2': True,
            'mean': True,
            'median': False,
            'std': False,
            'mad': False,
            'p10': False,
            'p90': False,
            'integrated': True,
            'frac_pos': False,
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

        assert 'area_um2' in result.columns
        intensity_cols = [col for col in result.columns if col.endswith(('_mean', '_integrated', '_median', '_std', '_mad', '_p10', '_p90', '_frac_pos'))]
        assert intensity_cols
        assert all(col.endswith(('_mean', '_integrated')) for col in intensity_cols)
        assert not any(col.endswith(('_median', '_std', '_mad', '_p10', '_p90', '_frac_pos')) for col in intensity_cols)
    
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

    def test_extract_features_reports_touching_cells_and_exact_channel_statistics(self):
        """Feature extraction should preserve exact per-cell stats for simple synthetic data."""
        mask = np.zeros((6, 5), dtype=np.uint32)
        mask[0:2, :] = 1
        mask[2:5, 1:4] = 2

        marker = np.zeros((6, 5), dtype=np.float32)
        marker[0:2, :] = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        marker[2:5, 1:4] = 3.0
        img_stack_hwc = marker[..., np.newaxis]

        selected_features = {
            'touches_edge': True,
            'mean': True,
            'median': True,
            'std': True,
            'mad': True,
            'p10': True,
            'p90': True,
            'integrated': True,
            'frac_pos': True,
        }
        acq_info = {
            'channels': ['Marker1'],
            'well': 'A1',
        }

        result = extract_features_for_acquisition(
            acq_id='test_1',
            mask=mask,
            selected_features=selected_features,
            acq_info=acq_info,
            acq_label='Test',
            img_stack=img_stack_hwc,
            arcsinh_enabled=False,
            cofactor=10.0,
            denoise_source='None',
            custom_denoise_settings=None,
            spillover_config=None,
            source_file='synthetic.ome.tif',
            excluded_channels=None,
        ).sort_values('cell_id').reset_index(drop=True)

        assert result['cell_id'].tolist() == [1, 2]
        assert result['touches_edge'].tolist() == [True, False]

        cell1_pixels = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        cell2_pixels = np.full(9, 3.0)
        expected_by_cell = {
            1: {
                'mean': float(np.mean(cell1_pixels)),
                'median': float(np.median(cell1_pixels)),
                'std': float(np.std(cell1_pixels)),
                'mad': float(np.median(np.abs(cell1_pixels - np.median(cell1_pixels)))),
                'p10': float(np.percentile(cell1_pixels, 10)),
                'p90': float(np.percentile(cell1_pixels, 90)),
                'integrated': float(np.sum(cell1_pixels)),
                'frac_pos': float(np.count_nonzero(cell1_pixels > 0) / cell1_pixels.size),
            },
            2: {
                'mean': float(np.mean(cell2_pixels)),
                'median': float(np.median(cell2_pixels)),
                'std': float(np.std(cell2_pixels)),
                'mad': float(np.median(np.abs(cell2_pixels - np.median(cell2_pixels)))),
                'p10': float(np.percentile(cell2_pixels, 10)),
                'p90': float(np.percentile(cell2_pixels, 90)),
                'integrated': float(np.sum(cell2_pixels)),
                'frac_pos': float(np.count_nonzero(cell2_pixels > 0) / cell2_pixels.size),
            },
        }

        for cell_id, expected in expected_by_cell.items():
            row = result.loc[result['cell_id'] == cell_id].iloc[0]
            for suffix, expected_value in expected.items():
                assert np.isclose(row[f'Marker1_{suffix}'], expected_value)

    def test_drop_excluded_channel_columns_removes_schema_columns(self):
        """Excluded channels should not remain as columns (including all-NaN columns)."""
        df = pd.DataFrame(
            {
                'acquisition_id': ['a1', 'a1'],
                'cell_id': [1, 2],
                'CD3_mean': [1.0, 2.0],
                'CD3_median': [1.1, 2.1],
                'CD4_mean': [np.nan, np.nan],  # Simulates concat-generated NA column
                'CD4_p90': [np.nan, np.nan],
                'area_um2': [50.0, 60.0],
            }
        )

        out = drop_excluded_channel_feature_columns(df, excluded_channels={'CD4'})

        assert 'CD4_mean' not in out.columns
        assert 'CD4_p90' not in out.columns
        assert 'CD3_mean' in out.columns
        assert 'area_um2' in out.columns

    def test_feature_percentiles_use_numpy_linear_interpolation(self):
        mask = np.ones((2, 5), dtype=np.uint16)
        marker = np.arange(10, dtype=np.float32).reshape(2, 5)

        result = extract_features_for_acquisition(
            acq_id='test_1',
            mask=mask,
            selected_features={'p10': True, 'p90': True},
            acq_info={'channels': ['Marker1'], 'well': 'A1'},
            acq_label='Test',
            img_stack=marker[..., np.newaxis],
            arcsinh_enabled=False,
            cofactor=10.0,
            source_file='synthetic.ome.tif',
        )

        assert result.loc[0, 'Marker1_p10'] == pytest.approx(np.percentile(marker, 10))
        assert result.loc[0, 'Marker1_p90'] == pytest.approx(np.percentile(marker, 90))
        assert result.loc[0, 'Marker1_p10'] == pytest.approx(0.9)
        assert result.loc[0, 'Marker1_p90'] == pytest.approx(8.1)
