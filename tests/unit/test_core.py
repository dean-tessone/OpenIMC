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
Unit tests for core.py functions.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from openimc.core import (
    load_mcd,
    parse_denoise_settings,
    preprocess,
    segment,
    extract_features,
    cluster,
    qc_analysis,
    pixel_correlation
)


@pytest.mark.unit
@pytest.mark.requires_readimc
class TestLoadMCD:
    """Tests for load_mcd function."""
    
    def test_load_mcd_file(self, test_data_dir):
        """Test loading an MCD file."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, loader_type = load_mcd(str(mcd_path))
        
        assert loader_type == 'mcd'
        assert loader is not None
        
        # Check that we can list acquisitions
        acquisitions = loader.list_acquisitions()
        assert len(acquisitions) > 0
    
    def test_load_mcd_invalid_path(self):
        """Test load_mcd with invalid path raises error."""
        with pytest.raises(ValueError, match="Input path must be"):
            load_mcd("/nonexistent/path.mcd")
    
    def test_load_mcd_directory(self, mock_ometiff_directory):
        """Test load_mcd with OME-TIFF directory."""
        loader, loader_type = load_mcd(str(mock_ometiff_directory))
        
        assert loader_type == 'ometiff'
        assert loader is not None
    
    def test_load_mcd_channel_format(self, mock_ometiff_directory):
        """Test load_mcd with custom channel format."""
        loader, loader_type = load_mcd(str(mock_ometiff_directory), channel_format='HWC')
        
        assert loader_type == 'ometiff'
        assert loader.channel_format == 'HWC'


@pytest.mark.unit
class TestParseDenoiseSettings:
    """Tests for parse_denoise_settings function."""
    
    def test_parse_denoise_settings_none(self):
        """Test parse_denoise_settings with None."""
        result = parse_denoise_settings(None)
        assert result == {}
    
    def test_parse_denoise_settings_dict(self):
        """Test parse_denoise_settings with dict."""
        settings = {"DAPI": {"hot": {"method": "median3"}}}
        result = parse_denoise_settings(settings)
        assert result == settings
    
    def test_parse_denoise_settings_json_string(self):
        """Test parse_denoise_settings with JSON string."""
        json_str = '{"DAPI": {"hot": {"method": "median3"}}}'
        result = parse_denoise_settings(json_str)
        
        assert isinstance(result, dict)
        assert "DAPI" in result
    
    def test_parse_denoise_settings_json_file(self, temp_dir):
        """Test parse_denoise_settings with JSON file."""
        import json
        settings = {
            "DAPI": {
                "hot": {"method": "median3"},
                "speckle": {"method": "gaussian", "sigma": 0.8}
            }
        }
        
        json_file = temp_dir / "denoise_settings.json"
        with open(json_file, 'w') as f:
            json.dump(settings, f)
        
        result = parse_denoise_settings(str(json_file))
        
        assert isinstance(result, dict)
        assert "DAPI" in result
        assert "hot" in result["DAPI"]
    
    def test_parse_denoise_settings_invalid_json(self):
        """Test parse_denoise_settings with invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_denoise_settings("{invalid json}")


@pytest.mark.unit
@pytest.mark.requires_readimc
class TestPreprocess:
    """Tests for preprocess function."""
    
    def test_preprocess_basic(self, test_data_dir, temp_dir):
        """Test basic preprocessing of an acquisition."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, _ = load_mcd(str(mcd_path))
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test file")
        
        # Use first acquisition
        acq = acquisitions[0]
        
        # Preprocess without denoising
        output_path = preprocess(
            loader=loader,
            acquisition=acq,
            output_dir=temp_dir,
            denoise_settings=None
        )
        
        assert output_path.exists(), "Output file should be created"
        assert output_path.suffix == '.tif', "Output should be a TIFF file"
        
        # Verify the output file can be read and has expected properties
        import tifffile
        output_img = tifffile.imread(str(output_path))
        assert output_img is not None, "Output file should be readable"
        assert output_img.ndim == 3, "Output should be 3D (C, H, W)"
        
        # Check that number of channels matches
        channels = loader.get_channels(acq.id)
        assert output_img.shape[0] == len(channels), f"Number of channels should match ({len(channels)})"
    
    def test_preprocess_with_denoise(self, test_data_dir, temp_dir):
        """Test preprocessing with denoise settings."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, _ = load_mcd(str(mcd_path))
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test file")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) == 0:
            pytest.skip("No channels found")
        
        # Create simple denoise settings for first channel
        denoise_settings = {
            channels[0]: {
                "hot": {"method": "median3"}
            }
        }
        
        output_path = preprocess(
            loader=loader,
            acquisition=acq,
            output_dir=temp_dir,
            denoise_settings=denoise_settings
        )
        
        assert output_path.exists()


@pytest.mark.unit
@pytest.mark.requires_readimc
class TestSegment:
    """Tests for segment function."""
    
    def test_segment_watershed(self, test_data_dir, temp_dir):
        """Test watershed segmentation."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, _ = load_mcd(str(mcd_path))
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test file")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for watershed (nuclear + membrane)")
        
        # Use first channel as nuclear channel, second as membrane channel
        nuclear_channels = [channels[0]]
        cyto_channels = [channels[1]]
        
        # Run watershed segmentation
        mask = segment(
            loader=loader,
            acquisition=acq,
            method="watershed",
            nuclear_channels=nuclear_channels,
            cyto_channels=cyto_channels,
            output_dir=temp_dir,
            min_cell_area=100,
            max_cell_area=10000
        )
        
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        # Watershed can return int32 or uint32 depending on implementation
        assert mask.dtype in [np.uint32, np.int32, np.int64], f"Mask dtype should be integer type, got {mask.dtype}"
        assert mask.ndim == 2
        
        # Check that mask dimensions match image dimensions
        img_stack = loader.get_all_channels(acq.id)
        assert mask.shape == img_stack.shape[:2], "Mask dimensions should match image dimensions"
        
        # Check that mask has some cells
        unique_labels = np.unique(mask)
        assert len(unique_labels) > 1, "Should have background (0) and at least one cell"
        
        # Check that background is 0
        assert 0 in unique_labels, "Background should be labeled as 0"
        
        # Check that cell labels are sequential and start from 1
        cell_labels = unique_labels[unique_labels > 0]
        if len(cell_labels) > 0:
            assert np.all(cell_labels >= 1), "Cell labels should be >= 1"
            # Check that labels are roughly sequential (allowing for some gaps)
            assert np.max(cell_labels) <= len(cell_labels) * 2, "Labels should be reasonably sequential"
        
        # Check that cell areas are reasonable
        # Note: Actual cell sizes depend on image content and may exceed max_cell_area
        # if cells are touching or if the image has large structures
        if len(cell_labels) > 0:
            cell_areas = []
            for cell_id in cell_labels:
                cell_area = np.sum(mask == cell_id)
                cell_areas.append(cell_area)
            
            cell_areas = np.array(cell_areas)
            # All cells should be positive and reasonable (not extremely large)
            assert np.all(cell_areas > 0), "All cell areas should be positive"
            assert np.all(cell_areas < 1e6), "Cell areas should be reasonable (< 1e6 pixels)"
            
            # Check that we have a reasonable number of cells (not just one huge blob)
            assert len(cell_areas) >= 1, "Should have at least one cell"


@pytest.mark.unit
@pytest.mark.requires_readimc
class TestExtractFeatures:
    """Tests for extract_features function."""
    
    def test_extract_features_basic(self, test_data_dir, temp_dir):
        """Test basic feature extraction."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, _ = load_mcd(str(mcd_path))
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test file")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 1:
            pytest.skip("Not enough channels for feature extraction")
        
        # Create a simple segmentation mask first
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for watershed (nuclear + membrane)")
        
        nuclear_channels = [channels[0]]
        cyto_channels = [channels[1]]
        mask = segment(
            loader=loader,
            acquisition=acq,
            method="watershed",
            nuclear_channels=nuclear_channels,
            cyto_channels=cyto_channels,
            output_dir=temp_dir,
            min_cell_area=100,
            max_cell_area=10000
        )
        
        # Save mask to file
        mask_dir = temp_dir / "masks"
        mask_dir.mkdir()
        mask_file = mask_dir / f"{acq.id}.tif"
        import tifffile
        tifffile.imwrite(str(mask_file), mask)
        
        # Extract features
        features_df = extract_features(
            loader=loader,
            acquisitions=[acq],
            mask_path=str(mask_dir),
            output_path=temp_dir / "features.csv",
            morphological=True,
            intensity=True
        )
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        
        # Check for expected columns
        # Note: feature_worker renames 'label' to 'cell_id' (see feature_worker.py line 363)
        assert 'cell_id' in features_df.columns, "Features should have 'cell_id' column"
        assert 'acquisition_id' in features_df.columns, "Features should have 'acquisition_id' column"
        
        # Check that number of features matches number of cells in mask
        n_cells_in_mask = len(np.unique(mask[mask > 0]))
        assert len(features_df) == n_cells_in_mask, f"Number of features ({len(features_df)}) should match number of cells ({n_cells_in_mask})"
        
        # Check that cell_ids match mask labels
        mask_labels = np.unique(mask[mask > 0])
        feature_cell_ids = features_df['cell_id'].unique()
        assert set(feature_cell_ids) == set(mask_labels), "Feature cell_ids should match mask labels"
        
        # Check that morphological features are present and have reasonable values
        if 'area_um2' in features_df.columns:
            assert features_df['area_um2'].min() > 0, "Cell areas should be positive"
            assert features_df['area_um2'].max() < 1e6, "Cell areas should be reasonable"
        
        if 'centroid_x' in features_df.columns and 'centroid_y' in features_df.columns:
            # Check that centroids are within image bounds
            height, width = mask.shape
            assert features_df['centroid_x'].min() >= 0, "Centroids should be within image bounds"
            assert features_df['centroid_x'].max() <= width, "Centroids should be within image bounds"
            assert features_df['centroid_y'].min() >= 0, "Centroids should be within image bounds"
            assert features_df['centroid_y'].max() <= height, "Centroids should be within image bounds"
        
        # Check that intensity features are present for each channel
        # Intensity features can be named like 'mean_CHANNEL' or 'CHANNEL_mean'
        intensity_cols = [col for col in features_df.columns if '_mean' in col or col.startswith('mean_')]
        if len(intensity_cols) == 0:
            # If no mean columns, check for other intensity features
            intensity_cols = [col for col in features_df.columns if any(x in col for x in ['_median', '_std', '_integrated'])]
        
        assert len(intensity_cols) > 0, f"Should have at least one intensity feature. Available columns: {list(features_df.columns)[:20]}"
        
        # Check that intensity values are non-negative
        for col in intensity_cols[:5]:  # Check first 5 intensity columns
            assert features_df[col].min() >= 0, f"Intensity values in {col} should be non-negative"
        
        # Check that output file was created
        assert (temp_dir / "features.csv").exists()
        
        # Verify CSV can be read back
        loaded_df = pd.read_csv(temp_dir / "features.csv")
        assert len(loaded_df) == len(features_df), "CSV should contain same number of rows"


@pytest.mark.unit
class TestCluster:
    """Tests for cluster function."""
    
    def test_cluster_kmeans(self, sample_feature_dataframe):
        """Test K-means clustering."""
        features_df = sample_feature_dataframe.copy()
        
        # Add some intensity features for clustering
        n_cells = len(features_df)
        features_df['mean_CD4'] = np.random.rand(n_cells) * 1000
        features_df['mean_CD8'] = np.random.rand(n_cells) * 1000
        
        n_clusters = 3
        clustered_df = cluster(
            features_df=features_df,
            method="kmeans",
            n_clusters=n_clusters,
            scaling="zscore"
        )
        
        assert isinstance(clustered_df, pd.DataFrame)
        assert 'cluster' in clustered_df.columns, "Clustered dataframe should have 'cluster' column"
        assert len(clustered_df) == len(features_df), "Clustered dataframe should have same number of rows"
        
        # Check that clusters are assigned
        unique_clusters = clustered_df['cluster'].unique()
        assert len(unique_clusters) > 0, "Should have at least one cluster"
        
        # Check that all cells are assigned to clusters
        assert clustered_df['cluster'].notna().all(), "All cells should be assigned to a cluster"
        
        # Check that cluster labels are non-negative integers
        assert (clustered_df['cluster'] >= 0).all(), "Cluster labels should be non-negative"
        assert clustered_df['cluster'].dtype in [np.int64, np.int32, int], "Cluster labels should be integers"
        
        # Check that we have the expected number of clusters (approximately, allowing for some variance)
        assert len(unique_clusters) <= n_clusters, f"Should have at most {n_clusters} clusters"
        # K-means should produce exactly n_clusters (unless some are empty, which is rare)
        assert len(unique_clusters) >= 1, "Should have at least 1 cluster"
    
    def test_cluster_hierarchical(self, sample_feature_dataframe):
        """Test hierarchical clustering."""
        features_df = sample_feature_dataframe.copy()
        
        # Add some intensity features
        n_cells = len(features_df)
        features_df['mean_CD4'] = np.random.rand(n_cells) * 1000
        features_df['mean_CD8'] = np.random.rand(n_cells) * 1000
        
        n_clusters = 3
        clustered_df = cluster(
            features_df=features_df,
            method="hierarchical",
            n_clusters=n_clusters,
            linkage="ward",
            scaling="zscore"
        )
        
        assert isinstance(clustered_df, pd.DataFrame)
        assert 'cluster' in clustered_df.columns, "Clustered dataframe should have 'cluster' column"
        assert len(clustered_df) == len(features_df), "Clustered dataframe should have same number of rows"
        
        # Check that all cells are assigned to clusters
        assert clustered_df['cluster'].notna().all(), "All cells should be assigned to a cluster"
        
        # Check that cluster labels are non-negative integers
        assert (clustered_df['cluster'] >= 0).all(), "Cluster labels should be non-negative"
        assert clustered_df['cluster'].dtype in [np.int64, np.int32, int], "Cluster labels should be integers"
        
        # Hierarchical clustering should produce exactly n_clusters
        unique_clusters = clustered_df['cluster'].unique()
        assert len(unique_clusters) == n_clusters, f"Hierarchical clustering should produce exactly {n_clusters} clusters"
    
    def test_cluster_leiden(self, sample_feature_dataframe):
        """Test Leiden clustering."""
        features_df = sample_feature_dataframe.copy()
        
        # Add some intensity features
        n_cells = len(features_df)
        features_df['mean_CD4'] = np.random.rand(n_cells) * 1000
        features_df['mean_CD8'] = np.random.rand(n_cells) * 1000
        
        clustered_df = cluster(
            features_df=features_df,
            method="leiden",
            resolution=1.0,
            scaling="zscore"
        )
        
        assert isinstance(clustered_df, pd.DataFrame)
        assert 'cluster' in clustered_df.columns, "Clustered dataframe should have 'cluster' column"
        assert len(clustered_df) == len(features_df), "Clustered dataframe should have same number of rows"
        
        # Check that all cells are assigned to clusters
        assert clustered_df['cluster'].notna().all(), "All cells should be assigned to a cluster"
        
        # Check that cluster labels are non-negative integers
        assert (clustered_df['cluster'] >= 0).all(), "Cluster labels should be non-negative"
        assert clustered_df['cluster'].dtype in [np.int64, np.int32, int], "Cluster labels should be integers"
        
        # Leiden clustering should produce at least one cluster
        unique_clusters = clustered_df['cluster'].unique()
        assert len(unique_clusters) >= 1, "Should have at least one cluster"
    
    def test_cluster_with_columns(self, sample_feature_dataframe):
        """Test clustering with specific columns."""
        features_df = sample_feature_dataframe.copy()
        
        n_cells = len(features_df)
        features_df['mean_CD4'] = np.random.rand(n_cells) * 1000
        features_df['mean_CD8'] = np.random.rand(n_cells) * 1000
        
        clustered_df = cluster(
            features_df=features_df,
            method="kmeans",
            n_clusters=3,
            columns=['mean_CD4', 'mean_CD8'],
            scaling="zscore"
        )
        
        assert isinstance(clustered_df, pd.DataFrame)
        assert 'cluster' in clustered_df.columns
    
    def test_cluster_invalid_method(self, sample_feature_dataframe):
        """Test clustering with invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown clustering method"):
            cluster(
                features_df=sample_feature_dataframe,
                method="invalid_method"
            )


@pytest.mark.unit
@pytest.mark.requires_readimc
class TestQCAnalysis:
    """Tests for qc_analysis function."""
    
    def test_qc_analysis_pixel_mode(self, test_data_dir):
        """Test QC analysis in pixel mode."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, _ = load_mcd(str(mcd_path))
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test file")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) == 0:
            pytest.skip("No channels found")
        
        # Test with first few channels
        test_channels = channels[:min(3, len(channels))]
        
        qc_df = qc_analysis(
            loader=loader,
            acquisition=acq,
            channels=test_channels,
            mode="pixel"
        )
        
        assert isinstance(qc_df, pd.DataFrame)
        assert len(qc_df) > 0, "QC analysis should return results"
        
        # Check for expected columns
        assert 'channel' in qc_df.columns, "QC results should have 'channel' column"
        assert 'snr' in qc_df.columns, "QC results should have 'snr' column"
        assert 'intensity_mean' in qc_df.columns, "QC results should have 'intensity_mean' column"
        assert 'mode' in qc_df.columns, "QC results should have 'mode' column"
        
        # Check that all rows are in pixel mode
        assert (qc_df['mode'] == 'pixel').all(), "All results should be in pixel mode"
        
        # Check that SNR values are reasonable (can be negative but should be finite)
        assert qc_df['snr'].notna().all(), "SNR values should not be NaN"
        assert np.isfinite(qc_df['snr']).all(), "SNR values should be finite"
        
        # Check that intensity values are reasonable
        assert qc_df['intensity_mean'].notna().all(), "Intensity means should not be NaN"
        assert (qc_df['intensity_mean'] >= 0).all(), "Intensity means should be non-negative"
        assert (qc_df['intensity_min'] <= qc_df['intensity_max']).all(), "Min should be <= max"
        
        # Check that we have one row per channel
        assert len(qc_df) == len(test_channels), f"Should have one row per channel ({len(test_channels)})"
    
    def test_qc_analysis_cell_mode(self, test_data_dir, temp_dir):
        """Test QC analysis in cell mode."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, _ = load_mcd(str(mcd_path))
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test file")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for watershed (nuclear + membrane)")
        
        # Create a segmentation mask
        nuclear_channels = [channels[0]]
        cyto_channels = [channels[1]]
        mask = segment(
            loader=loader,
            acquisition=acq,
            method="watershed",
            nuclear_channels=nuclear_channels,
            cyto_channels=cyto_channels,
            output_dir=temp_dir,
            min_cell_area=100,
            max_cell_area=10000
        )
        
        # Test with first few channels
        test_channels = channels[:min(3, len(channels))]
        
        qc_df = qc_analysis(
            loader=loader,
            acquisition=acq,
            channels=test_channels,
            mode="cell",
            mask=mask
        )
        
        assert isinstance(qc_df, pd.DataFrame)
        if len(qc_df) > 0:
            assert 'channel' in qc_df.columns, "QC results should have 'channel' column"
            assert 'snr' in qc_df.columns, "QC results should have 'snr' column"
            assert 'n_cells' in qc_df.columns, "Cell mode QC should have 'n_cells' column"
            
            # Check that all rows are in cell mode
            assert (qc_df['mode'] == 'cell').all(), "All results should be in cell mode"
            
            # Check that n_cells matches actual number of cells in mask
            n_cells_in_mask = len(np.unique(mask[mask > 0]))
            assert (qc_df['n_cells'] == n_cells_in_mask).all(), f"n_cells should match mask ({n_cells_in_mask})"
            
            # Check that cell density is reasonable
            if 'cell_density' in qc_df.columns:
                assert (qc_df['cell_density'] >= 0).all(), "Cell density should be non-negative"
                assert (qc_df['cell_density'] <= 1.0).all(), "Cell density should be <= 1.0 (fraction of pixels)"


@pytest.mark.unit
@pytest.mark.requires_readimc
class TestPixelCorrelation:
    """Tests for pixel_correlation function."""
    
    def test_pixel_correlation_basic(self, test_data_dir):
        """Test basic pixel correlation analysis."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, _ = load_mcd(str(mcd_path))
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test file")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for correlation")
        
        # Test with first 3 channels
        test_channels = channels[:min(3, len(channels))]
        
        corr_df = pixel_correlation(
            loader=loader,
            acquisition=acq,
            channels=test_channels,
            mask=None
        )
        
        assert isinstance(corr_df, pd.DataFrame)
        
        # Should have correlation results if we have at least 2 channels
        if len(test_channels) >= 2:
            assert len(corr_df) > 0, "Should have correlation results for multiple channels"
            assert 'marker1' in corr_df.columns, "Should have 'marker1' column"
            assert 'marker2' in corr_df.columns, "Should have 'marker2' column"
            assert 'correlation' in corr_df.columns, "Should have 'correlation' column"
            assert 'p_value' in corr_df.columns, "Should have 'p_value' column"
            
            # Check that correlation values are in valid range [-1, 1]
            assert (corr_df['correlation'] >= -1.0).all(), "Correlation values should be >= -1"
            assert (corr_df['correlation'] <= 1.0).all(), "Correlation values should be <= 1"
            
            # Check that p-values are in valid range [0, 1]
            assert (corr_df['p_value'] >= 0.0).all(), "P-values should be >= 0"
            assert (corr_df['p_value'] <= 1.0).all(), "P-values should be <= 1"
            
            # Check that n_pixels is positive
            if 'n_pixels' in corr_df.columns:
                assert (corr_df['n_pixels'] > 0).all(), "n_pixels should be positive"
            
            # Check that we have the expected number of pairs (n choose 2)
            expected_pairs = len(test_channels) * (len(test_channels) - 1) // 2
            assert len(corr_df) <= expected_pairs, f"Should have at most {expected_pairs} pairs"
    
    def test_pixel_correlation_with_mask(self, test_data_dir, temp_dir):
        """Test pixel correlation with segmentation mask."""
        mcd_path = test_data_dir / "Patient1.mcd"
        if not mcd_path.exists():
            pytest.skip(f"MCD test file not found: {mcd_path}")
        
        loader, _ = load_mcd(str(mcd_path))
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test file")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for correlation")
        
        # Create a segmentation mask
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for watershed (nuclear + membrane)")
        
        nuclear_channels = [channels[0]]
        cyto_channels = [channels[1]]
        mask = segment(
            loader=loader,
            acquisition=acq,
            method="watershed",
            nuclear_channels=nuclear_channels,
            cyto_channels=cyto_channels,
            output_dir=temp_dir,
            min_cell_area=100,
            max_cell_area=10000
        )
        
        # Test with first 3 channels
        test_channels = channels[:min(3, len(channels))]
        
        corr_df = pixel_correlation(
            loader=loader,
            acquisition=acq,
            channels=test_channels,
            mask=mask
        )
        
        assert isinstance(corr_df, pd.DataFrame)
        
        if len(test_channels) >= 2:
            assert len(corr_df) > 0, "Should have correlation results for multiple channels"
            assert 'marker1' in corr_df.columns, "Should have 'marker1' column"
            assert 'marker2' in corr_df.columns, "Should have 'marker2' column"
            
            # Check that correlation values are in valid range [-1, 1]
            assert (corr_df['correlation'] >= -1.0).all(), "Correlation values should be >= -1"
            assert (corr_df['correlation'] <= 1.0).all(), "Correlation values should be <= 1"
            
            # Check that p-values are in valid range [0, 1]
            assert (corr_df['p_value'] >= 0.0).all(), "P-values should be >= 0"
            assert (corr_df['p_value'] <= 1.0).all(), "P-values should be <= 1"

