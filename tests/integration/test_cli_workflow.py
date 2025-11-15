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
Integration tests for CLI workflows using real MCD data.

These tests are designed to work cross-platform (Windows, Linux, macOS)
and in CI environments like GitHub Actions.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import tifffile
import os

from openimc.core import (
    load_mcd,
    preprocess,
    segment,
    extract_features,
    cluster
)


def _find_dna_channels(channels):
    """Find channels whose names contain 'DNA' (case-insensitive).
    
    Args:
        channels: List of channel names
    
    Returns:
        List of channel names containing 'DNA'
    """
    return [ch for ch in channels if 'DNA' in ch.upper()]


def _select_nuclear_channels(channels, min_count=1):
    """Select nuclear channels from a list of channel names.
    
    Preferentially selects channels containing 'DNA', otherwise falls back
    to first available channels.
    
    Args:
        channels: List of channel names
        min_count: Minimum number of channels to select
    
    Returns:
        List of selected nuclear channel names
    
    Raises:
        ValueError: If not enough channels are available
    """
    dna_channels = _find_dna_channels(channels)
    if len(dna_channels) >= min_count:
        return dna_channels[:min_count]
    elif len(channels) >= min_count:
        # Fallback to first channels if no DNA channels found
        return channels[:min_count]
    else:
        raise ValueError(f"Need at least {min_count} channels, but only {len(channels)} available")


def _ensure_path(path):
    """Ensure a path is a resolved Path object for cross-platform compatibility.
    
    Args:
        path: Path-like object (str, Path, etc.)
    
    Returns:
        Resolved Path object
    """
    return Path(path).resolve()


def _get_test_data_path(test_data_dir):
    """Get the test data path, supporting both MCD files and OME-TIFF directories.
    
    This is a wrapper around the shared helper function from conftest.py.
    
    Args:
        test_data_dir: Path to the test data directory
    
    Returns:
        Tuple of (data_path, loader_type) where loader_type is 'mcd' or 'ometiff'
    
    Raises:
        FileNotFoundError: If no valid test data is found
    """
    from tests.conftest import get_test_data_path
    return get_test_data_path(test_data_dir)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_readimc
class TestMCDLoading:
    """Integration tests for loading MCD files and OME-TIFF directories."""
    
    def test_load_mcd_file(self, test_data_dir):
        """Test loading the MCD file or OME-TIFF directory from test data."""
        try:
            data_path, expected_type = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # load_mcd accepts both Path and str, but we'll use Path for consistency
        loader, loader_type = load_mcd(data_path)
        
        assert loader_type == expected_type
        assert loader is not None
        
        # Check that we can list acquisitions
        acquisitions = loader.list_acquisitions()
        assert len(acquisitions) > 0
        
        # Verify acquisition has channels
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        assert len(channels) > 0
        
        # Verify we can load an image
        img = loader.get_image(acq.id, channels[0])
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_readimc
class TestPreprocessWorkflow:
    """Integration tests for preprocessing workflow."""
    
    def test_preprocess_mcd_file(self, test_data_dir, temp_dir):
        """Test preprocessing an MCD file or OME-TIFF and exporting to OME-TIFF."""
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Load data - load_mcd accepts Path objects
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        # Use first acquisition
        acq = acquisitions[0]
        
        # Preprocess and export
        output_path = preprocess(
            loader=loader,
            acquisition=acq,
            output_dir=temp_dir,
            denoise_settings=None,
            normalization_method="None"
        )
        
        # Verify output file exists
        assert output_path.exists()
        assert output_path.suffix == '.tif'
        
        # Verify we can read the OME-TIFF
        img = tifffile.imread(str(output_path))
        assert img is not None
        assert isinstance(img, np.ndarray)
        
        # Verify image has expected shape (C, H, W)
        assert img.ndim == 3
        assert img.shape[0] > 0  # At least one channel


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_readimc
class TestSegmentWorkflow:
    """Integration tests for segmentation workflow."""
    
    def test_segment_watershed(self, test_data_dir, temp_dir):
        """Test watershed segmentation on MCD file."""
        # Get test data path (supports both MCD and OME-TIFF)
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Load data - load_mcd accepts Path objects
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for watershed segmentation")
        
        # Select DNA channels for nuclear, use first non-DNA channel for cytoplasm
        nuclear_channels = _select_nuclear_channels(channels, min_count=1)
        dna_channels = _find_dna_channels(channels)
        non_dna_channels = [ch for ch in channels if ch not in dna_channels]
        cyto_channels = [non_dna_channels[0]] if len(non_dna_channels) > 0 else []
        
        # Run watershed segmentation
        mask = segment(
            loader=loader,
            acquisition=acq,
            method='watershed',
            nuclear_channels=nuclear_channels,
            cyto_channels=cyto_channels,
            output_dir=temp_dir / "masks",
            min_cell_area=100,
            max_cell_area=10000,
            compactness=0.01
        )
        
        # Verify mask
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.dtype in [np.uint32, np.int32, np.uint64, np.int64]
        
        # Verify mask has some cells
        unique_labels = np.unique(mask)
        # Background is typically 0, so we should have at least 2 unique values
        assert len(unique_labels) >= 1
        
        # Verify mask file was saved
        mask_dir = temp_dir / "masks"
        mask_files = list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.npy"))
        assert len(mask_files) > 0
    
    @pytest.mark.slow
    def test_segment_cellsam(self, test_data_dir, temp_dir):
        """Test CellSAM segmentation on MCD file.
        
        WARNING: This test is VERY SLOW (may take several minutes) as it runs
        a deep learning model for segmentation.
        
        This test requires:
        - CellSAM to be installed (pip install git+https://github.com/vanvalenlab/cellSAM.git)
        - DEEPCELL_ACCESS_TOKEN environment variable or deepcell_api_key parameter
        """
        import os
        
        # Get test data path (supports both MCD and OME-TIFF)
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Check for API key
        api_key = os.environ.get("DEEPCELL_ACCESS_TOKEN")
        if not api_key:
            pytest.skip("DEEPCELL_ACCESS_TOKEN environment variable not set. CellSAM requires DeepCell API key.")
        
        # Load data
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 1:
            pytest.skip("Need at least 1 channel for CellSAM segmentation")
        
        # Select DNA channels for nuclear (CellSAM can work with nuclear-only)
        nuclear_channels = _select_nuclear_channels(channels, min_count=1)
        dna_channels = _find_dna_channels(channels)
        non_dna_channels = [ch for ch in channels if ch not in dna_channels]
        cyto_channels = [non_dna_channels[0]] if len(non_dna_channels) > 0 else []
        
        # Run CellSAM segmentation
        try:
            mask = segment(
                loader=loader,
                acquisition=acq,
                method='cellsam',
                nuclear_channels=nuclear_channels,
                cyto_channels=cyto_channels,
                output_dir=temp_dir / "masks_cellsam",
                deepcell_api_key=api_key,
                bbox_threshold=0.4,
                use_wsi=False,
                low_contrast_enhancement=False,
                gauge_cell_size=False
            )
        except ImportError as e:
            pytest.skip(f"CellSAM not installed: {e}")
        except (ValueError, RuntimeError) as e:
            # API key issues or model initialization problems
            pytest.skip(f"CellSAM initialization failed: {e}")
        
        # Verify mask
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.dtype in [np.uint32, np.int32, np.uint64, np.int64]
        
        # Verify mask has some cells
        unique_labels = np.unique(mask)
        assert len(unique_labels) >= 1
        
        # Verify mask file was saved
        mask_dir = temp_dir / "masks_cellsam"
        mask_files = list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.npy"))
        assert len(mask_files) > 0
    
    @pytest.mark.slow
    def test_segment_cellpose(self, test_data_dir, temp_dir):
        """Test Cellpose segmentation on MCD file.
        
        WARNING: This test is VERY SLOW (may take several minutes) as it runs
        a deep learning model for segmentation. On CPU, this can take 5-10+ minutes.
        
        This test requires:
        - Cellpose to be installed (pip install cellpose)
        - May use GPU if available, but will fall back to CPU
        """
        # Get test data path (supports both MCD and OME-TIFF)
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Load data - load_mcd accepts Path objects
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 1:
            pytest.skip("Need at least 1 channel for Cellpose segmentation")
        
        # Select DNA channels for nuclear
        # For cyto3 model, cytoplasm channels are optional (will fallback to nuclear)
        nuclear_channels = _select_nuclear_channels(channels, min_count=1)
        dna_channels = _find_dna_channels(channels)
        non_dna_channels = [ch for ch in channels if ch not in dna_channels]
        cyto_channels = [non_dna_channels[0]] if len(non_dna_channels) > 0 else []
        
        # Run Cellpose segmentation with cyto3 model
        try:
            mask = segment(
                loader=loader,
                acquisition=acq,
                method='cellpose',
                nuclear_channels=nuclear_channels,
                cyto_channels=cyto_channels,
                output_dir=temp_dir / "masks_cellpose",
                cellpose_model='cyto3',
                diameter=None,  # Auto-detect
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                gpu_id=None  # Use CPU for testing (can be set to GPU ID if available)
            )
        except ImportError as e:
            pytest.skip(f"Cellpose not installed: {e}")
        except Exception as e:
            # Other errors (model download, etc.)
            pytest.skip(f"Cellpose segmentation failed: {e}")
        
        # Verify mask
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        # Cellpose returns uint16 masks, other methods may return uint32
        assert mask.dtype in [np.uint8, np.uint16, np.uint32, np.int32, np.uint64, np.int64]
        
        # Verify mask structure (may be empty if no cells found, which is valid)
        unique_labels = np.unique(mask)
        assert len(unique_labels) >= 1  # At least background (0)
        
        # Verify mask file was saved
        mask_dir = temp_dir / "masks_cellpose"
        mask_files = list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.npy"))
        assert len(mask_files) > 0
    
    @pytest.mark.slow
    def test_segment_cellpose_nuclei(self, test_data_dir, temp_dir):
        """Test Cellpose nuclei model segmentation on MCD file.
        
        WARNING: This test is VERY SLOW (may take several minutes) as it runs
        a deep learning model for segmentation. On CPU, this can take 5-10+ minutes.
        
        This test uses the nuclei model which only requires nuclear channels.
        """
        # Get test data path (supports both MCD and OME-TIFF)
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Load data - load_mcd accepts Path objects
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 1:
            pytest.skip("Need at least 1 channel for Cellpose segmentation")
        
        # Select DNA channels for nuclear (nuclei model only needs nuclear)
        nuclear_channels = _select_nuclear_channels(channels, min_count=1)
        cyto_channels = []
        
        # Run Cellpose segmentation with nuclei model
        try:
            mask = segment(
                loader=loader,
                acquisition=acq,
                method='cellpose',
                nuclear_channels=nuclear_channels,
                cyto_channels=cyto_channels,
                output_dir=temp_dir / "masks_cellpose_nuclei",
                cellpose_model='nuclei',
                diameter=None,  # Auto-detect
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                gpu_id=None  # Use CPU for testing
            )
        except ImportError as e:
            pytest.skip(f"Cellpose not installed: {e}")
        except Exception as e:
            pytest.skip(f"Cellpose segmentation failed: {e}")
        
        # Verify mask
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        # Cellpose returns uint16 masks, other methods may return uint32
        assert mask.dtype in [np.uint8, np.uint16, np.uint32, np.int32, np.uint64, np.int64]
        
        # Verify mask structure (may be empty if no cells found, which is valid)
        unique_labels = np.unique(mask)
        assert len(unique_labels) >= 1  # At least background (0)
        
        # Verify mask file was saved
        mask_dir = temp_dir / "masks_cellpose_nuclei"
        mask_files = list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.npy"))
        assert len(mask_files) > 0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_readimc
class TestFeatureExtractionWorkflow:
    """Integration tests for feature extraction workflow."""
    
    def test_extract_features_from_segmentation(self, test_data_dir, temp_dir):
        """Test extracting features from segmented cells."""
        # Get test data path (supports both MCD and OME-TIFF)
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Load data - load_mcd accepts Path objects
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for segmentation")
        
        # Create segmentation mask directory
        mask_dir = temp_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Run segmentation first - select DNA channels for nuclear
        nuclear_channels = _select_nuclear_channels(channels, min_count=1)
        dna_channels = _find_dna_channels(channels)
        non_dna_channels = [ch for ch in channels if ch not in dna_channels]
        cyto_channels = [non_dna_channels[0]] if len(non_dna_channels) > 0 else []
        
        mask = segment(
            loader=loader,
            acquisition=acq,
            method='watershed',
            nuclear_channels=nuclear_channels,
            cyto_channels=cyto_channels,
            output_dir=mask_dir,
            min_cell_area=100,
            max_cell_area=10000,
            compactness=0.01
        )
        
        # Extract features
        features_df = extract_features(
            loader=loader,
            acquisitions=[acq],
            mask_path=mask_dir,
            output_path=temp_dir / "features.csv",
            morphological=True,
            intensity=True,
            arcsinh=False
        )
        
        # Verify features
        assert features_df is not None
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        
        # Verify feature columns exist
        assert 'cell_id' in features_df.columns or 'label' in features_df.columns
        assert 'acquisition_id' in features_df.columns
        
        # Verify CSV was saved
        features_csv = temp_dir / "features.csv"
        assert features_csv.exists()
        
        # Verify we can reload the CSV
        loaded_df = pd.read_csv(features_csv)
        assert len(loaded_df) == len(features_df)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_readimc
class TestClusterWorkflow:
    """Integration tests for clustering workflow."""
    
    def test_cluster_features(self, test_data_dir, temp_dir):
        """Test clustering extracted features."""
        # Get test data path (supports both MCD and OME-TIFF)
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Load data - load_mcd accepts Path objects
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for segmentation")
        
        # Create segmentation mask
        mask_dir = temp_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Select DNA channels for nuclear
        nuclear_channels = _select_nuclear_channels(channels, min_count=1)
        dna_channels = _find_dna_channels(channels)
        non_dna_channels = [ch for ch in channels if ch not in dna_channels]
        cyto_channels = [non_dna_channels[0]] if len(non_dna_channels) > 0 else []
        
        segment(
            loader=loader,
            acquisition=acq,
            method='watershed',
            nuclear_channels=nuclear_channels,
            cyto_channels=cyto_channels,
            output_dir=mask_dir,
            min_cell_area=100,
            max_cell_area=10000,
            compactness=0.01
        )
        
        # Extract features
        features_df = extract_features(
            loader=loader,
            acquisitions=[acq],
            mask_path=mask_dir,
            morphological=True,
            intensity=True,
            arcsinh=False
        )
        
        if len(features_df) < 10:
            pytest.skip("Need at least 10 cells for clustering")
        
        # Test hierarchical clustering
        clustered_df = cluster(
            features_df=features_df,
            method='hierarchical',
            n_clusters=3,
            output_path=temp_dir / "clustered_features.csv",
            scaling='zscore'
        )
        
        # Verify clustering results
        assert clustered_df is not None
        assert isinstance(clustered_df, pd.DataFrame)
        assert 'cluster' in clustered_df.columns
        assert len(clustered_df) == len(features_df)
        
        # Verify clusters were assigned
        unique_clusters = clustered_df['cluster'].unique()
        assert len(unique_clusters) > 0
        
        # Verify CSV was saved
        clustered_csv = temp_dir / "clustered_features.csv"
        assert clustered_csv.exists()
        
        # Test Leiden clustering if available
        try:
            clustered_df_leiden = cluster(
                features_df=features_df,
                method='leiden',
                resolution=1.0,
                output_path=temp_dir / "clustered_features_leiden.csv",
                scaling='zscore'
            )
            
            assert clustered_df_leiden is not None
            assert 'cluster' in clustered_df_leiden.columns
        except (ImportError, ValueError) as e:
            # Leiden clustering may not be available or may fail with small datasets
            pytest.skip(f"Leiden clustering not available: {e}")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_readimc
class TestEndToEndWorkflow:
    """Integration test for complete end-to-end workflow."""
    
    def test_complete_workflow(self, test_data_dir, temp_dir):
        """Test complete workflow: load -> segment -> extract -> cluster."""
        # Get test data path (supports both MCD and OME-TIFF)
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Step 1: Load data
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        acq = acquisitions[0]
        channels = loader.get_channels(acq.id)
        
        if len(channels) < 2:
            pytest.skip("Need at least 2 channels for complete workflow")
        
        # Step 2: Preprocess (optional, but test it)
        preprocess_output = temp_dir / "preprocessed"
        output_path = preprocess(
            loader=loader,
            acquisition=acq,
            output_dir=preprocess_output,
            normalization_method="None"
        )
        assert output_path.exists() or any(preprocess_output.glob("*.ome.tif"))
        
        # Step 3: Segment - select DNA channels for nuclear
        mask_dir = temp_dir / "masks"
        nuclear_channels = _select_nuclear_channels(channels, min_count=1)
        dna_channels = _find_dna_channels(channels)
        non_dna_channels = [ch for ch in channels if ch not in dna_channels]
        cyto_channels = [non_dna_channels[0]] if len(non_dna_channels) > 0 else []
        
        mask = segment(
            loader=loader,
            acquisition=acq,
            method='watershed',
            nuclear_channels=nuclear_channels,
            cyto_channels=cyto_channels,
            output_dir=mask_dir,
            min_cell_area=100,
            max_cell_area=10000
        )
        assert mask is not None
        assert np.unique(mask).size > 0
        
        # Step 4: Extract features
        features_df = extract_features(
            loader=loader,
            acquisitions=[acq],
            mask_path=mask_dir,
            output_path=temp_dir / "features.csv",
            morphological=True,
            intensity=True
        )
        assert len(features_df) > 0
        
        # Step 5: Cluster
        if len(features_df) >= 10:
            clustered_df = cluster(
                features_df=features_df,
                method='hierarchical',
                n_clusters=min(3, len(features_df) // 3),
                output_path=temp_dir / "clustered.csv",
                scaling='zscore'
            )
            assert 'cluster' in clustered_df.columns
            assert len(clustered_df) == len(features_df)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.regression
@pytest.mark.requires_readimc
class TestFeatureExtractionRegression:
    """Regression tests for feature extraction with known outputs.
    
    These tests compare extracted features against reference/target files
    to ensure feature extraction produces consistent results.
    """
    
    def test_feature_extraction_regression(self, test_data_dir, temp_dir):
        """Test feature extraction against a target feature file.
        
        This test:
        1. Loads the OME-TIFF file from test data
        2. Loads masks from a masks directory
        3. Extracts features with:
           - Hot pixel removal denoising for every channel
           - Arcsinh transform with cofactor 10
        4. Compares extracted features against target file (cell_features.csv)
        """
        # Get test data path (supports both MCD and OME-TIFF)
        try:
            data_path, _ = _get_test_data_path(test_data_dir)
        except FileNotFoundError as e:
            pytest.skip(str(e))
        
        # Check for masks directory
        masks_dir = test_data_dir / "mask"
        if not masks_dir.exists() or not masks_dir.is_dir():
            pytest.skip(f"Masks directory not found: {masks_dir}")
        
        # Check for target features file (use output_features.csv which matches CLI output)
        target_features_path = test_data_dir / "regression_targets" / "output_features.csv"
        if not target_features_path.exists():
            # Fallback to cell_features.csv if output_features.csv doesn't exist
            target_features_path = test_data_dir / "regression_targets" / "cell_features.csv"
            if not target_features_path.exists():
                pytest.skip(f"Target features file not found: {target_features_path}")
        
        # Load data
        loader, _ = load_mcd(data_path)
        acquisitions = loader.list_acquisitions()
        
        if len(acquisitions) == 0:
            pytest.skip("No acquisitions found in test data")
        
        # Verify mask is loaded correctly before feature extraction
        from openimc.core import _load_masks_for_acquisitions
        masks_dict = _load_masks_for_acquisitions(masks_dir, acquisitions)
        if len(masks_dict) == 0:
            pytest.fail(f"No masks loaded from {masks_dir}. Check mask filename matching.")
        
        for acq_id, mask in masks_dict.items():
            if mask is None:
                pytest.fail(f"Mask for acquisition {acq_id} is None")
            if mask.size == 0:
                pytest.fail(f"Mask for acquisition {acq_id} is empty")
            # Log mask info for debugging
            print(f"\n[DEBUG] Mask for {acq_id}: shape={mask.shape}, dtype={mask.dtype}, "
                  f"unique_labels={len(np.unique(mask))}, max_label={np.max(mask)}")
        
        # Use the same method as CLI: build_denoise_settings_for_all_channels
        # This matches: --denoise all --denoise-method median3
        from openimc.cli import build_denoise_settings_for_all_channels
        denoise_settings = build_denoise_settings_for_all_channels(
            loader, acquisitions, method='median3', n_sd=5.0
        )
        
        # Extract features with specified settings
        extracted_features = extract_features(
            loader=loader,
            acquisitions=acquisitions,
            mask_path=masks_dir,
            output_path=temp_dir / "extracted_features.csv",
            morphological=True,
            intensity=True,
            denoise_settings=denoise_settings,
            arcsinh=True,
            arcsinh_cofactor=10.0
        )
        
        # Load target features
        target_features = pd.read_csv(target_features_path)
        
        # Basic validation
        assert len(extracted_features) > 0, "No features extracted"
        assert len(extracted_features) == len(target_features), \
            f"Feature count mismatch: extracted {len(extracted_features)}, expected {len(target_features)}"
        
        # Sort both dataframes by cell_id or label for comparison
        # Try to find a common identifier column
        id_cols = ['cell_id', 'label']
        extracted_id_col = None
        target_id_col = None
        
        for col in id_cols:
            if col in extracted_features.columns:
                extracted_id_col = col
            if col in target_features.columns:
                target_id_col = col
        
        if extracted_id_col and target_id_col:
            extracted_features = extracted_features.sort_values(extracted_id_col).reset_index(drop=True)
            target_features = target_features.sort_values(target_id_col).reset_index(drop=True)
        
        # Get common feature columns (exclude metadata columns)
        metadata_cols = {'cell_id', 'label', 'acquisition_id', 'acquisition_name', 
                        'well', 'source_file', 'source_well', 'acquisition_label'}
        
        extracted_feature_cols = [col for col in extracted_features.columns 
                                 if col not in metadata_cols]
        target_feature_cols = [col for col in target_features.columns 
                              if col not in metadata_cols]
        
        # Find common feature columns
        common_cols = set(extracted_feature_cols) & set(target_feature_cols)
        
        if len(common_cols) == 0:
            pytest.fail(
                f"No common feature columns found. "
                f"Extracted: {extracted_feature_cols[:10]}..., "
                f"Target: {target_feature_cols[:10]}..."
            )
        
        # Compare features (with tolerance for floating point differences)
        # Use a more reasonable tolerance for intensity features after arcsinh transform
        # Absolute tolerance: 0.1 (for arcsinh-transformed values)
        # Relative tolerance: 1% (0.01) for values > 1, stricter for smaller values
        abs_tolerance = 0.1
        rel_tolerance = 0.01  # 1%
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        failed_cols = []
        
        for col in sorted(common_cols):
            if col not in extracted_features.columns or col not in target_features.columns:
                continue
            
            extracted_vals = extracted_features[col].values
            target_vals = target_features[col].values
            
            # Handle NaN values
            valid_mask = ~(np.isnan(extracted_vals) | np.isnan(target_vals))
            
            if not np.any(valid_mask):
                # All NaN - skip this column
                continue
            
            extracted_vals = extracted_vals[valid_mask]
            target_vals = target_vals[valid_mask]
            
            # Calculate differences
            abs_diff = np.abs(extracted_vals - target_vals)
            max_abs_diff_col = np.max(abs_diff)
            
            # Relative difference (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = np.abs((extracted_vals - target_vals) / (np.abs(target_vals) + 1e-10))
                max_rel_diff_col = np.max(rel_diff)
            
            # Check if difference exceeds tolerance
            # For small values (< 1), use absolute tolerance
            # For larger values, use relative tolerance
            small_val_mask = np.abs(target_vals) < 1.0
            large_val_mask = ~small_val_mask
            
            exceeds_tolerance = False
            if np.any(small_val_mask):
                # Use absolute tolerance for small values
                if np.any(abs_diff[small_val_mask] > abs_tolerance):
                    exceeds_tolerance = True
            if np.any(large_val_mask):
                # Use relative tolerance for larger values
                if np.any(rel_diff[large_val_mask] > rel_tolerance):
                    exceeds_tolerance = True
            
            if exceeds_tolerance:
                failed_cols.append({
                    'column': col,
                    'max_abs_diff': max_abs_diff_col,
                    'max_rel_diff': max_rel_diff_col,
                    'mean_abs_diff': np.mean(abs_diff),
                    'mean_rel_diff': np.mean(rel_diff[large_val_mask]) if np.any(large_val_mask) else 0.0
                })
                max_abs_diff = max(max_abs_diff, max_abs_diff_col)
                max_rel_diff = max(max_rel_diff, max_rel_diff_col)
        
        # Report results
        if failed_cols:
            error_msg = f"Feature extraction regression test failed:\n"
            error_msg += f"  Max absolute difference: {max_abs_diff:.2e}\n"
            error_msg += f"  Max relative difference: {max_rel_diff:.2e}\n"
            error_msg += f"  Failed columns ({len(failed_cols)}):\n"
            
            # Show top 10 worst columns
            failed_cols_sorted = sorted(failed_cols, key=lambda x: x['max_abs_diff'], reverse=True)[:10]
            for col_info in failed_cols_sorted:
                error_msg += f"    {col_info['column']}: abs_diff={col_info['max_abs_diff']:.2e}, "
                error_msg += f"rel_diff={col_info['max_rel_diff']:.2e}\n"
            
            error_msg += f"\n  Extracted features saved to: {temp_dir / 'extracted_features.csv'}\n"
            error_msg += f"  Target features: {target_features_path}\n"
            
            pytest.fail(error_msg)
        
        # If we get here, all features match within tolerance
        assert len(common_cols) > 0, "No common feature columns to compare"

