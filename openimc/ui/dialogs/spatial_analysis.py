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
Spatial Analysis Dialog for OpenIMC

This module provides shared utilities and factory functions for spatial analysis.
The actual dialog implementations are in separate files:
- simple_spatial_analysis.py: SimpleSpatialAnalysisDialog
- advanced_spatial_analysis.py: AdvancedSpatialAnalysisDialog
"""

# CRITICAL: Configure dask BEFORE any imports that might trigger dask.dataframe import
# This must be done at the very top, before any other imports
# Use environment variable approach as it's more reliable
import os
# Set environment variable before importing dask
os.environ.setdefault('DASK_DATAFRAME__QUERY_PLANNING', 'False')

# Also try direct config if dask is available
try:
    import dask
    # Set configuration before dask.dataframe is imported
    dask.config.set({'dataframe.query-planning': False})
except (ImportError, AttributeError):
    pass

from typing import Optional, Dict, Any, Tuple, List

import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from scipy.spatial import cKDTree, Delaunay
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from openimc.utils.logger import get_logger
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
try:
    from scipy import sparse as sp
    _HAVE_SPARSE = True
except Exception:
    _HAVE_SPARSE = False

try:
    import igraph as ig
    _HAVE_IGRAPH = True
except ImportError:
    _HAVE_IGRAPH = False

try:
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:
    _HAVE_SEABORN = False

try:
    # Dask should already be configured at the top of the file
    # Suppress FutureWarning about anndata.read_text deprecation and squidpy __version__ deprecation
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*read_text.*')
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*__version__.*')
        import squidpy as sq
        import scanpy as sc
        import anndata as ad
    _HAVE_SQUIDPY = True
except (ImportError, RuntimeError) as e:
    _HAVE_SQUIDPY = False
    sq = None
    sc = None
    ad = None
    # Log the error for debugging but don't fail module import
    import warnings
    warnings.warn(f"Failed to import squidpy: {e}. Squidpy features will be disabled.", ImportWarning)


def _get_vivid_colors(n):
    """
    Generate n vivid, distinct colors suitable for cluster visualization.
    Uses tab20, tab20b, tab20c for first 60 colors, then hsv for additional colors.
    
    Args:
        n: Number of colors needed
        
    Returns:
        Array of RGBA colors (n, 4)
    """
    colors = []
    
    # Use tab20, tab20b, tab20c for first 60 colors (vivid and distinct)
    if n <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n))
    elif n <= 40:
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, n - 20))
        ])
    elif n <= 60:
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, 20)),
            plt.cm.tab20c(np.linspace(0, 1, n - 40))
        ])
    else:
        # For more than 60 colors, use tab20 series + hsv for the rest
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, 20)),
            plt.cm.tab20c(np.linspace(0, 1, 20))
        ])
        # Use hsv colormap for additional colors, avoiding very dark/light values
        remaining = n - 60
        hsv_colors = plt.cm.hsv(np.linspace(0.1, 0.9, remaining))
        colors = np.vstack([colors, hsv_colors])
    
    return colors


def dataframe_to_anndata(
    df: pd.DataFrame,
    roi_id: Optional[str] = None,
    roi_column: str = 'acquisition_id',
    pixel_size_um: float = 1.0,
    parent=None
) -> Optional['ad.AnnData']:
    """
    Convert OpenIMC DataFrame to AnnData format for squidpy analysis.
    
    Args:
        df: Feature dataframe with cells as rows
        roi_id: Optional ROI identifier to filter to a single ROI
        roi_column: Column name for ROI identifier
        pixel_size_um: Pixel size in micrometers for coordinate conversion
        parent: Parent window to get pixel size if not provided
        
    Returns:
        AnnData object with spatial coordinates and features, or None if conversion fails
    """
    if not _HAVE_SQUIDPY:
        return None
    
    try:
        # Filter to specific ROI if provided
        if roi_id is not None and roi_column in df.columns:
            df = df[df[roi_column] == roi_id].copy()
        
        if df.empty:
            return None
        
        # Get pixel size from parent if not provided
        if pixel_size_um == 1.0 and parent is not None:
            try:
                if hasattr(parent, '_get_pixel_size_um'):
                    if roi_id is not None:
                        pixel_size_um = float(parent._get_pixel_size_um(roi_id))
                    else:
                        # Try to get from first ROI
                        roi_col = roi_column
                        if roi_col in df.columns:
                            first_roi = df[roi_col].iloc[0]
                            pixel_size_um = float(parent._get_pixel_size_um(first_roi))
            except Exception:
                pixel_size_um = 1.0
        
        # Ensure required columns exist
        required_cols = ['centroid_x', 'centroid_y', 'cell_id']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[DEBUG] dataframe_to_anndata: Missing required columns: {missing}")
            return None
        
        # Extract centroid coordinates
        centroid_coords = df[['centroid_x', 'centroid_y']].values
        print(f"[DEBUG] dataframe_to_anndata: Centroid coords shape: {centroid_coords.shape}, "
              f"range X: [{centroid_coords[:, 0].min():.2f}, {centroid_coords[:, 0].max():.2f}], "
              f"range Y: [{centroid_coords[:, 1].min():.2f}, {centroid_coords[:, 1].max():.2f}]")
        
        # Convert coordinates from pixels to micrometers
        # Note: pixel_size_um should be the size of one pixel in micrometers
        print(f"[DEBUG] dataframe_to_anndata: pixel_size_um: {pixel_size_um}")
        coords_um = centroid_coords * pixel_size_um
        print(f"[DEBUG] dataframe_to_anndata: Converted coords (µm) range X: [{coords_um[:, 0].min():.2f}, {coords_um[:, 0].max():.2f}], "
              f"range Y: [{coords_um[:, 1].min():.2f}, {coords_um[:, 1].max():.2f}]")
        
        # Identify feature columns (exclude metadata)
        metadata_cols = {
            'cell_id', 'acquisition_id', 'acquisition_label', 'source_file', 
            'source_well', 'label', 'centroid_x', 'centroid_y', 'cluster',
            'cluster_phenotype', 'cluster_id'
        }
        
        # Get intensity and morphology features
        # Filter to only include _mean features (as per user requirement)
        all_feature_cols = [col for col in df.columns if col not in metadata_cols]
        feature_cols = [col for col in all_feature_cols if col.endswith('_mean')]
        print(f"[DEBUG] dataframe_to_anndata: Found {len(feature_cols)} _mean features: {feature_cols[:5]}...")
        
        # Also include morphology features (they don't have _mean suffix)
        morpho_names = {
            'area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity',
            'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um',
            'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count'
        }
        morpho_cols = [col for col in all_feature_cols if col in morpho_names]
        feature_cols.extend(morpho_cols)
        print(f"[DEBUG] dataframe_to_anndata: Added {len(morpho_cols)} morphology features. Total: {len(feature_cols)}")
        
        # Create AnnData object
        # X: feature matrix (intensity and morphology features)
        X = df[feature_cols].values if feature_cols else np.zeros((len(df), 0))
        
        # obs: cell metadata
        obs = df[list(metadata_cols & set(df.columns))].copy()
        obs.index = df['cell_id'].astype(str).values
        
        # obsm: spatial coordinates
        obsm = {'spatial': coords_um}
        
        # var: feature names
        var = pd.DataFrame(index=feature_cols)
        
        # Create AnnData
        adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm)
        
        # Store cluster information in obs if available
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in df.columns:
                cluster_col = col
                break
        
        if cluster_col:
            adata.obs['cluster'] = df[cluster_col].values
        
        return adata
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


class SourceFileFilterDialog(QtWidgets.QDialog):
    """Dialog for selecting source files to filter spatial analysis data."""
    
    def __init__(self, available_source_files: set, selected_source_files: set, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Source File Filter")
        self.setModal(True)
        self.resize(500, 400)
        
        self.available_source_files = available_source_files
        self.selected_source_files = selected_source_files.copy() if selected_source_files else set()
        
        self._create_ui()
        self._populate_file_list()
    
    def _create_ui(self):
        """Create the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Info label
        info_label = QtWidgets.QLabel(
            "Select source files to include in spatial analysis.\n"
            "Leave all unchecked to include all files."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # File list
        self.source_file_list = QtWidgets.QListWidget()
        self.source_file_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        layout.addWidget(self.source_file_list)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.select_all_btn = QtWidgets.QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        button_layout.addWidget(self.deselect_all_btn)
        
        button_layout.addStretch()
        
        # OK/Cancel buttons
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _populate_file_list(self):
        """Populate the file list with available source files."""
        self.source_file_list.clear()
        source_files = sorted(self.available_source_files)
        
        for source_file in source_files:
            item = QtWidgets.QListWidgetItem(source_file)
            # Check if this file should be selected
            # If selected_source_files is empty, select all (default behavior)
            if not self.selected_source_files or source_file in self.selected_source_files:
                item.setCheckState(QtCore.Qt.Checked)
            else:
                item.setCheckState(QtCore.Qt.Unchecked)
            self.source_file_list.addItem(item)
    
    def _select_all(self):
        """Select all source files."""
        for i in range(self.source_file_list.count()):
            item = self.source_file_list.item(i)
            item.setCheckState(QtCore.Qt.Checked)
    
    def _deselect_all(self):
        """Deselect all source files."""
        for i in range(self.source_file_list.count()):
            item = self.source_file_list.item(i)
            item.setCheckState(QtCore.Qt.Unchecked)
    
    def get_selected_files(self):
        """Get the set of selected source files."""
        selected = set()
        for i in range(self.source_file_list.count()):
            item = self.source_file_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                selected.add(item.text())
        return selected


# Keep SpatialAnalysisDialog as an alias for backward compatibility
# It will default to Advanced if squidpy is available, otherwise Simple
# Use lazy imports to avoid circular import issues
def SpatialAnalysisDialog(feature_dataframe: pd.DataFrame, batch_corrected_dataframe=None, parent=None):
    """Factory function - returns Advanced if squidpy available, otherwise Simple."""
    # Lazy import to avoid circular dependencies
    from openimc.ui.dialogs.simple_spatial_analysis import SimpleSpatialAnalysisDialog
    if _HAVE_SQUIDPY:
        try:
            from openimc.ui.dialogs.advanced_spatial_analysis import AdvancedSpatialAnalysisDialog
            return AdvancedSpatialAnalysisDialog(feature_dataframe, batch_corrected_dataframe, parent)
        except (ImportError, RuntimeError):
            pass
    return SimpleSpatialAnalysisDialog(feature_dataframe, batch_corrected_dataframe, parent)

# For direct imports, also provide lazy access
def _get_SimpleSpatialAnalysisDialog():
    """Lazy getter for SimpleSpatialAnalysisDialog to avoid circular imports."""
    from openimc.ui.dialogs.simple_spatial_analysis import SimpleSpatialAnalysisDialog
    return SimpleSpatialAnalysisDialog

def _get_AdvancedSpatialAnalysisDialog():
    """Lazy getter for AdvancedSpatialAnalysisDialog to avoid circular imports."""
    if _HAVE_SQUIDPY:
        try:
            from openimc.ui.dialogs.advanced_spatial_analysis import AdvancedSpatialAnalysisDialog
            return AdvancedSpatialAnalysisDialog
        except (ImportError, RuntimeError):
            return None
    return None

# Create module-level aliases that will be populated on first access
SimpleSpatialAnalysisDialog = None
AdvancedSpatialAnalysisDialog = None

def __getattr__(name):
    """Lazy loading of dialog classes to avoid circular imports."""
    if name == 'SimpleSpatialAnalysisDialog':
        global SimpleSpatialAnalysisDialog
        if SimpleSpatialAnalysisDialog is None:
            from openimc.ui.dialogs.simple_spatial_analysis import SimpleSpatialAnalysisDialog as _Simple
            SimpleSpatialAnalysisDialog = _Simple
        return SimpleSpatialAnalysisDialog
    elif name == 'AdvancedSpatialAnalysisDialog':
        global AdvancedSpatialAnalysisDialog
        if AdvancedSpatialAnalysisDialog is None:
            if _HAVE_SQUIDPY:
                try:
                    from openimc.ui.dialogs.advanced_spatial_analysis import AdvancedSpatialAnalysisDialog as _Advanced
                    AdvancedSpatialAnalysisDialog = _Advanced
                except (ImportError, RuntimeError):
                    AdvancedSpatialAnalysisDialog = None
        return AdvancedSpatialAnalysisDialog
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Re-export classes for backward compatibility
__all__ = [
    'SpatialAnalysisDialog',
    'SimpleSpatialAnalysisDialog', 
    'AdvancedSpatialAnalysisDialog',
    'SourceFileFilterDialog',
    'dataframe_to_anndata',
    '_get_vivid_colors',
    '_HAVE_SQUIDPY',
    '_HAVE_SPARSE',
    '_HAVE_IGRAPH',
    '_HAVE_SEABORN'
]
