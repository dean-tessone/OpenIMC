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
Simple Spatial Analysis Dialog for OpenIMC

This module provides the simple spatial analysis dialog without squidpy dependencies.
"""

import os
os.environ.setdefault('DASK_DATAFRAME__QUERY_PLANNING', 'False')

from typing import Optional, Dict, Any, Tuple, List
import json
import numpy as np
import pandas as pd
import multiprocessing as mp
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
from openimc.core import spatial_enrichment, spatial_distance_distribution, build_spatial_graph
from openimc.ui.dialogs.spatial_analysis import (
    SourceFileFilterDialog,
    _get_vivid_colors,
    _HAVE_SPARSE,
    _HAVE_IGRAPH,
    _HAVE_SEABORN,
    _HAVE_SQUIDPY  # May be needed for some checks
)

try:
    from scipy import sparse as sp
except Exception:
    sp = None

try:
    import igraph as ig
except ImportError:
    ig = None

try:
    import seaborn as sns
    _HAVE_SEABORN_LOCAL = True
except ImportError:
    _HAVE_SEABORN_LOCAL = False
    sns = None


# Import worker functions from processing module
from openimc.processing.spatial_analysis_worker import (
    permutation_worker as _permutation_worker,
    distance_distribution_worker as _distance_distribution_worker,
    neighborhood_composition_worker as _neighborhood_composition_worker,
    ripley_worker as _ripley_worker
)

def _distance_distribution_worker(args):
    """
    Worker function for computing distance distributions for a single ROI.
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing:
            - roi_id: ROI identifier
            - roi_df: DataFrame with cells for this ROI
            - cluster_col: Name of cluster column
            - pixel_size_um: Pixel size in micrometers
            
    Returns:
        List of dictionaries with distance data for this ROI
    """
    roi_id, roi_df, cluster_col, pixel_size_um = args
    
    distance_data = []
    
    # Convert coordinates to micrometers
    coords_um = roi_df[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
    cell_ids = roi_df["cell_id"].astype(int).to_numpy()
    cell_clusters = roi_df[cluster_col].values
    
    # Get unique clusters in this ROI
    unique_clusters = sorted(roi_df[cluster_col].unique())
    
    # Create KDTree for efficient nearest neighbor search
    tree = cKDTree(coords_um)
    
    # Pre-compute cluster masks for efficiency
    cluster_masks = {}
    for cluster in unique_clusters:
        cluster_masks[cluster] = (cell_clusters == cluster)
    
    # For each cell, find nearest neighbor of each cluster type
    for pos_idx in range(len(roi_df)):
        cell_id = int(cell_ids[pos_idx])
        cell_cluster = cell_clusters[pos_idx]
        cell_coord = coords_um[pos_idx]
        
        # Find nearest neighbor for each cluster type
        for target_cluster in unique_clusters:
            # Get mask for target cluster
            target_mask = cluster_masks[target_cluster]
            
            if target_cluster == cell_cluster:
                # For same cluster, find nearest neighbor excluding self
                # Create mask excluding current cell
                target_mask_excluding_self = target_mask.copy()
                target_mask_excluding_self[pos_idx] = False
                
                if np.sum(target_mask_excluding_self) < 1:
                    continue
                
                # Get coordinates of target cells (excluding self)
                target_coords = coords_um[target_mask_excluding_self]
                target_cell_ids = cell_ids[target_mask_excluding_self]
                
                # Use KDTree query to find nearest neighbor
                # Query with k=2 to get self and nearest neighbor, then take the second one
                if len(target_coords) > 0:
                    # Create a tree for target cells only
                    target_tree = cKDTree(target_coords)
                    min_distance, nearest_target_idx = target_tree.query(cell_coord, k=1)
                    nearest_cell_id = int(target_cell_ids[nearest_target_idx])
                else:
                    min_distance = float('inf')
                    nearest_cell_id = None
            else:
                # For different clusters, find nearest neighbor
                if not np.any(target_mask):
                    continue
                
                # Get coordinates of target cells
                target_coords = coords_um[target_mask]
                target_cell_ids = cell_ids[target_mask]
                
                # Use KDTree query to find nearest neighbor
                # Create a temporary tree for just the target cluster
                target_tree = cKDTree(target_coords)
                min_distance, nearest_target_idx = target_tree.query(cell_coord, k=1)
                nearest_cell_id = int(target_cell_ids[nearest_target_idx])
            
            # Record the nearest neighbor distance
            if min_distance != float('inf') and nearest_cell_id is not None:
                distance_data.append({
                    'roi_id': roi_id,
                    'cell_A_id': cell_id,
                    'cell_A_cluster': cell_cluster,
                    'nearest_B_cluster': target_cluster,
                    'nearest_B_dist_um': float(min_distance),
                    'nearest_B_cell_id': nearest_cell_id
                })
    
    return distance_data


def _neighborhood_composition_worker(args):
    """
    Worker function for computing neighborhood composition for a single ROI.
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing:
            - roi_id: ROI identifier
            - roi_df: DataFrame with cells for this ROI
            - roi_edges: DataFrame with edges for this ROI
            - cluster_col: Name of cluster column
            - unique_clusters: List of unique cluster IDs
            
    Returns:
        List of dictionaries with neighborhood composition data for this ROI
    """
    roi_id, roi_df, roi_edges, cluster_col, unique_clusters = args
    
    neighborhood_data = []
    
    # Create efficient cell_id to cluster mapping
    cell_to_cluster = dict(zip(roi_df['cell_id'], roi_df[cluster_col]))
    
    # Build adjacency list efficiently
    cell_to_neighbors = defaultdict(list)
    for _, edge in roi_edges.iterrows():
        cell_a, cell_b = int(edge['cell_id_A']), int(edge['cell_id_B'])
        cell_to_neighbors[cell_a].append(cell_b)
        cell_to_neighbors[cell_b].append(cell_a)
    
    # Vectorized neighborhood composition computation
    for _, cell_row in roi_df.iterrows():
        cell_id = int(cell_row['cell_id'])
        cell_cluster = cell_row[cluster_col]
        
        # Initialize composition vector using actual cluster IDs
        composition = {f'frac_cluster_{cluster}': 0.0 for cluster in unique_clusters}
        
        if cell_id in cell_to_neighbors:
            neighbors = cell_to_neighbors[cell_id]
            if neighbors:
                # Vectorized neighbor cluster lookup
                neighbor_clusters = [cell_to_cluster.get(nb_id) for nb_id in neighbors]
                neighbor_clusters = [c for c in neighbor_clusters if c is not None]
                
                if neighbor_clusters:
                    # Vectorized cluster counting
                    total_neighbors = len(neighbor_clusters)
                    for cluster in unique_clusters:
                        cluster_count = neighbor_clusters.count(cluster)
                        composition[f'frac_cluster_{cluster}'] = cluster_count / total_neighbors
        
        # Add cell information
        row_data = {
            'cell_id': cell_id,
            'roi_id': roi_id,
            'cluster_id': cell_cluster,
        }
        row_data.update(composition)
        neighborhood_data.append(row_data)
    
    return neighborhood_data


def _ripley_worker(args):
    """
    Worker function for computing Ripley K/L functions for a single ROI and cluster.
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing:
            - roi_id: ROI identifier
            - cluster: Cluster ID
            - cluster_coords: Array of coordinates for this cluster
            - coords_um: All coordinates in ROI (for edge correction)
            - roi_area: Area of ROI
            - radius_steps: Array of radius values to compute
            - pixel_size_um: Pixel size in micrometers
            
    Returns:
        List of dictionaries with Ripley data for this ROI and cluster
    """
    roi_id, cluster, cluster_coords, coords_um, roi_area, radius_steps, pixel_size_um = args
    
    ripley_data = []
    n_points = len(cluster_coords)
    
    if n_points < 2:
        return ripley_data
    
    # Point density
    lambda_density = n_points / roi_area if roi_area > 0 else 0
    
    # Compute K function for this cluster
    for r in radius_steps:
        # Count points within radius r with edge correction
        k_sum = 0
        for i, point in enumerate(cluster_coords):
            distances = np.sqrt(np.sum((cluster_coords - point)**2, axis=1))
            # Exclude the point itself
            within_radius = (distances <= r) & (distances > 0)
            
            # Apply edge correction (isotropic correction)
            within_radius_indices = np.where(within_radius)[0]
            edge_correction = np.ones(len(within_radius_indices))
            for j, idx in enumerate(within_radius_indices):
                if distances[idx] > 0:
                    # Check if circle of radius r around point intersects ROI boundary
                    x, y = point
                    
                    # Distance to each boundary
                    dist_to_left = x - np.min(coords_um[:, 0])
                    dist_to_right = np.max(coords_um[:, 0]) - x
                    dist_to_bottom = y - np.min(coords_um[:, 1])
                    dist_to_top = np.max(coords_um[:, 1]) - y
                    
                    # Edge correction factor (simplified isotropic correction)
                    if dist_to_left < r or dist_to_right < r or dist_to_bottom < r or dist_to_top < r:
                        # Partial edge correction - use fraction of circle within ROI
                        edge_correction[j] = 0.5  # Simplified correction
            
            k_sum += np.sum(edge_correction)
        
        # Corrected Ripley K function
        if lambda_density > 0 and n_points > 1:
            k_obs = k_sum / (lambda_density * n_points)
        else:
            k_obs = 0
        
        # Expected K under complete spatial randomness (CSR)
        k_exp = np.pi * r**2
        
        # L function (corrected formula)
        if k_obs > 0:
            l_obs = np.sqrt(k_obs / np.pi) - r
        else:
            l_obs = -r
        
        l_exp = 0  # Expected L under CSR
        
        ripley_data.append({
            'roi_id': roi_id,
            'cell_type': cluster,
            'r_um': r,
            'K_obs': k_obs,
            'K_exp': k_exp,
            'L_obs': l_obs,
            'L_exp': l_exp,
            'lambda_density': lambda_density,
            'n_points': n_points,
            'roi_area': roi_area
        })
    
    return ripley_data


class SimpleSpatialAnalysisDialog(QtWidgets.QDialog):
    """Simple Spatial Analysis Dialog - original implementation without squidpy."""
    def __init__(self, feature_dataframe: pd.DataFrame, batch_corrected_dataframe=None, parent=None):
        print("[DEBUG] SimpleSpatialAnalysisDialog.__init__: Starting initialization...")
        super().__init__(parent)
        self.setWindowTitle("Simple Spatial Analysis")
        self.setMinimumSize(900, 650)
        
        # Set size to 90% of parent window size if parent exists
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)

        self.original_feature_dataframe = feature_dataframe
        self.batch_corrected_dataframe = batch_corrected_dataframe
        if batch_corrected_dataframe is not None and not batch_corrected_dataframe.empty:
            self.feature_dataframe = batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = feature_dataframe.copy()
        self.edge_df: Optional[pd.DataFrame] = None
        self.adj_matrices: Dict[str, sp.csr_matrix] = {} if sp else {}
        self.cell_id_to_gid: Dict[Tuple[str, int], int] = {}
        self.gid_to_cell_id: Dict[int, Tuple[str, int]] = {}
        self.metadata: Dict[str, Any] = {}
        self.cluster_summary_df: Optional[pd.DataFrame] = None
        self.enrichment_df: Optional[pd.DataFrame] = None
        self.distance_df: Optional[pd.DataFrame] = None
        self.rng_seed: int = 42
        
        self.spatial_viz_cache: Dict[str, Any] = {}
        
        self.enrichment_analysis_run = False
        self.distance_analysis_run = False
        self.spatial_viz_run = False
        self.community_analysis_run = False
        
        self.cluster_annotation_map = {}
        self.selected_source_files = set()
        self.available_source_files = set()

        self._create_ui()
        
        if hasattr(self, 'source_file_status_label'):
            self._update_source_file_status_label()
    
    def _get_roi_column(self):
        """Get the appropriate ROI column name."""
        if self.feature_dataframe is not None and 'source_well' in self.feature_dataframe.columns:
            return 'source_well'
        return 'acquisition_id'
    
    def _on_feature_set_changed(self):
        """Handle feature set selection change."""
        if not hasattr(self, 'feature_set_combo'):
            return
        
        selected = self.feature_set_combo.currentText()
        if selected == "Batch-Corrected Features" and self.batch_corrected_dataframe is not None:
            self.feature_dataframe = self.batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = self.original_feature_dataframe.copy()
        
        self._clear_analysis_cache()
        self._update_source_file_filter()
        
    
    def _get_filtered_dataframe(self):
        """Get the filtered dataframe based on selected source files."""
        df = self.feature_dataframe.copy()
        if ('source_file' in df.columns and 
            hasattr(self, 'selected_source_files') and 
            self.selected_source_files and 
            len(self.selected_source_files) > 0):
            df = df[df['source_file'].isin(self.selected_source_files)]
        return df
    
    def _get_source_files_for_logging(self):
        """Get source file names from the filtered dataframe for logging."""
        filtered_df = self._get_filtered_dataframe()
        if 'source_file' in filtered_df.columns:
            source_files = sorted(filtered_df['source_file'].dropna().unique())
            if len(source_files) == 1:
                return source_files[0]
            elif len(source_files) > 1:
                return f"{len(source_files)} files: {', '.join(source_files[:3])}" + ("..." if len(source_files) > 3 else "")
        if self.parent() is not None and hasattr(self.parent(), 'current_path'):
            return os.path.basename(self.parent().current_path) if self.parent().current_path else None
        return None
    
    def _open_source_file_filter_dialog(self):
        """Open the source file filter dialog."""
        dlg = SourceFileFilterDialog(
            self.available_source_files,
            self.selected_source_files,
            self
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            selected = dlg.get_selected_files()
            if len(selected) == len(self.available_source_files):
                self.selected_source_files = set()
            else:
                self.selected_source_files = selected
            self._update_source_file_status_label()
            self._clear_analysis_cache()
    
    def _update_source_file_status_label(self):
        """Update the source file status label."""
        if not hasattr(self, 'source_file_status_label'):
            return
        
        if not self.selected_source_files:
            self.source_file_status_label.setText("All files")
        else:
            count = len(self.selected_source_files)
            total = len(self.available_source_files)
            if count == 1:
                file_name = list(self.selected_source_files)[0]
                self.source_file_status_label.setText(f"1 file: {file_name}")
            else:
                self.source_file_status_label.setText(f"{count} of {total} files")
    
    def _clear_analysis_cache(self):
        """Clear cached analyses when filtering changes."""
        self.edge_df = None
        self.adj_matrices = {}
        self.cell_id_to_gid = {}
        self.gid_to_cell_id = {}
        self.cluster_summary_df = None
        self.enrichment_df = None
        self.distance_df = None
        self.spatial_viz_cache = {}
        
        self.enrichment_analysis_run = False
        self.distance_analysis_run = False
        self.spatial_viz_run = False
        self.community_analysis_run = False
        
        # Clear distance cluster list
        if hasattr(self, 'distance_cluster_list'):
            self.distance_cluster_list.clear()
            self.distance_select_all_btn.setEnabled(False)
            self.distance_deselect_all_btn.setEnabled(False)
    
    def _update_source_file_filter(self):
        """Update source file filter when feature set changes."""
        if 'source_file' in self.feature_dataframe.columns:
            source_files = sorted(self.feature_dataframe['source_file'].dropna().unique())
            self.available_source_files = set(source_files)
            
            if hasattr(self, 'source_file_status_label'):
                self._update_source_file_status_label()
            
            if self.selected_source_files:
                self.selected_source_files = {
                    f for f in self.selected_source_files 
                    if f in self.available_source_files
                }
    
    def _create_ui(self):
        """Create the UI."""
        layout = QtWidgets.QVBoxLayout(self)

        if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
            feature_set_layout = QtWidgets.QHBoxLayout()
            feature_set_layout.addWidget(QtWidgets.QLabel("Feature Set:"))
            self.feature_set_combo = QtWidgets.QComboBox()
            self.feature_set_combo.addItem("Original Features")
            self.feature_set_combo.addItem("Batch-Corrected Features")
            self.feature_set_combo.setCurrentText("Batch-Corrected Features")
            self.feature_set_combo.setToolTip("Choose between original or batch-corrected feature sets")
            self.feature_set_combo.currentTextChanged.connect(self._on_feature_set_changed)
            feature_set_layout.addWidget(self.feature_set_combo)
            feature_set_layout.addStretch()
            layout.addLayout(feature_set_layout)

        if 'source_file' in self.feature_dataframe.columns:
            source_files = sorted(self.feature_dataframe['source_file'].dropna().unique())
            self.available_source_files = set(source_files)
            
            if len(source_files) > 1:
                source_file_layout = QtWidgets.QHBoxLayout()
                source_file_layout.addWidget(QtWidgets.QLabel("Source Files:"))
                self.source_file_status_label = QtWidgets.QLabel("All files")
                self.source_file_status_label.setToolTip("Click 'Configure...' to filter source files")
                source_file_layout.addWidget(self.source_file_status_label)
                
                self.source_file_config_btn = QtWidgets.QPushButton("Configure...")
                self.source_file_config_btn.clicked.connect(self._open_source_file_filter_dialog)
                source_file_layout.addWidget(self.source_file_config_btn)
                source_file_layout.addStretch()
                layout.addLayout(source_file_layout)

        params_group = QtWidgets.QGroupBox("Spatial Graph Construction")
        params_layout = QtWidgets.QGridLayout(params_group)

        self.graph_mode_combo = QtWidgets.QComboBox()
        self.graph_mode_combo.addItems(["kNN", "Radius", "Delaunay"])
        self.graph_mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 64)
        self.k_spin.setValue(20)
        
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 500.0)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setValue(20.0)
        
        self.n_perm_spin = QtWidgets.QSpinBox()
        self.n_perm_spin.setRange(10, 1000)
        self.n_perm_spin.setValue(100)
        
        # Workers spinbox - default to max workers - 2, but at least 1
        try:
            cpu_count = mp.cpu_count()
        except (NotImplementedError, RuntimeError):
            cpu_count = 4  # Fallback to 4 if cpu_count fails
        max_workers = max(1, cpu_count - 2)
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, cpu_count)
        self.workers_spin.setValue(max_workers)
        self.workers_spin.setToolTip("Number of parallel workers for permutation tests")

        params_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        params_layout.addWidget(self.graph_mode_combo, 0, 1)
        
        self.k_label = QtWidgets.QLabel("k:")
        params_layout.addWidget(self.k_label, 0, 2)
        params_layout.addWidget(self.k_spin, 0, 3)
        
        self.radius_label = QtWidgets.QLabel("Radius (µm):")
        params_layout.addWidget(self.radius_label, 0, 4)
        params_layout.addWidget(self.radius_spin, 0, 5)
        
        params_layout.addWidget(QtWidgets.QLabel("Random Seed:"), 0, 6)
        self.seed_spinbox = QtWidgets.QSpinBox()
        self.seed_spinbox.setRange(0, 2**31 - 1)
        self.seed_spinbox.setValue(42)
        self.seed_spinbox.setToolTip("Random seed for reproducibility (default: 42)")
        params_layout.addWidget(self.seed_spinbox, 0, 7)
        
        self.build_graph_btn = QtWidgets.QPushButton("Build Graph")
        self.build_graph_btn.setToolTip("Build the spatial graph using the selected mode and parameters")
        params_layout.addWidget(self.build_graph_btn, 0, 8)
        
        self._on_mode_changed()
        layout.addWidget(params_group)

        action_row = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export Results…")
        self.export_btn.setEnabled(False)
        self.export_graph_btn = QtWidgets.QPushButton("Export Graph…")
        self.export_graph_btn.setEnabled(False)
        action_row.addWidget(self.export_btn)
        action_row.addWidget(self.export_graph_btn)
        action_row.addStretch(1)
        
        # Advanced analysis button
        self.advanced_analysis_btn = QtWidgets.QPushButton("Advanced analysis")
        self.advanced_analysis_btn.setToolTip(
            "Open Advanced Spatial Analysis using Squidpy for more sophisticated spatial analysis methods, "
            "including neighborhood enrichment, co-occurrence analysis, spatial autocorrelation, and Ripley functions. "
            "Requires squidpy to be installed."
        )
        action_row.addWidget(self.advanced_analysis_btn)
        layout.addLayout(action_row)

        self.tabs = QtWidgets.QTabWidget()
        
        # Pairwise Enrichment tab
        self.enrichment_tab = QtWidgets.QWidget()
        enrichment_layout = QtWidgets.QVBoxLayout(self.enrichment_tab)
        
        enrichment_desc = QtWidgets.QLabel("Tests for significant spatial co-occurrence or avoidance between cluster pairs using permutation tests. Results show z-scores and p-values.")
        enrichment_desc.setWordWrap(True)
        enrichment_desc.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        enrichment_layout.addWidget(enrichment_desc)
        
        enrichment_params = QtWidgets.QHBoxLayout()
        enrichment_params.addWidget(QtWidgets.QLabel("Permutations:"))
        enrichment_params.addWidget(self.n_perm_spin)
        enrichment_params.addWidget(QtWidgets.QLabel("Workers:"))
        enrichment_params.addWidget(self.workers_spin)
        self.enrichment_run_btn = QtWidgets.QPushButton("Run Enrichment Analysis")
        self.enrichment_save_btn = QtWidgets.QPushButton("Save Plot")
        self.enrichment_save_btn.setEnabled(False)
        enrichment_params.addWidget(self.enrichment_run_btn)
        enrichment_params.addWidget(self.enrichment_save_btn)
        enrichment_params.addStretch()
        enrichment_layout.addLayout(enrichment_params)
        
        self.enrichment_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        enrichment_layout.addWidget(self.enrichment_canvas)
        self.tabs.addTab(self.enrichment_tab, "Pairwise Enrichment")
        
        # Distance Distributions tab
        self.distance_tab = QtWidgets.QWidget()
        distance_layout = QtWidgets.QVBoxLayout(self.distance_tab)
        
        distance_desc = QtWidgets.QLabel("Computes nearest neighbor distances between all cluster pairs. Results show distance distributions as violin plots to assess spatial relationships.")
        distance_desc.setWordWrap(True)
        distance_desc.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        distance_layout.addWidget(distance_desc)
        
        distance_btn_layout = QtWidgets.QHBoxLayout()
        distance_btn_layout.addWidget(QtWidgets.QLabel("Workers:"))
        self.distance_workers_spin = QtWidgets.QSpinBox()
        try:
            cpu_count = mp.cpu_count()
        except (NotImplementedError, RuntimeError):
            cpu_count = 4
        max_workers = max(1, cpu_count - 2)
        self.distance_workers_spin.setRange(1, cpu_count)
        self.distance_workers_spin.setValue(max_workers)
        self.distance_workers_spin.setToolTip("Number of parallel workers for distance computation")
        distance_btn_layout.addWidget(self.distance_workers_spin)
        self.distance_run_btn = QtWidgets.QPushButton("Run Distance Analysis")
        self.distance_save_btn = QtWidgets.QPushButton("Save Plot")
        self.distance_save_btn.setEnabled(False)
        distance_btn_layout.addWidget(self.distance_run_btn)
        distance_btn_layout.addWidget(self.distance_save_btn)
        distance_btn_layout.addStretch()
        distance_layout.addLayout(distance_btn_layout)
        
        # Cluster selection for filtering distance distributions
        distance_cluster_layout = QtWidgets.QHBoxLayout()
        distance_cluster_layout.addWidget(QtWidgets.QLabel("Filter clusters:"))
        self.distance_cluster_list = QtWidgets.QListWidget()
        self.distance_cluster_list.setToolTip("Select clusters to display in distance distribution plot. Only pairs involving selected clusters will be shown.")
        self.distance_cluster_list.setMaximumHeight(100)
        self.distance_cluster_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        distance_cluster_layout.addWidget(self.distance_cluster_list)
        
        distance_cluster_btn_layout = QtWidgets.QVBoxLayout()
        self.distance_select_all_btn = QtWidgets.QPushButton("Select All")
        self.distance_deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        self.distance_select_all_btn.setEnabled(False)
        self.distance_deselect_all_btn.setEnabled(False)
        self.distance_select_all_btn.clicked.connect(self._on_distance_select_all)
        self.distance_deselect_all_btn.clicked.connect(self._on_distance_deselect_all)
        distance_cluster_btn_layout.addWidget(self.distance_select_all_btn)
        distance_cluster_btn_layout.addWidget(self.distance_deselect_all_btn)
        distance_cluster_btn_layout.addStretch()
        distance_cluster_layout.addLayout(distance_cluster_btn_layout)
        distance_cluster_layout.addStretch()
        distance_layout.addLayout(distance_cluster_layout)
        
        # Connect selection changes to update plot
        self.distance_cluster_list.itemSelectionChanged.connect(self._on_distance_cluster_selection_changed)
        
        self.distance_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        distance_layout.addWidget(self.distance_canvas)
        self.tabs.addTab(self.distance_tab, "Distance Distributions")
        
        # Spatial Visualization tab
        self.spatial_viz_tab = QtWidgets.QWidget()
        spatial_viz_layout = QtWidgets.QVBoxLayout(self.spatial_viz_tab)
        
        spatial_viz_desc = QtWidgets.QLabel("Creates spatial scatter plots of cells colored by cluster or feature values. Results show the spatial distribution of cells across ROIs.")
        spatial_viz_desc.setWordWrap(True)
        spatial_viz_desc.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        spatial_viz_layout.addWidget(spatial_viz_desc)
        
        spatial_viz_controls = QtWidgets.QHBoxLayout()
        self.roi_label = QtWidgets.QLabel("ROI:")
        spatial_viz_controls.addWidget(self.roi_label)
        self.roi_combo = QtWidgets.QComboBox()
        spatial_viz_controls.addWidget(self.roi_combo)
        
        spatial_viz_controls.addWidget(QtWidgets.QLabel("Color by:"))
        self.spatial_color_combo = QtWidgets.QComboBox()
        self.spatial_color_combo.setEditable(True)
        self.spatial_color_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.spatial_color_combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.spatial_color_combo.setToolTip("Search and select feature for color encoding")
        spatial_viz_controls.addWidget(self.spatial_color_combo)
        
        spatial_viz_controls.addWidget(QtWidgets.QLabel("Point Size:"))
        self.spatial_point_size_spin = QtWidgets.QDoubleSpinBox()
        self.spatial_point_size_spin.setRange(0.1, 10.0)
        self.spatial_point_size_spin.setSingleStep(0.1)
        self.spatial_point_size_spin.setValue(1.0)
        self.spatial_point_size_spin.setDecimals(1)
        self.spatial_point_size_spin.setToolTip("Multiplier for point sizes (1.0 = default, increase for larger points)")
        spatial_viz_controls.addWidget(self.spatial_point_size_spin)
        
        self.spatial_show_edges_check = QtWidgets.QCheckBox("Show edges")
        self.spatial_show_edges_check.setChecked(False)
        self.spatial_show_edges_check.setToolTip("Display edges between cells (can be slow for large datasets)")
        spatial_viz_controls.addWidget(self.spatial_show_edges_check)
        
        self.spatial_viz_run_btn = QtWidgets.QPushButton("Generate Spatial Plot")
        self.spatial_viz_save_btn = QtWidgets.QPushButton("Save Plot")
        self.spatial_viz_save_btn.setEnabled(False)
        spatial_viz_controls.addWidget(self.spatial_viz_run_btn)
        spatial_viz_controls.addWidget(self.spatial_viz_save_btn)
        spatial_viz_controls.addStretch()
        spatial_viz_layout.addLayout(spatial_viz_controls)
        
        self.spatial_viz_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        spatial_viz_layout.addWidget(self.spatial_viz_canvas)
        self.tabs.addTab(self.spatial_viz_tab, "Spatial Visualization")
        
        # Spatial Community Analysis tab
        self.community_tab = QtWidgets.QWidget()
        community_layout = QtWidgets.QVBoxLayout(self.community_tab)
        
        community_desc = QtWidgets.QLabel("Identifies spatially coherent communities of cells using graph-based clustering. Results show community assignments and spatial organization patterns.")
        community_desc.setWordWrap(True)
        community_desc.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        community_layout.addWidget(community_desc)
        
        community_controls = QtWidgets.QHBoxLayout()
        community_controls.addWidget(QtWidgets.QLabel("ROI:"))
        self.community_roi_combo = QtWidgets.QComboBox()
        community_controls.addWidget(self.community_roi_combo)
        community_controls.addWidget(QtWidgets.QLabel("Min cells:"))
        self.min_cells_spin = QtWidgets.QSpinBox()
        self.min_cells_spin.setRange(1, 100)
        self.min_cells_spin.setValue(5)
        community_controls.addWidget(self.min_cells_spin)
        self.community_run_btn = QtWidgets.QPushButton("Run Community Analysis")
        self.community_save_btn = QtWidgets.QPushButton("Save Plot")
        self.community_save_btn.setEnabled(False)
        community_controls.addWidget(self.community_run_btn)
        community_controls.addWidget(self.community_save_btn)
        community_controls.addStretch()
        community_layout.addLayout(community_controls)
        
        exclusion_layout = QtWidgets.QHBoxLayout()
        exclusion_layout.addWidget(QtWidgets.QLabel("Exclude cell types:"))
        self.exclude_clusters_check = QtWidgets.QCheckBox("Enable exclusion")
        exclusion_layout.addWidget(self.exclude_clusters_check)
        self.exclude_clusters_list = QtWidgets.QListWidget()
        self.exclude_clusters_list.setMaximumHeight(100)
        self.exclude_clusters_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        exclusion_layout.addWidget(self.exclude_clusters_list)
        exclusion_layout.addStretch()
        community_layout.addLayout(exclusion_layout)
        
        self.community_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        community_layout.addWidget(self.community_canvas)
        self.tabs.addTab(self.community_tab, "Spatial Communities")
        
        layout.addWidget(self.tabs, 1)
        
        # Set default tab to the leftmost tab (index 0)
        self.tabs.setCurrentIndex(0)

        # Wire signals
        self.build_graph_btn.clicked.connect(self._on_build_graph_clicked)
        self.enrichment_run_btn.clicked.connect(self._run_enrichment_analysis)
        self.distance_run_btn.clicked.connect(self._run_distance_analysis)
        self.enrichment_save_btn.clicked.connect(self._save_enrichment_plot)
        self.distance_save_btn.clicked.connect(self._save_distance_plot)
        self.spatial_viz_run_btn.clicked.connect(self._run_spatial_visualization)
        self.spatial_viz_save_btn.clicked.connect(self._save_spatial_viz_plot)
        self.spatial_color_combo.currentTextChanged.connect(self._on_spatial_viz_option_changed)
        self.spatial_point_size_spin.valueChanged.connect(self._on_spatial_viz_option_changed)
        self.spatial_show_edges_check.toggled.connect(self._on_spatial_viz_option_changed)
        self.community_run_btn.clicked.connect(self._run_community_analysis)
        self.community_save_btn.clicked.connect(self._save_community_plot)
        self.export_btn.clicked.connect(self._export_results)
        self.export_graph_btn.clicked.connect(self._export_graph)
        self.advanced_analysis_btn.clicked.connect(self._open_advanced_analysis)
        
        self._update_tab_states()
        self._load_cluster_annotations()
        self._populate_roi_combo()
        self._populate_spatial_color_options()
        self._populate_community_roi_combo()
        self._populate_exclude_clusters_list()
        
        self.annotation_timer = QtCore.QTimer()
        self.annotation_timer.timeout.connect(self._check_annotation_updates)
        self.annotation_timer.start(2000)

    def _on_mode_changed(self):
        """Handle mode change to show/hide relevant controls."""
        mode = self.graph_mode_combo.currentText()
        is_knn = mode == "kNN"
        is_delaunay = mode == "Delaunay"

        self.k_label.setVisible(is_knn)
        self.k_spin.setVisible(is_knn)
        
        self.radius_label.setVisible(not is_knn and not is_delaunay)
        self.radius_spin.setVisible(not is_knn and not is_delaunay)

    def _update_tab_states(self):
        """Update tab enabled/disabled states based on analysis progress."""
        # Check if graph is built
        graph_built = self.edge_df is not None and not self.edge_df.empty
        
        # Enable visualization tabs when graph is built
        self.tabs.setTabEnabled(0, graph_built)  # Pairwise Enrichment
        self.tabs.setTabEnabled(1, graph_built)  # Distance Distributions
        self.tabs.setTabEnabled(2, graph_built)  # Spatial Visualization
        self.tabs.setTabEnabled(3, graph_built)  # Spatial Communities
        
        # Enable run buttons when graph is built
        self.enrichment_run_btn.setEnabled(graph_built)
        self.distance_run_btn.setEnabled(graph_built)
        self.spatial_viz_run_btn.setEnabled(graph_built)
        self.community_run_btn.setEnabled(graph_built)
        
        # Save buttons depend on their respective analyses being run
        self.enrichment_save_btn.setEnabled(self.enrichment_analysis_run)
        self.distance_save_btn.setEnabled(self.distance_analysis_run)
        self.spatial_viz_save_btn.setEnabled(self.spatial_viz_run)
        self.community_save_btn.setEnabled(self.community_analysis_run)

        self.export_graph_btn.setEnabled(graph_built)

    def _load_cluster_annotations(self):
        """Load cluster annotations from parent dialog if available."""
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, 'cluster_annotation_map'):
                self.cluster_annotation_map = parent.cluster_annotation_map.copy()
            elif parent is not None and hasattr(parent, '_get_cluster_display_name'):
                pass
            
            if self.feature_dataframe is not None and 'cluster_phenotype' in self.feature_dataframe.columns:
                phenotype_map = {}
                for cluster_id in self.feature_dataframe['cluster'].unique():
                    if pd.notna(cluster_id):
                        phenotype_rows = self.feature_dataframe[
                            (self.feature_dataframe['cluster'] == cluster_id) & 
                            (self.feature_dataframe['cluster_phenotype'].notna()) &
                            (self.feature_dataframe['cluster_phenotype'] != '')
                        ]
                        if not phenotype_rows.empty:
                            phenotype_map[cluster_id] = phenotype_rows['cluster_phenotype'].iloc[0]
                
                if phenotype_map:
                    self.cluster_annotation_map.update(phenotype_map)
                    
        except Exception as e:
            pass

    def _get_cluster_display_name(self, cluster_id):
        """Return display label for a cluster id, using annotation if available."""
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, '_get_cluster_display_name'):
                return parent._get_cluster_display_name(cluster_id)
        except Exception:
            pass
        
        if isinstance(self.cluster_annotation_map, dict) and cluster_id in self.cluster_annotation_map and self.cluster_annotation_map[cluster_id]:
            name = self.cluster_annotation_map[cluster_id]
            return name.replace('_', ' ')
            
        return f"Cluster {int(cluster_id)}"

    def _check_annotation_updates(self):
        """Periodically check for cluster annotation updates from parent."""
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, 'cluster_annotation_map'):
                new_map = parent.cluster_annotation_map
                if new_map != self.cluster_annotation_map:
                    self.cluster_annotation_map = new_map.copy()
                    # Update plots if they're already generated
                    if self.enrichment_analysis_run:
                        self._update_enrichment_plot()
        except Exception:
            pass

    def _validate_data(self):
        """Validate that required data is available for analysis."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "Feature dataframe is empty.")
            return False
        
        filtered_df = self._get_filtered_dataframe()
        if filtered_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", 
                "No data available for selected source files.")
            return False
        
        roi_col = self._get_roi_column()
        required_cols = {roi_col, "cell_id", "centroid_x", "centroid_y"}
        missing = [c for c in required_cols if c not in self.feature_dataframe.columns]
        if missing:
            QtWidgets.QMessageBox.critical(self, "Missing columns", 
                f"Missing required columns: {', '.join(missing)}")
            return False
        
        return True

    # NOTE: The full implementations of these methods are in the original spatial_analysis.py file.
    # They need to be copied from lines 2321-4868. For now, importing from the original file.
    # TODO: Extract full method implementations from spatial_analysis.py
    
    def _build_spatial_graph(self):
        """Build the spatial graph (edges and adjacency matrices) using core.build_spatial_graph."""
        if hasattr(self, 'edge_df') and self.edge_df is not None and not self.edge_df.empty:
            return True
        
        mode = self.graph_mode_combo.currentText()
        k = int(self.k_spin.value())
        radius_um = float(self.radius_spin.value())
        
        # Get seed from UI and set for reproducibility
        self.rng_seed = self.seed_spinbox.value()
        random.seed(self.rng_seed)
        np.random.seed(self.rng_seed)

        try:
            # Initialize global cell ID mapping
            self.cell_id_to_gid = {}
            self.gid_to_cell_id = {}
            self.adj_matrices = {}
            
            parent = self.parent() if hasattr(self, 'parent') else None

            # Get filtered dataframe (respects source file filter)
            filtered_df = self._get_filtered_dataframe()
            
            # Get pixel size (use first ROI's pixel size as default)
            pixel_size_um = 1.0
            roi_col = self._get_roi_column()
            if roi_col and roi_col in filtered_df.columns:
                first_roi = filtered_df[roi_col].iloc[0] if len(filtered_df) > 0 else None
                if first_roi is not None and parent is not None and hasattr(parent, '_get_pixel_size_um'):
                    try:
                        pixel_size_um = float(parent._get_pixel_size_um(first_roi))
                    except Exception:
                        pixel_size_um = 1.0
            
            # Use core.build_spatial_graph to build edges
            edges_df, _ = build_spatial_graph(
                features_df=filtered_df,
                method=mode,
                k_neighbors=k,
                radius=radius_um if mode == "Radius" else None,
                pixel_size_um=pixel_size_um,
                roi_column=roi_col,
                detect_communities=False,  # We don't need communities here
                community_seed=self.rng_seed,
                output_path=None
            )
            
            self.edge_df = edges_df
            
            # Build adjacency matrices from edges (GUI-specific for visualization)
            if _HAVE_SPARSE and not self.edge_df.empty:
                roi_groups = list(filtered_df.groupby(roi_col)) if roi_col and roi_col in filtered_df.columns else [(None, filtered_df)]
                global_id_counter = 0
                
                for roi_id, roi_df in roi_groups:
                    roi_id_str = str(roi_id) if roi_id is not None else "global"
                    roi_edges = self.edge_df[self.edge_df['roi_id'] == roi_id_str] if 'roi_id' in self.edge_df.columns else self.edge_df
                    
                    if roi_edges.empty:
                        continue
                    
                    # Get cell IDs for this ROI
                    cell_ids = roi_df["cell_id"].astype(int).to_numpy() if 'cell_id' in roi_df.columns else roi_df.index.values
                    n_cells = len(cell_ids)
                    
                    # Build global ID mapping for this ROI
                    roi_cell_to_gid = {}
                    for i, cell_id in enumerate(cell_ids):
                        gid = global_id_counter + i
                        self.cell_id_to_gid[(roi_id_str, int(cell_id))] = gid
                        self.gid_to_cell_id[gid] = (roi_id_str, int(cell_id))
                        roi_cell_to_gid[int(cell_id)] = gid
                    
                    global_id_counter += n_cells
                    
                    # Build adjacency matrix from edges
                    rows, cols, data = [], [], []
                    for _, edge in roi_edges.iterrows():
                        src_cell_id = int(edge['cell_id_A'])
                        dst_cell_id = int(edge['cell_id_B'])
                        
                        if src_cell_id in roi_cell_to_gid and dst_cell_id in roi_cell_to_gid:
                            src_gid = roi_cell_to_gid[src_cell_id]
                            dst_gid = roi_cell_to_gid[dst_cell_id]
                            
                            # Convert global IDs to local indices for this ROI
                            src_local = src_gid - (global_id_counter - n_cells)
                            dst_local = dst_gid - (global_id_counter - n_cells)
                            
                            # Add both directions (undirected graph)
                            rows.extend([src_local, dst_local])
                            cols.extend([dst_local, src_local])
                            data.extend([1.0, 1.0])
                    
                    if rows:
                        adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
                        self.adj_matrices[roi_id_str] = adj_matrix.tocsr()

            # Update metadata
            roi_groups = list(filtered_df.groupby(roi_col)) if roi_col and roi_col in filtered_df.columns else [(None, filtered_df)]
            self.metadata.update({
                "mode": mode,
                "k": k,
                "radius_um": radius_um,
                "rng_seed": self.rng_seed,
                "num_edges": int(len(self.edge_df)),
                "num_rois": len(roi_groups),
                "pixel_size_um": pixel_size_um,
            })

            # Log graph construction
            logger = get_logger()
            acquisitions = [roi_id for roi_id, _ in roi_groups] if roi_groups else []
            params = {
                "mode": mode,
                "k": k,
                "radius_um": radius_um,
                "seed": self.seed_spinbox.value(),
                "num_edges": int(len(self.edge_df)),
                "num_rois": len(roi_groups),
                "pixel_size_um": pixel_size_um
            }
            # Get source file names from dataframe
            source_file = self._get_source_files_for_logging()
            logger.log_spatial_analysis(
                analysis_type="graph_construction",
                parameters=params,
                acquisitions=acquisitions,
                notes=f"Built spatial graph with {len(self.edge_df)} edges across {len(roi_groups)} ROIs",
                source_file=source_file
            )
            
            # Enable export graph button now that graph is built
            self._update_tab_states()
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Spatial Graph Error", f"Error: {str(e)}\n\nCheck console for detailed debug information.")
            return False
    
    def _on_build_graph_clicked(self):
        """Handle build graph button click."""
        if not self._validate_data():
            return
        
        self.build_graph_btn.setEnabled(False)
        try:
            if self._build_spatial_graph():
                QtWidgets.QMessageBox.information(self, "Graph Built", "Spatial graph built successfully.")
            else:
                QtWidgets.QMessageBox.warning(self, "Graph Build Failed", "Failed to build spatial graph.")
        finally:
            self.build_graph_btn.setEnabled(True)
    
    def _run_enrichment_analysis(self):
        """Run pairwise enrichment analysis."""
        
        if not self._validate_data():
            return
            
        if self.edge_df is None or self.edge_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Graph", "Please build the spatial graph first using the 'Build Graph' button.")
            return
            
        try:
            n_perm = int(self.n_perm_spin.value())
            self._compute_pairwise_enrichment(n_perm=n_perm)
            
            self.enrichment_analysis_run = True
            
            
            # Log enrichment analysis
            logger = get_logger()
            filtered_df = self._get_filtered_dataframe()
            roi_col = self._get_roi_column()
            acquisitions = list(filtered_df[roi_col].unique()) if roi_col in filtered_df.columns else []
            params = {
                "n_permutations": n_perm,
                "seed": self.seed_spinbox.value()
            }
            # Get source file names from dataframe
            source_file = self._get_source_files_for_logging()
            logger.log_spatial_analysis(
                analysis_type="pairwise_enrichment",
                parameters=params,
                acquisitions=acquisitions,
                notes=f"Pairwise enrichment analysis with {n_perm} permutations",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Enrichment Analysis", "Enrichment analysis completed successfully.")
            
            # Update visualization
            self._update_enrichment_plot()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Enrichment Analysis Error", f"Error: {str(e)}")
    
    def _run_distance_analysis(self):
        """Run distance distribution analysis."""
        
        if not self._validate_data():
            return
            
        if self.edge_df is None or self.edge_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Graph", "Please build the spatial graph first using the 'Build Graph' button.")
            return
            
        try:
            self._compute_distance_distributions()
            
            self.distance_analysis_run = True
            
            
            # Log distance analysis
            logger = get_logger()
            filtered_df = self._get_filtered_dataframe()
            roi_col = self._get_roi_column()
            acquisitions = list(filtered_df[roi_col].unique()) if roi_col in filtered_df.columns else []
            # Get source file names from dataframe
            source_file = self._get_source_files_for_logging()
            logger.log_spatial_analysis(
                analysis_type="distance_distribution",
                parameters={},
                acquisitions=acquisitions,
                notes="Distance distribution analysis completed",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Distance Analysis", "Distance analysis completed successfully.")
            
            # Populate cluster list and update visualization
            self._populate_distance_cluster_list()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Distance Analysis Error", f"Error: {str(e)}")
    
    def _compute_pairwise_enrichment(self, n_perm=100):
        """Compute pairwise interaction enrichment analysis using core function."""
        if self.edge_df is None or self.edge_df.empty:
            return
            
        # Use detected cluster column (already validated in _run_analysis)
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        if cluster_col is None:
            QtWidgets.QMessageBox.warning(
                self, 
                "No Cluster Column", 
                "No cluster column found. Please ensure your data has a 'cluster', 'cluster_phenotype', or 'cluster_id' column."
            )
            return
        
        # Get filtered dataframe (respects source file filter)
        filtered_df = self._get_filtered_dataframe()
        
        # Get ROI column
        roi_col = self._get_roi_column()
        seed = self.seed_spinbox.value()
        
        # Show progress dialog
        progress_dlg = QtWidgets.QProgressDialog(
            "Computing enrichment analysis...",
            "Cancel",
            0,
            0,
            self
        )
        progress_dlg.setWindowTitle("Enrichment Analysis")
        progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.setValue(0)
        QtWidgets.QApplication.processEvents()
        
        try:
            # Use core spatial_enrichment function
            self.enrichment_df = spatial_enrichment(
                features_df=filtered_df,
                edges_df=self.edge_df,
                cluster_column=cluster_col,
                n_permutations=n_perm,
                seed=seed,
                roi_column=roi_col,
                output_path=None  # Don't save, we'll use the dataframe directly
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Enrichment Analysis Error",
                f"Error computing enrichment: {str(e)}"
            )
            import traceback
            traceback.print_exc()
            self.enrichment_df = pd.DataFrame()
        finally:
            progress_dlg.close()
    
    def _compute_distance_distributions(self):
        """Compute distance distribution analysis using core function."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return
        
        if self.edge_df is None or self.edge_df.empty:
            return
            
        # Use detected cluster column (already validated in _run_analysis)
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        if cluster_col is None:
            QtWidgets.QMessageBox.warning(
                self, 
                "No Cluster Column", 
                "No cluster column found. Please ensure your data has a 'cluster', 'cluster_phenotype', or 'cluster_id' column."
            )
            return
        
        # Get pixel size for distance conversion
        parent = self.parent() if hasattr(self, 'parent') else None
        
        # Get filtered dataframe (respects source file filter)
        filtered_df = self._get_filtered_dataframe()
        
        # Get ROI column
        roi_col = self._get_roi_column()
        
        # Show progress dialog
        progress_dlg = QtWidgets.QProgressDialog(
            "Computing distance distributions...",
            "Cancel",
            0,
            0,
            self
        )
        progress_dlg.setWindowTitle("Distance Distribution Analysis")
        progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        progress_dlg.setMinimumDuration(0)
        progress_dlg.setValue(0)
        QtWidgets.QApplication.processEvents()
        
        try:
            # Use core spatial_distance_distribution function
            self.distance_df = spatial_distance_distribution(
                features_df=filtered_df,
                edges_df=self.edge_df,
                cluster_column=cluster_col,
                roi_column=roi_col,
                output_path=None  # Don't save, we'll use the dataframe directly
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Distance Distribution Error",
                f"Error computing distance distributions: {str(e)}"
            )
            import traceback
            traceback.print_exc()
            self.distance_df = pd.DataFrame()
        finally:
            progress_dlg.close()
    
    def _populate_roi_combo(self):
        """Populate ROI combo box."""
        self.roi_combo.clear()
        filtered_df = self._get_filtered_dataframe()
        roi_col = self._get_roi_column()
        unique_rois = sorted(filtered_df[roi_col].unique())
        for roi_id in unique_rois:
            self.roi_combo.addItem(str(roi_id), roi_id)
    
    def _populate_spatial_color_options(self):
        """Populate spatial color combo box with available features."""
        self.spatial_color_combo.clear()
        filtered_df = self._get_filtered_dataframe()
        
        # Add cluster columns
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in filtered_df.columns:
                self.spatial_color_combo.addItem(col)
        
        # Add numeric feature columns
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['cell_id', 'centroid_x', 'centroid_y']:
                self.spatial_color_combo.addItem(col)
    
    def _populate_community_roi_combo(self):
        """Populate community ROI combo box."""
        self.community_roi_combo.clear()
        filtered_df = self._get_filtered_dataframe()
        roi_col = self._get_roi_column()
        unique_rois = sorted(filtered_df[roi_col].unique())
        for roi_id in unique_rois:
            self.community_roi_combo.addItem(str(roi_id), roi_id)
    
    def _populate_exclude_clusters_list(self):
        """Populate exclude clusters list."""
        self.exclude_clusters_list.clear()
        filtered_df = self._get_filtered_dataframe()
        
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in filtered_df.columns:
                cluster_col = col
                break
        
        if cluster_col:
            unique_clusters = sorted(filtered_df[cluster_col].dropna().unique())
            for cluster_id in unique_clusters:
                display_name = self._get_cluster_display_name(cluster_id)
                item = QtWidgets.QListWidgetItem(display_name)
                item.setData(QtCore.Qt.UserRole, cluster_id)
                self.exclude_clusters_list.addItem(item)
    
    def _run_spatial_visualization(self):
        """Run spatial visualization for selected ROI."""
        selected_roi = self.roi_combo.currentData()
        if not selected_roi:
            QtWidgets.QMessageBox.warning(self, "No ROI Selected", "Please select an ROI to visualize.")
            return
            
        self._create_spatial_visualization(selected_roi, force_regenerate=True)
        self.spatial_viz_run = True
        self._update_tab_states()
    
    def _on_spatial_viz_option_changed(self):
        """Handle spatial visualization option changes."""
        if self.spatial_viz_run:
            selected_roi = self.roi_combo.currentData()
            if selected_roi:
                self._create_spatial_visualization(selected_roi, force_regenerate=True)
    
    def _create_spatial_visualization(self, roi_id, force_regenerate=False):
        """Create spatial visualization for a specific ROI."""
        if not force_regenerate and roi_id in self.spatial_viz_cache:
            cached_data = self.spatial_viz_cache[roi_id]
            self._render_spatial_visualization(roi_id, cached_data)
            return
            
        # Get filtered dataframe
        filtered_df = self._get_filtered_dataframe()
        roi_col = self._get_roi_column()
        roi_df = filtered_df[filtered_df[roi_col] == roi_id].copy()
        
        if roi_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", f"No data available for ROI {roi_id}.")
            return
            
        # Get color option
        color_option = self.spatial_color_combo.currentText() if hasattr(self, 'spatial_color_combo') else 'cluster'
        
        # Get pixel size
        parent = self.parent() if hasattr(self, 'parent') else None
        pixel_size_um = 1.0
        try:
            if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                pixel_size_um = float(parent._get_pixel_size_um(roi_id))
        except Exception:
            pixel_size_um = 1.0
        
        # Get coordinates
        coords_um = roi_df[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
        
        # Get color values
        if color_option in roi_df.columns:
            color_values = roi_df[color_option].values
        else:
            color_values = None
        
        # Get edges if graph is built AND checkbox is enabled (skip computation if not needed)
        edges_um = None
        show_edges = (hasattr(self, 'spatial_show_edges_check') and 
                     self.spatial_show_edges_check.isChecked())
        if show_edges and self.edge_df is not None and not self.edge_df.empty:
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)]
            if not roi_edges.empty:
                # Map cell IDs to coordinates
                cell_to_coord = dict(zip(roi_df['cell_id'], coords_um))
                edges_um = []
                for _, edge in roi_edges.iterrows():
                    cell_a = int(edge['cell_id_A'])
                    cell_b = int(edge['cell_id_B'])
                    if cell_a in cell_to_coord and cell_b in cell_to_coord:
                        edges_um.append((cell_to_coord[cell_a], cell_to_coord[cell_b]))
        
        # Cache the data
        cache_data = {
            'coords_um': coords_um,
            'color_values': color_values,
            'color_option': color_option,
            'edges_um': edges_um,
            'roi_df': roi_df
        }
        self.spatial_viz_cache[roi_id] = cache_data
        
        # Render
        self._render_spatial_visualization(roi_id, cache_data)
    
    def _render_spatial_visualization(self, roi_id, cache_data):
        """Render the spatial visualization on the canvas."""
        self.spatial_viz_canvas.figure.clear()
        ax = self.spatial_viz_canvas.figure.add_subplot(111)
        
        coords_um = cache_data['coords_um']
        color_values = cache_data['color_values']
        color_option = cache_data['color_option']
        edges_um = cache_data.get('edges_um')
        
        # Plot edges first (if available AND checkbox is enabled)
        show_edges = (hasattr(self, 'spatial_show_edges_check') and 
                     self.spatial_show_edges_check.isChecked())
        if edges_um and show_edges:
            for edge_start, edge_end in edges_um:
                ax.plot([edge_start[0], edge_end[0]], [edge_start[1], edge_end[1]], 
                       'gray', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Get point size multiplier
        point_size = (self.spatial_point_size_spin.value() 
                     if hasattr(self, 'spatial_point_size_spin') else 1.0)
        
        # Plot cells
        legend_handles = []
        if color_values is not None:
            # Color by cluster/feature
            unique_values = np.unique(color_values)
            n_values = len(unique_values)
            colors = _get_vivid_colors(n_values)
            value_to_color = {val: colors[i] for i, val in enumerate(unique_values)}
            
            # Check if coloring by cluster to create legend with cluster names
            is_cluster_coloring = color_option in ['cluster', 'cluster_id', 'cluster_phenotype']
            
            for i, coord in enumerate(coords_um):
                value = color_values[i]
                color = value_to_color.get(value, 'gray')
                ax.scatter(coord[0], coord[1], c=[color], s=20*point_size, alpha=0.7, zorder=2)
            
            # Create legend if coloring by cluster
            if is_cluster_coloring:
                for val in sorted(unique_values):
                    color = value_to_color.get(val, 'gray')
                    label = self._get_cluster_display_name(val)
                    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                     markerfacecolor=color, markersize=8, 
                                                     label=label, alpha=0.7))
        else:
            # Default color
            ax.scatter(coords_um[:, 0], coords_um[:, 1], c='blue', s=20*point_size, alpha=0.7, zorder=2)
        
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title(f'Spatial Visualization: ROI {roi_id} (colored by {color_option})')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Set y=0 at the top (image coordinates)
        ax.grid(True, alpha=0.3)
        
        # Add legend if we have cluster coloring
        if legend_handles:
            ncol = 1 if len(legend_handles) <= 10 else 2
            ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                     frameon=True, fancybox=True, shadow=True, ncol=ncol)
        
        self.spatial_viz_canvas.figure.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend
        self.spatial_viz_canvas.draw()
    
    def _run_community_analysis(self):
        """Run community detection analysis on spatial graph."""
        if self.edge_df is None or self.edge_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Graph", "Please build the spatial graph first.")
            return
            
        selected_roi = self.community_roi_combo.currentData() if hasattr(self, 'community_roi_combo') else None
        if not selected_roi:
            QtWidgets.QMessageBox.warning(self, "No ROI Selected", "Please select an ROI for community analysis.")
            return
            
        if not _HAVE_IGRAPH:
            QtWidgets.QMessageBox.warning(self, "igraph Required", 
                "Community analysis requires igraph. Please install it: pip install python-igraph")
            return
            
        try:
            # Get filtered dataframe
            filtered_df = self._get_filtered_dataframe()
            roi_col = self._get_roi_column()
            roi_df = filtered_df[filtered_df[roi_col] == selected_roi].copy()
            
            if roi_df.empty:
                return
                
            # Get edges for this ROI
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(selected_roi)]
            if roi_edges.empty:
                QtWidgets.QMessageBox.warning(self, "No Edges", f"No edges found for ROI {selected_roi}.")
                return
            
            # Build igraph graph
            cell_ids = roi_df['cell_id'].astype(int).tolist()
            cell_id_to_idx = {cell_id: i for i, cell_id in enumerate(cell_ids)}
            
            g = ig.Graph()
            g.add_vertices(len(cell_ids))
            
            for _, edge in roi_edges.iterrows():
                cell_a = int(edge['cell_id_A'])
                cell_b = int(edge['cell_id_B'])
                if cell_a in cell_id_to_idx and cell_b in cell_id_to_idx:
                    idx_a = cell_id_to_idx[cell_a]
                    idx_b = cell_id_to_idx[cell_b]
                    g.add_edge(idx_a, idx_b)
            
            # Run community detection (Louvain algorithm)
            communities = g.community_multilevel()
            community_labels = communities.membership
            
            # Store results
            roi_df['community'] = [community_labels[cell_id_to_idx[int(cid)]] for cid in roi_df['cell_id']]
            
            # Update visualization
            self._update_community_plot(selected_roi, roi_df, community_labels)
            
            self.community_analysis_run = True
            QtWidgets.QMessageBox.information(self, "Community Analysis", 
                f"Detected {len(set(community_labels))} communities in ROI {selected_roi}.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Community Analysis Error", f"Error: {str(e)}")
    
    def _update_community_plot(self, roi_id, roi_df, community_labels):
        """Update community plot visualization."""
        self.community_canvas.figure.clear()
        ax = self.community_canvas.figure.add_subplot(111)
        
        # Get pixel size
        parent = self.parent() if hasattr(self, 'parent') else None
        pixel_size_um = 1.0
        try:
            if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                pixel_size_um = float(parent._get_pixel_size_um(roi_id))
        except Exception:
            pixel_size_um = 1.0
        
        # Get coordinates
        coords_um = roi_df[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
        
        # Color by community
        n_communities = len(set(community_labels))
        colors = _get_vivid_colors(n_communities)
        community_colors = {comm: colors[i] for i, comm in enumerate(set(community_labels))}
        
        for i, coord in enumerate(coords_um):
            comm = community_labels[i]
            color = community_colors.get(comm, 'gray')
            ax.scatter(coord[0], coord[1], c=[color], s=20, alpha=0.7)
        
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title(f'Spatial Communities: ROI {roi_id} ({n_communities} communities)')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Set y=0 at the top (image coordinates)
        ax.grid(True, alpha=0.3)
        
        self.community_canvas.figure.tight_layout()
        self.community_canvas.draw()
    
    def _save_enrichment_plot(self):
        """Save the enrichment plot."""
        if save_figure_with_options(self.enrichment_canvas.figure, "pairwise_enrichment.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _save_distance_plot(self):
        """Save the distance plot."""
        if save_figure_with_options(self.distance_canvas.figure, "distance_distributions.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _save_spatial_viz_plot(self):
        """Save the spatial visualization plot."""
        if save_figure_with_options(self.spatial_viz_canvas.figure, "spatial_visualization.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _save_community_plot(self):
        """Save the community plot."""
        if save_figure_with_options(self.community_canvas.figure, "spatial_communities.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _export_results(self):
        """Export analysis results to CSV files."""
        from PyQt5.QtWidgets import QFileDialog
        
        if not any([self.enrichment_df is not None, self.distance_df is not None]):
            QtWidgets.QMessageBox.warning(self, "No Results", "No analysis results to export. Please run analyses first.")
            return
        
        # Get export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        
        try:
            # Export enrichment results
            if self.enrichment_df is not None and not self.enrichment_df.empty:
                file_path = os.path.join(export_dir, "pairwise_enrichment.csv")
                self.enrichment_df.to_csv(file_path, index=False)
            
            # Export distance distributions
            if self.distance_df is not None and not self.distance_df.empty:
                file_path = os.path.join(export_dir, "distance_distributions.csv")
                self.distance_df.to_csv(file_path, index=False)
            
            QtWidgets.QMessageBox.information(self, "Export Complete", 
                f"Results exported to:\n{export_dir}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def _export_graph(self):
        """Export spatial graph to CSV file."""
        from PyQt5.QtWidgets import QFileDialog
        
        if self.edge_df is None or self.edge_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Graph", "No spatial graph to export. Please build the graph first.")
            return
        
        # Get export file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Spatial Graph", "spatial_graph.csv", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            self.edge_df.to_csv(file_path, index=False)
            QtWidgets.QMessageBox.information(self, "Export Complete", 
                f"Spatial graph exported to:\n{file_path}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export graph: {str(e)}")
    
    def _open_advanced_analysis(self):
        """Open the advanced spatial analysis dialog and close this simple dialog."""
        # Check if AdvancedSpatialAnalysisDialog is available
        if not _HAVE_SQUIDPY:
            QtWidgets.QMessageBox.warning(
                self,
                "Squidpy Not Available",
                "Advanced Spatial Analysis requires squidpy, which is not installed.\n\n"
                "Please install squidpy to use advanced spatial analysis features."
            )
            return
        
        try:
            from openimc.ui.dialogs.advanced_spatial_analysis import AdvancedSpatialAnalysisDialog
        except (ImportError, RuntimeError) as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Squidpy Not Available",
                f"Advanced Spatial Analysis requires squidpy, which could not be imported.\n\n"
                f"Error: {str(e)}\n\n"
                "Please install squidpy to use advanced spatial analysis features."
            )
            return
        
        # Get parent to access the method for opening advanced dialog
        parent = self.parent()
        if parent is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Error",
                "Cannot open advanced analysis: no parent window found."
            )
            return
        
        # Close this simple dialog
        self.close()
        
        # Open advanced dialog using parent's method
        if hasattr(parent, '_open_advanced_spatial_dialog'):
            parent._open_advanced_spatial_dialog()
        else:
            # Fallback: create dialog directly
            advanced_dialog = AdvancedSpatialAnalysisDialog(
                self.original_feature_dataframe,
                batch_corrected_dataframe=self.batch_corrected_dataframe,
                parent=parent
            )
            advanced_dialog.setModal(False)
            advanced_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            advanced_dialog.show()
    
    def _update_enrichment_plot(self):
        """Update enrichment plot with pairwise enrichment data."""
        if self.enrichment_df is None or self.enrichment_df.empty:
            return
            
        self.enrichment_canvas.figure.clear()
        ax = self.enrichment_canvas.figure.add_subplot(111)
        
        # Aggregate across ROIs if multiple
        if 'roi_id' in self.enrichment_df.columns:
            # Average z-scores across ROIs
            enrichment_agg = self.enrichment_df.groupby(['cluster_A', 'cluster_B']).agg({
                'z_score': 'mean',
                'p_value': lambda x: np.mean(x < 0.05)  # Fraction significant
            }).reset_index()
        else:
            enrichment_agg = self.enrichment_df.copy()
        
        # Get unique clusters
        all_clusters = set(enrichment_agg['cluster_A'].unique()) | set(enrichment_agg['cluster_B'].unique())
        unique_clusters = sorted(all_clusters)
        n_clusters = len(unique_clusters)
        
        if n_clusters == 0:
            return
            
        # Create heatmap matrix
        heatmap_data = np.zeros((n_clusters, n_clusters))
        pvalue_data = np.ones((n_clusters, n_clusters))
        
        for _, row in enrichment_agg.iterrows():
            cluster_a = row['cluster_A']
            cluster_b = row['cluster_B']
            
            i = unique_clusters.index(cluster_a)
            j = unique_clusters.index(cluster_b)
            
            heatmap_data[i, j] = row['z_score']
            if 'p_value' in row:
                pvalue_data[i, j] = row['p_value']
        
        # Create symmetric matrix (average both directions)
        heatmap_data = (heatmap_data + heatmap_data.T) / 2
        pvalue_data = np.minimum(pvalue_data, pvalue_data.T)
        
        # Create heatmap
        vmax = np.max(np.abs(heatmap_data)) if np.max(np.abs(heatmap_data)) > 0 else 1
        # Use TwoSlopeNorm to center colormap at 0 for diverging data
        try:
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', norm=norm)
        except ImportError:
            # Fallback for older matplotlib versions
            im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_clusters))
        ax.set_yticks(np.arange(n_clusters))
        ax.set_xticklabels([self._get_cluster_display_name(c) for c in unique_clusters], rotation=45, ha='right')
        ax.set_yticklabels([self._get_cluster_display_name(c) for c in unique_clusters])
        
        # Add colorbar
        self.enrichment_canvas.figure.colorbar(im, ax=ax, label='Z-Score')
        
        # Add text annotations with significance markers
        for i in range(n_clusters):
            for j in range(n_clusters):
                z_score = heatmap_data[i, j]
                p_val = pvalue_data[i, j]
                sig_marker = '*' if p_val < 0.05 else ''
                text = ax.text(j, i, f'{z_score:.2f}{sig_marker}',
                             ha="center", va="center", 
                             color="white" if abs(z_score) > vmax/2 else "black",
                             fontweight='bold' if p_val < 0.05 else 'normal')
        
        ax.set_title("Pairwise Enrichment: Z-Scores (Positive = Enriched, Negative = Depleted)")
        ax.set_xlabel("Cluster B")
        ax.set_ylabel("Cluster A")
        
        self.enrichment_canvas.figure.tight_layout()
        self.enrichment_canvas.draw()
    
    def _update_distance_plot(self):
        """Update distance plot with distance distribution data."""
        if self.distance_df is None or self.distance_df.empty:
            return
            
        self.distance_canvas.figure.clear()
        ax = self.distance_canvas.figure.add_subplot(111)
        
        # Get selected clusters for filtering
        selected_clusters = self._get_selected_distance_clusters()
        
        # If no clusters are selected, show message
        if not selected_clusters:
            ax.text(0.5, 0.5, 'No clusters selected.\nPlease select clusters to view distance distributions.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            self.distance_canvas.figure.tight_layout()
            self.distance_canvas.draw()
            return
        
        # Get unique cluster pairs
        unique_pairs = self.distance_df[['cell_A_cluster', 'nearest_B_cluster']].drop_duplicates()
        
        # Filter pairs based on selected clusters (show pairs where both clusters are selected)
        unique_pairs = unique_pairs[
            (unique_pairs['cell_A_cluster'].isin(selected_clusters)) &
            (unique_pairs['nearest_B_cluster'].isin(selected_clusters))
        ]
        
        # Create violin/box plot for each pair
        plot_data = []
        plot_labels = []
        
        for _, pair_row in unique_pairs.iterrows():
            cluster_a = pair_row['cell_A_cluster']
            cluster_b = pair_row['nearest_B_cluster']
            
            pair_data = self.distance_df[
                (self.distance_df['cell_A_cluster'] == cluster_a) &
                (self.distance_df['nearest_B_cluster'] == cluster_b)
            ]['nearest_B_dist_um'].values
            
            if len(pair_data) > 0:
                plot_data.append(pair_data)
                plot_labels.append(f"{self._get_cluster_display_name(cluster_a)} → {self._get_cluster_display_name(cluster_b)}")
        
        if not plot_data:
            ax.text(0.5, 0.5, 'No data to display for selected clusters.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            self.distance_canvas.figure.tight_layout()
            self.distance_canvas.draw()
            return
            
        # Create box plot
        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        
        # Color boxes
        colors = _get_vivid_colors(len(plot_data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Distance to Nearest Neighbor (µm)')
        title = 'Distance Distribution: Nearest Neighbor Distances Between Cluster Pairs'
        if selected_clusters:
            title += f' (Filtered: {len(selected_clusters)} clusters)'
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        self.distance_canvas.figure.tight_layout()
        self.distance_canvas.draw()
    
    def _get_selected_distance_clusters(self):
        """Get list of selected cluster IDs from the distance cluster list."""
        selected_clusters = []
        for i in range(self.distance_cluster_list.count()):
            item = self.distance_cluster_list.item(i)
            if item.isSelected():
                cluster_id = item.data(QtCore.Qt.UserRole)
                selected_clusters.append(cluster_id)
        return selected_clusters
    
    def _on_distance_cluster_selection_changed(self):
        """Handle distance cluster selection change and update plot."""
        if self.distance_df is not None and not self.distance_df.empty:
            self._update_distance_plot()
    
    def _on_distance_select_all(self):
        """Select all clusters in the distance cluster list."""
        self.distance_cluster_list.selectAll()
    
    def _on_distance_deselect_all(self):
        """Deselect all clusters in the distance cluster list."""
        self.distance_cluster_list.clearSelection()
    
    def _populate_distance_cluster_list(self):
        """Populate the distance cluster list with available clusters."""
        if self.distance_df is None or self.distance_df.empty:
            return
        
        # Get all unique clusters from both cell_A_cluster and nearest_B_cluster
        all_clusters = set(self.distance_df['cell_A_cluster'].unique()) | set(self.distance_df['nearest_B_cluster'].unique())
        all_clusters = sorted(all_clusters)
        
        self.distance_cluster_list.blockSignals(True)
        self.distance_cluster_list.clear()
        
        for cluster in all_clusters:
            cluster_name = self._get_cluster_display_name(cluster)
            item = QtWidgets.QListWidgetItem(cluster_name)
            item.setData(QtCore.Qt.UserRole, cluster)
            item.setSelected(True)  # Select all by default
            self.distance_cluster_list.addItem(item)
        
        self.distance_cluster_list.blockSignals(False)
        
        # Enable buttons
        self.distance_select_all_btn.setEnabled(True)
        self.distance_deselect_all_btn.setEnabled(True)
        
        # Update plot with initial selection (all selected)
        self._update_distance_plot()

    def reset_analysis_state(self):
        """Reset all analysis state - clear results and allow restart."""
        # Clear all dataframes
        self.edge_df = None
        self.adj_matrices = {}
        self.cell_id_to_gid = {}
        self.gid_to_cell_id = {}
        self.metadata = {}
        self.cluster_summary_df = None
        self.enrichment_df = None
        self.distance_df = None
        self.spatial_viz_cache = {}
        
        # Reset analysis flags
        self.enrichment_analysis_run = False
        self.distance_analysis_run = False
        self.spatial_viz_run = False
        self.community_analysis_run = False
        
        # Clear all canvas figures
        if hasattr(self, 'enrichment_canvas'):
            self.enrichment_canvas.figure.clear()
            self.enrichment_canvas.draw()
        if hasattr(self, 'distance_canvas'):
            self.distance_canvas.figure.clear()
            self.distance_canvas.draw()
        if hasattr(self, 'spatial_viz_canvas'):
            self.spatial_viz_canvas.figure.clear()
            self.spatial_viz_canvas.draw()
        if hasattr(self, 'community_canvas'):
            self.community_canvas.figure.clear()
            self.community_canvas.draw()
        
        # Update tab states to disable tabs that require graph
        self._update_tab_states()
        
        # Update status labels
        if hasattr(self, 'graph_status_label'):
            self.graph_status_label.setText("Graph not created")
            self.graph_status_label.setStyleSheet("")
    
    def refresh_dataframe(self):
        """Refresh the feature dataframe from parent window."""
        parent = self.parent()
        if parent is None:
            return
        
        # Get updated dataframes from parent
        if hasattr(parent, 'feature_dataframe') and parent.feature_dataframe is not None:
            self.original_feature_dataframe = parent.feature_dataframe.copy()
            
            # Update feature_dataframe based on batch correction preference
            if hasattr(parent, 'batch_corrected_dataframe') and parent.batch_corrected_dataframe is not None and not parent.batch_corrected_dataframe.empty:
                self.batch_corrected_dataframe = parent.batch_corrected_dataframe.copy()
                if hasattr(self, 'feature_set_combo') and self.feature_set_combo.currentText() == "Batch-Corrected Features":
                    self.feature_dataframe = self.batch_corrected_dataframe.copy()
                else:
                    self.feature_dataframe = self.original_feature_dataframe.copy()
            else:
                self.batch_corrected_dataframe = None
                self.feature_dataframe = self.original_feature_dataframe.copy()
        
        # Update cluster annotation map if available
        if hasattr(parent, 'cluster_annotation_map'):
            self.cluster_annotation_map = parent.cluster_annotation_map.copy()
        
        # Refresh ROI combo boxes and other UI elements that depend on dataframe
        self._populate_roi_combo()
        if hasattr(self, '_populate_exclude_clusters_list'):
            self._populate_exclude_clusters_list()
        if hasattr(self, '_populate_spatial_color_options'):
            self._populate_spatial_color_options()
    
    def on_clusters_changed(self):
        """Handle cluster changes - reset analysis and refresh dataframe."""
        # Show info message
        reply = QtWidgets.QMessageBox.information(
            self,
            "Clusters Changed",
            "Cluster assignments have been updated. All spatial analysis results will be cleared.\n\n"
            "You can now rebuild the spatial graph and rerun analyses with the new cluster assignments.",
            QtWidgets.QMessageBox.Ok
        )
        
        # Refresh dataframe first
        self.refresh_dataframe()
        
        # Reset all analysis state
        self.reset_analysis_state()

    def closeEvent(self, event):
        """Handle dialog closing to clean up resources."""
        if hasattr(self, 'annotation_timer'):
            self.annotation_timer.stop()
        event.accept()

