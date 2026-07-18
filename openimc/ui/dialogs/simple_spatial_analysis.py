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
os.environ.setdefault('DASK_DATAFRAME__QUERY_PLANNING', 'True')

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
from matplotlib.collections import LineCollection, PathCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
from collections import defaultdict
from openimc.utils.logger import get_logger
from openimc.ui.cluster_utils import (
    canonicalize_cluster_id,
    extract_cluster_annotation_map_from_dataframe,
    format_default_cluster_label,
    get_cluster_display_name,
    is_cluster_column,
    map_cluster_series_to_display_names,
    normalize_cluster_annotation_map,
    sort_cluster_values,
)
from openimc.ui.figure_layout import dense_heatmap_style, fit_canvas_and_draw, refresh_canvas
from openimc.ui.utils import benjamini_hochberg_adjust
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.progress_dialog import (
    run_blocking_task_with_progress,
    run_blocking_task_with_progress_then_finalize,
)
from openimc.ui.dialogs.label_customization_dialogs import (
    edit_cluster_annotation_map,
    edit_feature_label_map,
)
from openimc.core import spatial_enrichment, spatial_distance_distribution, build_spatial_graph
from openimc.ui.dialogs.spatial_analysis import (
    SourceFileFilterDialog,
    _get_vivid_colors,
    _HAVE_SPARSE,
    _HAVE_IGRAPH,
    _HAVE_SEABORN,
    _HAVE_SQUIDPY,
    squidpy_available,
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
    def __init__(self, feature_dataframe: pd.DataFrame, batch_corrected_dataframe=None, clustered_cells_dataframe=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simple Spatial Analysis")
        self.setMinimumSize(900, 650)
        
        # Set size to 90% of parent window size if parent exists
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)

        self.original_feature_dataframe = feature_dataframe  # Full dataset
        self.batch_corrected_dataframe = batch_corrected_dataframe
        self.clustered_cells_dataframe = clustered_cells_dataframe  # Saved state only (for initialization)
        
        # ALWAYS start with full feature_dataframe - do NOT pre-filter
        # Filtering will be applied dynamically when needed based on current clustering state
        if batch_corrected_dataframe is not None and not batch_corrected_dataframe.empty:
            self.feature_dataframe = batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = feature_dataframe.copy()
        
        # Note: clustered_cells_dataframe is stored but NOT used to filter feature_dataframe
        # We want users to have access to all cells for new analyses
        # Filtering happens dynamically in _get_filtered_dataframe() based on current clustering dialog state
        self.edge_df: Optional[pd.DataFrame] = None
        self.adj_matrices: Dict[str, sp.csr_matrix] = {} if sp else {}
        self.cell_id_to_gid: Dict[Tuple[str, int], int] = {}
        self.gid_to_cell_id: Dict[int, Tuple[str, int]] = {}
        self.metadata: Dict[str, Any] = {}
        self.cluster_summary_df: Optional[pd.DataFrame] = None
        self.enrichment_df: Optional[pd.DataFrame] = None
        self.distance_df: Optional[pd.DataFrame] = None
        self.community_results_by_roi: Dict[str, pd.DataFrame] = {}
        self.rng_seed: int = 42
        
        self.spatial_viz_cache: Dict[str, Any] = {}
        
        self.enrichment_analysis_run = False
        self.distance_analysis_run = False
        self.spatial_viz_run = False
        self.community_analysis_run = False
        
        self.cluster_annotation_map = {}
        self.feature_label_map = {}
        self.selected_source_files = set()
        self.available_source_files = set()
        self._spatial_viz_layout_rect = [0.0, 0.0, 0.95, 1.0]

        self._create_ui()
        self._plot_resize_in_progress = False
        self._plot_resize_timer = QtCore.QTimer(self)
        self._plot_resize_timer.setSingleShot(True)
        self._plot_resize_timer.timeout.connect(self._refresh_current_plot_after_resize)
        if hasattr(self, 'tabs'):
            self.tabs.currentChanged.connect(self._queue_plot_resize_refresh)
        
        if hasattr(self, 'source_file_status_label'):
            self._update_source_file_status_label()

    def _fit_canvas(self, canvas, *, rect=None, pad: float = 0.9):
        """Fit the supplied canvas to the live tab geometry and redraw it."""
        if canvas is None or canvas.width() < 10 or canvas.height() < 10:
            return
        fit_canvas_and_draw(canvas, rect=rect, pad=pad, allow_text_compaction=True)

    def _finalize_canvas_render(self, canvas, *, rect=None, pad: float = 0.95):
        """Force an immediate draw plus a queued second fit after layout settles."""
        if canvas is None:
            return
        try:
            canvas.show()
            canvas.updateGeometry()
            refresh_canvas(canvas, draw=False)
        except Exception:
            pass

        self._fit_canvas(canvas, rect=rect, pad=pad)

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)

        def _second_pass():
            if canvas is None or not canvas.figure.axes:
                return
            self._fit_canvas(canvas, rect=rect, pad=pad)

        QtCore.QTimer.singleShot(0, _second_pass)

    def _current_plot_canvas_layout(self):
        """Return the active canvas and layout reservation for the current tab."""
        if not hasattr(self, 'tabs'):
            return None
        current_tab = self.tabs.currentWidget()
        if current_tab is getattr(self, 'enrichment_tab', None) and self.enrichment_canvas.figure.axes:
            return self.enrichment_canvas, None, 0.95
        if current_tab is getattr(self, 'distance_tab', None) and self.distance_canvas.figure.axes:
            return self.distance_canvas, None, 0.95
        if current_tab is getattr(self, 'spatial_viz_tab', None) and self.spatial_viz_canvas.figure.axes:
            rect = self._calculate_spatial_viz_layout_rect()
            return self.spatial_viz_canvas, rect, 0.95
        if current_tab is getattr(self, 'community_tab', None) and self.community_canvas.figure.axes:
            return self.community_canvas, None, 0.95
        return None

    def _measure_spatial_viz_artist_bbox(self, artist):
        """Measure an artist in display coordinates using the live spatial canvas renderer."""
        if artist is None or not hasattr(self, 'spatial_viz_canvas'):
            return None
        canvas = self.spatial_viz_canvas
        figure = getattr(canvas, 'figure', None)
        if figure is None or getattr(figure, 'canvas', None) is None:
            return None
        try:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            return artist.get_window_extent(renderer=renderer)
        except Exception:
            return None

    def _spatial_viz_legend_kwargs(self):
        """Return compact defaults for the external categorical legend."""
        return {
            'loc': 'center left',
            'bbox_to_anchor': (1.01, 0.5),
            'frameon': True,
            'fancybox': True,
            'shadow': False,
            'fontsize': 8,
            'borderaxespad': 0.0,
            'borderpad': 0.35,
            'labelspacing': 0.35,
            'handletextpad': 0.35,
            'columnspacing': 0.75,
            'markerscale': 0.95,
        }

    def _choose_spatial_viz_legend_columns(self, ax, legend_handles, legend_kwargs=None):
        """Prefer the narrowest legend that still fits within the canvas height."""
        if not legend_handles:
            return 1
        if legend_kwargs is None:
            legend_kwargs = self._spatial_viz_legend_kwargs()

        figure = self.spatial_viz_canvas.figure
        best_ncol = 1
        best_height_fraction = float('inf')
        target_height_fraction = 0.78
        max_cols = min(4, len(legend_handles))

        for ncol in range(1, max_cols + 1):
            legend = ax.legend(handles=legend_handles, ncol=ncol, **legend_kwargs)
            bbox = self._measure_spatial_viz_artist_bbox(legend)
            try:
                legend.remove()
            except Exception:
                pass

            if bbox is None:
                continue

            height_fraction = bbox.height / max(1.0, float(figure.bbox.height))
            if height_fraction < best_height_fraction:
                best_height_fraction = height_fraction
                best_ncol = ncol
            if height_fraction <= target_height_fraction:
                return ncol

        return best_ncol

    def _calculate_spatial_viz_layout_rect(self, ax=None, *, base_right: float = 0.95):
        """Reserve only the right-side width actually needed by the spatial legend/colorbar."""
        if not hasattr(self, 'spatial_viz_canvas'):
            return None
        figure = self.spatial_viz_canvas.figure
        if not figure.axes:
            return self._spatial_viz_layout_rect

        rect = [0.0, 0.0, float(base_right), 1.0]
        if ax is None:
            ax = figure.axes[0]

        legend = ax.get_legend()
        if legend is not None:
            bbox = self._measure_spatial_viz_artist_bbox(legend)
            if bbox is not None:
                reserve_fraction = (bbox.width / max(1.0, float(figure.bbox.width))) + 0.02
                reserve_fraction = min(0.36, max(0.12, reserve_fraction))
                rect[2] = max(0.60, min(rect[2], 1.0 - reserve_fraction))
                return rect
            return [0.0, 0.0, 0.82, 1.0]

        if len(figure.axes) > 1:
            rect[2] = min(rect[2], 0.88)

        return rect

    def _queue_plot_resize_refresh(self, *_args):
        """Debounce plot reflow while the dialog is being resized or tabs switch."""
        if self._plot_resize_in_progress or not self.isVisible():
            return
        if self._current_plot_canvas_layout() is None:
            return
        self._plot_resize_timer.start(140)

    def _refresh_current_plot_after_resize(self):
        """Refit the currently visible plot to the latest canvas size."""
        if self._plot_resize_in_progress:
            return
        layout = self._current_plot_canvas_layout()
        if layout is None:
            return
        canvas, rect, pad = layout
        try:
            self._plot_resize_in_progress = True
            self._fit_canvas(canvas, rect=rect, pad=pad)
        finally:
            self._plot_resize_in_progress = False

    def resizeEvent(self, event):
        """Keep the active spatial plot fitted to the dialog canvas."""
        super().resizeEvent(event)
        if event is None:
            return
        try:
            old_size = event.oldSize()
            new_size = event.size()
            if old_size.isValid() and new_size == old_size:
                return
        except Exception:
            pass
        self._queue_plot_resize_refresh()
    
    def _get_roi_column(self):
        """Get the appropriate ROI column name."""
        if self.feature_dataframe is not None and 'source_well' in self.feature_dataframe.columns:
            return 'source_well'
        return 'acquisition_id'

    def _has_batch_corrected_features(self) -> bool:
        """Return whether a batch-corrected feature table is available."""
        return isinstance(self.batch_corrected_dataframe, pd.DataFrame) and not self.batch_corrected_dataframe.empty

    def get_active_feature_set_key(self) -> str:
        """Return the active feature-set key used by the dialog."""
        if hasattr(self, 'feature_set_combo') and self.feature_set_combo.currentText() == "Batch-Corrected Features":
            return "batch_corrected"
        return "original"

    def _feature_set_text_for_key(self, feature_set_key: Optional[str] = None) -> str:
        """Map an internal feature-set key to the current UI label."""
        if feature_set_key == "batch_corrected" and self._has_batch_corrected_features():
            return "Batch-Corrected Features"
        return "Original Features" if self._has_batch_corrected_features() else "Loaded Features"

    def _refresh_feature_set_combo(self, preferred_feature_set: Optional[str] = None):
        """Rebuild feature-set options to match currently available data."""
        if not hasattr(self, 'feature_set_combo'):
            return

        target_text = self._feature_set_text_for_key(preferred_feature_set or self.get_active_feature_set_key())
        self.feature_set_combo.blockSignals(True)
        self.feature_set_combo.clear()

        if self._has_batch_corrected_features():
            self.feature_set_combo.addItem("Original Features")
            self.feature_set_combo.addItem("Batch-Corrected Features")
            self.feature_set_combo.setToolTip("Choose between original or batch-corrected feature sets")
        else:
            self.feature_set_combo.addItem("Loaded Features")
            self.feature_set_combo.setToolTip("Only one feature set is currently available (the loaded feature table)")

        self.feature_set_combo.setCurrentText(target_text)
        self.feature_set_combo.blockSignals(False)

    def _update_active_feature_dataframe(self):
        """Align the active feature dataframe to the current feature-set selection."""
        if self.get_active_feature_set_key() == "batch_corrected" and self._has_batch_corrected_features():
            self.feature_dataframe = self.batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = self.original_feature_dataframe.copy()

    def _on_feature_set_changed(self):
        """Handle feature set selection change."""
        if not hasattr(self, 'feature_set_combo'):
            return

        self._update_active_feature_dataframe()
        self._clear_analysis_cache()
        self._update_source_file_filter()

        parent = self.parent()
        if parent is not None and hasattr(parent, '_set_analysis_feature_set_preference'):
            parent._set_analysis_feature_set_preference(self.get_active_feature_set_key(), source_dialog=self)
        
    
    def _get_filtered_dataframe(self):
        """Get the filtered dataframe based on selected source files and cell filters."""
        df = self.feature_dataframe.copy()
        
        # Apply source file filtering
        if ('source_file' in df.columns and 
            hasattr(self, 'selected_source_files') and 
            self.selected_source_files and 
            len(self.selected_source_files) > 0):
            df = df[df['source_file'].isin(self.selected_source_files)]
        
        # Apply cell filters from clustering dialog if available
        filter_settings = self._get_filter_settings_from_parent()
        if filter_settings:
            df = self._apply_cell_filters(df, filter_settings)
        
        return df
    
    def _get_filter_settings_from_parent(self):
        """Get filter settings from parent's clustering dialog if available."""
        parent = self.parent()
        if parent is not None and hasattr(parent, 'clustering_dialog'):
            clustering_dialog = parent.clustering_dialog
            if clustering_dialog is not None and hasattr(clustering_dialog, 'filter_settings'):
                return clustering_dialog.filter_settings
        return None
    
    def _apply_cell_filters(self, df, filter_settings):
        """Apply cell filtering based on filter settings.
        
        Args:
            df: DataFrame to filter
            filter_settings: Dictionary with filter settings
        
        Returns:
            Filtered DataFrame
        """
        if filter_settings is None:
            return df
        
        filtered_df = df.copy()
        
        # Exclude cells touching edge
        if filter_settings.get('exclude_edge_cells', False):
            if 'touches_edge' in filtered_df.columns:
                # Use .eq(False) instead of ~ to avoid numpy boolean subtraction issues
                filtered_df = filtered_df[filtered_df['touches_edge'].astype(bool).eq(False)]
            elif 'touches_border' in filtered_df.columns:
                # Fallback to touches_border if touches_edge not available
                filtered_df = filtered_df[filtered_df['touches_border'].astype(bool).eq(False)]
        
        # Filter by area
        if 'area_um2' in filtered_df.columns:
            min_area = filter_settings.get('min_area')
            max_area = filter_settings.get('max_area')
            
            if min_area is not None:
                filtered_df = filtered_df[filtered_df['area_um2'] >= min_area]
            if max_area is not None:
                filtered_df = filtered_df[filtered_df['area_um2'] <= max_area]
        
        return filtered_df
    
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

        feature_set_layout = QtWidgets.QHBoxLayout()
        feature_set_layout.addWidget(QtWidgets.QLabel("Feature Set:"))
        self.feature_set_combo = QtWidgets.QComboBox()
        self._refresh_feature_set_combo(
            preferred_feature_set="batch_corrected" if self._has_batch_corrected_features() else "original"
        )
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
        self.export_btn = QtWidgets.QPushButton("Export All Results…")
        self.export_btn.setEnabled(False)
        self.export_graph_btn = QtWidgets.QPushButton("Export Graph…")
        self.export_graph_btn.setEnabled(False)
        action_row.addWidget(self.export_btn)
        action_row.addWidget(self.export_graph_btn)
        self.cluster_labels_btn = QtWidgets.QPushButton("Customize Cluster Names…")
        self.cluster_labels_btn.setToolTip("Set custom display names for spatial-analysis cluster labels.")
        action_row.addWidget(self.cluster_labels_btn)
        self.feature_labels_btn = QtWidgets.QPushButton("Customize Feature Labels…")
        self.feature_labels_btn.setToolTip("Set custom display names for features used in spatial analysis.")
        action_row.addWidget(self.feature_labels_btn)
        action_row.addStretch(1)
        
        # Advanced analysis button
        self.advanced_analysis_btn = QtWidgets.QPushButton("Advanced analysis")
        self.advanced_analysis_btn.setToolTip(
            "Open Advanced Spatial Analysis using Squidpy for more sophisticated spatial analysis methods, "
            "including neighborhood enrichment, co-occurrence analysis, spatial autocorrelation, and Ripley functions. "
            "Requires squidpy to be installed."
        )
        self.advanced_analysis_btn.setStyleSheet(
            """
            QPushButton {
                font-weight: 700;
                padding: 8px 14px;
                border: 1px solid #2d6a4f;
                border-radius: 6px;
                color: #123524;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #e9f7ef,
                    stop: 1 #cfead8
                );
            }
            QPushButton:hover {
                background: #d8f1e0;
            }
            QPushButton:pressed {
                background: #c4e3cf;
            }
            """
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
        
        # Warning label about computation time
        enrichment_warning = QtWidgets.QLabel("⚠ Warning: Large datasets with high numbers of permutations may take a long time to compute.")
        enrichment_warning.setWordWrap(True)
        enrichment_warning.setStyleSheet("color: #d9534f; font-weight: bold; padding: 5px; background-color: #fcf8e3; border: 1px solid #f0ad4e; border-radius: 3px;")
        enrichment_layout.addWidget(enrichment_warning)
        
        enrichment_params = QtWidgets.QHBoxLayout()
        enrichment_params.addWidget(QtWidgets.QLabel("Permutations:"))
        enrichment_params.addWidget(self.n_perm_spin)
        enrichment_params.addWidget(QtWidgets.QLabel("Workers:"))
        enrichment_params.addWidget(self.workers_spin)
        self.enrichment_run_btn = QtWidgets.QPushButton("Run Enrichment Analysis")
        self.enrichment_save_btn = QtWidgets.QPushButton("Save Plot")
        self.enrichment_save_btn.setEnabled(False)
        self.enrichment_export_btn = QtWidgets.QPushButton("Export Results…")
        self.enrichment_export_btn.setEnabled(False)
        enrichment_params.addWidget(self.enrichment_run_btn)
        enrichment_params.addWidget(self.enrichment_save_btn)
        enrichment_params.addWidget(self.enrichment_export_btn)
        enrichment_params.addStretch()
        enrichment_layout.addLayout(enrichment_params)
        
        self.enrichment_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        enrichment_layout.addWidget(self.enrichment_canvas)
        self.tabs.addTab(self.enrichment_tab, "Pairwise Enrichment")
        
        # Distance Distributions tab
        self.distance_tab = QtWidgets.QWidget()
        distance_layout = QtWidgets.QVBoxLayout(self.distance_tab)
        
        distance_desc = QtWidgets.QLabel(
            "Computes nearest neighbor distances from each cell to each cluster type. "
            "For each cell in cluster A, finds the distance to the nearest cell in cluster B. "
            "Results show distance distributions as box plots to assess spatial relationships between clusters. "
            "Shorter distances indicate closer spatial proximity, longer distances indicate spatial separation."
        )
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
        self.distance_export_btn = QtWidgets.QPushButton("Export Results…")
        self.distance_export_btn.setEnabled(False)
        distance_btn_layout.addWidget(self.distance_run_btn)
        distance_btn_layout.addWidget(self.distance_save_btn)
        distance_btn_layout.addWidget(self.distance_export_btn)
        distance_btn_layout.addStretch()
        distance_layout.addLayout(distance_btn_layout)
        
        # Cluster selection for displaying distance distributions
        distance_cluster_layout = QtWidgets.QHBoxLayout()
        distance_cluster_layout.addWidget(QtWidgets.QLabel("Select clusters to display:"))
        self.distance_cluster_list = QtWidgets.QListWidget()
        self.distance_cluster_list.setToolTip(
            "Select clusters to include in the distance distribution plot. "
            "The plot will show distances from cells in selected clusters to their nearest neighbors "
            "in selected clusters (including self-distances)."
        )
        self.distance_cluster_list.setMaximumHeight(100)
        self.distance_cluster_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        distance_cluster_layout.addWidget(self.distance_cluster_list)
        
        # Option to show/hide self-pairs
        self.distance_show_self_pairs_check = QtWidgets.QCheckBox("Show self-distances\n(A→A pairs)")
        self.distance_show_self_pairs_check.setChecked(True)
        self.distance_show_self_pairs_check.setToolTip(
            "When checked, shows distances from cells to nearest neighbors in the same cluster. "
            "When unchecked, only shows distances between different clusters."
        )
        self.distance_show_self_pairs_check.toggled.connect(self._on_distance_cluster_selection_changed)
        
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
        distance_cluster_layout.addWidget(self.distance_show_self_pairs_check)
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
        if self.spatial_color_combo.completer() is not None:
            self.spatial_color_combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            self.spatial_color_combo.completer().setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            self.spatial_color_combo.completer().setFilterMode(QtCore.Qt.MatchContains)
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
        self.community_export_btn = QtWidgets.QPushButton("Export Results…")
        self.community_export_btn.setEnabled(False)
        community_controls.addWidget(self.community_run_btn)
        community_controls.addWidget(self.community_save_btn)
        community_controls.addWidget(self.community_export_btn)
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

        # Wire signals
        self.build_graph_btn.clicked.connect(self._on_build_graph_clicked)
        self.enrichment_run_btn.clicked.connect(self._run_enrichment_analysis)
        self.distance_run_btn.clicked.connect(self._run_distance_analysis)
        self.enrichment_save_btn.clicked.connect(self._save_enrichment_plot)
        self.distance_save_btn.clicked.connect(self._save_distance_plot)
        self.enrichment_export_btn.clicked.connect(self._export_enrichment_results)
        self.distance_export_btn.clicked.connect(self._export_distance_results)
        self.spatial_viz_run_btn.clicked.connect(self._run_spatial_visualization)
        self.spatial_viz_save_btn.clicked.connect(self._save_spatial_viz_plot)
        self.spatial_color_combo.currentTextChanged.connect(self._on_spatial_viz_option_changed)
        self.spatial_point_size_spin.valueChanged.connect(self._on_spatial_viz_option_changed)
        self.spatial_show_edges_check.toggled.connect(self._on_spatial_viz_option_changed)
        self.community_run_btn.clicked.connect(self._run_community_analysis)
        self.community_save_btn.clicked.connect(self._save_community_plot)
        self.community_export_btn.clicked.connect(self._export_community_results)
        self.export_btn.clicked.connect(self._export_results)
        self.export_graph_btn.clicked.connect(self._export_graph)
        self.cluster_labels_btn.clicked.connect(self._open_cluster_labels_dialog)
        self.feature_labels_btn.clicked.connect(self._open_feature_labels_dialog)
        self.advanced_analysis_btn.clicked.connect(self._open_advanced_analysis)
        
        self._update_tab_states()
        
        # Set default tab to Pairwise Enrichment (index 0) AFTER updating tab states
        # Temporarily enable the tab if needed to set it, then disable if graph not built
        if not (self.edge_df is not None and not self.edge_df.empty):
            self.tabs.setTabEnabled(0, True)
        self.tabs.setCurrentIndex(0)
        if not (self.edge_df is not None and not self.edge_df.empty):
            self.tabs.setTabEnabled(0, False)
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
        self.enrichment_export_btn.setEnabled(self.enrichment_analysis_run)
        self.distance_export_btn.setEnabled(self.distance_analysis_run)
        self.community_export_btn.setEnabled(bool(self.community_results_by_roi))
        self.export_btn.setEnabled(
            self.enrichment_analysis_run
            or self.distance_analysis_run
            or bool(self.community_results_by_roi)
        )

        self.export_graph_btn.setEnabled(graph_built)

    def _sorted_cluster_ids(self, clusters) -> List[Any]:
        """Return cluster ids in numeric cluster-id order."""
        return sort_cluster_values(clusters, annotation_map=self.cluster_annotation_map, canonical=False)

    def _get_cluster_color_map(self, clusters) -> Dict[Any, Any]:
        """Return a stable cluster-to-color mapping for categorical plots."""
        ordered_clusters = self._sorted_cluster_ids(clusters)
        palette = _get_vivid_colors(len(ordered_clusters))
        return {
            cluster: palette[idx]
            for idx, cluster in enumerate(ordered_clusters)
        }

    def _load_cluster_annotations(self):
        """Load cluster and feature display names from the clustering dialog if available."""
        try:
            label_source = self._get_label_source_dialog()
            if label_source is not None and hasattr(label_source, 'cluster_annotation_map'):
                self.cluster_annotation_map = normalize_cluster_annotation_map(label_source.cluster_annotation_map or {})
            if label_source is not None and hasattr(label_source, 'feature_label_map'):
                self.feature_label_map = dict(label_source.feature_label_map or {})
            
            loaded_annotations = {}
            for dataframe in (
                self.original_feature_dataframe,
                self.batch_corrected_dataframe,
                self.feature_dataframe,
                self.clustered_cells_dataframe,
            ):
                loaded_annotations.update(extract_cluster_annotation_map_from_dataframe(dataframe))
            if loaded_annotations:
                self.cluster_annotation_map.update(loaded_annotations)
            
            self._apply_cluster_annotations_to_dataframes()
        except Exception:
            pass

    def _get_label_source_dialog(self):
        """Return the best available dialog or parent object for shared label maps."""
        parent = self.parent()
        if parent is None:
            return None
        if hasattr(parent, 'clustering_dialog') and parent.clustering_dialog is not None:
            return parent.clustering_dialog
        return parent

    def _apply_cluster_annotations_to_dataframes(self):
        """Mirror the current cluster display names into cluster_phenotype columns when present."""
        for dataframe_attr in (
            'feature_dataframe',
            'original_feature_dataframe',
            'batch_corrected_dataframe',
            'clustered_cells_dataframe',
        ):
            df = getattr(self, dataframe_attr, None)
            if df is None or 'cluster' not in df.columns:
                continue
            phenotype_series = map_cluster_series_to_display_names(
                df['cluster'],
                annotation_map=self.cluster_annotation_map,
            )
            df.loc[:, 'cluster_phenotype'] = phenotype_series

    def _format_default_cluster_name(self, cluster_id) -> str:
        """Return the default display label for an unannotated cluster id."""
        return format_default_cluster_label(cluster_id)

    def _get_feature_display_name(self, feature_name: str) -> str:
        """Return the custom display name for a feature when available."""
        return self.feature_label_map.get(feature_name, feature_name)

    def _get_feature_name_from_display(self, label: str) -> str:
        """Resolve a display label back to its underlying feature name when possible."""
        if not label:
            return label
        for feature_name, display_name in self.feature_label_map.items():
            if display_name == label:
                return feature_name
        return label

    def _get_spatial_feature_columns(self) -> List[str]:
        """Return numeric mean-intensity feature columns for spatial coloring."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return []
        exclude = {
            'cell_id',
            'centroid_x',
            'centroid_y',
            'cluster',
            'cluster_id',
            'cluster_phenotype',
            'acquisition_id',
        }
        numeric_cols = self.feature_dataframe.select_dtypes(include=[np.number]).columns
        return [
            column
            for column in numeric_cols
            if column not in exclude and str(column).endswith('_mean')
        ]

    def _get_selected_spatial_color_option(self) -> str:
        """Return the actual selected column for the spatial color selector."""
        if not hasattr(self, 'spatial_color_combo'):
            return 'cluster'
        current_index = self.spatial_color_combo.currentIndex()
        if current_index >= 0:
            actual_value = self.spatial_color_combo.itemData(current_index)
            if actual_value:
                return actual_value
        current_text = self.spatial_color_combo.currentText().strip()
        cluster_display_map = {
            'Cluster': 'cluster',
            'Cluster ID': 'cluster_id',
            'Cluster phenotype': 'cluster_phenotype',
        }
        if current_text in cluster_display_map:
            return cluster_display_map[current_text]
        return self._get_feature_name_from_display(current_text)

    def _get_color_option_display_name(self, color_option: str) -> str:
        """Return a readable label for the selected spatial color option."""
        if color_option == 'cluster':
            return 'Cluster'
        if color_option == 'cluster_id':
            return 'Cluster ID'
        if color_option == 'cluster_phenotype':
            return 'Cluster phenotype'
        return self._get_feature_display_name(color_option)

    def _is_continuous_spatial_color_option(self, color_option: str, roi_df: Optional[pd.DataFrame]) -> bool:
        """Return True when a spatial color option should use a scalar colormap."""
        if roi_df is None or roi_df.empty or color_option not in roi_df.columns:
            return False
        if is_cluster_column(color_option):
            return False
        return bool(pd.api.types.is_numeric_dtype(roi_df[color_option]))

    def _sync_label_maps_to_parent(self):
        """Push cluster and feature display names back to the clustering dialog when possible."""
        label_source = self._get_label_source_dialog()
        if label_source is None:
            return
        if hasattr(label_source, 'cluster_annotation_map'):
            label_source.cluster_annotation_map = normalize_cluster_annotation_map(self.cluster_annotation_map)
            if hasattr(label_source, '_apply_cluster_annotations'):
                try:
                    label_source._apply_cluster_annotations()
                except Exception:
                    pass
        if hasattr(label_source, 'feature_label_map'):
            label_source.feature_label_map = dict(self.feature_label_map)

    def _refresh_label_dependent_views(self):
        """Refresh controls and plots that depend on cluster or feature display names."""
        if hasattr(self, '_populate_spatial_color_options'):
            self._populate_spatial_color_options()
        if hasattr(self, '_populate_exclude_clusters_list'):
            self._populate_exclude_clusters_list()
        if self.enrichment_analysis_run:
            self._update_enrichment_plot()
        if self.distance_df is not None and not self.distance_df.empty:
            self._populate_distance_cluster_list()

        if self.spatial_viz_run and hasattr(self, 'roi_combo'):
            roi_id = self.roi_combo.currentData()
            if roi_id in self.spatial_viz_cache:
                self._render_spatial_visualization(roi_id, self.spatial_viz_cache[roi_id])

    def _open_cluster_labels_dialog(self):
        """Open a dialog to customize cluster display names for spatial plots."""
        filtered_df = self._get_filtered_dataframe()
        if filtered_df is None or filtered_df.empty or 'cluster' not in filtered_df.columns:
            QtWidgets.QMessageBox.warning(self, "No Clusters", "No cluster assignments are available to relabel.")
            return
        cluster_ids = sort_cluster_values(filtered_df['cluster'].dropna().unique(), annotation_map=self.cluster_annotation_map, canonical=True)
        updated_map = edit_cluster_annotation_map(self, cluster_ids, self.cluster_annotation_map)
        if updated_map is None:
            return
        self.cluster_annotation_map = normalize_cluster_annotation_map(updated_map)
        self._apply_cluster_annotations_to_dataframes()
        self._sync_label_maps_to_parent()
        self._refresh_label_dependent_views()
        QtWidgets.QMessageBox.information(self, "Labels Applied", "Cluster names have been updated for spatial analysis.")

    def _open_feature_labels_dialog(self):
        """Open a dialog to customize feature display names for spatial plots."""
        feature_columns = self._get_spatial_feature_columns()
        if not feature_columns:
            QtWidgets.QMessageBox.warning(self, "No Features", "No numeric features are available to relabel.")
            return
        updated_map = edit_feature_label_map(self, feature_columns, self.feature_label_map)
        if updated_map is None:
            return
        self.feature_label_map = updated_map
        self._sync_label_maps_to_parent()
        self._refresh_label_dependent_views()
        QtWidgets.QMessageBox.information(self, "Labels Applied", "Feature labels have been updated for spatial analysis.")

    def _get_cluster_display_name(self, cluster_id):
        """Return display label for a cluster id, using annotation if available."""
        local_map = self.cluster_annotation_map or {}
        if local_map:
            canonical_cluster_id = canonicalize_cluster_id(cluster_id, annotation_map=local_map)
            if canonical_cluster_id in local_map:
                return get_cluster_display_name(cluster_id, annotation_map=local_map)

        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, '_get_cluster_display_name'):
                return parent._get_cluster_display_name(cluster_id)
        except Exception:
            pass

        return get_cluster_display_name(cluster_id, annotation_map=local_map)

    def _check_annotation_updates(self):
        """Periodically pull shared label maps from the clustering dialog."""
        try:
            label_source = self._get_label_source_dialog()
            maps_changed = False
            if label_source is not None and hasattr(label_source, 'cluster_annotation_map'):
                new_cluster_map = normalize_cluster_annotation_map(label_source.cluster_annotation_map or {})
                if new_cluster_map != self.cluster_annotation_map:
                    self.cluster_annotation_map = new_cluster_map
                    maps_changed = True
            if label_source is not None and hasattr(label_source, 'feature_label_map'):
                new_feature_map = dict(label_source.feature_label_map or {})
                if new_feature_map != self.feature_label_map:
                    self.feature_label_map = new_feature_map
                    maps_changed = True
            if maps_changed:
                self._apply_cluster_annotations_to_dataframes()
                self._refresh_label_dependent_views()
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
            parent = self.parent() if hasattr(self, 'parent') else None

            # Get filtered dataframe (respects source file filter)
            filtered_df = self._get_filtered_dataframe()
            
            # Get pixel size (use first ROI's pixel size as default, default to 1.0 um if not available)
            pixel_size_um = 1.0
            roi_col = self._get_roi_column()
            if roi_col and roi_col in filtered_df.columns:
                first_roi = filtered_df[roi_col].iloc[0] if len(filtered_df) > 0 else None
                if first_roi is not None and parent is not None and hasattr(parent, '_get_pixel_size_um'):
                    try:
                        pixel_size = parent._get_pixel_size_um(first_roi)
                        if pixel_size is not None:
                            pixel_size_um = float(pixel_size)
                        # If None, keep default 1.0
                    except Exception:
                        # Default to 1.0 um if pixel size cannot be retrieved (e.g., MCD file not loaded)
                        pixel_size_um = 1.0

            def _graph_task():
                edges_df, _ = build_spatial_graph(
                    features_df=filtered_df,
                    method=mode,
                    k_neighbors=k,
                    radius=radius_um if mode == "Radius" else None,
                    pixel_size_um=pixel_size_um,
                    roi_column=roi_col,
                    detect_communities=False,
                    community_seed=self.rng_seed,
                    output_path=None,
                )

                cell_id_to_gid = {}
                gid_to_cell_id = {}
                adj_matrices = {}

                if _HAVE_SPARSE and not edges_df.empty:
                    roi_groups_local = list(filtered_df.groupby(roi_col)) if roi_col and roi_col in filtered_df.columns else [(None, filtered_df)]
                    global_id_counter = 0

                    for roi_id, roi_df in roi_groups_local:
                        roi_id_str = str(roi_id) if roi_id is not None else "global"
                        roi_edges = edges_df[edges_df['roi_id'] == roi_id_str] if 'roi_id' in edges_df.columns else edges_df

                        if roi_edges.empty:
                            continue

                        cell_ids = roi_df["cell_id"].astype(int).to_numpy() if 'cell_id' in roi_df.columns else roi_df.index.values
                        n_cells = len(cell_ids)

                        roi_cell_to_gid = {}
                        for i, cell_id in enumerate(cell_ids):
                            gid = global_id_counter + i
                            cell_id_to_gid[(roi_id_str, int(cell_id))] = gid
                            gid_to_cell_id[gid] = (roi_id_str, int(cell_id))
                            roi_cell_to_gid[int(cell_id)] = gid

                        global_id_counter += n_cells

                        rows, cols, data = [], [], []
                        for _, edge in roi_edges.iterrows():
                            src_cell_id = int(edge['cell_id_A'])
                            dst_cell_id = int(edge['cell_id_B'])

                            if src_cell_id in roi_cell_to_gid and dst_cell_id in roi_cell_to_gid:
                                src_gid = roi_cell_to_gid[src_cell_id]
                                dst_gid = roi_cell_to_gid[dst_cell_id]
                                src_local = src_gid - (global_id_counter - n_cells)
                                dst_local = dst_gid - (global_id_counter - n_cells)
                                rows.extend([src_local, dst_local])
                                cols.extend([dst_local, src_local])
                                data.extend([1.0, 1.0])

                        if rows:
                            adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
                            adj_matrices[roi_id_str] = adj_matrix.tocsr()

                return edges_df, cell_id_to_gid, gid_to_cell_id, adj_matrices

            edges_df, cell_id_to_gid, gid_to_cell_id, adj_matrices = run_blocking_task_with_progress(
                parent=self,
                window_title="Building Spatial Graph",
                initial_message="Building spatial graph",
                detail_text=(
                    "Computing graph edges and adjacency matrices.\n"
                    "Large datasets may take several minutes."
                ),
                task=_graph_task,
            )

            self.edge_df = edges_df
            self.cell_id_to_gid = cell_id_to_gid
            self.gid_to_cell_id = gid_to_cell_id
            self.adj_matrices = adj_matrices

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

        cluster_col = self._get_active_cluster_column()
        if cluster_col is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Cluster Column",
                "No cluster column found. Please ensure your data has a 'cluster', 'cluster_phenotype', or 'cluster_id' column."
            )
            return

        try:
            n_perm = int(self.n_perm_spin.value())
            filtered_df = self._get_filtered_dataframe()
            roi_col = self._get_roi_column()
            seed = self.seed_spinbox.value()
            n_workers = int(self.workers_spin.value())

            def _enrichment_task():
                return self._compute_pairwise_enrichment(
                    n_perm=n_perm,
                    filtered_df=filtered_df,
                    cluster_col=cluster_col,
                    roi_col=roi_col,
                    seed=seed,
                    n_workers=n_workers,
                )

            def _finalize_enrichment(results_df, _progress):
                self.enrichment_df = results_df if results_df is not None else pd.DataFrame()
                self.enrichment_analysis_run = bool(
                    self.enrichment_df is not None and not self.enrichment_df.empty
                )
                self._update_tab_states()
                self._update_enrichment_plot()

            run_blocking_task_with_progress_then_finalize(
                parent=self,
                window_title="Enrichment Analysis",
                initial_message="Computing enrichment analysis",
                detail_text="Running permutation-based enrichment across ROIs.",
                task=_enrichment_task,
                finalize=_finalize_enrichment,
                finishing_message="Rendering enrichment plot",
                finishing_detail_text="Drawing the fitted heatmap in the current tab.",
            )

            logger = get_logger()
            acquisitions = list(filtered_df[roi_col].unique()) if roi_col in filtered_df.columns else []
            params = {
                "n_permutations": n_perm,
                "seed": self.seed_spinbox.value()
            }
            source_file = self._get_source_files_for_logging()
            logger.log_spatial_analysis(
                analysis_type="pairwise_enrichment",
                parameters=params,
                acquisitions=acquisitions,
                notes=f"Pairwise enrichment analysis with {n_perm} permutations",
                source_file=source_file
            )
            
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

        cluster_col = self._get_active_cluster_column()
        if cluster_col is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Cluster Column",
                "No cluster column found. Please ensure your data has a 'cluster', 'cluster_phenotype', or 'cluster_id' column."
            )
            return

        try:
            filtered_df = self._get_filtered_dataframe()
            roi_col = self._get_roi_column()
            pixel_size_um = self._get_default_pixel_size_um(filtered_df, roi_col)
            n_workers = int(self.distance_workers_spin.value()) if hasattr(self, 'distance_workers_spin') else None

            def _distance_task():
                return self._compute_distance_distributions(
                    filtered_df=filtered_df,
                    cluster_col=cluster_col,
                    roi_col=roi_col,
                    pixel_size_um=pixel_size_um,
                    n_workers=n_workers,
                )

            def _finalize_distance(results_df, _progress):
                self.distance_df = results_df if results_df is not None else pd.DataFrame()
                self.distance_analysis_run = bool(
                    self.distance_df is not None and not self.distance_df.empty
                )
                self._update_tab_states()
                self._populate_distance_cluster_list()

            run_blocking_task_with_progress_then_finalize(
                parent=self,
                window_title="Distance Distribution Analysis",
                initial_message="Computing distance distributions",
                detail_text="Calculating nearest-neighbor distances across ROIs.",
                task=_distance_task,
                finalize=_finalize_distance,
                finishing_message="Rendering distance plot",
                finishing_detail_text="Drawing the fitted cluster-distance distributions.",
            )

            logger = get_logger()
            acquisitions = list(filtered_df[roi_col].unique()) if roi_col in filtered_df.columns else []
            source_file = self._get_source_files_for_logging()
            logger.log_spatial_analysis(
                analysis_type="distance_distribution",
                parameters={},
                acquisitions=acquisitions,
                notes="Distance distribution analysis completed",
                source_file=source_file
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Distance Analysis Error", f"Error: {str(e)}")
    
    def _get_active_cluster_column(self) -> Optional[str]:
        """Return the preferred cluster-label column for spatial analysis."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                return col
        return None

    def _get_default_pixel_size_um(self, filtered_df: pd.DataFrame, roi_col: str) -> float:
        """Return the first available pixel size for the filtered dataset."""
        parent = self.parent() if hasattr(self, 'parent') else None
        pixel_size_um = 1.0
        if roi_col and roi_col in filtered_df.columns:
            first_roi = filtered_df[roi_col].iloc[0] if len(filtered_df) > 0 else None
            if first_roi is not None and parent is not None and hasattr(parent, '_get_pixel_size_um'):
                try:
                    pixel_size = parent._get_pixel_size_um(first_roi)
                    if pixel_size is not None:
                        pixel_size_um = float(pixel_size)
                except Exception:
                    pixel_size_um = 1.0
        return pixel_size_um

    def _compute_pairwise_enrichment(
        self,
        *,
        n_perm=100,
        filtered_df: Optional[pd.DataFrame] = None,
        cluster_col: Optional[str] = None,
        roi_col: Optional[str] = None,
        seed: Optional[int] = None,
        n_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute pairwise interaction enrichment analysis using the core function."""
        if self.edge_df is None or self.edge_df.empty:
            return pd.DataFrame()

        if cluster_col is None:
            cluster_col = self._get_active_cluster_column()
        if cluster_col is None:
            raise ValueError("No cluster column found for enrichment analysis.")

        if filtered_df is None:
            filtered_df = self._get_filtered_dataframe()
        if roi_col is None:
            roi_col = self._get_roi_column()
        if seed is None:
            seed = self.seed_spinbox.value()
        if n_workers is None:
            n_workers = int(self.workers_spin.value())

        return spatial_enrichment(
            features_df=filtered_df,
            edges_df=self.edge_df,
            cluster_column=cluster_col,
            n_permutations=n_perm,
            seed=seed,
            roi_column=roi_col,
            output_path=None,
            n_workers=n_workers,
        )

    def _compute_distance_distributions(
        self,
        *,
        filtered_df: Optional[pd.DataFrame] = None,
        cluster_col: Optional[str] = None,
        roi_col: Optional[str] = None,
        pixel_size_um: Optional[float] = None,
        n_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute distance distribution analysis using the core function."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return pd.DataFrame()
        if self.edge_df is None or self.edge_df.empty:
            return pd.DataFrame()

        if cluster_col is None:
            cluster_col = self._get_active_cluster_column()
        if cluster_col is None:
            raise ValueError("No cluster column found for distance analysis.")

        if filtered_df is None:
            filtered_df = self._get_filtered_dataframe()
        if roi_col is None:
            roi_col = self._get_roi_column()
        if pixel_size_um is None:
            pixel_size_um = self._get_default_pixel_size_um(filtered_df, roi_col)
        if n_workers is None and hasattr(self, 'distance_workers_spin'):
            n_workers = int(self.distance_workers_spin.value())

        return spatial_distance_distribution(
            features_df=filtered_df,
            edges_df=self.edge_df,
            cluster_column=cluster_col,
            roi_column=roi_col,
            output_path=None,
            pixel_size_um=pixel_size_um,
            n_workers=n_workers,
        )
    
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
        current_actual = self._get_selected_spatial_color_option()
        self.spatial_color_combo.blockSignals(True)
        self.spatial_color_combo.clear()
        filtered_df = self._get_filtered_dataframe()
        
        # Add cluster columns
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in filtered_df.columns:
                self.spatial_color_combo.addItem(self._get_color_option_display_name(col), col)
        
        # Add numeric feature columns
        for col in self._get_spatial_feature_columns():
            if col in filtered_df.columns:
                self.spatial_color_combo.addItem(self._get_feature_display_name(col), col)

        if current_actual:
            for idx in range(self.spatial_color_combo.count()):
                if self.spatial_color_combo.itemData(idx) == current_actual:
                    self.spatial_color_combo.setCurrentIndex(idx)
                    break
        self.spatial_color_combo.blockSignals(False)
    
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
            unique_clusters = sort_cluster_values(
                filtered_df[cluster_col].dropna().unique(),
                annotation_map=self.cluster_annotation_map,
                canonical=False,
            )
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

        try:
            color_option = self._get_selected_spatial_color_option()
            show_edges = (
                hasattr(self, 'spatial_show_edges_check')
                and self.spatial_show_edges_check.isChecked()
            )
            filtered_df = self._get_filtered_dataframe()
            roi_col = self._get_roi_column()
            pixel_size_um = 1.0
            parent = self.parent() if hasattr(self, 'parent') else None
            try:
                if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                    pixel_size = parent._get_pixel_size_um(selected_roi)
                    if pixel_size is not None:
                        pixel_size_um = float(pixel_size)
            except Exception:
                pixel_size_um = 1.0
            def _prepare_visualization():
                return self._build_spatial_visualization_cache_data(
                    selected_roi,
                    filtered_df=filtered_df,
                    roi_col=roi_col,
                    pixel_size_um=pixel_size_um,
                    color_option=color_option,
                    show_edges=show_edges,
                )

            def _finalize_visualization(cache_data, _progress):
                self.spatial_viz_cache[selected_roi] = cache_data
                self._render_spatial_visualization(selected_roi, cache_data)
                self.spatial_viz_run = True
                self._update_tab_states()

            run_blocking_task_with_progress_then_finalize(
                parent=self,
                window_title="Spatial Visualization",
                initial_message="Preparing spatial visualization",
                detail_text="Collecting ROI coordinates and overlay data.",
                task=_prepare_visualization,
                finalize=_finalize_visualization,
                finishing_message="Rendering spatial visualization",
                finishing_detail_text="Drawing cells and graph edges on the canvas.",
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Visualization Error", f"Error: {str(e)}")
    
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

        cache_data = self._build_spatial_visualization_cache_data(roi_id)
        self.spatial_viz_cache[roi_id] = cache_data
        self._render_spatial_visualization(roi_id, cache_data)

    def _build_spatial_visualization_cache_data(
        self,
        roi_id,
        filtered_df=None,
        roi_col=None,
        pixel_size_um=None,
        color_option=None,
        show_edges=None,
    ):
        """Build cache data for a spatial visualization (safe to run in a worker thread)."""
        # Get filtered dataframe
        if filtered_df is None:
            filtered_df = self._get_filtered_dataframe()
        if roi_col is None:
            roi_col = self._get_roi_column()
        roi_df = filtered_df[filtered_df[roi_col] == roi_id].copy()
        
        if roi_df.empty:
            raise ValueError(f"No data available for ROI {roi_id}.")
            
        # Get color option
        if color_option is None:
            color_option = self._get_selected_spatial_color_option()
        
        # Get pixel size (default to 1.0 um if not available, e.g., MCD file not loaded)
        if pixel_size_um is None:
            parent = self.parent() if hasattr(self, 'parent') else None
            pixel_size_um = 1.0
            try:
                if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                    pixel_size = parent._get_pixel_size_um(roi_id)
                    if pixel_size is not None:
                        pixel_size_um = float(pixel_size)
            except Exception:
                pixel_size_um = 1.0
        
        # Get coordinates
        coords_um = roi_df[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
        
        # Get color values
        color_kind = 'categorical'
        color_labels = {}
        if color_option in roi_df.columns:
            if self._is_continuous_spatial_color_option(color_option, roi_df):
                color_kind = 'continuous'
                color_values = pd.to_numeric(roi_df[color_option], errors='coerce').to_numpy(dtype=float)
            else:
                raw_values = roi_df[color_option].values
                if is_cluster_column(color_option):
                    raw_values = roi_df['cluster'].values if 'cluster' in roi_df.columns else raw_values
                    color_values = np.asarray(
                        [
                            (
                                canonical_value
                                if canonical_value is not None
                                else '__missing__'
                            )
                            for value in raw_values
                            for canonical_value in [canonicalize_cluster_id(value, annotation_map=self.cluster_annotation_map)]
                        ],
                        dtype=object,
                    )
                    for cluster_value in sort_cluster_values(
                        color_values,
                        annotation_map=self.cluster_annotation_map,
                        canonical=True,
                    ):
                        color_labels[cluster_value] = self._get_cluster_display_name(cluster_value)
                else:
                    color_values = pd.Series(raw_values, dtype='object').where(
                        pd.Series(raw_values, dtype='object').notna(),
                        '__missing__',
                    ).to_numpy(dtype=object)
        else:
            color_values = None
        
        # Get edges if graph is built AND checkbox is enabled (skip computation if not needed)
        edges_um = None
        if show_edges is None:
            show_edges = (hasattr(self, 'spatial_show_edges_check') and 
                         self.spatial_show_edges_check.isChecked())
        if show_edges and self.edge_df is not None and not self.edge_df.empty:
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)]
            if not roi_edges.empty:
                coord_df = pd.DataFrame(
                    {
                        'cell_id': roi_df['cell_id'].astype(int).values,
                        'x_um': coords_um[:, 0],
                        'y_um': coords_um[:, 1],
                    }
                )
                edge_coords = (
                    roi_edges[['cell_id_A', 'cell_id_B']]
                    .astype(int)
                    .merge(
                        coord_df.rename(
                            columns={'cell_id': 'cell_id_A', 'x_um': 'x_a', 'y_um': 'y_a'}
                        ),
                        on='cell_id_A',
                        how='inner',
                    )
                    .merge(
                        coord_df.rename(
                            columns={'cell_id': 'cell_id_B', 'x_um': 'x_b', 'y_um': 'y_b'}
                        ),
                        on='cell_id_B',
                        how='inner',
                    )
                )
                if not edge_coords.empty:
                    edges_um = edge_coords[['x_a', 'y_a', 'x_b', 'y_b']].to_numpy(dtype=float).reshape(-1, 2, 2)
        
        # Cache the data
        cache_data = {
            'coords_um': coords_um,
            'color_values': color_values,
            'color_option': color_option,
            'color_kind': color_kind,
            'color_labels': color_labels,
            'edges_um': edges_um,
            'roi_df': roi_df
        }
        return cache_data
    
    def _render_spatial_visualization(self, roi_id, cache_data):
        """Render the spatial visualization on the canvas."""
        self.spatial_viz_canvas.figure.clear()
        ax = self.spatial_viz_canvas.figure.add_subplot(111)
        
        coords_um = cache_data['coords_um']
        color_values = cache_data['color_values']
        color_option = cache_data['color_option']
        color_kind = cache_data.get('color_kind', 'categorical')
        color_labels = cache_data.get('color_labels', {})
        edges_um = cache_data.get('edges_um')
        
        # Plot edges first (if available AND checkbox is enabled)
        show_edges = (hasattr(self, 'spatial_show_edges_check') and 
                     self.spatial_show_edges_check.isChecked())
        if edges_um is not None and len(edges_um) > 0 and show_edges:
            ax.add_collection(
                LineCollection(
                    edges_um,
                    colors='gray',
                    alpha=0.3,
                    linewidths=0.5,
                    zorder=1,
                )
            )
        
        # Get point size multiplier
        point_size = (self.spatial_point_size_spin.value() 
                     if hasattr(self, 'spatial_point_size_spin') else 1.0)
        
        # Plot cells
        legend_handles = []
        rect = [0.0, 0.0, 0.95, 1.0]
        if color_values is not None:
            if color_kind == 'continuous':
                numeric_values = np.asarray(color_values, dtype=float)
                valid_mask = np.isfinite(numeric_values)
                if np.any(valid_mask):
                    vmin = float(np.nanmin(numeric_values[valid_mask]))
                    vmax = float(np.nanmax(numeric_values[valid_mask]))
                    if np.isclose(vmin, vmax):
                        vmax = vmin + 1e-6
                    scatter = ax.scatter(
                        coords_um[valid_mask, 0],
                        coords_um[valid_mask, 1],
                        c=numeric_values[valid_mask],
                        cmap='viridis',
                        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                        s=20 * point_size,
                        alpha=0.8,
                        zorder=2,
                        edgecolors='none',
                    )
                    if np.any(~valid_mask):
                        ax.scatter(
                            coords_um[~valid_mask, 0],
                            coords_um[~valid_mask, 1],
                            c='lightgray',
                            s=20 * point_size,
                            alpha=0.5,
                            zorder=2,
                            edgecolors='none',
                        )
                    colorbar = self.spatial_viz_canvas.figure.colorbar(
                        scatter,
                        ax=ax,
                        fraction=0.046,
                        pad=0.03,
                    )
                    colorbar.set_label(self._get_color_option_display_name(color_option))
                    rect = [0.0, 0.0, 0.88, 1.0]
                else:
                    ax.scatter(coords_um[:, 0], coords_um[:, 1], c='lightgray', s=20 * point_size, alpha=0.6, zorder=2)
            else:
                color_series = pd.Series(color_values, dtype='object')
                prepared_values = color_series.where(color_series.notna(), '__missing__').to_numpy(dtype=object)
                if is_cluster_column(color_option):
                    unique_values = sort_cluster_values(
                        prepared_values,
                        annotation_map=self.cluster_annotation_map,
                        canonical=True,
                    )
                else:
                    unique_values = sorted(pd.unique(prepared_values), key=lambda value: str(value))
                colors = _get_vivid_colors(len(unique_values))
                value_to_color = {val: colors[i] for i, val in enumerate(unique_values)}

                for value in unique_values:
                    mask = prepared_values == value
                    color = value_to_color.get(value, 'gray')
                    ax.scatter(
                        coords_um[mask, 0],
                        coords_um[mask, 1],
                        c=[color],
                        s=20 * point_size,
                        alpha=0.7,
                        zorder=2,
                        edgecolors='none',
                    )

                if is_cluster_column(color_option):
                    for value in unique_values:
                        color = value_to_color.get(value, 'gray')
                        label = color_labels.get(value)
                        if not label:
                            label = 'Unassigned' if value == '__missing__' else self._get_cluster_display_name(value)
                        legend_handles.append(
                            plt.Line2D(
                                [0],
                                [0],
                                marker='o',
                                color='w',
                                markerfacecolor=color,
                                markersize=7,
                                label=label,
                                alpha=0.7,
                            )
                        )
        else:
            # Default color
            ax.scatter(coords_um[:, 0], coords_um[:, 1], c='blue', s=20*point_size, alpha=0.7, zorder=2)
        
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        color_label = self._get_color_option_display_name(color_option)
        ax.set_title(f'Spatial Visualization: ROI {roi_id} (colored by {color_label})')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Set y=0 at the top (image coordinates)
        ax.grid(True, alpha=0.3)
        
        # Add legend if we have cluster coloring
        if legend_handles:
            legend_kwargs = self._spatial_viz_legend_kwargs()
            ncol = self._choose_spatial_viz_legend_columns(ax, legend_handles, legend_kwargs=legend_kwargs)
            ax.legend(handles=legend_handles, ncol=ncol, **legend_kwargs)
            rect = self._calculate_spatial_viz_layout_rect(ax=ax, base_right=rect[2])

        self._spatial_viz_layout_rect = rect
        self._finalize_canvas_render(self.spatial_viz_canvas, rect=rect, pad=0.95)
    
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

            def _community_task():
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

                communities = g.community_multilevel()
                return communities.membership, cell_id_to_idx

            def _finalize_community(result, _progress):
                community_labels, cell_id_to_idx = result
                community_df = roi_df.copy()
                community_df['community'] = [community_labels[cell_id_to_idx[int(cid)]] for cid in community_df['cell_id']]
                community_sizes = community_df['community'].value_counts()
                community_df['community_size'] = community_df['community'].map(community_sizes).astype(int)
                self.community_results_by_roi[str(selected_roi)] = community_df.copy()
                self._update_community_plot(selected_roi, community_df, community_labels)
                self.community_analysis_run = True
                self._update_tab_states()

            community_labels, _ = run_blocking_task_with_progress_then_finalize(
                parent=self,
                window_title="Community Analysis",
                initial_message="Running community detection",
                detail_text="Building graph partitions for the selected ROI.",
                task=_community_task,
                finalize=_finalize_community,
                finishing_message="Rendering community plot",
                finishing_detail_text="Coloring cells by detected spatial community.",
            )

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
        
        # Get pixel size (default to 1.0 um if not available, e.g., MCD file not loaded)
        parent = self.parent() if hasattr(self, 'parent') else None
        pixel_size_um = 1.0
        try:
            if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                pixel_size = parent._get_pixel_size_um(roi_id)
                if pixel_size is not None:
                    pixel_size_um = float(pixel_size)
                # If None, keep default 1.0
        except Exception:
            # Default to 1.0 um if pixel size cannot be retrieved (e.g., MCD file not loaded)
            pixel_size_um = 1.0
        
        # Get coordinates
        coords_um = roi_df[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
        
        # Color by community
        n_communities = len(set(community_labels))
        colors = _get_vivid_colors(n_communities)
        unique_communities = sorted(set(community_labels))
        community_colors = {comm: colors[i] for i, comm in enumerate(unique_communities)}
        community_labels_array = np.asarray(community_labels)

        for comm in unique_communities:
            mask = community_labels_array == comm
            color = community_colors.get(comm, 'gray')
            ax.scatter(
                coords_um[mask, 0],
                coords_um[mask, 1],
                c=[color],
                s=20,
                alpha=0.7,
                edgecolors='none',
            )
        
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title(f'Spatial Communities: ROI {roi_id} ({n_communities} communities)')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Set y=0 at the top (image coordinates)
        ax.grid(True, alpha=0.3)
        
        self._fit_canvas(self.community_canvas, pad=0.95)
    
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

    def _prepare_enrichment_export_df(self) -> pd.DataFrame:
        """Return pairwise enrichment results with adjusted p-values and labels."""
        if self.enrichment_df is None or self.enrichment_df.empty:
            return pd.DataFrame()

        export_df = self.enrichment_df.copy()
        if 'p_value' in export_df.columns and 'p_value_adjusted' not in export_df.columns:
            export_df['p_value_adjusted'] = np.nan
            if 'roi_id' in export_df.columns:
                for _, roi_index in export_df.groupby('roi_id', sort=False).groups.items():
                    roi_pvals = export_df.loc[roi_index, 'p_value'].to_numpy(dtype=float, copy=False)
                    export_df.loc[roi_index, 'p_value_adjusted'] = benjamini_hochberg_adjust(roi_pvals)
            else:
                export_df['p_value_adjusted'] = benjamini_hochberg_adjust(
                    export_df['p_value'].to_numpy(dtype=float, copy=False)
                )

        if 'cluster_A' in export_df.columns:
            export_df['cluster_A_label'] = export_df['cluster_A'].map(self._get_cluster_display_name)
        if 'cluster_B' in export_df.columns:
            export_df['cluster_B_label'] = export_df['cluster_B'].map(self._get_cluster_display_name)

        sort_columns = [column for column in ['roi_id', 'cluster_A', 'cluster_B'] if column in export_df.columns]
        if sort_columns:
            export_df = export_df.sort_values(sort_columns).reset_index(drop=True)
        return export_df

    def _prepare_distance_export_df(self) -> pd.DataFrame:
        """Return distance results annotated with cluster display labels."""
        if self.distance_df is None or self.distance_df.empty:
            return pd.DataFrame()

        export_df = self.distance_df.copy()
        if 'cell_A_cluster' in export_df.columns:
            export_df['cell_A_cluster_label'] = export_df['cell_A_cluster'].map(self._get_cluster_display_name)
        if 'nearest_B_cluster' in export_df.columns:
            export_df['nearest_B_cluster_label'] = export_df['nearest_B_cluster'].map(self._get_cluster_display_name)

        sort_columns = [
            column for column in ['roi_id', 'cell_A_cluster', 'cell_A_id', 'nearest_B_cluster']
            if column in export_df.columns
        ]
        if sort_columns:
            export_df = export_df.sort_values(sort_columns).reset_index(drop=True)
        return export_df

    def _prepare_community_export_df(self, roi_id=None, *, all_rois: bool = False) -> pd.DataFrame:
        """Return community assignments for the selected ROI or all analyzed ROIs."""
        if not self.community_results_by_roi:
            return pd.DataFrame()

        if all_rois:
            export_df = pd.concat(self.community_results_by_roi.values(), ignore_index=True, sort=False)
        else:
            if roi_id is None:
                roi_id = self.community_roi_combo.currentData() if hasattr(self, 'community_roi_combo') else None
            export_df = self.community_results_by_roi.get(str(roi_id), pd.DataFrame()).copy()

        if export_df.empty:
            return export_df

        if 'cluster' in export_df.columns:
            export_df['cluster_label'] = export_df['cluster'].map(self._get_cluster_display_name)
        roi_col = self._get_roi_column()
        if roi_col in export_df.columns and 'roi_id' not in export_df.columns:
            export_df['roi_id'] = export_df[roi_col].astype(str)

        preferred_columns = [
            'roi_id',
            'cell_id',
            'cluster',
            'cluster_label',
            'community',
            'community_size',
            'centroid_x',
            'centroid_y',
        ]
        remaining_columns = [column for column in export_df.columns if column not in preferred_columns]
        ordered_columns = [column for column in preferred_columns if column in export_df.columns] + remaining_columns
        export_df = export_df.loc[:, ordered_columns]

        sort_columns = [column for column in ['roi_id', 'community', 'cell_id'] if column in export_df.columns]
        if sort_columns:
            export_df = export_df.sort_values(sort_columns).reset_index(drop=True)
        return export_df

    def _export_dataframe_to_csv(
        self,
        dataframe: pd.DataFrame,
        *,
        title: str,
        default_filename: str,
        success_message: str,
    ) -> bool:
        """Export a dataframe to a user-selected CSV file."""
        if dataframe is None or dataframe.empty:
            return False

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            title,
            default_filename,
            "CSV Files (*.csv)"
        )
        if not file_path:
            return False

        try:
            dataframe.to_csv(file_path, index=False)
            QtWidgets.QMessageBox.information(self, "Export Complete", f"{success_message}\n{file_path}")
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
            return False

    def _export_enrichment_results(self):
        """Export pairwise enrichment results to CSV."""
        export_df = self._prepare_enrichment_export_df()
        if export_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Results", "No pairwise enrichment results are available to export.")
            return
        self._export_dataframe_to_csv(
            export_df,
            title="Export Pairwise Enrichment Results",
            default_filename="pairwise_enrichment.csv",
            success_message="Pairwise enrichment results exported to:",
        )

    def _export_distance_results(self):
        """Export distance-distribution results to CSV."""
        export_df = self._prepare_distance_export_df()
        if export_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Results", "No distance-distribution results are available to export.")
            return
        self._export_dataframe_to_csv(
            export_df,
            title="Export Distance Distribution Results",
            default_filename="distance_distributions.csv",
            success_message="Distance-distribution results exported to:",
        )

    def _export_community_results(self):
        """Export community assignments for the selected ROI to CSV."""
        selected_roi = self.community_roi_combo.currentData() if hasattr(self, 'community_roi_combo') else None
        export_df = self._prepare_community_export_df(selected_roi)
        if export_df.empty:
            if selected_roi is not None:
                message = f"No spatial community results are available for ROI {selected_roi}. Please run the analysis first."
            else:
                message = "No spatial community results are available to export. Please run the analysis first."
            QtWidgets.QMessageBox.warning(self, "No Results", message)
            return
        filename_suffix = str(selected_roi) if selected_roi is not None else "all_rois"
        self._export_dataframe_to_csv(
            export_df,
            title="Export Spatial Community Results",
            default_filename=f"spatial_communities_{filename_suffix}.csv",
            success_message="Spatial community assignments exported to:",
        )

    def _export_results(self):
        """Export analysis results to CSV files."""
        from PyQt5.QtWidgets import QFileDialog

        export_payloads = []
        enrichment_export_df = self._prepare_enrichment_export_df()
        if not enrichment_export_df.empty:
            export_payloads.append(("pairwise_enrichment.csv", enrichment_export_df))

        distance_export_df = self._prepare_distance_export_df()
        if not distance_export_df.empty:
            export_payloads.append(("distance_distributions.csv", distance_export_df))

        community_export_df = self._prepare_community_export_df(all_rois=True)
        if not community_export_df.empty:
            export_payloads.append(("spatial_communities.csv", community_export_df))

        if not export_payloads:
            QtWidgets.QMessageBox.warning(self, "No Results", "No analysis results to export. Please run analyses first.")
            return
        
        # Get export directory
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        
        try:
            for filename, export_df in export_payloads:
                file_path = os.path.join(export_dir, filename)
                export_df.to_csv(file_path, index=False)
            
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
        if not (_HAVE_SQUIDPY or squidpy_available()):
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
        self.enrichment_canvas.figure.clear()
        ax = self.enrichment_canvas.figure.add_subplot(111)

        if self.enrichment_df is None or self.enrichment_df.empty:
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                'Run pairwise enrichment to display the cluster-cluster heatmap.',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=12,
            )
            self._finalize_canvas_render(self.enrichment_canvas, pad=0.95)
            return

        # Aggregate across ROIs if multiple
        if 'roi_id' in self.enrichment_df.columns:
            enrichment_agg = (
                self.enrichment_df
                .groupby(['cluster_A', 'cluster_B'], as_index=False)
                .agg({'z_score': 'mean', 'p_value': 'mean'})
            )
        else:
            enrichment_agg = self.enrichment_df.copy()

        all_clusters = set(enrichment_agg['cluster_A'].unique()) | set(enrichment_agg['cluster_B'].unique())
        unique_clusters = self._sorted_cluster_ids(all_clusters)
        n_clusters = len(unique_clusters)

        if n_clusters == 0:
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                'No enrichment results are available for the current selection.',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=12,
            )
            self._finalize_canvas_render(self.enrichment_canvas, pad=0.95)
            return

        cluster_lookup = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
        heatmap_data = np.zeros((n_clusters, n_clusters), dtype=float)
        pvalue_data = np.ones((n_clusters, n_clusters), dtype=float)

        for row in enrichment_agg.itertuples(index=False):
            i = cluster_lookup[row.cluster_A]
            j = cluster_lookup[row.cluster_B]
            heatmap_data[i, j] = float(row.z_score)
            heatmap_data[j, i] = float(row.z_score)
            p_value = float(getattr(row, 'p_value', 1.0))
            pvalue_data[i, j] = p_value
            pvalue_data[j, i] = p_value

        cluster_labels = [self._get_cluster_display_name(c) for c in unique_clusters]
        style = dense_heatmap_style(
            n_rows=n_clusters,
            n_cols=n_clusters,
            row_labels=cluster_labels,
            col_labels=cluster_labels,
            base_tick_fontsize=10.0,
            base_annotation_fontsize=9.0,
            allow_annotations=True,
        )

        vmax = float(np.max(np.abs(heatmap_data))) if np.max(np.abs(heatmap_data)) > 0 else 1.0
        annotation_data = False
        if style['show_annotations']:
            annotation_data = np.empty_like(heatmap_data, dtype=object)
            for row_idx in range(n_clusters):
                for col_idx in range(n_clusters):
                    sig_marker = '*' if pvalue_data[row_idx, col_idx] < 0.05 else ''
                    annotation_data[row_idx, col_idx] = f"{heatmap_data[row_idx, col_idx]:.1f}{sig_marker}"

        heatmap_df = pd.DataFrame(heatmap_data, index=cluster_labels, columns=cluster_labels)
        if _HAVE_SEABORN_LOCAL:
            sns.heatmap(
                heatmap_df,
                cmap='RdBu_r',
                center=0,
                vmin=-vmax,
                vmax=vmax,
                square=style['square_cells'],
                linewidths=style['linewidths'],
                cbar_kws={
                    'label': 'Z-Score',
                    'shrink': style['colorbar_shrink'],
                    'fraction': style['colorbar_fraction'],
                    'pad': style['colorbar_pad'],
                },
                ax=ax,
                annot=annotation_data if style['show_annotations'] else False,
                fmt='',
                annot_kws={
                    'size': style['annotation_fontsize'],
                    'weight': 'normal',
                    'color': 'black',
                },
                xticklabels=True,
                yticklabels=True,
            )
        else:
            from matplotlib.colors import TwoSlopeNorm

            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', norm=norm)
            self.enrichment_canvas.figure.colorbar(
                im,
                ax=ax,
                fraction=style['colorbar_fraction'],
                pad=style['colorbar_pad'],
                shrink=style['colorbar_shrink'],
            )
            ax.set_xticks(np.arange(n_clusters))
            ax.set_yticks(np.arange(n_clusters))
            ax.set_xticklabels(cluster_labels)
            ax.set_yticklabels(cluster_labels)
            if style['show_annotations']:
                for row_idx in range(n_clusters):
                    for col_idx in range(n_clusters):
                        ax.text(
                            col_idx,
                            row_idx,
                            annotation_data[row_idx, col_idx],
                            ha='center',
                            va='center',
                            fontsize=style['annotation_fontsize'],
                        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=style['x_rotation'],
            ha='right',
            fontsize=style['tick_fontsize'],
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=style['tick_fontsize'])
        ax.set_title(
            "Pairwise Enrichment: Z-Scores (Positive = Enriched, Negative = Depleted)",
            fontsize=style['title_fontsize'],
            pad=10,
        )
        ax.set_xlabel("Cluster B", fontsize=style['axis_fontsize'])
        ax.set_ylabel("Cluster A", fontsize=style['axis_fontsize'])
        ax.tick_params(axis='both', labelsize=style['tick_fontsize'])

        colorbar = ax.collections[0].colorbar if ax.collections else None
        if colorbar is not None:
            colorbar.ax.tick_params(labelsize=style['colorbar_fontsize'])
            colorbar.set_label('Z-Score', fontsize=style['axis_fontsize'])

        self._finalize_canvas_render(self.enrichment_canvas, pad=0.95)
    
    def _update_distance_plot(self):
        """Update distance plot with distance distribution data."""
        self.distance_canvas.figure.clear()
        fig = self.distance_canvas.figure

        if self.distance_df is None or self.distance_df.empty:
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.text(
                0.5,
                0.5,
                'Run distance analysis to display fitted nearest-neighbor distributions.',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=12,
            )
            self._finalize_canvas_render(self.distance_canvas, pad=0.95)
            return

        selected_clusters = self._get_selected_distance_clusters()
        show_self_pairs = (hasattr(self, 'distance_show_self_pairs_check') and 
                           self.distance_show_self_pairs_check.isChecked())

        if not selected_clusters:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No clusters selected.\nPlease select clusters to view distance distributions.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            self._finalize_canvas_render(self.distance_canvas, pad=0.95)
            return

        plot_df = self.distance_df[
            self.distance_df['cell_A_cluster'].isin(selected_clusters)
        ].copy()

        if not show_self_pairs:
            plot_df = plot_df[
                plot_df['cell_A_cluster'] != plot_df['nearest_B_cluster']
            ]

        if plot_df.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No distance data to display for the selected clusters.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            self._finalize_canvas_render(self.distance_canvas, pad=0.95)
            return

        source_clusters = [
            cluster for cluster in self._sorted_cluster_ids(selected_clusters)
            if cluster in set(plot_df['cell_A_cluster'].dropna().unique())
        ]
        all_clusters = set(plot_df['cell_A_cluster'].dropna().unique()) | set(plot_df['nearest_B_cluster'].dropna().unique())
        color_map = self._get_cluster_color_map(all_clusters)

        n_sources = len(source_clusters)
        n_cols = 1 if n_sources == 1 else 2
        n_rows = int(np.ceil(n_sources / n_cols))
        axes = fig.subplots(n_rows, n_cols, squeeze=False, sharex=True)
        all_axes = axes.flatten()
        global_max = float(plot_df['nearest_B_dist_um'].max()) if not plot_df.empty else 1.0
        global_max = max(global_max, 1.0)

        for axis in all_axes:
            axis.set_visible(False)

        for plot_idx, source_cluster in enumerate(source_clusters):
            axis = all_axes[plot_idx]
            axis.set_visible(True)
            source_df = plot_df[plot_df['cell_A_cluster'] == source_cluster]
            target_clusters = self._sorted_cluster_ids(source_df['nearest_B_cluster'].unique())
            target_labels = [self._get_cluster_display_name(cluster) for cluster in target_clusters]
            style = dense_heatmap_style(
                n_rows=max(1, len(target_labels)),
                n_cols=1,
                row_labels=target_labels,
                col_labels=[self._get_cluster_display_name(source_cluster)],
                base_tick_fontsize=10.0,
                allow_annotations=False,
            )

            box_data = []
            box_colors = []
            y_labels = []
            for target_cluster in target_clusters:
                distances = source_df.loc[
                    source_df['nearest_B_cluster'] == target_cluster,
                    'nearest_B_dist_um',
                ].to_numpy(dtype=float, copy=False)
                if distances.size == 0:
                    continue
                box_data.append(distances)
                box_colors.append(color_map.get(target_cluster, '#808080'))
                y_labels.append(self._get_cluster_display_name(target_cluster))

            if not box_data:
                axis.text(
                    0.5,
                    0.5,
                    'No data for this source cluster.',
                    ha='center',
                    va='center',
                    transform=axis.transAxes,
                    fontsize=style['tick_fontsize'] + 1.0,
                )
                axis.set_axis_off()
                continue

            boxplot = axis.boxplot(
                box_data,
                orientation='horizontal',
                patch_artist=True,
                tick_labels=y_labels,
                widths=0.64,
                showfliers=False,
                medianprops={'color': '#1f2933', 'linewidth': 1.4},
                whiskerprops={'color': '#6b7280', 'linewidth': 1.0},
                capprops={'color': '#6b7280', 'linewidth': 1.0},
                boxprops={'linewidth': 1.0, 'edgecolor': '#6b7280'},
            )

            for patch, color in zip(boxplot['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.74)
                patch.set_edgecolor(color)

            axis.set_xlim(0, global_max * 1.05)
            axis.grid(True, alpha=0.25, axis='x')
            axis.set_title(
                f"From {self._get_cluster_display_name(source_cluster)}",
                fontsize=style['title_fontsize'],
                pad=8,
            )
            axis.tick_params(axis='x', labelsize=style['tick_fontsize'])
            axis.tick_params(axis='y', labelsize=style['tick_fontsize'])
            if plot_idx % n_cols == 0:
                axis.set_ylabel('Target Cluster', fontsize=style['axis_fontsize'])
            if plot_idx // n_cols == n_rows - 1:
                axis.set_xlabel('Distance to Nearest Cell (µm)', fontsize=style['axis_fontsize'])

        hidden_suffix = " (self-distances hidden)" if not show_self_pairs else ""
        fig.suptitle(
            f"Distance Distributions by Source Cluster{hidden_suffix}",
            fontsize=11,
            y=0.98,
        )

        self._finalize_canvas_render(self.distance_canvas, pad=0.95)
    
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
            self.distance_cluster_list.clear()
            self.distance_select_all_btn.setEnabled(False)
            self.distance_deselect_all_btn.setEnabled(False)
            self._update_distance_plot()
            return
        
        # Get all unique clusters from both cell_A_cluster and nearest_B_cluster
        # Filter out NaN values
        clusters_a = [c for c in self.distance_df['cell_A_cluster'].unique() if pd.notna(c)]
        clusters_b = [c for c in self.distance_df['nearest_B_cluster'].unique() if pd.notna(c)]
        all_clusters = set(clusters_a) | set(clusters_b)
        all_clusters = self._sorted_cluster_ids(all_clusters)
        
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
        self.community_results_by_roi = {}
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
            else:
                self.batch_corrected_dataframe = None

        self._refresh_feature_set_combo(preferred_feature_set=self.get_active_feature_set_key())
        self._update_active_feature_dataframe()

        label_source = self._get_label_source_dialog()
        if label_source is not None and hasattr(label_source, 'cluster_annotation_map'):
            self.cluster_annotation_map = dict(label_source.cluster_annotation_map or {})
        if label_source is not None and hasattr(label_source, 'feature_label_map'):
            self.feature_label_map = dict(label_source.feature_label_map or {})
        self._apply_cluster_annotations_to_dataframes()
        
        # Refresh ROI combo boxes and other UI elements that depend on dataframe
        self._populate_roi_combo()
        if hasattr(self, '_populate_exclude_clusters_list'):
            self._populate_exclude_clusters_list()
        if hasattr(self, '_populate_spatial_color_options'):
            self._populate_spatial_color_options()

    def apply_feature_set_preference(self, feature_set_key: Optional[str]):
        """Apply a shared feature-set preference without forcing unnecessary resets."""
        target_text = self._feature_set_text_for_key(feature_set_key)
        if hasattr(self, 'feature_set_combo'):
            if self.feature_set_combo.currentText() != target_text:
                self.feature_set_combo.setCurrentText(target_text)
                return

        self._update_active_feature_dataframe()
        self._update_source_file_filter()
    
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
