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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import kruskal, mannwhitneyu
from skimage.measure import regionprops
import json
import math
from openimc.utils.logger import get_logger
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.plot_config_dialog import PlotConfigDialog
from openimc.ui.dialogs.progress_dialog import run_blocking_task_with_progress
from openimc.ui.figure_layout import (
    fit_figure_to_canvas,
    refresh_canvas,
    should_use_nonblocking_canvas_refresh,
    sync_figure_to_canvas,
)
from openimc.ui.cluster_utils import (
    canonicalize_cluster_id,
    cluster_sort_key,
    extract_cluster_annotation_map_from_dataframe,
    format_default_cluster_label,
    get_cluster_display_name,
    normalize_cluster_annotation_map,
    sort_cluster_values,
)
from openimc.core import cluster

# Optional seaborn for enhanced clustering visualization
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:
    _HAVE_SEABORN = False

# Optional leidenalg for Louvain clustering
try:
    import leidenalg
    import igraph as ig
    _HAVE_LEIDEN = True
except ImportError:
    _HAVE_LEIDEN = False

# Optional UMAP for dimensionality reduction
try:
    import umap
    _HAVE_UMAP = True
except ImportError:
    _HAVE_UMAP = False

# Optional HDBSCAN for density-based clustering
try:
    import hdbscan
    _HAVE_HDBSCAN = True
except ImportError:
    _HAVE_HDBSCAN = False

# Optional scikit-learn for k-means and metrics
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

# Optional t-SNE for dimensionality reduction
try:
    from sklearn.manifold import TSNE
    _HAVE_TSNE = True
except ImportError:
    _HAVE_TSNE = False

_HEATMAP_SCALING_DEFAULT_TEXT = "Z-score"
_HEATMAP_SCALING_MAP = {
    "None (no scaling)": "none",
    "Z-score": "zscore",
    "MAD (Median Absolute Deviation)": "mad",
}


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


def _get_patient_colors(n):
    """
    Generate n distinct high-contrast colors for patient/source annotation.
    Uses vibrant, high-contrast color palettes to ensure patient annotations
    are visually distinct and easily distinguishable.
    
    Args:
        n: Number of colors needed
        
    Returns:
        Array of RGBA colors (n, 4)
    """
    colors = []
    
    # Use high-contrast color palettes: tab20, tab20c, Set1, Set2, Dark2
    # These provide much better contrast than pastel colors
    if n <= 20:
        # tab20 provides 20 high-contrast colors
        colors = plt.cm.tab20(np.linspace(0, 1, n))
    elif n <= 40:
        # Combine tab20 and tab20c for 40 colors
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20c(np.linspace(0, 1, n - 20))
        ])
    elif n <= 56:
        # Add Set1 (9 colors) and Set2 (8 colors) for more variety
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20c(np.linspace(0, 1, 20)),
            plt.cm.Set1(np.linspace(0, 1, min(9, n - 40))),
            plt.cm.Set2(np.linspace(0, 1, min(8, max(0, n - 49))))
        ])
    else:
        # For more than 56 colors, use all available high-contrast palettes + hsv
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20c(np.linspace(0, 1, 20)),
            plt.cm.Set1(np.linspace(0, 1, 9)),
            plt.cm.Set2(np.linspace(0, 1, 8)),
            plt.cm.Dark2(np.linspace(0, 1, 8))
        ])
        # Use hsv colormap for additional colors with high saturation
        remaining = n - 65
        if remaining > 0:
            # Use full saturation HSV colors for maximum contrast
            hsv_colors = plt.cm.hsv(np.linspace(0, 1, remaining))
            colors = np.vstack([colors, hsv_colors])
    
    return colors


# --------------------------
# Cell Clustering Dialog
# --------------------------
class CellClusteringDialog(QtWidgets.QDialog):
    def __init__(self, feature_dataframe, normalization_config=None, batch_corrected_dataframe=None, clustered_cells_dataframe=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Clustering Analysis")
        self.setModal(True)
        
        # Set size to 90% of parent window if available
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)
        
        self.setMinimumSize(800, 600)
        self.original_feature_dataframe = feature_dataframe  # Store original (full dataset)
        self.batch_corrected_dataframe = batch_corrected_dataframe  # Store batch-corrected
        # Fallback: if batch-corrected features were loaded elsewhere (e.g. restored/imported session),
        # pick them up from the parent window so users can still choose feature source.
        if (self.batch_corrected_dataframe is None or self.batch_corrected_dataframe.empty) and parent is not None:
            parent_batch_df = getattr(parent, 'batch_corrected_dataframe', None)
            if isinstance(parent_batch_df, pd.DataFrame) and not parent_batch_df.empty:
                self.batch_corrected_dataframe = parent_batch_df.copy()
        # Default to batch-corrected features if available, otherwise use original
        if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
            self.feature_dataframe = self.batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = feature_dataframe  # Active dataframe (can be switched)
        self.normalization_config = normalization_config
        
        # Initialize clustering state
        self.cluster_labels = None
        self.clustered_data = None
        self.clustered_data_unscaled = None  # Store original unscaled data for heatmap display
        
        # If clustered_cells_dataframe is provided (from saved state), use it ONLY for initialization
        # This restores the previous clustering state but does NOT limit future clustering operations
        # When user clicks "Run Clustering", they will use the full feature_dataframe with new filters
        if clustered_cells_dataframe is not None and not clustered_cells_dataframe.empty:
            print(f"[CellClusteringDialog] Initializing with saved clustered_cells: {len(clustered_cells_dataframe)} cells")
            self.clustered_data = clustered_cells_dataframe.copy()
            self.clustered_data_unscaled = self.clustered_data.copy()
            # Extract cluster_labels for compatibility
            if 'cluster' in self.clustered_data.columns:
                self.cluster_labels = self.clustered_data['cluster'].values
                print(f"[CellClusteringDialog] Restored clusters: {sorted(self.clustered_data['cluster'].unique())}")
        # Note: feature_dataframe remains FULL - users can run new clustering with different filters
        self.umap_embedding = None
        self.tsne_embedding = None
        self.cluster_annotation_map = {}
        self.cluster_backend_names = {}  # Kept in sync with persisted cluster phenotype names
        self.original_cluster_assignments = None  # Store original cluster assignments before merging
        self.clustering_scaling_method = None  # Store scaling method used for clustering
        self.actual_clustering_method = None  # Store the clustering method actually used when clustering was run
        self.actual_dendrogram_mode = None  # Store the dendrogram mode actually used when clustering was run
        self.patient_annotation_map = {}  # Store custom patient/source file labels
        self.patient_cohort_map = {}  # Store patient -> cohort mapping (e.g., {'patient1': 'cohort_A', 'patient2': 'cohort_A'})
        self.cohort_colors = {}  # Store cohort -> color mapping (will be auto-generated)
        self.use_cohort_coloring = False  # Toggle for using cohort coloring vs individual coloring
        self.feature_label_map = {}  # Store custom feature labels for y-axis ticks (friendly names)
        self._de_render_context = 'gui'
        self.patient_legend_label = 'Patient/Source'  # Custom label for patient annotation legend
        # Initialize patient annotation column with default priority (source_file, batch_group, source_well, then metadata columns)
        self.patient_annotation_column = None
        if self.feature_dataframe is not None:
            # First check standard columns
            for col in ['source_file', 'batch_group', 'source_well', 'cohort']:
                if col in self.feature_dataframe.columns:
                    self.patient_annotation_column = col
                    break
            
            # If no standard column found, check for metadata columns
            if self.patient_annotation_column is None:
                metadata_cols = self._get_metadata_columns(self.feature_dataframe)
                if metadata_cols:
                    # Prefer columns that might be batch identifiers (PID, patient_id, etc.)
                    priority_metadata = [col for col in metadata_cols 
                                       if any(keyword in col.lower() for keyword in ['pid', 'patient', 'batch', 'sample', 'subject'])]
                    if priority_metadata:
                        self.patient_annotation_column = priority_metadata[0]
                    else:
                        self.patient_annotation_column = metadata_cols[0]
        self.patient_annotation_enabled = False  # Track whether patient annotation is enabled
        self.feature_tick_fontsize = 8  # Font size for feature labels on y-axis (deprecated, use y_tick_fontsize)
        self.x_tick_fontsize = 10  # Font size for x-axis tick labels
        self.y_tick_fontsize = 8  # Font size for y-axis tick labels
        self.legend_nrows = 1  # Number of rows for legend layout
        self.legend_ncols = 1  # Number of columns for legend layout
        self.legend_fontsize = 8  # Font size for legend text
        self.cluster_map_orientation = 'landscape'  # 'portrait' or 'landscape' for Cluster Map
        self.cluster_map_dendrogram = 'Columns only'  # 'Both rows and columns', 'Rows only', 'Columns only', 'No dendrogram'
        self.cluster_map_zscore_method = 'Mean'  # 'Mean', 'Median', 'Max', 'Min' for cluster aggregation
        self.cluster_map_cell_size = 14  # Approximate pixel size per cluster-map cell
        self.cluster_map_colorbar_width = 0.08  # Colorbar thickness in inches
        self.cluster_map_colorbar_position = 'Upper right'  # 'Upper right', 'Upper left', 'Right side'
        self.cluster_map_colorbar_orientation = 'Vertical'  # 'Vertical' or 'Horizontal'
        self.cluster_map_show_legend = False  # Keep cluster map legend off by default to maximize plot area
        self.gating_rules = []  # list of dict: {name, logic, conditions: [{column, op, threshold}]}
        self.llm_phenotype_cache = {}  # Cache for LLM phenotype suggestions
        # Ensure cache is always a dict, never None
        if not isinstance(self.llm_phenotype_cache, dict):
            self.llm_phenotype_cache = {}
        self.seed = 42  # Default seed for reproducibility
        self.statistical_results = {}  # Store statistical test results for export: {marker: [(cluster1, cluster2, p_val, adj_p_val)]}
        self.statistical_results_summary = {}  # Store per-marker omnibus and pairwise testing metadata
        self.filter_settings = None  # Store filter settings from feature selector
        self.selected_display_features = None  # Store selected features for display (separate from clustering features)
        self.de_cluster_filter_selection = None  # Selected clusters for differential expression (None = all)
        self.umap_large_dataset_threshold = 250000  # Warn when UMAP uses very large point counts
        self._last_umap_plot_warning_n = None  # Avoid repeating the same render warning constantly
        self._active_view_name = 'Heatmap'
        self._view_resize_in_progress = False
        self._view_resize_timer = QtCore.QTimer(self)
        self._view_resize_timer.setSingleShot(True)
        self._view_resize_timer.timeout.connect(self._refresh_current_view_after_resize)
        
        self._create_ui()
        self._setup_plot()
        self._on_clustering_type_changed()  # Initialize UI state
        self._on_leiden_mode_changed()  # Initialize Leiden mode state
        
        # Write cohorts to features if they exist (after UI is created so method is available)
        if self.patient_cohort_map:
            self._write_cohorts_to_features()
        
        # Check if cluster columns exist and auto-draw heatmap if they do
        self._check_and_auto_draw_heatmap()

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
            self.feature_set_combo.setToolTip("Choose between original and batch-corrected feature sets")
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

    def refresh_dataframe(self):
        """Refresh source dataframes from the parent window and preserve selection."""
        parent = self.parent()
        if parent is not None and hasattr(parent, 'feature_dataframe') and parent.feature_dataframe is not None:
            self.original_feature_dataframe = parent.feature_dataframe.copy()

        if parent is not None and hasattr(parent, 'batch_corrected_dataframe'):
            parent_batch_df = getattr(parent, 'batch_corrected_dataframe', None)
            if isinstance(parent_batch_df, pd.DataFrame) and not parent_batch_df.empty:
                self.batch_corrected_dataframe = parent_batch_df.copy()
            else:
                self.batch_corrected_dataframe = None

        self._refresh_feature_set_combo(preferred_feature_set=self.get_active_feature_set_key())
        self._update_active_feature_dataframe()
        self._update_clustering_settings_summary()

    def apply_feature_set_preference(self, feature_set_key: Optional[str]):
        """Apply a shared feature-set preference without forcing unnecessary resets."""
        target_text = self._feature_set_text_for_key(feature_set_key)
        if hasattr(self, 'feature_set_combo'):
            if self.feature_set_combo.currentText() != target_text:
                self.feature_set_combo.setCurrentText(target_text)
                return

        self._update_active_feature_dataframe()
        self._update_clustering_settings_summary()

    def _get_logging_dataframe(self):
        """Return the dataframe that best represents the current clustering context."""
        if self.clustered_data is not None and not self.clustered_data.empty:
            return self.clustered_data
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            return self.feature_dataframe
        return None

    def _get_logging_acquisitions(self) -> List[str]:
        """Collect acquisition IDs for analysis-step logging."""
        df = self._get_logging_dataframe()
        if df is None or 'acquisition_id' not in df.columns:
            return []
        return [str(v) for v in df['acquisition_id'].dropna().unique().tolist()]

    def _get_source_files_for_logging(self) -> List[str]:
        """Collect source file basenames for analysis-step logging."""
        df = self._get_logging_dataframe()
        source_files = []
        if df is not None and 'source_file' in df.columns:
            seen = set()
            for value in df['source_file'].dropna().tolist():
                basename = os.path.basename(str(value))
                if basename and basename not in seen:
                    seen.add(basename)
                    source_files.append(basename)

        if not source_files and self.parent() is not None and hasattr(self.parent(), 'current_path'):
            current_path = getattr(self.parent(), 'current_path', None)
            if current_path:
                source_files = [os.path.basename(current_path)]

        return source_files

    def _get_source_file_summary_for_logging(self):
        """Return a compact human-readable source file summary for log headers."""
        source_files = self._get_source_files_for_logging()
        if not source_files:
            return None
        if len(source_files) == 1:
            return source_files[0]
        if len(source_files) <= 3:
            return ", ".join(source_files)
        return ", ".join(source_files[:3]) + f" and {len(source_files) - 3} more"

    def _load_cluster_annotations_from_dataframes(self):
        """Recover persisted phenotype names from loaded clustered dataframes."""
        loaded_annotations = {}
        for dataframe in (self.feature_dataframe, self.clustered_data):
            loaded_annotations.update(extract_cluster_annotation_map_from_dataframe(dataframe))

        if not loaded_annotations:
            return

        self.cluster_annotation_map = normalize_cluster_annotation_map(
            {
                **self.cluster_annotation_map,
                **loaded_annotations,
            }
        )
        self.cluster_backend_names = dict(self.cluster_annotation_map)
    
    def _check_and_auto_draw_heatmap(self):
        """Check if cluster columns exist and auto-draw heatmap if they do.
        
        This is called during initialization when loading a saved file.
        If clustered_data was already initialized (from clustered_cells_dataframe),
        we use that directly. Otherwise, we extract from feature_dataframe.
        """
        # If clustered_data is already initialized (from clustered_cells_dataframe), use it
        if self.clustered_data is not None and not self.clustered_data.empty:
            print(f"[_check_and_auto_draw_heatmap] Using pre-initialized clustered_data: {len(self.clustered_data)} cells")
            # clustered_data and clustered_data_unscaled should already be set
            # Just need to set display features and draw
            self._load_cluster_annotations_from_dataframes()
            if hasattr(self, 'last_features_used') and self.last_features_used:
                available_last_features = [f for f in self.last_features_used if f in self.clustered_data.columns]
                if available_last_features:
                    self.selected_display_features = available_last_features
            
            # Set view to Heatmap and draw it
            if hasattr(self, 'view_combo'):
                self.view_combo.setCurrentText('Heatmap')
            self._show_heatmap()
            self._update_cluster_action_buttons()
            return
        
        # Otherwise, extract from feature_dataframe (backward compatibility)
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return
        
        # Check for cluster columns
        cluster_cols = ['cluster', 'cluster_id', 'cluster_phenotype']
        has_cluster_col = any(col in self.feature_dataframe.columns for col in cluster_cols)
        
        if not has_cluster_col:
            return
        
        # Find which cluster column exists
        cluster_col = None
        for col in cluster_cols:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        if cluster_col is None:
            return
        
        # Check for morphometric and _mean intensity features
        all_cols = self.feature_dataframe.columns.tolist()
        
        # Morphometric features (common names)
        # Exclude centroid_x and centroid_y by default
        morpho_keywords = ['area', 'perimeter', 'eccentricity', 'solidity', 'extent', 
                          'major_axis', 'minor_axis', 'orientation', 'convex_area',
                          'equivalent_diameter', 'bbox']
        morpho_features = [col for col in all_cols 
                          if any(keyword in col.lower() for keyword in morpho_keywords)
                          and pd.api.types.is_numeric_dtype(self.feature_dataframe[col])
                          and col not in ['centroid_x', 'centroid_y']]
        
        # Mean intensity features
        mean_features = [col for col in all_cols 
                       if col.endswith('_mean') 
                       and pd.api.types.is_numeric_dtype(self.feature_dataframe[col])]
        
        # Include touches_edge if it exists
        touches_edge_features = []
        if 'touches_edge' in all_cols and pd.api.types.is_numeric_dtype(self.feature_dataframe['touches_edge']):
            touches_edge_features = ['touches_edge']
        
        # Combine morphometric, mean, and touches_edge features
        display_features = morpho_features + mean_features + touches_edge_features
        
        if not display_features:
            # Show warning if no morphometric or mean features found
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Features",
                "The loaded features file contains cluster information, but no morphometric "
                "or _mean intensity features were found.\n\n"
                "Please re-cluster the data to generate the required features for visualization."
            )
            return
        
        # Set up clustered_data from existing feature dataframe
        # IMPORTANT: Only include cells that were actually clustered (cluster != 0 and not NaN)
        # This respects the original filtering (e.g., edge cells excluded during clustering)
        print(f"[_check_and_auto_draw_heatmap] Extracting clustered_data from feature_dataframe")
        valid_cluster_mask = (self.feature_dataframe[cluster_col] != 0) & (self.feature_dataframe[cluster_col].notna())
        self.clustered_data = self.feature_dataframe[valid_cluster_mask].copy()
        print(f"[_check_and_auto_draw_heatmap] Extracted {len(self.clustered_data)} cells with valid clusters")
        
        # Ensure cluster column is named 'cluster' for consistency
        if cluster_col != 'cluster':
            if 'cluster' in self.clustered_data.columns:
                # If both exist, prefer the numeric one
                if cluster_col in ['cluster_id']:
                    self.clustered_data['cluster'] = self.clustered_data[cluster_col]
            else:
                self.clustered_data['cluster'] = self.clustered_data[cluster_col]
        
        # Convert cluster column to numeric if it's not already
        if not pd.api.types.is_numeric_dtype(self.clustered_data['cluster']):
            # Try to convert to numeric, creating a mapping if needed
            unique_clusters = self.clustered_data['cluster'].unique()
            # Filter out NaN values for mapping
            unique_clusters = [c for c in unique_clusters if pd.notna(c)]
            cluster_map = {
                val: idx + 1
                for idx, val in enumerate(self._sorted_cluster_ids(unique_clusters, canonical=False))
            }
            self.clustered_data['cluster'] = self.clustered_data['cluster'].map(cluster_map)
        
        # Ensure cluster column is numeric and handle NaN/inf values
        self.clustered_data['cluster'] = pd.to_numeric(self.clustered_data['cluster'], errors='coerce')
        # Replace inf and fill NaN with 0 (unassigned)
        self.clustered_data['cluster'] = self.clustered_data['cluster'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Store unscaled data
        self.clustered_data_unscaled = self.clustered_data.copy()
        self._load_cluster_annotations_from_dataframes()
        
        # Set selected display features to morphometric and mean features
        # Prioritize last_features_used if available (from previous clustering session)
        if hasattr(self, 'last_features_used') and self.last_features_used:
            # Filter to only include features that exist in current dataframe
            available_last_features = [f for f in self.last_features_used if f in self.feature_dataframe.columns]
            if available_last_features:
                self.selected_display_features = available_last_features
            else:
                self.selected_display_features = display_features
        else:
            self.selected_display_features = display_features
        
        # Set view to Heatmap and draw it
        if hasattr(self, 'view_combo'):
            self.view_combo.setCurrentText('Heatmap')
        self._show_heatmap()
        self._update_cluster_action_buttons()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Cell Clustering Analysis")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12pt; }")
        layout.addWidget(title_label)
        
        # Options panel
        options_group = QtWidgets.QGroupBox("Clustering Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        options_grid = QtWidgets.QGridLayout()
        options_grid.setHorizontalSpacing(10)
        options_grid.setVerticalSpacing(6)
        options_grid.setColumnStretch(1, 1)
        options_grid.setColumnStretch(3, 1)
        options_row = 0
        
        # Feature source selector
        self.feature_set_label = QtWidgets.QLabel("Feature Set:")
        options_grid.addWidget(self.feature_set_label, options_row, 0)
        self.feature_set_combo = QtWidgets.QComboBox()
        self.feature_set_combo.currentTextChanged.connect(self._on_feature_set_changed)
        self._refresh_feature_set_combo(
            preferred_feature_set="batch_corrected" if self._has_batch_corrected_features() else "original"
        )
        options_grid.addWidget(self.feature_set_combo, options_row, 1)
        options_row += 1
        
        # (Aggregation and morphometric inclusion moved to Feature Selector dialog)
        
        # Clustering method type (first)
        options_grid.addWidget(QtWidgets.QLabel("Clustering Method:"), options_row, 0)
        self.clustering_type = QtWidgets.QComboBox()
        clustering_types = []
        if _HAVE_LEIDEN:
            clustering_types.append("Leiden")
            clustering_types.append("Louvain")
        if _HAVE_SKLEARN:
            clustering_types.append("K-means")
        if _HAVE_HDBSCAN:
            clustering_types.append("HDBSCAN")
        clustering_types.append("Hierarchical")
        self.clustering_type.addItems(clustering_types)
        # Set Leiden as default if available, otherwise fall back to Hierarchical
        if _HAVE_LEIDEN:
            self.clustering_type.setCurrentText("Leiden")
        else:
            self.clustering_type.setCurrentText("Hierarchical")
        self.clustering_type.currentTextChanged.connect(self._on_clustering_type_changed)
        options_grid.addWidget(self.clustering_type, options_row, 1)
        
        # Feature scaling method (for clustering)
        options_grid.addWidget(QtWidgets.QLabel("Feature Scaling:"), options_row, 2)
        self.clustering_scaling_combo = QtWidgets.QComboBox()
        self.clustering_scaling_combo.addItems(["None (no scaling)", "Z-score", "MAD (Median Absolute Deviation)"])
        self.clustering_scaling_combo.setCurrentText("None (no scaling)")
        self.clustering_scaling_combo.setToolTip("Scaling method applied to features before clustering")
        self.clustering_scaling_combo.currentTextChanged.connect(self._update_clustering_settings_summary)
        options_grid.addWidget(self.clustering_scaling_combo, options_row, 3)
        options_row += 1
        
        # PCA feature representation options
        self.pca_options_group = QtWidgets.QGroupBox("Feature Representation")
        pca_options_layout = QtWidgets.QGridLayout(self.pca_options_group)
        pca_options_layout.setHorizontalSpacing(10)
        pca_options_layout.setVerticalSpacing(6)
        pca_options_layout.setColumnStretch(1, 1)
        pca_options_layout.setColumnStretch(3, 1)

        self.use_pca_checkbox = QtWidgets.QCheckBox("Cluster on principal components")
        self.use_pca_checkbox.setChecked(False)
        self.use_pca_checkbox.setToolTip("Project selected, scaled features into PC space before clustering")
        if not _HAVE_SKLEARN:
            self.use_pca_checkbox.setEnabled(False)
            self.use_pca_checkbox.setToolTip("PCA clustering requires scikit-learn")
        self.use_pca_checkbox.toggled.connect(self._on_pca_controls_changed)
        pca_options_layout.addWidget(self.use_pca_checkbox, 0, 0, 1, 2)

        self.pca_mode_label = QtWidgets.QLabel("PC selection:")
        self.pca_mode_combo = QtWidgets.QComboBox()
        self.pca_mode_combo.addItem("Variance retained", "variance")
        self.pca_mode_combo.addItem("Number of PCs", "components")
        self.pca_mode_combo.setCurrentIndex(0)
        self.pca_mode_combo.currentIndexChanged.connect(self._on_pca_controls_changed)
        pca_options_layout.addWidget(self.pca_mode_label, 1, 0)
        pca_options_layout.addWidget(self.pca_mode_combo, 1, 1)

        self.pca_variance_label = QtWidgets.QLabel("Variance retained:")
        self.pca_variance_spinbox = QtWidgets.QDoubleSpinBox()
        self.pca_variance_spinbox.setRange(1.0, 100.0)
        self.pca_variance_spinbox.setDecimals(1)
        self.pca_variance_spinbox.setSingleStep(1.0)
        self.pca_variance_spinbox.setSuffix("%")
        self.pca_variance_spinbox.setValue(95.0)
        self.pca_variance_spinbox.setToolTip("Target cumulative variance retained by PCA")
        self.pca_variance_spinbox.valueChanged.connect(self._on_pca_controls_changed)
        pca_options_layout.addWidget(self.pca_variance_label, 1, 2)
        pca_options_layout.addWidget(self.pca_variance_spinbox, 1, 3)

        self.pca_n_components_label = QtWidgets.QLabel("Number of PCs:")
        self.pca_n_components_spinbox = QtWidgets.QSpinBox()
        self.pca_n_components_spinbox.setRange(1, 1000)
        self.pca_n_components_spinbox.setValue(10)
        self.pca_n_components_spinbox.setToolTip("Fixed number of principal components to retain")
        self.pca_n_components_spinbox.valueChanged.connect(self._on_pca_controls_changed)
        pca_options_layout.addWidget(self.pca_n_components_label, 2, 2)
        pca_options_layout.addWidget(self.pca_n_components_spinbox, 2, 3)
        self._on_pca_controls_changed()

        # Random seed
        options_grid.addWidget(QtWidgets.QLabel("Random Seed:"), options_row, 0)
        self.seed_spinbox = QtWidgets.QSpinBox()
        self.seed_spinbox.setRange(0, 2**31 - 1)
        self.seed_spinbox.setValue(42)
        self.seed_spinbox.setToolTip("Random seed for reproducibility (default: 42)")
        options_grid.addWidget(self.seed_spinbox, options_row, 1)
        
        # Number of clusters (for hierarchical and k-means)
        self.n_clusters_label = QtWidgets.QLabel("Number of clusters:")
        options_grid.addWidget(self.n_clusters_label, options_row, 2)
        self.n_clusters = QtWidgets.QSpinBox()
        self.n_clusters.setRange(2, 50)
        self.n_clusters.setValue(5)
        options_grid.addWidget(self.n_clusters, options_row, 3)
        options_row += 1
        
        # K-range search button (for hierarchical and k-means)
        self.k_range_btn = QtWidgets.QPushButton("Find Optimal K")
        self.k_range_btn.setToolTip("Search over a range of k values and plot elbow/silhouette scores")
        self.k_range_btn.clicked.connect(self._open_k_range_dialog)
        options_grid.addWidget(self.k_range_btn, options_row, 0, 1, 2)
        
        # Hierarchical method selection (initially visible)
        self.hierarchical_label = QtWidgets.QLabel("Linkage Method:")
        self.hierarchical_method = QtWidgets.QComboBox()
        self.hierarchical_method.addItems(["ward", "complete", "average", "single"])
        self.hierarchical_method.setCurrentText("ward")
        options_grid.addWidget(self.hierarchical_label, options_row, 2)
        options_grid.addWidget(self.hierarchical_method, options_row, 3)
        options_row += 1
        
        # Leiden clustering options (initially hidden)
        self.leiden_options_group = QtWidgets.QGroupBox("Leiden/Louvain Options")
        leiden_options_layout = QtWidgets.QGridLayout(self.leiden_options_group)
        leiden_options_layout.setHorizontalSpacing(10)
        leiden_options_layout.setVerticalSpacing(6)
        leiden_options_layout.setColumnStretch(1, 1)
        leiden_options_layout.setColumnStretch(3, 1)
        
        # Resolution vs Modularity choice
        self.leiden_mode_group = QtWidgets.QButtonGroup()
        self.resolution_radio = QtWidgets.QRadioButton("Resolution")
        self.modularity_radio = QtWidgets.QRadioButton("Modularity")
        self.resolution_radio.setChecked(True)
        self.leiden_mode_group.addButton(self.resolution_radio)
        self.leiden_mode_group.addButton(self.modularity_radio)
        leiden_options_layout.addWidget(self.resolution_radio, 0, 0)
        leiden_options_layout.addWidget(self.modularity_radio, 0, 1)
        
        # N neighbors parameter for graph construction
        self.n_neighbors_label = QtWidgets.QLabel("N neighbors:")
        self.n_neighbors_spinbox = QtWidgets.QSpinBox()
        self.n_neighbors_spinbox.setRange(5, 100)
        self.n_neighbors_spinbox.setValue(15)
        self.n_neighbors_spinbox.setToolTip("Number of neighbors for k-NN graph construction")
        leiden_options_layout.addWidget(self.n_neighbors_label, 1, 0)
        leiden_options_layout.addWidget(self.n_neighbors_spinbox, 1, 1)
        
        # Resolution parameter
        self.resolution_label = QtWidgets.QLabel("Resolution:")
        self.resolution_spinbox = QtWidgets.QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.1, 5.0)
        self.resolution_spinbox.setSingleStep(0.1)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setDecimals(1)
        leiden_options_layout.addWidget(self.resolution_label, 1, 2)
        leiden_options_layout.addWidget(self.resolution_spinbox, 1, 3)
        
        # Distance metric selection
        self.leiden_metric_label = QtWidgets.QLabel("Distance metric:")
        self.leiden_metric_combo = QtWidgets.QComboBox()
        self.leiden_metric_combo.addItems(["euclidean", "manhattan", "cosine"])
        self.leiden_metric_combo.setCurrentText("euclidean")
        self.leiden_metric_combo.setToolTip("Distance metric to use for k-NN graph construction")
        leiden_options_layout.addWidget(self.leiden_metric_label, 2, 0)
        leiden_options_layout.addWidget(self.leiden_metric_combo, 2, 1)
        
        # Jaccard weighting option (PhenoGraph-like)
        self.jaccard_checkbox = QtWidgets.QCheckBox("Use Jaccard weighting")
        self.jaccard_checkbox.setToolTip("Weight graph edges with Jaccard similarity (PhenoGraph-like implementation)")
        self.jaccard_checkbox.setChecked(False)
        leiden_options_layout.addWidget(self.jaccard_checkbox, 2, 2, 1, 2)
        
        # Connect radio button changes
        self.resolution_radio.toggled.connect(self._on_leiden_mode_changed)
        self.modularity_radio.toggled.connect(self._on_leiden_mode_changed)
        
        self.leiden_options_group.setVisible(False)
        options_layout.addLayout(options_grid)
        options_layout.addWidget(self.pca_options_group)
        options_layout.addWidget(self.leiden_options_group)

        # HDBSCAN clustering options (initially hidden)
        self.hdbscan_options_group = QtWidgets.QGroupBox("HDBSCAN Clustering Options")
        hdbscan_options_layout = QtWidgets.QGridLayout(self.hdbscan_options_group)
        hdbscan_options_layout.setHorizontalSpacing(10)
        hdbscan_options_layout.setVerticalSpacing(6)
        hdbscan_options_layout.setColumnStretch(1, 1)
        hdbscan_options_layout.setColumnStretch(3, 1)
        
        # Min cluster size
        self.min_cluster_size_label = QtWidgets.QLabel("Min cluster size:")
        self.min_cluster_size_spinbox = QtWidgets.QSpinBox()
        self.min_cluster_size_spinbox.setRange(2, 1000)
        self.min_cluster_size_spinbox.setValue(10)
        self.min_cluster_size_spinbox.setToolTip("Minimum size of clusters; smaller clusters will be discarded as noise")
        hdbscan_options_layout.addWidget(self.min_cluster_size_label, 0, 0)
        hdbscan_options_layout.addWidget(self.min_cluster_size_spinbox, 0, 1)
        
        # Min samples
        self.min_samples_label = QtWidgets.QLabel("Min samples:")
        self.min_samples_spinbox = QtWidgets.QSpinBox()
        self.min_samples_spinbox.setRange(1, 100)
        self.min_samples_spinbox.setValue(5)
        self.min_samples_spinbox.setToolTip("Number of samples in a neighborhood for a point to be considered a core point")
        hdbscan_options_layout.addWidget(self.min_samples_label, 0, 2)
        hdbscan_options_layout.addWidget(self.min_samples_spinbox, 0, 3)
        
        # Cluster selection method (EOM vs Leaf)
        self.cluster_selection_label = QtWidgets.QLabel("Cluster selection method:")
        self.cluster_selection_combo = QtWidgets.QComboBox()
        self.cluster_selection_combo.addItems(["eom", "leaf"])
        self.cluster_selection_combo.setCurrentText("eom")
        self.cluster_selection_combo.setToolTip("eom: Excess of Mass (default, more conservative)\nleaf: Leaf (more aggressive, creates smaller clusters)")
        hdbscan_options_layout.addWidget(self.cluster_selection_label, 1, 0)
        hdbscan_options_layout.addWidget(self.cluster_selection_combo, 1, 1)
        
        # Metric selection (only euclidean and manhattan for HDBSCAN)
        self.metric_label = QtWidgets.QLabel("Distance metric:")
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(["euclidean", "manhattan"])
        self.metric_combo.setCurrentText("euclidean")
        self.metric_combo.setToolTip("Distance metric to use for clustering")
        hdbscan_options_layout.addWidget(self.metric_label, 1, 2)
        hdbscan_options_layout.addWidget(self.metric_combo, 1, 3)
        
        self.hdbscan_options_group.setVisible(False)
        options_layout.addWidget(self.hdbscan_options_group)

        # Dendrogram mode (only for hierarchical methods)
        self.dendro_label = QtWidgets.QLabel("Dendrogram:")
        self.dendro_mode = QtWidgets.QComboBox()
        self.dendro_mode.addItems(["Rows only", "Rows and columns"]) 
        self.dendro_mode.setCurrentText("Rows and columns")  # Default to both dendrograms
        options_grid.addWidget(self.dendro_label, options_row, 0)
        options_grid.addWidget(self.dendro_mode, options_row, 1)
        
        # Run clustering button
        self.run_btn = QtWidgets.QPushButton("Run Clustering")
        self.run_btn.clicked.connect(self._run_clustering)
        
        # Save clustering output button
        self.save_output_btn = QtWidgets.QPushButton("Save Clustering Output")
        self.save_output_btn.clicked.connect(self._save_clustering_output)
        self.save_output_btn.setEnabled(False)
        self.save_output_btn.setToolTip("Save CSV with all features, cluster labels, and manual annotations")

        # Keep full clustering options in a popup dialog to maximize plot space.
        self._clustering_options_group = options_group
        self._build_clustering_settings_dialog()

        # Compact top control row (plot-first layout).
        controls_row = QtWidgets.QHBoxLayout()
        self.cluster_settings_btn = QtWidgets.QPushButton("Clustering Settings...")
        self.cluster_settings_btn.setToolTip("Open clustering configuration (method, resolution, feature set, and related options)")
        self.cluster_settings_btn.clicked.connect(self._open_clustering_settings_dialog)
        controls_row.addWidget(self.cluster_settings_btn)

        self.settings_summary_label = QtWidgets.QLabel("")
        self.settings_summary_label.setStyleSheet("QLabel { color: #666; }")
        controls_row.addWidget(self.settings_summary_label, 1)

        controls_row.addWidget(self.run_btn)
        controls_row.addWidget(self.save_output_btn)
        layout.addLayout(controls_row)
        
        # Plot area (Step 2: Visualization)
        plot_group = QtWidgets.QGroupBox("Visualization")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        
        # Create matplotlib canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        # Visualization controls
        viz_layout = QtWidgets.QHBoxLayout()
        viz_layout.addWidget(QtWidgets.QLabel("View:"))
        self.view_combo = QtWidgets.QComboBox()
        view_items = ["Heatmap", "Cluster Map", "UMAP", "Stacked Bars", "Differential Expression", "Boxplot/Violin Plot"]
        if _HAVE_TSNE:
            view_items.insert(3, "t-SNE")  # Insert after UMAP
        self.view_combo.addItems(view_items)
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        viz_layout.addWidget(self.view_combo)

        # Color-by control (UMAP/t-SNE only) - multi-select for faceted plotting
        self.color_by_label = QtWidgets.QLabel("Color by (select multiple for faceted plots):")
        viz_layout.addWidget(self.color_by_label)
        # Search/filter box for color-by options
        self.color_by_search = QtWidgets.QLineEdit()
        self.color_by_search.setPlaceholderText("Search/filter options...")
        self.color_by_search.textChanged.connect(self._filter_color_by_options)
        viz_layout.addWidget(self.color_by_search)
        self.color_by_listwidget = QtWidgets.QListWidget()
        self.color_by_listwidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.color_by_listwidget.setMaximumHeight(100)
        self.color_by_listwidget.itemSelectionChanged.connect(self._on_color_by_changed)
        viz_layout.addWidget(self.color_by_listwidget)
        # Keep combo for backward compatibility but hide it
        self.color_by_combo = QtWidgets.QComboBox()
        self.color_by_combo.setVisible(False)
        
        # Cohort coloring toggle (only show if cohorts exist)
        self.use_cohort_checkbox = QtWidgets.QCheckBox("Use cohort coloring")
        self.use_cohort_checkbox.setToolTip("When enabled, patients/batch groups assigned to cohorts will share colors. When disabled, each patient/batch group gets its own color.")
        self.use_cohort_checkbox.setChecked(False)
        self.use_cohort_checkbox.stateChanged.connect(self._on_cohort_coloring_changed)
        self.use_cohort_checkbox.setVisible(False)  # Hidden by default, shown when cohorts exist
        viz_layout.addWidget(self.use_cohort_checkbox)

        # Point size control (UMAP/t-SNE only)
        self.point_size_label = QtWidgets.QLabel("Point size:")
        viz_layout.addWidget(self.point_size_label)
        self.point_size_spinbox = QtWidgets.QSpinBox()
        self.point_size_spinbox.setMinimum(1)
        self.point_size_spinbox.setMaximum(200)
        self.point_size_spinbox.setValue(18)
        self.point_size_spinbox.setToolTip("Size of points in scatter plot")
        self.point_size_spinbox.valueChanged.connect(self._on_point_style_changed)
        viz_layout.addWidget(self.point_size_spinbox)

        # Point alpha control (UMAP/t-SNE only)
        self.point_alpha_label = QtWidgets.QLabel("Point alpha:")
        viz_layout.addWidget(self.point_alpha_label)
        self.point_alpha_spinbox = QtWidgets.QDoubleSpinBox()
        self.point_alpha_spinbox.setMinimum(0.0)
        self.point_alpha_spinbox.setMaximum(1.0)
        self.point_alpha_spinbox.setSingleStep(0.1)
        self.point_alpha_spinbox.setValue(0.8)
        self.point_alpha_spinbox.setDecimals(2)
        self.point_alpha_spinbox.setToolTip("Transparency of points (0.0 = transparent, 1.0 = opaque)")
        self.point_alpha_spinbox.valueChanged.connect(self._on_point_style_changed)
        viz_layout.addWidget(self.point_alpha_spinbox)

        # Show legend checkbox (UMAP/t-SNE/Stacked Bars)
        self.show_legend_checkbox = QtWidgets.QCheckBox("Show legend")
        self.show_legend_checkbox.setChecked(True)
        self.show_legend_checkbox.setToolTip("Show/hide legend in plots (legend is also shown in heatmap)")
        self.show_legend_checkbox.stateChanged.connect(self._on_legend_changed)
        viz_layout.addWidget(self.show_legend_checkbox)

        # Remake UMAP button (UMAP only)
        self.remake_umap_btn = QtWidgets.QPushButton("Remake UMAP")
        self.remake_umap_btn.setToolTip("Regenerate UMAP with new parameters (features, scaling, n_neighbors)")
        self.remake_umap_btn.clicked.connect(self._remake_umap)
        viz_layout.addWidget(self.remake_umap_btn)

        # Group-by for stacked bars (Stacked Bars only)
        self.group_by_label = QtWidgets.QLabel("Group by:")
        viz_layout.addWidget(self.group_by_label)
        self.group_by_combo = QtWidgets.QComboBox()
        candidate_cols = [
            'roi', 'ROI', 'slide', 'Slide', 'condition', 'Condition',
            'acquisition_name', 'well', 'acquisition_id'
        ]
        available_group_cols = [c for c in candidate_cols if c in self.feature_dataframe.columns]
        
        # Build priority columns list in order (all should be options, not replacements)
        priority_cols = []
        if 'source_file' in self.feature_dataframe.columns:
            priority_cols.append('source_file')
        if 'batch_group' in self.feature_dataframe.columns:
            priority_cols.append('batch_group')
        if 'cohort' in self.feature_dataframe.columns:
            priority_cols.append('cohort')
        if 'source_well' in self.feature_dataframe.columns:
            priority_cols.append('source_well')
        
        # Add priority columns at the beginning, avoiding duplicates
        # Insert in reverse order so they appear in the correct priority order
        for col in reversed(priority_cols):
            if col not in available_group_cols:
                available_group_cols.insert(0, col)
        
        # Add source_file_acquisition_id if both source_file and acquisition_id exist
        if 'source_file' in self.feature_dataframe.columns and 'acquisition_id' in self.feature_dataframe.columns:
            # Create merged column if it doesn't exist
            if 'source_file_acquisition_id' not in self.feature_dataframe.columns:
                # Use assign() to avoid DataFrame fragmentation warnings
                self.feature_dataframe = self.feature_dataframe.assign(
                    source_file_acquisition_id=(
                        self.feature_dataframe['source_file'].astype(str) + '_' + 
                        self.feature_dataframe['acquisition_id'].astype(str)
                    )
                )
            if 'source_file_acquisition_id' not in available_group_cols:
                available_group_cols.insert(0, 'source_file_acquisition_id')
        if not available_group_cols:
            available_group_cols = ['acquisition_name'] if 'acquisition_name' in self.feature_dataframe.columns else []
        
        # Add metadata columns for grouping
        metadata_cols = self._get_metadata_columns(self.feature_dataframe)
        if metadata_cols:
            if available_group_cols:
                available_group_cols.extend(metadata_cols)
            else:
                available_group_cols = metadata_cols
        
        for col in available_group_cols:
            self.group_by_combo.addItem(col)
        viz_layout.addWidget(self.group_by_combo)
        
        # View type selector for stacked bars (Fraction vs Total enumeration)
        self.stacked_bars_view_type_label = QtWidgets.QLabel("View type:")
        viz_layout.addWidget(self.stacked_bars_view_type_label)
        self.stacked_bars_view_type_combo = QtWidgets.QComboBox()
        self.stacked_bars_view_type_combo.addItems(["Fraction", "Total enumeration"])
        self.stacked_bars_view_type_combo.setCurrentText("Fraction")
        self.stacked_bars_view_type_combo.currentTextChanged.connect(self._on_stacked_bars_view_type_changed)
        viz_layout.addWidget(self.stacked_bars_view_type_combo)
        
        # Cluster filter button for stacked bars
        self.stacked_bars_filter_btn = QtWidgets.QPushButton("Filter Clusters...")
        self.stacked_bars_filter_btn.setToolTip("Select which clusters to display in the stacked bars plot")
        self.stacked_bars_filter_btn.clicked.connect(self._open_stacked_bars_filter_dialog)
        viz_layout.addWidget(self.stacked_bars_filter_btn)
        
        # Initialize stacked bars filter selection (None means show all clusters)
        self.stacked_bars_filter_selection = None
        
        # Connect group_by_combo to refresh plot when changed
        self.group_by_combo.currentTextChanged.connect(self._on_group_by_changed)

        # Colormap selector (for heatmaps and differential expression)
        # Note: For Heatmap, colormap is also available in PlotConfigDialog
        # But we keep it here for Differential Expression since Configure Plot only shows for Heatmap
        self.colormap_label = QtWidgets.QLabel("Colormap:")
        viz_layout.addWidget(self.colormap_label)
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems([
            "RdBu_r (Red-White-Blue)",
            "viridis (Purple-Green-Yellow)", 
            "plasma (Purple-Pink-Yellow)",
            "inferno (Purple-Red-Yellow)",
            "Blues (Light-Dark Blue)",
            "Reds (Light-Dark Red)",
            "Greens (Light-Dark Green)",
            "Oranges (Light-Dark Orange)",
            "Purples (Light-Dark Purple)"
        ])
        self.colormap_combo.setCurrentText("RdBu_r (Red-White-Blue)")
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        viz_layout.addWidget(self.colormap_combo)

        # Heatmap scaling selector (for heatmap only)
        self.heatmap_scaling_label = QtWidgets.QLabel("Heatmap Scaling:")
        viz_layout.addWidget(self.heatmap_scaling_label)
        self.heatmap_scaling_combo = QtWidgets.QComboBox()
        self.heatmap_scaling_combo.addItems(["None (no scaling)", "Z-score", "MAD (Median Absolute Deviation)"])
        self.heatmap_scaling_combo.setCurrentText(_HEATMAP_SCALING_DEFAULT_TEXT)
        self.heatmap_scaling_combo.setToolTip("Scaling method applied to features in the heatmap display")
        self.heatmap_scaling_combo.currentTextChanged.connect(self._on_heatmap_scaling_changed)
        viz_layout.addWidget(self.heatmap_scaling_combo)

        # Configure plot button (opens PlotConfigDialog)
        # Note: Heatmap source, heatmap filter, heatmap scaling, patient annotation,
        # and patient label customization are now available in PlotConfigDialog (Heatmap only)
        self.configure_plot_btn = QtWidgets.QPushButton("Configure Plot...")
        self.configure_plot_btn.setToolTip("Open plot configuration dialog to customize font sizes, labels, and other plot settings")
        self.configure_plot_btn.clicked.connect(self._open_plot_config_dialog)
        viz_layout.addWidget(self.configure_plot_btn)

        # Customize feature labels button (for Differential Expression, Stacked Bars, Boxplot/Violin Plot)
        self.feature_labels_btn = QtWidgets.QPushButton("Customize Feature Labels...")
        self.feature_labels_btn.setToolTip("Set custom display names for features in visualizations (e.g., 'Vimentin_mean' -> 'Mean Vimentin')")
        self.feature_labels_btn.clicked.connect(self._open_feature_labels_dialog)
        viz_layout.addWidget(self.feature_labels_btn)

        # Top N markers selector (for differential expression only)
        # Note: Also available in PlotConfigDialog, but kept here for quick access
        self.top_n_label = QtWidgets.QLabel("Top N:")
        viz_layout.addWidget(self.top_n_label)
        self.top_n_spinbox = QtWidgets.QSpinBox()
        self.top_n_spinbox.setMinimum(1)
        self.top_n_spinbox.setMaximum(20)
        self.top_n_spinbox.setValue(5)
        self.top_n_spinbox.valueChanged.connect(self._on_top_n_changed)
        viz_layout.addWidget(self.top_n_spinbox)
        self.de_filter_btn = QtWidgets.QPushButton("Select Clusters...")
        self.de_filter_btn.setToolTip("Select which clusters to include in differential expression analysis")
        self.de_filter_btn.clicked.connect(self._open_de_cluster_filter_dialog)
        viz_layout.addWidget(self.de_filter_btn)
        self.de_flip_axes_checkbox = QtWidgets.QCheckBox("Flip axes")
        self.de_flip_axes_checkbox.setToolTip(
            "Transpose the differential expression heatmap so clusters are shown on the Y axis and features on the X axis."
        )
        self.de_flip_axes_checkbox.setChecked(False)
        self.de_flip_axes_checkbox.stateChanged.connect(self._on_de_axis_orientation_changed)
        viz_layout.addWidget(self.de_flip_axes_checkbox)
        self.de_show_values_checkbox = QtWidgets.QCheckBox("Show z-score values")
        self.de_show_values_checkbox.setToolTip(
            "Overlay the numeric z-score values inside the differential expression heatmap cells."
        )
        self.de_show_values_checkbox.setChecked(True)
        self.de_show_values_checkbox.stateChanged.connect(self._on_de_axis_orientation_changed)
        viz_layout.addWidget(self.de_show_values_checkbox)
        self.de_show_boxes_checkbox = QtWidgets.QCheckBox("Show top-marker boxes")
        self.de_show_boxes_checkbox.setToolTip(
            "Draw black boxes around the per-cluster top markers in the differential expression heatmap."
        )
        self.de_show_boxes_checkbox.setChecked(True)
        self.de_show_boxes_checkbox.stateChanged.connect(self._on_de_axis_orientation_changed)
        viz_layout.addWidget(self.de_show_boxes_checkbox)

        # Note: Marker selection, plot type, and statistical testing options
        # are also available in PlotConfigDialog, but kept here for quick access
        # Marker selection for boxplot/violin plot
        self.marker_select_label = QtWidgets.QLabel("Markers:")
        viz_layout.addWidget(self.marker_select_label)
        self.marker_select_btn = QtWidgets.QPushButton("Select Markers...")
        self.marker_select_btn.setToolTip("Select markers to visualize")
        self.marker_select_btn.clicked.connect(self._open_marker_selection_dialog)
        viz_layout.addWidget(self.marker_select_btn)
        self.selected_markers = []  # Store selected markers

        # Plot type selector (for boxplot/violin plot only)
        self.plot_type_label = QtWidgets.QLabel("Plot type:")
        viz_layout.addWidget(self.plot_type_label)
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems(["Violin Plot", "Boxplot"])
        self.plot_type_combo.setCurrentText("Violin Plot")
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        viz_layout.addWidget(self.plot_type_combo)

        # Statistical testing checkbox (for boxplot/violin plot only)
        self.stats_test_checkbox = QtWidgets.QCheckBox("Show statistical tests")
        self.stats_test_checkbox.setToolTip(
            "Run Kruskal-Wallis omnibus tests and BH-corrected pairwise Mann-Whitney U comparisons."
        )
        self.stats_test_checkbox.setChecked(False)
        self.stats_test_checkbox.stateChanged.connect(self._on_stats_test_changed)
        viz_layout.addWidget(self.stats_test_checkbox)

        # Statistical test mode selector
        self.stats_mode_label = QtWidgets.QLabel("Test mode:")
        viz_layout.addWidget(self.stats_mode_label)
        self.stats_mode_combo = QtWidgets.QComboBox()
        self.stats_mode_combo.addItems(["Pairwise (all pairs)", "One vs Others"])
        self.stats_mode_combo.currentTextChanged.connect(self._on_stats_mode_changed)
        viz_layout.addWidget(self.stats_mode_combo)

        # Cluster selector for one-vs-others mode
        self.stats_cluster_label = QtWidgets.QLabel("Reference cluster:")
        viz_layout.addWidget(self.stats_cluster_label)
        self.stats_cluster_combo = QtWidgets.QComboBox()
        self.stats_cluster_combo.currentTextChanged.connect(self._on_stats_cluster_changed)
        viz_layout.addWidget(self.stats_cluster_combo)

        # Export statistical results button
        self.stats_export_btn = QtWidgets.QPushButton("Export Stats")
        self.stats_export_btn.setToolTip("Export statistical test results (raw and adjusted p-values)")
        self.stats_export_btn.clicked.connect(self._export_statistical_results)
        self.stats_export_btn.setEnabled(False)
        viz_layout.addWidget(self.stats_export_btn)

        viz_layout.addStretch()

        # Save current plot
        self.save_plot_btn = QtWidgets.QPushButton("Save Plot")
        self.save_plot_btn.clicked.connect(self._save_current_plot)
        self.save_plot_btn.setEnabled(False)
        viz_layout.addWidget(self.save_plot_btn)

        # Close
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        viz_layout.addWidget(self.close_btn)

        plot_layout.addLayout(viz_layout)
        layout.addWidget(plot_group)

        # Step 3: Phenotype tools
        phenotype_group = QtWidgets.QGroupBox("Phenotype Annotation / Exploration")
        phenotype_layout = QtWidgets.QHBoxLayout(phenotype_group)
        self.annotate_btn = QtWidgets.QPushButton("Annotate Phenotypes")
        self.annotate_btn.clicked.connect(self._open_annotation_dialog)
        self.annotate_btn.setEnabled(False)
        phenotype_layout.addWidget(self.annotate_btn)
        
        # Merge clusters button
        self.merge_clusters_btn = QtWidgets.QPushButton("Merge Clusters")
        self.merge_clusters_btn.clicked.connect(self._open_merge_clusters_dialog)
        self.merge_clusters_btn.setEnabled(False)
        self.merge_clusters_btn.setToolTip("Merge two clusters into one")
        phenotype_layout.addWidget(self.merge_clusters_btn)
        

        self.explore_btn = QtWidgets.QPushButton("Explore Clusters")
        self.explore_btn.clicked.connect(self._explore_clusters)
        self.explore_btn.setEnabled(False)
        phenotype_layout.addWidget(self.explore_btn)

        # Manual gating entry point (Step 1/3 entry kept here for linear flow)
        self.gating_btn = QtWidgets.QPushButton("Manual Gating")
        self.gating_btn.clicked.connect(self._open_gating_dialog)
        phenotype_layout.addWidget(self.gating_btn)

        phenotype_layout.addStretch()
        layout.addWidget(phenotype_group)
        self._update_clustering_settings_summary()

    def _build_clustering_settings_dialog(self):
        """Create a popup dialog that hosts clustering settings controls."""
        self.clustering_settings_dialog = QtWidgets.QDialog(self)
        self.clustering_settings_dialog.setWindowTitle("Clustering Settings")
        self.clustering_settings_dialog.setModal(True)
        self.clustering_settings_dialog.setMinimumSize(380, 260)

        if self.parent() is not None:
            parent_size = self.parent().size()
            dialog_width = max(560, int(parent_size.width() * 0.55))
            dialog_height = max(420, int(parent_size.height() * 0.65))
            self.clustering_settings_dialog.resize(dialog_width, dialog_height)

        dialog_layout = QtWidgets.QVBoxLayout(self.clustering_settings_dialog)
        help_label = QtWidgets.QLabel(
            "Configure clustering parameters and feature source here. "
            "Then click 'Run Clustering' in the main window."
        )
        help_label.setWordWrap(True)
        dialog_layout.addWidget(help_label)

        self.clustering_settings_scroll_area = QtWidgets.QScrollArea()
        self.clustering_settings_scroll_area.setWidgetResizable(True)
        self.clustering_settings_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.clustering_settings_scroll_area.setWidget(self._clustering_options_group)
        dialog_layout.addWidget(self.clustering_settings_scroll_area, stretch=1)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addStretch()
        done_btn = QtWidgets.QPushButton("Done")
        done_btn.clicked.connect(self.clustering_settings_dialog.accept)
        buttons_layout.addWidget(done_btn)
        dialog_layout.addLayout(buttons_layout)

    def _open_clustering_settings_dialog(self):
        """Show the clustering settings popup dialog."""
        if not hasattr(self, 'clustering_settings_dialog') or self.clustering_settings_dialog is None:
            return
        self._on_clustering_type_changed()
        self._on_leiden_mode_changed()
        self.clustering_settings_dialog.exec_()
        self._update_clustering_settings_summary()

    def _update_clustering_settings_summary(self):
        """Update compact summary text shown next to the settings button."""
        if not hasattr(self, 'settings_summary_label'):
            return
        if not hasattr(self, 'clustering_type'):
            self.settings_summary_label.setText("")
            return

        method = self.clustering_type.currentText()
        feature_set = self.feature_set_combo.currentText() if hasattr(self, 'feature_set_combo') else "Original Features"
        scaling = self.clustering_scaling_combo.currentText() if hasattr(self, 'clustering_scaling_combo') else "None"
        use_pca, pca_mode, pca_variance, pca_n_components = self._get_pca_settings() if hasattr(self, '_get_pca_settings') else (False, "variance", 0.95, None)
        if use_pca and pca_mode == "variance":
            representation_text = f"PCA ({pca_variance * 100:.1f}% variance)"
        elif use_pca:
            representation_text = f"PCA ({pca_n_components} PCs)"
        else:
            representation_text = "Raw features"

        if method == "Hierarchical":
            method_detail = self.hierarchical_method.currentText() if hasattr(self, 'hierarchical_method') else "ward"
            method_text = f"{method} ({method_detail})"
        elif method == "Leiden":
            if hasattr(self, 'resolution_radio') and self.resolution_radio.isChecked():
                res = self.resolution_spinbox.value() if hasattr(self, 'resolution_spinbox') else 1.0
                method_text = f"Leiden (res={res:.1f})"
            else:
                method_text = "Leiden (modularity)"
        elif method == "Louvain":
            neighbors = self.n_neighbors_spinbox.value() if hasattr(self, 'n_neighbors_spinbox') else 15
            method_text = f"Louvain (n_neighbors={neighbors})"
        elif method == "HDBSCAN":
            min_cluster_size = self.min_cluster_size_spinbox.value() if hasattr(self, 'min_cluster_size_spinbox') else 10
            method_text = f"HDBSCAN (min_cluster_size={min_cluster_size})"
        elif method == "K-means":
            n_clusters = self.n_clusters.value() if hasattr(self, 'n_clusters') else 5
            method_text = f"K-means (k={n_clusters})"
        else:
            method_text = method

        summary = f"{feature_set} | {representation_text} | {method_text} | {scaling}"
        self.settings_summary_label.setText(summary)
        self.settings_summary_label.setToolTip(summary)

    def _hard_reset_canvas(self):
        """Clear figure and sync it to canvas size to avoid overdraw artifacts."""
        if not hasattr(self, 'figure') or not hasattr(self, 'canvas'):
            return
        if getattr(self.canvas, 'figure', None) is not self.figure:
            self.canvas.figure = self.figure
        try:
            sync_figure_to_canvas(self.figure, self.canvas)
        except Exception:
            # Size sync is best-effort; continue with clear-only if unavailable.
            pass
        self.figure.clear()

    def _flush_canvas(self, force_layout_refresh: bool = False):
        """Force a full repaint so stale pixels are not left on the canvas.

        Args:
            force_layout_refresh: When True, trigger a tiny programmatic resize
                nudge to emulate the manual resize that fixes stale cluster-map
                paints on some systems.
        """
        if not hasattr(self, 'canvas'):
            return
        nonblocking_refresh = should_use_nonblocking_canvas_refresh()
        refresh_canvas(self.canvas, draw=True)
        if force_layout_refresh:
            if nonblocking_refresh:
                try:
                    self.canvas.updateGeometry()
                    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 20)
                    if getattr(self.canvas, 'figure', None) is self.figure:
                        sync_figure_to_canvas(self.figure, self.canvas)
                    if hasattr(self.canvas, 'resize_event'):
                        try:
                            self.canvas.resize_event()
                        except Exception:
                            pass
                except Exception:
                    pass
                refresh_canvas(self.canvas, draw=True)
            else:
                try:
                    old_w = max(2, int(self.canvas.width()))
                    old_h = max(2, int(self.canvas.height()))
                    # Nudge the canvas by 1px and back to force a full Qt repaint cycle.
                    self.canvas.resize(old_w + 1, old_h + 1)
                    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 20)
                    self.canvas.resize(old_w, old_h)
                    if hasattr(self.canvas, 'resize_event'):
                        try:
                            self.canvas.resize_event()
                        except Exception:
                            pass
                except Exception:
                    # Best-effort fallback only.
                    pass
                refresh_canvas(self.canvas, draw=True)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 20)

    def _fit_standard_figure_to_canvas(self, rect=None, pad: float = 0.8, allow_text_compaction: bool = True):
        """Fit the current figure to the live canvas for non-clustermap views."""
        fit_figure_to_canvas(
            self.figure,
            self.canvas,
            rect=rect,
            pad=pad,
            max_passes=3,
            allow_text_compaction=allow_text_compaction,
        )

    def _finalize_standard_plot(
        self,
        *,
        rect=None,
        pad: float = 0.8,
        allow_text_compaction: bool = True,
        force_layout_refresh: bool = False,
    ):
        """Apply canvas-fit layout to the current figure and redraw."""
        self._fit_standard_figure_to_canvas(rect=rect, pad=pad, allow_text_compaction=allow_text_compaction)
        self._flush_canvas(force_layout_refresh=force_layout_refresh)

    def _measure_figure_text_overflow(self):
        """Measure how far visible text extends outside the figure canvas.

        Returns:
            Dict with left/right/top/bottom overflow fractions in figure coordinates.
        """
        overflow = {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}
        if not hasattr(self, 'figure') or getattr(self.figure, 'canvas', None) is None:
            return overflow

        try:
            self.figure.canvas.draw()
            renderer = self.figure.canvas.get_renderer()
            fig_bbox = self.figure.bbox
            fig_width = max(1.0, float(fig_bbox.width))
            fig_height = max(1.0, float(fig_bbox.height))

            for text_artist in self.figure.findobj(match=Text):
                if not text_artist.get_visible():
                    continue
                text_value = text_artist.get_text()
                if not isinstance(text_value, str) or not text_value.strip():
                    continue
                try:
                    bbox = text_artist.get_window_extent(renderer=renderer)
                except Exception:
                    continue
                if bbox.width <= 0 or bbox.height <= 0:
                    continue
                if bbox.x0 < fig_bbox.x0:
                    overflow['left'] = max(overflow['left'], (fig_bbox.x0 - bbox.x0) / fig_width)
                if bbox.x1 > fig_bbox.x1:
                    overflow['right'] = max(overflow['right'], (bbox.x1 - fig_bbox.x1) / fig_width)
                if bbox.y0 < fig_bbox.y0:
                    overflow['bottom'] = max(overflow['bottom'], (fig_bbox.y0 - bbox.y0) / fig_height)
                if bbox.y1 > fig_bbox.y1:
                    overflow['top'] = max(overflow['top'], (bbox.y1 - fig_bbox.y1) / fig_height)
        except Exception:
            pass

        return overflow

    def _view_supports_resize_refresh(self) -> bool:
        """Return True when the current visualization can be safely redrawn after resize."""
        view = getattr(self, '_active_view_name', None)
        if not view and hasattr(self, 'view_combo'):
            view = self.view_combo.currentText()
        if not view:
            return False
        if view in {'Heatmap', 'Cluster Map', 'Differential Expression', 'Boxplot/Violin Plot', 'Stacked Bars'}:
            return self.clustered_data is not None and not self.clustered_data.empty
        if view == 'UMAP':
            return getattr(self, 'umap_embedding', None) is not None
        if view == 't-SNE':
            return getattr(self, 'tsne_embedding', None) is not None
        return False

    def _queue_view_resize_refresh(self):
        """Debounce plot redraws so the current view reflows after resizing."""
        if self._view_resize_in_progress:
            return
        if not self.isVisible():
            return
        if not self._view_supports_resize_refresh():
            return
        self._view_resize_timer.start(140)

    def _refresh_current_view_after_resize(self):
        """Redraw the active visualization using the new canvas size after a resize."""
        if self._view_resize_in_progress:
            return
        if not self._view_supports_resize_refresh():
            return
        view = getattr(self, '_active_view_name', None)
        if not view and hasattr(self, 'view_combo'):
            view = self.view_combo.currentText()
        if not view:
            return
        try:
            self._view_resize_in_progress = True
            if view == 'Heatmap':
                self._create_heatmap()
            elif view == 'Cluster Map':
                self._create_cluster_map()
            elif view == 'UMAP':
                self._create_umap_plot()
            elif view == 't-SNE':
                self._create_tsne_plot()
            elif view == 'Stacked Bars':
                self._show_stacked_bars()
            elif view == 'Differential Expression':
                self._show_differential_expression()
            elif view == 'Boxplot/Violin Plot':
                self._show_boxplot_violin()
        finally:
            self._view_resize_in_progress = False

    def resizeEvent(self, event):
        """Reflow the current plot when the dialog size changes."""
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
        self._queue_view_resize_refresh()

    def _setup_plot(self):
        """Setup the matplotlib plot."""
        self._hard_reset_canvas()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Click 'Run Clustering' to generate heatmap", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        self._flush_canvas()
        self._update_viz_controls_visibility()
        self._update_cluster_action_buttons()

    def _has_annotatable_clusters(self) -> bool:
        """Return True when clustered_data contains at least one valid (non-zero) cluster."""
        if self.clustered_data is None or self.clustered_data.empty:
            return False
        if 'cluster' not in self.clustered_data.columns:
            return False
        try:
            cluster_vals = pd.to_numeric(self.clustered_data['cluster'], errors='coerce')
            return bool((cluster_vals.notna() & (cluster_vals != 0)).any())
        except Exception:
            # If cluster values are not cleanly numeric but data exists, allow tools.
            return True

    def _update_cluster_action_buttons(self):
        """Enable/disable controls that depend on clustered cells."""
        has_clusters = self._has_annotatable_clusters()
        masks_available = self._has_segmentation_masks_for_cluster_explorer()
        if hasattr(self, 'annotate_btn'):
            self.annotate_btn.setEnabled(has_clusters)
        if hasattr(self, 'merge_clusters_btn'):
            self.merge_clusters_btn.setEnabled(has_clusters)
        if hasattr(self, 'explore_btn'):
            self.explore_btn.setEnabled(has_clusters and masks_available)
            if has_clusters and not masks_available:
                self.explore_btn.setToolTip(
                    "Load segmentation masks to enable Cluster Explorer previews."
                )
            else:
                self.explore_btn.setToolTip("")

    def _has_segmentation_masks_for_cluster_explorer(self) -> bool:
        """Return whether segmentation masks are available for Cluster Explorer."""
        parent_window = self.parent()
        if parent_window is None:
            return False
        segmentation_masks = getattr(parent_window, "segmentation_masks", None)
        if segmentation_masks is None:
            return False
        try:
            return len(segmentation_masks) > 0
        except Exception:
            try:
                return bool(list(segmentation_masks.keys()))
            except Exception:
                return False
        
    def _on_clustering_type_changed(self):
        """Handle clustering type change to show/hide relevant controls."""
        clustering_type = self.clustering_type.currentText()
        is_leiden = clustering_type == "Leiden"
        is_louvain = clustering_type == "Louvain"
        is_hierarchical = clustering_type == "Hierarchical"
        is_hdbscan = clustering_type == "HDBSCAN"
        is_kmeans = clustering_type == "K-means"
        
        # Show/hide Leiden options group (also used for Louvain)
        self.leiden_options_group.setVisible(is_leiden or is_louvain)
        
        # Show/hide HDBSCAN options group
        self.hdbscan_options_group.setVisible(is_hdbscan)
        
        # Show/hide hierarchical method selection
        self.hierarchical_label.setVisible(is_hierarchical)
        self.hierarchical_method.setVisible(is_hierarchical)
        
        # Show/hide number of clusters for hierarchical and k-means
        if hasattr(self, 'n_clusters_label'):
            self.n_clusters_label.setVisible(is_hierarchical or is_kmeans)
        if hasattr(self, 'n_clusters'):
            self.n_clusters.setVisible(is_hierarchical or is_kmeans)
        
        # Show/hide k-range search button for hierarchical and k-means
        if hasattr(self, 'k_range_btn'):
            self.k_range_btn.setVisible(is_hierarchical or is_kmeans)
        
        # Show/hide dendrogram controls for hierarchical methods
        self.dendro_label.setVisible(is_hierarchical)
        self.dendro_mode.setVisible(is_hierarchical)
        self._update_viz_controls_visibility()
        self._update_clustering_settings_summary()
    
    def _on_feature_set_changed(self):
        """Handle feature set selection change."""
        if not hasattr(self, 'feature_set_combo'):
            return

        self._update_active_feature_dataframe()

        # Clear existing clustering results when switching feature sets
        self.cluster_labels = None
        self.clustered_data = None
        self.clustered_data_unscaled = None
        self.umap_embedding = None
        self.tsne_embedding = None
        # Clear stored clustering method and dendrogram mode so new clustering uses current UI settings
        self.actual_clustering_method = None
        self.actual_dendrogram_mode = None
        
        # Clear plots if they exist
        if hasattr(self, 'figure') and hasattr(self, 'canvas'):
            self._hard_reset_canvas()
            # Add placeholder text
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Feature set changed. Click 'Run Clustering' to generate heatmap", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self._flush_canvas()
        self._update_cluster_action_buttons()
        self._update_clustering_settings_summary()

        parent = self.parent()
        if parent is not None and hasattr(parent, '_set_analysis_feature_set_preference'):
            parent._set_analysis_feature_set_preference(self.get_active_feature_set_key(), source_dialog=self)
    
    def _on_leiden_mode_changed(self):
        """Handle Leiden clustering mode change (resolution vs modularity)."""
        use_resolution = self.resolution_radio.isChecked()
        self.resolution_label.setVisible(use_resolution)
        self.resolution_spinbox.setVisible(use_resolution)
        self._update_clustering_settings_summary()

    def _get_pca_settings(self):
        """Return PCA clustering settings from the UI."""
        if not hasattr(self, 'use_pca_checkbox'):
            return False, "variance", 0.95, None
        use_pca = self.use_pca_checkbox.isChecked() and self.use_pca_checkbox.isEnabled()
        mode = self.pca_mode_combo.currentData() if hasattr(self, 'pca_mode_combo') else "variance"
        if mode not in {"variance", "components"}:
            mode = "variance"
        variance = self.pca_variance_spinbox.value() / 100.0 if hasattr(self, 'pca_variance_spinbox') else 0.95
        n_components = self.pca_n_components_spinbox.value() if hasattr(self, 'pca_n_components_spinbox') else None
        return use_pca, mode, variance, n_components

    def _on_pca_controls_changed(self, *_args):
        """Enable the relevant PCA controls and refresh the compact summary."""
        if not hasattr(self, 'use_pca_checkbox'):
            return
        use_pca, mode, _variance, _n_components = self._get_pca_settings()
        if hasattr(self, 'pca_mode_combo'):
            self.pca_mode_combo.setEnabled(use_pca)
        for widget_name in ('pca_variance_label', 'pca_variance_spinbox'):
            if hasattr(self, widget_name):
                getattr(self, widget_name).setEnabled(use_pca and mode == "variance")
        for widget_name in ('pca_n_components_label', 'pca_n_components_spinbox'):
            if hasattr(self, widget_name):
                getattr(self, widget_name).setEnabled(use_pca and mode == "components")
        self._update_clustering_settings_summary()
        
    def _run_clustering(self):
        """Run the clustering analysis."""
        original_run_btn_text = self.run_btn.text() if hasattr(self, 'run_btn') else "Run Clustering"
        if hasattr(self, 'run_btn'):
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Running...")
        try:
            # Preserve selected display features and view state before resetting
            saved_view = None
            saved_selected_features = None
            saved_heatmap_scaling = None
            if hasattr(self, 'view_combo'):
                saved_view = self.view_combo.currentText()
            if hasattr(self, 'selected_display_features') and self.selected_display_features is not None:
                saved_selected_features = self.selected_display_features.copy() if hasattr(self.selected_display_features, 'copy') else set(self.selected_display_features)
            if hasattr(self, 'heatmap_scaling_combo'):
                saved_heatmap_scaling = self.heatmap_scaling_combo.currentText()
            
            # Reset cluster merges when re-clustering (restore original assignments if they exist)
            if self.original_cluster_assignments is not None:
                self._reset_cluster_merges()
            
            # Reset custom cluster names when re-clustering
            self.cluster_annotation_map = {}
            self.cluster_backend_names = {}
            # Clear LLM phenotype cache when re-clustering
            self.llm_phenotype_cache = {}
            # Reset cluster-specific view filters because cluster IDs may change after re-clustering
            self.stacked_bars_filter_selection = None
            self.de_cluster_filter_selection = None
            
            # Clear any existing cluster phenotype data
            if hasattr(self, 'clustered_data') and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
                self.clustered_data = self.clustered_data.drop('cluster_phenotype', axis=1)
            if 'cluster_phenotype' in self.feature_dataframe.columns:
                self.feature_dataframe = self.feature_dataframe.drop('cluster_phenotype', axis=1)
            
            # Get options
            # Defaults for backward compatibility (now controlled by selector)
            agg_method = "mean"
            include_morpho = True
            n_clusters = self.n_clusters.value()
            clustering_type = self.clustering_type.currentText()
            
            # Determine the actual clustering method
            if clustering_type == "Leiden":
                cluster_method = "leiden"
            elif clustering_type == "Louvain":
                cluster_method = "louvain"
            elif clustering_type == "K-means":
                cluster_method = "kmeans"
            elif clustering_type == "HDBSCAN":
                cluster_method = "hdbscan"
            else:
                cluster_method = "hierarchical"
            
            # Prepare data
            # Allow user to select features interactively
            available_cols = self._list_available_feature_columns(include_morpho)
            from openimc.ui.dialogs.feature_selector_dialog import FeatureSelectorDialog
            selector = FeatureSelectorDialog(available_cols, self)
            # Pre-populate filter settings if available
            if self.filter_settings is not None:
                selector.set_filter_settings(self.filter_settings)
            # Pre-populate selected features if available (from saved state)
            if hasattr(self, 'last_features_used') and self.last_features_used is not None:
                # Only set features that are actually available in the current dataset
                available_features = [f for f in self.last_features_used if f in available_cols]
                if available_features:
                    selector.set_selected_features(available_features)
            if selector.exec_() != QtWidgets.QDialog.Accepted:
                return
            selected_columns = selector.get_selected_columns()
            
            # Get filter settings
            filter_settings = selector.get_filter_settings()
            self.filter_settings = filter_settings  # Store for use in UMAP/spatial analyses

            # Apply filters to feature dataframe before clustering
            filtered_df = self._apply_filters(self.feature_dataframe.copy(), filter_settings)
            if filtered_df.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No cells remain after applying filters.")
                return
            
            # Get scaling method
            scaling_text = self.clustering_scaling_combo.currentText()
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            scaling_method = scaling_map.get(scaling_text, "zscore")
            use_pca, pca_mode, pca_variance, pca_n_components = self._get_pca_settings()
            
            result = self._prepare_clustering_data(agg_method, include_morpho, selected_columns, scaling_method, filtered_df, filter_settings)
            
            if result is None:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for clustering.")
                return
            
            data, data_unscaled = result
            
            if data is None or data.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for clustering.")
                return
            
            # Clear canvas before clustering
            self.figure.clear()
            self.canvas.draw()

            # Perform clustering in background to keep UI responsive.
            # Freeze the exact post-validation feature set. Keeping a distinct
            # snapshot prevents later UI refreshes from changing audit/logging
            # metadata for the clustering operation that just ran.
            features_used_for_clustering = list(data.columns)
            seed = self.seed_spinbox.value()
            full_data = self.feature_dataframe.loc[data.index].copy()

            cluster_kwargs = {
                "features_df": full_data,
                "method": cluster_method,
                "columns": list(features_used_for_clustering),
                "scaling": scaling_method,
                "output_path": None,
                "seed": seed,
                "n_clusters": n_clusters,
                "use_pca": use_pca,
                "pca_mode": pca_mode,
                "pca_variance": pca_variance,
                "pca_n_components": pca_n_components,
            }

            if cluster_method == "leiden":
                cluster_kwargs["resolution"] = self.resolution_spinbox.value() if self.resolution_radio.isChecked() else 1.0
                cluster_kwargs["n_neighbors"] = self.n_neighbors_spinbox.value()
                cluster_kwargs["metric"] = self.leiden_metric_combo.currentText()
                cluster_kwargs["use_jaccard"] = self.jaccard_checkbox.isChecked()
            elif cluster_method == "louvain":
                cluster_kwargs["n_neighbors"] = self.n_neighbors_spinbox.value()
                cluster_kwargs["metric"] = self.leiden_metric_combo.currentText()
                cluster_kwargs["use_jaccard"] = self.jaccard_checkbox.isChecked()
            elif cluster_method == "hdbscan":
                cluster_kwargs["min_cluster_size"] = self.min_cluster_size_spinbox.value()
                cluster_kwargs["min_samples"] = self.min_samples_spinbox.value()
                cluster_kwargs["cluster_selection_method"] = self.cluster_selection_combo.currentText()
                cluster_kwargs["hdbscan_metric"] = self.metric_combo.currentText()
            elif cluster_method == "hierarchical":
                cluster_kwargs["linkage"] = self.hierarchical_method.currentText()

            clustered_df = self._run_core_cluster_with_progress(cluster_kwargs, cluster_method)
            pca_metadata = getattr(clustered_df, 'attrs', {}).get('pca_metadata', {})

            # Extract cluster labels and build clustered matrix for downstream visualizations.
            self.cluster_labels = clustered_df['cluster'].astype(int).values
            self.clustered_data = data.copy()
            self.clustered_data['cluster'] = self.cluster_labels.astype(int)
            self.clustered_data = self.clustered_data.sort_values('cluster')
            
            # Ensure cluster column is integer type to avoid boolean subtraction issues
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                self.clustered_data['cluster'] = self.clustered_data['cluster'].astype(int)
            
            # Store original cluster assignments before any merging
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                self.original_cluster_assignments = self.clustered_data['cluster'].copy()
            
            # Store features used for clustering so they can be auto-selected for display
            self.last_features_used = list(features_used_for_clustering)
            
            # Auto-select these features for heatmap display
            self.selected_display_features = list(features_used_for_clustering)
            
            # Store unscaled data with same structure as clustered_data
            if self.clustered_data is not None:
                # Align unscaled data with clustered_data indices and add cluster column
                self.clustered_data_unscaled = data_unscaled.loc[self.clustered_data.index].copy()
                if 'cluster' in self.clustered_data.columns:
                    self.clustered_data_unscaled['cluster'] = self.clustered_data['cluster'].astype(int).values
                # Copy any other non-feature columns from clustered_data
                for col in self.clustered_data.columns:
                    if col not in self.clustered_data_unscaled.columns and col != 'cluster':
                        if col in ['acquisition_id', 'manual_phenotype']:
                            self.clustered_data_unscaled[col] = self.clustered_data[col].values
            
            # Automatically add cluster column to main feature dataframe
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                # Ensure cluster column exists in main dataframe
                if 'cluster' not in self.feature_dataframe.columns:
                    self.feature_dataframe['cluster'] = 0  # Initialize with default value
                
                # Update cluster assignments for the clustered cells (ensure integer type)
                self.feature_dataframe.loc[self.clustered_data.index, 'cluster'] = self.clustered_data['cluster'].astype(int).values
            else:
                pass
            
            # Log clustering operation
            logger = get_logger()
            n_clusters_found = len(np.unique(self.cluster_labels)) if self.cluster_labels is not None else n_clusters
            
            # Get scaling method for logging
            scaling_text = self.clustering_scaling_combo.currentText()
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            scaling_method = scaling_map.get(scaling_text, "zscore")
            
            params = {
                "method": cluster_method,
                "n_clusters": n_clusters,
                "n_clusters_found": int(n_clusters_found),
                "aggregation_method": agg_method,
                "include_morphological": include_morpho,
                "scaling_method": scaling_method,
                "distance_metric": "euclidean",
                "feature_representation": pca_metadata.get("feature_representation", "principal_components" if use_pca else "raw_features"),
                "pca_selection_mode": pca_metadata.get("pca_selection_mode"),
                "pca_requested_variance": pca_metadata.get("pca_requested_variance"),
                "pca_requested_n_components": pca_metadata.get("pca_requested_n_components"),
                "pca_n_components_retained": pca_metadata.get("pca_n_components_retained"),
                "pca_variance_retained": pca_metadata.get("pca_variance_retained"),
                "pca_input_feature_count": pca_metadata.get("pca_input_feature_count", len(features_used_for_clustering)),
                "n_cells": int(len(self.clustered_data)) if self.clustered_data is not None else 0,
                "source_files": self._get_source_files_for_logging(),
            }
            
            if cluster_method == "leiden":
                if self.resolution_radio.isChecked():
                    params["resolution_parameter"] = self.resolution_spinbox.value()
                else:
                    params["optimization_method"] = "modularity"
                params["seed"] = self.seed_spinbox.value()
                params["n_neighbors"] = self.n_neighbors_spinbox.value()
                params["distance_metric"] = self.leiden_metric_combo.currentText()
            elif cluster_method == "louvain":
                params["seed"] = self.seed_spinbox.value()
                params["n_neighbors"] = self.n_neighbors_spinbox.value()
                params["distance_metric"] = self.leiden_metric_combo.currentText()
            elif cluster_method == "hdbscan":
                params["min_cluster_size"] = self.min_cluster_size_spinbox.value()
                params["min_samples"] = self.min_samples_spinbox.value()
                params["cluster_selection_method"] = self.cluster_selection_combo.currentText()
                params["metric"] = self.metric_combo.currentText()
                params["distance_metric"] = self.metric_combo.currentText()
                params["seed"] = self.seed_spinbox.value()
            elif cluster_method == "kmeans":
                params["seed"] = self.seed_spinbox.value()
                params["n_init"] = 10
            else:
                params["linkage_method"] = self.hierarchical_method.currentText()
                # Hierarchical clustering is deterministic, but we log seed for consistency
                params["seed"] = self.seed_spinbox.value()

            self.last_clustering_method = cluster_method
            self.last_clustering_params = params.copy()
            
            # Get acquisition IDs from clustered data
            acquisitions = self._get_logging_acquisitions()
            source_file = self._get_source_file_summary_for_logging()
            
            logger.log_clustering(
                method=cluster_method,
                parameters=params,
                features_used=list(features_used_for_clustering),
                n_clusters=int(n_clusters_found),
                acquisitions=acquisitions,
                notes=f"Clustered {len(self.clustered_data) if self.clustered_data is not None else 0} cells into {n_clusters_found} clusters",
                source_file=source_file
            )
            
            # Store the scaling method used for clustering.
            clustering_scaling_text = self.clustering_scaling_combo.currentText()
            self.clustering_scaling_method = clustering_scaling_text
            
            # Restore the view and display scaling after clustering. The
            # feature selection deliberately remains the exact feature set
            # used for the new clustering result.
            if saved_view and hasattr(self, 'view_combo'):
                # Only restore if we were viewing heatmap
                if saved_view == 'Heatmap' and saved_selected_features:
                    # Restore heatmap scaling if it was set
                    if saved_heatmap_scaling and hasattr(self, 'heatmap_scaling_combo'):
                        self.heatmap_scaling_combo.blockSignals(True)
                        self.heatmap_scaling_combo.setCurrentText(saved_heatmap_scaling)
                        self.heatmap_scaling_combo.blockSignals(False)
                    # Restore view to heatmap
                    self.view_combo.blockSignals(True)
                    self.view_combo.setCurrentText('Heatmap')
                    self.view_combo.blockSignals(False)
                    # Redraw heatmap with preserved features
                    if hasattr(self, '_show_heatmap'):
                        try:
                            self._show_heatmap()
                        except Exception as e:
                            print(f"Warning: Could not redraw heatmap after clustering: {e}")
            
            # Store the clustering method and dendrogram mode actually used
            self.actual_clustering_method = self.clustering_type.currentText()
            if hasattr(self, 'dendro_mode') and self.dendro_mode.isVisible():
                self.actual_dendrogram_mode = self.dendro_mode.currentText()
            else:
                self.actual_dendrogram_mode = None
            
            # Only create heatmap if we don't have a preserved view/features
            # (The preservation logic above will handle redrawing if needed)
            if not (saved_view == 'Heatmap' and saved_selected_features):
                # Default to heatmap view after clustering
                try:
                    if self.clustered_data is not None:
                        if 'cluster' in self.clustered_data.columns:
                            pass
                    self._create_heatmap()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    QtWidgets.QMessageBox.critical(self, "Plot Generation Error", 
                        f"Error generating heatmap after clustering:\n{str(e)}\n\nSee console for details.")
                    # Still try to show something
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, f"Clustering completed but plot generation failed.\nError: {str(e)}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    self.canvas.draw()
            
            # Force canvas refresh
            try:
                self.canvas.draw()
            except Exception as e:
                import traceback
                traceback.print_exc()
            
            # Enable buttons
            self.annotate_btn.setEnabled(True)
            self.merge_clusters_btn.setEnabled(True)
            self.save_plot_btn.setEnabled(True)
            self.save_output_btn.setEnabled(True)
            self._update_cluster_action_buttons()
            
            # Update statistical cluster combo if it exists
            if hasattr(self, 'stats_cluster_combo'):
                self._update_stats_cluster_combo()
            # If UMAP was previously run, keep that available
            # Otherwise, selecting UMAP will prompt to run

            # Auto-apply annotations if already loaded for these cluster ids
            if self.cluster_annotation_map:
                self._apply_cluster_annotations()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Clustering Error", f"Error during clustering: {str(e)}")
        finally:
            if hasattr(self, 'run_btn'):
                self.run_btn.setEnabled(True)
                self.run_btn.setText(original_run_btn_text)

    def _run_core_cluster_with_progress(self, cluster_kwargs, cluster_method):
        """Run core clustering in a worker thread while keeping the UI responsive."""
        phase_map = {
            "leiden": "Building k-NN graph and optimizing Leiden partitions",
            "louvain": "Building k-NN graph and optimizing Louvain partitions",
            "hdbscan": "Running HDBSCAN density clustering",
            "hierarchical": "Running hierarchical clustering",
            "ward": "Running hierarchical clustering",
            "complete": "Running hierarchical clustering",
            "average": "Running hierarchical clustering",
            "single": "Running hierarchical clustering",
            "kmeans": "Running k-means clustering",
        }
        phase_text = phase_map.get(cluster_method, "Running clustering")

        progress = QtWidgets.QProgressDialog("", None, 0, 0, self)
        progress.setWindowTitle("Clustering In Progress")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        QtWidgets.QApplication.processEvents()

        timer = QtCore.QElapsedTimer()
        timer.start()

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(cluster, **cluster_kwargs)

                while not future.done():
                    elapsed_s = max(0, int(timer.elapsed() / 1000))
                    dots = "." * ((elapsed_s % 3) + 1)
                    seed_value = cluster_kwargs.get('seed', 'N/A')
                    progress.setLabelText(
                        f"{phase_text}{dots}\n"
                        f"Cells: {len(cluster_kwargs['features_df']):,} | "
                        f"Features: {len(cluster_kwargs['columns'])} | "
                        f"Seed: {seed_value}\n"
                        f"Elapsed: {elapsed_s}s"
                    )
                    QtWidgets.QApplication.processEvents()
                    QtCore.QThread.msleep(100)

                return future.result()
        finally:
            progress.close()
    
    def _list_available_feature_columns(self, include_morpho):
        marker_cols = [col for col in self.feature_dataframe.columns 
                      if any(col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        morpho_cols = []
        if include_morpho:
            morpho_cols = [col for col in self.feature_dataframe.columns 
                          if col in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity', 
                                   'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um', 
                                   'aspect_ratio', 'bbox_area_um2', 'touches_border', 'touches_edge', 'holes_count']]
        # Note: centroid_x and centroid_y are excluded from clustering as they are spatial coordinates
        return sorted(set(marker_cols + morpho_cols))

    def _apply_filters(self, df, filter_settings):
        """Apply cell filtering based on filter settings.
        
        Args:
            df: DataFrame to filter
            filter_settings: Dictionary with filter settings from FeatureSelectorDialog
        
        Returns:
            Filtered DataFrame
        """
        if filter_settings is None:
            return df
        
        initial_count = len(df)
        filtered_df = df.copy()
        
        # Exclude cells touching edge
        if filter_settings.get('exclude_edge_cells', False):
            before_edge_filter = len(filtered_df)
            if 'touches_edge' in filtered_df.columns:
                # Count cells touching edge before filtering
                edge_cells_count = filtered_df['touches_edge'].astype(bool).sum()
                # Use .eq(False) instead of ~ to avoid numpy boolean subtraction issues
                filtered_df = filtered_df[filtered_df['touches_edge'].astype(bool).eq(False)]
                after_edge_filter = len(filtered_df)
            elif 'touches_border' in filtered_df.columns:
                # Count cells touching border before filtering
                border_cells_count = filtered_df['touches_border'].astype(bool).sum()
                # Fallback to touches_border if touches_edge not available
                filtered_df = filtered_df[filtered_df['touches_border'].astype(bool).eq(False)]
                after_edge_filter = len(filtered_df)
        
        # Filter by area
        if 'area_um2' in filtered_df.columns:
            before_area_filter = len(filtered_df)
            min_area = filter_settings.get('min_area')
            max_area = filter_settings.get('max_area')
            
            if min_area is not None:
                filtered_df = filtered_df[filtered_df['area_um2'] >= min_area]
            if max_area is not None:
                before_max = len(filtered_df)
                filtered_df = filtered_df[filtered_df['area_um2'] <= max_area]
        
        final_count = len(filtered_df)
        if initial_count != final_count:
            pass
        else:
            pass
        
        return filtered_df
    
    def _apply_percentile_censoring(self, data, filter_settings):
        """Apply percentile censoring to data to remove outliers.
        
        Per-channel censoring: For each channel/feature, compute the 99th percentile across all cells,
        then set any value above that threshold to the 99th percentile.
        
        Args:
            data: pandas DataFrame with feature data
            filter_settings: Dictionary with filter settings including percentile censoring options
        
        Returns:
            Censored pandas DataFrame
        """
        if filter_settings is None:
            return data
        
        if not filter_settings.get('enable_percentile_censoring', False):
            return data
        
        data_censored = data.copy()
        censor_both_ends = filter_settings.get('censor_both_ends', False)
        
        censored_cols = []
        total_values_censored = 0
        total_values = 0
        
        # Apply censoring column by column (per-channel censoring)
        for col in data_censored.columns:
            col_data = data_censored[col].values
            
            # Skip non-numeric columns (including boolean)
            if col_data.dtype == bool:
                continue
            try:
                # Try to convert to float to check if numeric
                test_data = col_data.astype(np.float64)
            except (ValueError, TypeError):
                # Not numeric, skip this column
                continue
            
            # Convert to float64 to avoid dtype issues
            col_data = col_data.astype(np.float64)
            
            # Skip if all values are NaN or infinite
            finite_mask = np.isfinite(col_data)
            if not np.any(finite_mask):
                continue
            
            finite_data = col_data[finite_mask]
            total_values += len(finite_data)
            
            try:
                if censor_both_ends:
                    # Censor at both 1st and 99th percentiles
                    p1 = np.percentile(finite_data, 1)
                    p99 = np.percentile(finite_data, 99)
                    original_max = np.max(finite_data)
                    original_min = np.min(finite_data)
                    
                    # Count values that will be censored
                    values_above_p99 = np.sum(col_data > p99)
                    values_below_p1 = np.sum(col_data < p1)
                    values_censored = values_above_p99 + values_below_p1
                    
                    data_censored[col] = np.clip(col_data, p1, p99).astype(np.float64)
                    
                    if values_censored > 0:
                        censored_cols.append(col)
                        total_values_censored += values_censored
                else:
                    # Censor at 99th percentile only (cap values at 99th percentile)
                    # This is the standard IMC approach: "censored at the 99th percentile"
                    p99 = np.percentile(finite_data, 99)
                    original_max = np.max(finite_data)
                    
                    # Count values that will be censored (values above p99)
                    values_above_p99 = np.sum(col_data > p99)
                    
                    if values_above_p99 > 0:
                        censored_cols.append(col)
                        total_values_censored += values_above_p99
                    
                    # Use np.minimum to clip from above (equivalent to clip with None lower bound)
                    data_censored[col] = np.minimum(col_data, p99).astype(np.float64)
            except (ValueError, TypeError) as e:
                # If percentile calculation fails, skip this column
                print(f"[CENSORING WARNING] Failed to apply percentile censoring to column {col}: {e}")
                continue
        
        if censored_cols:
            pass
        else:
            pass
        
        return data_censored
    
    def _prepare_clustering_data(self, agg_method, include_morpho, selected_columns, scaling_method="zscore", filtered_df=None, filter_settings=None):
        """Prepare data for clustering.
        
        Args:
            agg_method: Aggregation method (not used currently)
            include_morpho: Whether to include morphological features (not used currently)
            selected_columns: List of column names to use for clustering
            scaling_method: Scaling method - 'zscore' or 'mad'
            filtered_df: Optional pre-filtered dataframe (if None, uses self.feature_dataframe)
        """
        # Use filtered dataframe if provided, otherwise use original
        working_df = filtered_df if filtered_df is not None else self.feature_dataframe
        
        feature_cols = list(selected_columns or [])
        
        if not feature_cols:
            return None
        
        # Check if all selected columns exist in the dataframe
        missing_cols = [col for col in feature_cols if col not in working_df.columns]
        if missing_cols:
            return None
        
        # Extract data
        data = working_df[feature_cols].copy()
        # Handle missing/infinite values safely
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median(numeric_only=True))
        
        # Apply percentile censoring if enabled (before scaling)
        if filter_settings is not None:
            data = self._apply_percentile_censoring(data, filter_settings)
        
        # Ensure all columns are numeric (float64) to avoid boolean subtraction issues
        for col in data.columns:
            if data[col].dtype == bool:
                # Convert boolean to int then float
                data[col] = data[col].astype(int).astype(np.float64)
            elif not np.issubdtype(data[col].dtype, np.number):
                # Convert non-numeric to float64 if possible
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)
                except (ValueError, TypeError):
                    # If conversion fails, drop the column
                    data = data.drop(columns=[col])
        
        # Store unscaled data for heatmap display (before scaling)
        data_unscaled = data.copy()
        
        # Apply selected scaling method (Z-score or MAD)
        data = self._apply_scaling(data, scaling_method)
        
        # Drop any residual non-finite rows/cols
        data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').dropna(axis=1, how='any')
        
        # Guard: require at least 2 rows and 2 columns to compute distances
        if data.shape[0] < 2 or data.shape[1] < 2:
            return None
        
        # Store unscaled data, aligning indices and columns with scaled data after dropna
        # Only keep rows and columns that remain after dropna
        data_unscaled = data_unscaled.loc[data.index, data.columns]
        
        return data, data_unscaled
    
    def _apply_scaling(self, data, scaling_method):
        """Apply scaling to data for UMAP.
        
        Args:
            data: pandas DataFrame to scale
            scaling_method: str, one of 'none', 'zscore', 'mad'
        
        Returns:
            Scaled pandas DataFrame
        """
        if scaling_method == 'none':
            return data.copy()
        
        data_scaled = data.astype(float).copy()
        
        if scaling_method == 'zscore':
            # Z-score normalization: (x - mean) / std
            data_means = data_scaled.mean()
            data_stds = data_scaled.std(ddof=0)
            
            # Handle columns with zero variance or NaN std/mean
            zero_var_cols = (data_stds == 0) | data_stds.isna() | data_means.isna()
            if zero_var_cols.any():
                # Set zero variance/NaN columns to 0 (centered but not scaled)
                data_scaled.loc[:, zero_var_cols] = 0.0
                non_zero_var_cols = ~zero_var_cols
                if non_zero_var_cols.any():
                    normalized_data = (data_scaled.loc[:, non_zero_var_cols] - data_means[non_zero_var_cols]) / data_stds[non_zero_var_cols]
                    data_scaled.loc[:, non_zero_var_cols] = normalized_data
            else:
                # Normalize all columns
                data_scaled = (data_scaled - data_means) / data_stds
        
        elif scaling_method == 'mad':
            # MAD (Median Absolute Deviation) scaling: (x - median) / MAD
            # MAD = median(|x - median(x)|)
            data_medians = data_scaled.median()
            
            # Calculate MAD for each column
            mad_values = {}
            for col in data_scaled.columns:
                col_data = data_scaled[col].values
                median_val = data_medians[col]
                # Handle NaN median
                if pd.isna(median_val):
                    mad_values[col] = 0.0
                else:
                    mad = np.median(np.abs(col_data - median_val))
                    # Handle NaN MAD
                    if pd.isna(mad):
                        mad_values[col] = 0.0
                    else:
                        mad_values[col] = mad
            
            # Convert to Series for vectorized operations
            mad_series = pd.Series(mad_values)
            
            # Handle columns with zero MAD or NaN (all values are the same or invalid)
            zero_mad_cols = (mad_series == 0) | mad_series.isna() | data_medians.isna()
            if zero_mad_cols.any():
                # Set zero MAD/NaN columns to 0 (centered but not scaled)
                data_scaled.loc[:, zero_mad_cols] = 0
                non_zero_mad_cols = ~zero_mad_cols
                if non_zero_mad_cols.any():
                    # Scale non-zero MAD columns
                    for col in data_scaled.columns[non_zero_mad_cols]:
                        data_scaled[col] = (data_scaled[col] - data_medians[col]) / mad_series[col]
            else:
                # Scale all columns
                for col in data_scaled.columns:
                    data_scaled[col] = (data_scaled[col] - data_medians[col]) / mad_series[col]
        
        # Handle any infinities that might have been introduced
        data_scaled = data_scaled.replace([np.inf, -np.inf], np.nan)
        
        if data_scaled.shape[0] > 0 and data_scaled.shape[1] > 0:
            pass
        
        return data_scaled

    def _get_heatmap_scaling_method(self) -> str:
        """Return the selected heatmap/cluster-map scaling method."""
        scaling_text = _HEATMAP_SCALING_DEFAULT_TEXT
        if hasattr(self, 'heatmap_scaling_combo'):
            scaling_text = self.heatmap_scaling_combo.currentText()
        return _HEATMAP_SCALING_MAP.get(scaling_text, "zscore")
    
    def _perform_clustering(self, data, n_clusters, method):
        """Perform clustering using specified method."""
        import time
        
        t0 = time.time()
        clustering_type = self.clustering_type.currentText()
        
        # Get selected columns from data
        selected_columns = list(data.columns)
        
        # Get scaling method
        scaling_text = self.clustering_scaling_combo.currentText()
        scaling_map = {
            "None (no scaling)": "none",
            "Z-score": "zscore",
            "MAD (Median Absolute Deviation)": "mad"
        }
        scaling_method = scaling_map.get(scaling_text, "zscore")
        
        # Get seed
        seed = self.seed_spinbox.value()
        
        # Prepare full dataframe with all columns (for core function)
        # The core function needs the full dataframe but will use only selected columns
        # Use the indices from the filtered/scaled data to get the corresponding rows from the original dataframe
        full_data = self.feature_dataframe.loc[data.index].copy()
        
        if clustering_type == "Leiden":
            # Use core.cluster for Leiden
            resolution = self.resolution_spinbox.value() if self.resolution_radio.isChecked() else 1.0
            n_neighbors = self.n_neighbors_spinbox.value()
            metric = self.leiden_metric_combo.currentText()
            
            use_jaccard = self.jaccard_checkbox.isChecked()
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="leiden",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                resolution=resolution,
                seed=seed,
                n_neighbors=n_neighbors,
                metric=metric,
                use_jaccard=use_jaccard
            )
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
            
        elif clustering_type == "Louvain":
            # Use core.cluster for Louvain
            n_neighbors = self.n_neighbors_spinbox.value()
            metric = self.leiden_metric_combo.currentText()
            
            use_jaccard = self.jaccard_checkbox.isChecked()
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="louvain",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                seed=seed,
                n_neighbors=n_neighbors,
                metric=metric,
                use_jaccard=use_jaccard
            )
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
            
        elif clustering_type == "HDBSCAN":
            # Use core.cluster for HDBSCAN
            min_cluster_size = self.min_cluster_size_spinbox.value()
            min_samples = self.min_samples_spinbox.value()
            cluster_selection_method = self.cluster_selection_combo.currentText()
            hdbscan_metric = self.metric_combo.currentText()
            
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="hdbscan",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                seed=seed,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
                hdbscan_metric=hdbscan_metric
            )
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
            
        elif clustering_type == "K-means":
            # Use core.cluster for K-means
            
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="kmeans",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                n_clusters=n_clusters,
                seed=seed,
                n_init=10  # Use 10 initializations (efficient default)
            )
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
        else:  # Hierarchical
            # Use core.cluster for hierarchical
            linkage_method = method if isinstance(method, str) else "ward"
            
            t1 = time.time()
            clustered_df = cluster(
                features_df=full_data,
                method="hierarchical",
                columns=selected_columns,
                scaling=scaling_method,
                output_path=None,  # Don't save here
                n_clusters=n_clusters,
                linkage=linkage_method,
                seed=seed
            )
            # Extract cluster labels and ensure integer type to avoid boolean subtraction issues
            cluster_labels = clustered_df['cluster'].astype(int).values
            # Get data subset with clusters
            clustered_data = data.copy()
            clustered_data['cluster'] = cluster_labels.astype(int)
            clustered_data = clustered_data.sort_values('cluster')
            return clustered_data, cluster_labels
    
    def _perform_hierarchical_clustering(self, data, n_clusters, method):
        """Perform hierarchical clustering."""
        # Calculate distance matrix
        distances = pdist(data.values, metric='euclidean')
        
        # Perform linkage
        linkage_matrix = linkage(distances, method=method)
        
        # Get cluster labels and ensure integer type to avoid boolean subtraction issues
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust').astype(int)
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels.astype(int)
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_kmeans_clustering(self, data, n_clusters):
        """Perform K-means clustering."""
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn is required for K-means clustering")
        
        # Get seed from UI
        seed = self.seed_spinbox.value()
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(data.values)
        
        # Convert to 1-based labels and ensure integer type to avoid boolean subtraction issues
        cluster_labels = (cluster_labels + 1).astype(int)
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels.astype(int)
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_leiden_clustering(self, data):
        """Perform Leiden clustering using k-NN graph."""
        if not _HAVE_LEIDEN:
            raise ImportError("leidenalg and igraph are required for Leiden clustering")
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn is required for k-NN graph construction")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Get n_neighbors, metric, and Jaccard option from UI
        n_neighbors = self.n_neighbors_spinbox.value()
        metric = self.leiden_metric_combo.currentText()
        use_jaccard = self.jaccard_checkbox.isChecked()
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(data.values)
        distances, indices = nbrs.kneighbors(data.values)
        
        # Create graph from k-NN
        n = data.shape[0]
        edges = []
        weights = []
        
        if use_jaccard:
            # Compute neighbor sets for Jaccard similarity (PhenoGraph-like)
            # Each node's neighbor set includes itself and its k-nearest neighbors
            neighbor_sets = [set(indices[i]) | {i} for i in range(n)]
            
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        edges.append((i, neighbor_idx))
                        # Compute Jaccard similarity: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
                        intersection = len(neighbor_sets[i] & neighbor_sets[neighbor_idx])
                        union = len(neighbor_sets[i] | neighbor_sets[neighbor_idx])
                        jaccard = intersection / union if union > 0 else 0.0
                        weights.append(jaccard)
        else:
            # Use inverse distance weighting (default)
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        edges.append((i, neighbor_idx))
                        # Convert distance to similarity (inverse, normalized)
                        weight = 1.0 / (1.0 + distances[i][j_idx])
                        weights.append(weight)
        
        # Create igraph (undirected - convert to symmetric)
        edge_set = set()
        symmetric_edges = []
        symmetric_weights = []
        for (i, j), w in zip(edges, weights):
            if (i, j) not in edge_set and (j, i) not in edge_set:
                edge_set.add((i, j))
                symmetric_edges.append((i, j))
                symmetric_weights.append(w)
        
        g = ig.Graph(n)
        g.add_edges(symmetric_edges)
        g.es['weight'] = symmetric_weights
        
        # Get seed from UI
        seed = self.seed_spinbox.value()
        
        # Perform Leiden clustering
        if self.resolution_radio.isChecked():
            # Use resolution parameter
            resolution = self.resolution_spinbox.value()
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=resolution,
                seed=seed,
            )
        else:
            # Use modularity optimization
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights='weight',
                seed=seed,
            )
        
        # Get cluster labels
        cluster_labels = np.array(partition.membership) + 1  # Start from 1
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_louvain_clustering(self, data):
        """Perform Louvain clustering using k-NN graph."""
        if not _HAVE_LEIDEN:
            raise ImportError("leidenalg and igraph are required for Louvain clustering")
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn is required for k-NN graph construction")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Get n_neighbors, metric, and Jaccard option from UI
        n_neighbors = self.n_neighbors_spinbox.value()
        metric = self.leiden_metric_combo.currentText()
        use_jaccard = self.jaccard_checkbox.isChecked()
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(data.values)
        distances, indices = nbrs.kneighbors(data.values)
        
        # Create graph from k-NN
        n = data.shape[0]
        edges = []
        weights = []
        
        if use_jaccard:
            # Compute neighbor sets for Jaccard similarity (PhenoGraph-like)
            # Each node's neighbor set includes itself and its k-nearest neighbors
            neighbor_sets = [set(indices[i]) | {i} for i in range(n)]
            
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        edges.append((i, neighbor_idx))
                        # Compute Jaccard similarity: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
                        intersection = len(neighbor_sets[i] & neighbor_sets[neighbor_idx])
                        union = len(neighbor_sets[i] | neighbor_sets[neighbor_idx])
                        jaccard = intersection / union if union > 0 else 0.0
                        weights.append(jaccard)
        else:
            # Use inverse distance weighting (default)
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        edges.append((i, neighbor_idx))
                        # Convert distance to similarity (inverse, normalized)
                        weight = 1.0 / (1.0 + distances[i][j_idx])
                        weights.append(weight)
        
        # Create igraph (undirected - convert to symmetric)
        edge_set = set()
        symmetric_edges = []
        symmetric_weights = []
        for (i, j), w in zip(edges, weights):
            if (i, j) not in edge_set and (j, i) not in edge_set:
                edge_set.add((i, j))
                symmetric_edges.append((i, j))
                symmetric_weights.append(w)
        
        g = ig.Graph(n)
        g.add_edges(symmetric_edges)
        g.es['weight'] = symmetric_weights
        
        # Get seed from UI
        seed = self.seed_spinbox.value()
        
        # Perform Louvain clustering (using ModularityVertexPartition)
        # Louvain is essentially modularity optimization
        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights='weight',
            seed=seed,
        )
        
        # Get cluster labels
        cluster_labels = np.array(partition.membership) + 1  # Start from 1
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_hdbscan_clustering(self, data):
        """Perform HDBSCAN clustering."""
        if not _HAVE_HDBSCAN:
            raise ImportError("hdbscan is required for HDBSCAN clustering")
        
        # Get parameters from UI
        min_cluster_size = self.min_cluster_size_spinbox.value()
        min_samples = self.min_samples_spinbox.value()
        cluster_selection_method = self.cluster_selection_combo.currentText()
        metric = self.metric_combo.currentText()
        seed = self.seed_spinbox.value()
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric=metric,
            core_dist_n_jobs=1  # Use single thread for stability
        )
        
        # Fit and get cluster labels
        cluster_labels = clusterer.fit_predict(data.values)
        
        # HDBSCAN uses -1 for noise points, convert to 0-based then 1-based
        # First convert -1 to 0, then add 1 to all labels
        cluster_labels = cluster_labels + 1  # -1 becomes 0, others become 1-based
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster (noise points will be at the beginning with cluster 0)
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _create_heatmap(self):
        """Create the heatmap visualization."""
        self._active_view_name = 'Heatmap'
        try:
            self._hard_reset_canvas()
            
            # Check if clustered_data exists
            if self.clustered_data is None:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No clustered data available.\nPlease run clustering first.", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title("Heatmap")
                self._flush_canvas()
                return
            
            if 'cluster' in self.clustered_data.columns:
                pass
            
            scaling_method = self._get_heatmap_scaling_method()
            
            # Use unscaled data if available, otherwise use clustered_data
            base_data = self.clustered_data_unscaled if self.clustered_data_unscaled is not None else self.clustered_data
            if base_data is not None and 'cluster' in base_data.columns:
                pass
            
            # IMPORTANT: Filter out cells with cluster 0 (unclustered/filtered cells)
            # This ensures that cells excluded during clustering (e.g., touching edge) don't appear in heatmap
            if 'cluster' in base_data.columns:
                valid_cluster_mask = (base_data['cluster'] != 0) & (base_data['cluster'].notna())
                if not valid_cluster_mask.any():
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, "No clustered cells available (all cells have cluster=0).\nPlease run clustering first.", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title("Heatmap")
                    self._flush_canvas()
                    return
                base_data = base_data[valid_cluster_mask].copy()
                print(f"[_create_heatmap] Filtered to {len(base_data)} cells with valid clusters (cluster != 0)")
            
            # Additional safeguard: If filter_settings exist, reapply them to ensure excluded cells don't appear
            # This handles cases where clustered_data might have been corrupted or incorrectly restored
            if hasattr(self, 'filter_settings') and self.filter_settings is not None:
                try:
                    # Check for exclude_edge_cells filter
                    if self.filter_settings.get('exclude_edge_cells', False):
                        if 'touches_edge' in base_data.columns:
                            edge_mask = base_data['touches_edge'] == 0
                            cells_before = len(base_data)
                            base_data = base_data[edge_mask].copy()
                            cells_after = len(base_data)
                            if cells_before != cells_after:
                                print(f"[_create_heatmap] Filtered out {cells_before - cells_after} edge-touching cells (from {cells_before} to {cells_after})")
                    
                    # Check for area filters
                    if 'min_area' in self.filter_settings and self.filter_settings['min_area'] is not None:
                        if 'area' in base_data.columns:
                            area_mask = base_data['area'] >= self.filter_settings['min_area']
                            base_data = base_data[area_mask].copy()
                    
                    if 'max_area' in self.filter_settings and self.filter_settings['max_area'] is not None:
                        if 'area' in base_data.columns:
                            area_mask = base_data['area'] <= self.filter_settings['max_area']
                            base_data = base_data[area_mask].copy()
                    
                    if base_data.empty:
                        ax = self.figure.add_subplot(111)
                        ax.text(0.5, 0.5, "No cells remain after applying filter settings.", 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12)
                        ax.set_title("Heatmap")
                        self._flush_canvas()
                        return
                except Exception as filter_err:
                    print(f"[_create_heatmap] Warning: Could not apply filter settings: {filter_err}")
                    # Continue anyway with unfiltered data
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
        # Determine source and prepare data ordering and optional grouping
        source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
        data_to_plot = base_data.copy()
        
        # Ensure patient annotation column is available
        # Determine which column to use (selected or default priority)
        patient_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            patient_col = self.patient_annotation_column
        else:
            # Default priority order
            for col in ['source_file', 'batch_group', 'source_well', 'cohort']:
                if col in self.feature_dataframe.columns:
                    patient_col = col
                    break
        
        if patient_col and patient_col not in data_to_plot.columns and patient_col in self.feature_dataframe.columns:
            # Merge patient annotation column from feature_dataframe by index
            patient_col_series = self.feature_dataframe[patient_col].reindex(data_to_plot.index)
            data_to_plot[patient_col] = patient_col_series.values
        
        group_col = 'cluster'
        legend_labels = None
        if source == 'Manual Gates' and 'manual_phenotype' in data_to_plot.columns:
            groups = self._get_manual_groups_series()
            if groups is not None:
                data_to_plot = data_to_plot.copy()
                data_to_plot['__group__'] = groups.values
                group_col = '__group__'
                # Apply filter by names if set
                if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                    data_to_plot = self._apply_heatmap_filter(data_to_plot, group_col)
                # Sort by group label
                data_to_plot = data_to_plot.sort_values(group_col)
                legend_labels = sorted(data_to_plot[group_col].unique())
            else:
                group_col = 'cluster'
        else:
            # Clusters source: optionally filter by selected clusters (by display name or id)
            try:
                if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                    # Ensure cluster column is integer before filtering
                    # Handle NaN/inf values properly
                    if data_to_plot['cluster'].dtype == bool:
                        data_to_plot['cluster'] = data_to_plot['cluster'].astype(int)
                    elif data_to_plot['cluster'].dtype.name.startswith('object'):
                        data_to_plot['cluster'] = pd.to_numeric(data_to_plot['cluster'], errors='coerce').fillna(0).astype(int)
                    else:
                        # Replace inf and NaN before converting to int
                        data_to_plot['cluster'] = data_to_plot['cluster'].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
                    
                    wanted_ids = set()
                    for cid in self._sorted_cluster_ids(base_data['cluster'].unique(), canonical=False):
                        name = self._get_cluster_display_name(cid)
                        if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                            wanted_ids.add(int(cid))  # Ensure integer
                    if wanted_ids:
                        mask = data_to_plot['cluster'].isin(sorted(wanted_ids))
                        data_to_plot = data_to_plot[mask]
                # Ensure cluster is integer before sorting
                if data_to_plot['cluster'].dtype != int and not data_to_plot['cluster'].dtype.name.startswith('int'):
                    # Handle NaN/inf values before converting to int
                    data_to_plot['cluster'] = pd.to_numeric(data_to_plot['cluster'], errors='coerce')
                    # Fill NaN/inf with 0, then convert to int
                    data_to_plot['cluster'] = data_to_plot['cluster'].replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
                data_to_plot = data_to_plot.sort_values('cluster')
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise

        # Prepare feature columns before scaling
        feature_cols = self._select_feature_columns(data_to_plot)
        
        # Apply selected scaling to feature data
        feature_data = data_to_plot[feature_cols].copy()
        if feature_data.shape[0] > 0 and feature_data.shape[1] > 0:
            pass
        feature_data_scaled = self._apply_scaling(feature_data, scaling_method)
        if feature_data_scaled.shape[0] > 0 and feature_data_scaled.shape[1] > 0:
            pass
        feature_data_scaled = feature_data_scaled.fillna(0)  # Handle any NaN from scaling
        
        # Create data_to_plot with scaled features
        data_to_plot_scaled = data_to_plot.copy()
        for col in feature_cols:
            if col in feature_data_scaled.columns:
                data_to_plot_scaled[col] = feature_data_scaled[col].values

        # Use Scanpy-style heatmap (replaces seaborn)
        self._create_scanpy_style_heatmap(data_to_plot, data_to_plot_scaled, feature_cols, 
                                          group_col, source, scaling_method)
        return
    
    def _create_scanpy_style_heatmap(self, data_to_plot, data_to_plot_scaled, feature_cols, 
                                     group_col, source, scaling_method):
        """Create a Scanpy-style heatmap with improved layout and spacing."""
        self._hard_reset_canvas()
        
        # Prepare heatmap data
        heatmap_data = data_to_plot_scaled[feature_cols].values
        
        # Ensure heatmap_data is numeric (convert boolean/object to float)
        if heatmap_data.dtype == bool:
            heatmap_data = heatmap_data.astype(float)
        elif heatmap_data.dtype.name.startswith('object'):
            # Try to convert to numeric, fill NaN with 0
            heatmap_data = pd.DataFrame(heatmap_data).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        elif not np.issubdtype(heatmap_data.dtype, np.number):
            heatmap_data = heatmap_data.astype(float)
        
        # Determine if dendrograms should be applied (but don't show them)
        # Use the actual clustering method that was used when clustering was run, not the current UI selection
        clustering_type = self.actual_clustering_method if hasattr(self, 'actual_clustering_method') and self.actual_clustering_method is not None else (self.clustering_type.currentText() if hasattr(self, 'clustering_type') else 'Hierarchical')
        is_leiden = clustering_type == "Leiden"
        is_louvain = clustering_type == "Louvain"
        is_hdbscan = clustering_type == "HDBSCAN"
        
        # Apply clustering for reordering but don't show dendrograms
        if is_leiden or is_louvain or is_hdbscan:
            row_cluster = False
            col_cluster = False
            linkage_method = None
        else:
            row_cluster = True
            # Use the actual dendrogram mode that was used when clustering was run, not the current UI selection
            dendro_mode = self.actual_dendrogram_mode if hasattr(self, 'actual_dendrogram_mode') and self.actual_dendrogram_mode is not None else (self.dendro_mode.currentText() if hasattr(self, 'dendro_mode') else "Rows and columns")
            col_cluster = (dendro_mode == "Rows and columns")
            linkage_method = self.hierarchical_method.currentText() if hasattr(self, 'hierarchical_method') else "ward"
        
        # Apply hierarchical clustering if needed (for reordering only)
        from scipy.cluster.hierarchy import linkage, leaves_list
        
        row_indices = np.arange(len(feature_cols))
        col_indices = np.arange(heatmap_data.shape[0])
        
        if row_cluster:
            # Cluster features (rows)
            row_linkage = linkage(heatmap_data.T, method=linkage_method, metric='euclidean')
            row_indices = leaves_list(row_linkage)
        
        if col_cluster:
            # Cluster cells (columns)
            col_linkage = linkage(heatmap_data, method=linkage_method, metric='euclidean')
            col_indices = leaves_list(col_linkage)
        
        # Reorder data based on clustering
        heatmap_data_reordered = heatmap_data[np.ix_(col_indices, row_indices)]
        feature_cols_reordered = [feature_cols[i] for i in row_indices]
        
        # Reorder annotation bar to match column clustering
        group_values = data_to_plot[group_col].values
        group_values_reordered = group_values[col_indices]
        
        # Create group color mapping (shared with Cluster Map so cluster IDs keep the same colors).
        unique_groups = self._sort_group_values(data_to_plot[group_col].unique())
        cluster_color_map = self._build_group_color_map(unique_groups, source=source, group_col=group_col)
        
        # Create reordered cell colors for annotation bar (match column clustering)
        # Convert to proper RGB array for imshow
        cell_colors_rgb = [cluster_color_map.get(val, (0.7, 0.7, 0.7)) for val in group_values_reordered]
        
        # Check if patient annotation is enabled
        # Determine which column to use for patient annotation
        # Priority: selected column, or source_file, batch_group, source_well
        patient_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            patient_col = self.patient_annotation_column
        else:
            # Default priority order
            for col in ['source_file', 'batch_group', 'source_well', 'cohort']:
                if col in data_to_plot.columns:
                    patient_col = col
                    break
        
        show_patient_annotation = (self.patient_annotation_enabled and
                                   patient_col is not None and patient_col in data_to_plot.columns)
        
        # Prepare patient annotation data if enabled
        patient_values_reordered = None
        patient_color_map = {}
        cohort_color_map = {}
        use_cohorts = False
        if show_patient_annotation:
            # Get patient annotation values from selected column and reorder to match column clustering
            patient_values = data_to_plot[patient_col].values
            patient_values_reordered = patient_values[col_indices]
            
            # Get unique patient values
            unique_patients = sorted([f for f in data_to_plot[patient_col].unique() if pd.notna(f)])
            
            # Check if cohorts are defined and cohort coloring is enabled
            cohorts_used = set()
            if self.use_cohort_coloring:
                for patient in unique_patients:
                    if patient in self.patient_cohort_map:
                        cohorts_used.add(self.patient_cohort_map[patient])
            
            if cohorts_used:
                # Use cohort-based coloring
                use_cohorts = True
                unique_cohorts = sorted(cohorts_used)
                
                # Generate colors for cohorts
                cohort_colors_raw = _get_patient_colors(len(unique_cohorts))
                for i, cohort_name in enumerate(unique_cohorts):
                    color = cohort_colors_raw[i]
                    if len(color) == 4:
                        rgb = tuple(color[:3])
                    elif len(color) == 3:
                        rgb = tuple(color)
                    else:
                        rgb = (color[0], color[1], color[2])
                    cohort_color_map[cohort_name] = rgb
                    self.cohort_colors[cohort_name] = rgb
                
                # Map patients to cohort colors (or individual colors if not in cohort)
                unassigned_patients = [p for p in unique_patients if p not in self.patient_cohort_map]
                if unassigned_patients:
                    # Generate colors for unassigned patients
                    unassigned_colors_raw = _get_patient_colors(len(unassigned_patients))
                    for i, patient_file in enumerate(unassigned_patients):
                        color = unassigned_colors_raw[i]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        patient_color_map[patient_file] = rgb
                
                # Create mapping: patient -> color (via cohort if assigned)
                for patient_file in unique_patients:
                    if patient_file in self.patient_cohort_map:
                        cohort = self.patient_cohort_map[patient_file]
                        patient_color_map[patient_file] = cohort_color_map[cohort]
            else:
                # Use individual patient colors (original behavior)
                patient_colors_raw = _get_patient_colors(len(unique_patients))
                for i, patient_file in enumerate(unique_patients):
                    color = patient_colors_raw[i]
                    if len(color) == 4:
                        rgb = tuple(color[:3])
                    elif len(color) == 3:
                        rgb = tuple(color)
                    else:
                        rgb = (color[0], color[1], color[2])
                    patient_color_map[patient_file] = rgb
            
            # Create reordered patient colors for annotation bar
            patient_colors_rgb = [patient_color_map.get(val, (0.8, 0.8, 0.8)) for val in patient_values_reordered]
        
        # Create layout - adjust based on whether patient annotation is shown
        if show_patient_annotation:
            # Colorbar, patient annotation, cell annotation, heatmap
            gs = self.figure.add_gridspec(
                nrows=4, ncols=2, 
                height_ratios=[0.02, 0.04, 0.06, 0.88],  # Colorbar, patient annotation, cell annotation, heatmap
                width_ratios=[0.88, 0.12],  # Heatmap area, legend
                hspace=0.03, wspace=0.02,  # Space between elements
                left=0.15, right=0.98, top=0.88, bottom=0.12  # More top margin for colorbar label and ticks
            )
            heatmap_row = 3
            cell_annotation_row = 2
            patient_annotation_row = 1
        else:
            # Colorbar, annotation bar, heatmap
            gs = self.figure.add_gridspec(
                nrows=3, ncols=2, 
                height_ratios=[0.02, 0.06, 0.92],  # Colorbar, annotation bar, heatmap
                width_ratios=[0.88, 0.12],  # Heatmap area, legend
                hspace=0.03, wspace=0.02,  # Space between elements
                left=0.15, right=0.98, top=0.88, bottom=0.12  # More top margin for colorbar label and ticks
            )
            heatmap_row = 2
            cell_annotation_row = 1
            patient_annotation_row = None
        
        # Patient annotation bar (if enabled)
        if show_patient_annotation:
            ax_patient_annotation = self.figure.add_subplot(gs[patient_annotation_row, 0])
            patient_annotation_array = np.array(patient_colors_rgb).reshape(1, -1, 3)
            ax_patient_annotation.imshow(patient_annotation_array, aspect='auto', interpolation='nearest', 
                                         extent=[0, len(patient_colors_rgb), 0, 1])
            ax_patient_annotation.set_xlim(0, len(patient_colors_rgb))
            ax_patient_annotation.set_xticks([])
            ax_patient_annotation.set_yticks([])
            ax_patient_annotation.spines['top'].set_visible(False)
            ax_patient_annotation.spines['right'].set_visible(False)
            ax_patient_annotation.spines['bottom'].set_visible(False)
            ax_patient_annotation.spines['left'].set_visible(False)
            self._annotate_heatmap_bar_axis(
                ax_patient_annotation,
                self._get_heatmap_annotation_bar_label(source, patient=True),
            )
        
        # Cell annotation bar below colorbar (or below patient annotation if enabled)
        ax_annotation = self.figure.add_subplot(gs[cell_annotation_row, 0])
        # Create a simple colored bar - convert to proper RGB array
        annotation_array = np.array(cell_colors_rgb).reshape(1, -1, 3)
        ax_annotation.imshow(annotation_array, aspect='auto', interpolation='nearest', extent=[0, len(cell_colors_rgb), 0, 1])
        ax_annotation.set_xlim(0, len(cell_colors_rgb))
        ax_annotation.set_xticks([])
        ax_annotation.set_yticks([])
        ax_annotation.spines['top'].set_visible(False)
        ax_annotation.spines['right'].set_visible(False)
        ax_annotation.spines['bottom'].set_visible(False)
        ax_annotation.spines['left'].set_visible(False)
        self._annotate_heatmap_bar_axis(
            ax_annotation,
            self._get_heatmap_annotation_bar_label(source),
        )
        
        # Main heatmap
        ax_heatmap = self.figure.add_subplot(gs[heatmap_row, 0])
        
        # Get colormap
        colormap_name = self._get_colormap_name()
        
        # Create heatmap with reordered data
        # Ensure heatmap_data_reordered is numeric before percentile calculation
        if heatmap_data_reordered.dtype == bool:
            heatmap_data_reordered = heatmap_data_reordered.astype(float)
        elif not np.issubdtype(heatmap_data_reordered.dtype, np.number):
            heatmap_data_reordered = heatmap_data_reordered.astype(float)
        
        # Remove any NaN or Inf values before percentile calculation
        heatmap_data_reordered_clean = np.nan_to_num(heatmap_data_reordered, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            vmin_val = np.percentile(heatmap_data_reordered_clean, 2)
            vmax_val = np.percentile(heatmap_data_reordered_clean, 98)
        except Exception as e:
            # Fallback to min/max if percentile fails
            vmin_val = np.nanmin(heatmap_data_reordered_clean)
            vmax_val = np.nanmax(heatmap_data_reordered_clean)
        
        im = ax_heatmap.imshow(
            heatmap_data_reordered.T, 
            aspect='auto', 
            cmap=colormap_name, 
            interpolation='nearest',
            vmin=vmin_val,
            vmax=vmax_val
        )
        
        # Colorbar at top (horizontal)
        ax_cbar = self.figure.add_subplot(gs[0, 0])
        cbar = self.figure.colorbar(im, cax=ax_cbar, orientation='horizontal')
        # Move ticks and label to top of colorbar
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=8, top=True, labeltop=True, bottom=False, labelbottom=False)
        cbar.set_label('Normalized Feature Value', fontsize=8, labelpad=10)
        
        # Set feature labels on y-axis with proper spacing
        n_features = len(feature_cols_reordered)
        ax_heatmap.set_yticks(np.arange(n_features))
        # Use custom feature labels if available
        feature_labels_display = [self._get_feature_display_name(f) for f in feature_cols_reordered]
        ax_heatmap.set_yticklabels(feature_labels_display, fontsize=self.feature_tick_fontsize, rotation=0)
        ax_heatmap.set_ylabel('Features', fontsize=10, fontweight='bold')
        
        # Ensure all labels are visible
        ax_heatmap.tick_params(axis='y', which='major', labelsize=self.feature_tick_fontsize, pad=2)
        for label in ax_heatmap.get_yticklabels():
            label.set_visible(True)
        
        # Remove x-axis labels (cells)
        ax_heatmap.set_xticks([])
        ax_heatmap.set_xlabel('Cells', fontsize=10, fontweight='bold')
        
        # Set proper limits
        ax_heatmap.set_xlim(-0.5, heatmap_data_reordered.shape[0] - 0.5)
        ax_heatmap.set_ylim(-0.5, n_features - 0.5)
        
        # Remove spines for cleaner look
        ax_heatmap.spines['top'].set_visible(False)
        ax_heatmap.spines['right'].set_visible(False)
        ax_heatmap.spines['bottom'].set_visible(False)
        ax_heatmap.spines['left'].set_visible(False)
        
        # Legend on the right - adjust layout based on patient annotation
        # Check if legend should be shown
        show_legend = self.show_legend_checkbox.isChecked() if hasattr(self, 'show_legend_checkbox') else True
        
        if show_patient_annotation:
            # Create nested gridspec for two legends (patient on top, clusters below)
            legend_gs = gs[heatmap_row, 1].subgridspec(2, 1, hspace=0.0, height_ratios=[0.4, 0.6])
            
            # Patient legend on top
            ax_patient_legend = self.figure.add_subplot(legend_gs[0])
            ax_patient_legend.axis('off')
            if show_legend:
                patient_legend_elements = []
                if use_cohorts and cohort_color_map:
                    # Show cohorts in legend instead of individual patients
                    for cohort_name in sorted(cohort_color_map.keys()):
                        color = cohort_color_map[cohort_name]
                        # Get list of patients in this cohort for display
                        patients_in_cohort = [p for p in unique_patients if p in self.patient_cohort_map and self.patient_cohort_map[p] == cohort_name]
                        if patients_in_cohort:
                            # Show cohort name with patient count or list
                            if len(patients_in_cohort) <= 3:
                                patient_labels = [self._get_patient_display_name(p) for p in patients_in_cohort]
                                label = f"{cohort_name} ({', '.join(patient_labels)})"
                            else:
                                label = f"{cohort_name} ({len(patients_in_cohort)} patients)"
                            patient_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label, edgecolor='black', linewidth=0.5))
                    
                    # Also show unassigned patients if any
                    unassigned_patients = [p for p in unique_patients if p not in self.patient_cohort_map]
                    for patient_file in sorted(unassigned_patients):
                        if patient_file in patient_color_map:
                            color = patient_color_map[patient_file]
                            label = self._get_patient_display_name(patient_file)
                            patient_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label, edgecolor='black', linewidth=0.5))
                else:
                    # Show individual patients (original behavior)
                    for patient_file in sorted(patient_color_map.keys()):
                        color = patient_color_map[patient_file]
                        # Use custom patient label (helper function handles custom labels and defaults)
                        label = self._get_patient_display_name(patient_file)
                        patient_legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label, edgecolor='black', linewidth=0.5))
                
                if patient_legend_elements:
                    # Use configurable legend font size
                    legend_fontsize = getattr(self, 'legend_fontsize', 8)
                    legend_title = 'Cohorts' if use_cohorts and cohort_color_map else self.patient_legend_label
                    ax_patient_legend.legend(handles=patient_legend_elements, loc='upper left', frameon=True, fontsize=legend_fontsize, 
                                            title=legend_title, title_fontsize=legend_fontsize + 1)
            
            # Cluster legend below
            ax_cluster_legend = self.figure.add_subplot(legend_gs[1])
            ax_cluster_legend.axis('off')
            if show_legend:
                legend_elements = []
                if source == 'Manual Gates':
                    for key in unique_groups:
                        color = cluster_color_map[key]
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=str(key), edgecolor='black', linewidth=0.5))
                else:
                    for key in unique_groups:
                        color = cluster_color_map[key]
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=self._get_cluster_display_name(key), edgecolor='black', linewidth=0.5))
                
                if legend_elements:
                    # Use configurable legend layout
                    n_clusters = len(legend_elements)
                    # Get legend configuration from dialog settings
                    legend_ncols = getattr(self, 'legend_ncols', 1)
                    legend_nrows = getattr(self, 'legend_nrows', 1)
                    legend_fontsize = getattr(self, 'legend_fontsize', 8)
                    
                    # Calculate ncol based on desired rows if specified, otherwise use configured columns
                    if legend_nrows > 1:
                        ncol = max(1, (n_clusters + legend_nrows - 1) // legend_nrows)
                    else:
                        ncol = legend_ncols
                    
                    ax_cluster_legend.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=legend_fontsize, 
                                            title='Clusters' if source == 'Clusters' else 'Groups', title_fontsize=legend_fontsize + 1, ncol=ncol)
        else:
            # Single legend area
            ax_legend = self.figure.add_subplot(gs[heatmap_row, 1])
            ax_legend.axis('off')  # Hide axes for legend area
            
            if show_legend:
                # Add legend for groups/clusters - vertical layout
                # Use the same color mapping as annotation bar (sorted for consistency)
                legend_elements = []
                if source == 'Manual Gates':
                    for key in unique_groups:
                        color = cluster_color_map[key]
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=str(key), edgecolor='black', linewidth=0.5))
                else:
                    for key in unique_groups:
                        color = cluster_color_map[key]
                        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=self._get_cluster_display_name(key), edgecolor='black', linewidth=0.5))
                # Place legend vertically to the right of colorbar
                # Use configurable legend layout
                n_clusters = len(legend_elements)
                # Get legend configuration from dialog settings
                legend_ncols = getattr(self, 'legend_ncols', 1)
                legend_nrows = getattr(self, 'legend_nrows', 1)
                legend_fontsize = getattr(self, 'legend_fontsize', 8)
                
                # Calculate ncol based on desired rows if specified, otherwise use configured columns
                if legend_nrows > 1:
                    ncol = max(1, (n_clusters + legend_nrows - 1) // legend_nrows)
                else:
                    ncol = legend_ncols
                
                ax_legend.legend(handles=legend_elements, loc='center left', frameon=True, fontsize=legend_fontsize, 
                                 title='Clusters' if source == 'Clusters' else 'Groups', title_fontsize=legend_fontsize + 1, ncol=ncol)
        
        self._flush_canvas()
    
    def _create_cluster_map(self):
        """Create the cluster map visualization (one equal-sized box per cluster)."""
        self._active_view_name = 'Cluster Map'
        try:
            self._hard_reset_canvas()
            
            # Check if clustered_data exists
            if self.clustered_data is None:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No clustered data available.\nPlease run clustering first.", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title("Cluster Map")
                self._flush_canvas()
                return
            
            
            scaling_method = self._get_heatmap_scaling_method()
            
            # Use unscaled data if available, otherwise use clustered_data
            base_data = self.clustered_data_unscaled if self.clustered_data_unscaled is not None else self.clustered_data
            
            # Prepare data - filter clusters if needed
            data_to_plot = base_data.copy()
            
            # Ensure cluster column is integer
            if 'cluster' in data_to_plot.columns:
                if data_to_plot['cluster'].dtype != int and not data_to_plot['cluster'].dtype.name.startswith('int'):
                    # Handle NaN/inf properly before converting
                    data_to_plot['cluster'] = pd.to_numeric(data_to_plot['cluster'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
            
            # Apply filter if set
            if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                wanted_ids = set()
                for cid in self._sorted_cluster_ids(base_data['cluster'].unique(), canonical=False):
                    name = self._get_cluster_display_name(cid)
                    if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                        wanted_ids.add(int(cid))
                if wanted_ids:
                    data_to_plot = data_to_plot[data_to_plot['cluster'].isin(sorted(wanted_ids))]
            
            # Select feature columns
            feature_cols = self._select_feature_columns(data_to_plot)
            
            # Aggregate data per cluster using selected method
            zscore_method = getattr(self, 'cluster_map_zscore_method', 'Mean')
            if zscore_method == 'Mean':
                cluster_aggregated = data_to_plot.groupby('cluster')[feature_cols].mean()
            elif zscore_method == 'Median':
                cluster_aggregated = data_to_plot.groupby('cluster')[feature_cols].median()
            elif zscore_method == 'Max':
                cluster_aggregated = data_to_plot.groupby('cluster')[feature_cols].max()
            elif zscore_method == 'Min':
                cluster_aggregated = data_to_plot.groupby('cluster')[feature_cols].min()
            else:
                cluster_aggregated = data_to_plot.groupby('cluster')[feature_cols].mean()  # Default to mean
            
            # Apply scaling to aggregated data
            feature_data_scaled = self._apply_scaling(cluster_aggregated, scaling_method)
            feature_data_scaled = feature_data_scaled.fillna(0)  # Handle any NaN from scaling
            
            # Ensure feature_cols matches the scaled data columns
            scaled_feature_cols = list(feature_data_scaled.columns)
            
            # Create the cluster map visualization
            self._create_cluster_map_visualization(feature_data_scaled, scaled_feature_cols, cluster_aggregated.index)
            return
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    def _create_cluster_map_visualization(self, cluster_data_scaled, feature_cols, cluster_ids):
        """Create the actual cluster map visualization with one equal-sized box per cluster."""
        self._hard_reset_canvas()
        
        # Get orientation setting
        orientation = getattr(self, 'cluster_map_orientation', 'landscape')
        is_landscape = (orientation == 'landscape')
        
        # Respect cluster-map dendrogram settings from Plot Configuration.
        dendrogram_setting = getattr(self, 'cluster_map_dendrogram', None)
        valid_dendrogram_settings = {'Both rows and columns', 'Rows only', 'Columns only', 'No dendrogram'}
        if dendrogram_setting not in valid_dendrogram_settings:
            # Backward-compatible fallback for older sessions where only clustering-level mode was saved.
            dendro_mode = getattr(self, 'actual_dendrogram_mode', None)
            if dendro_mode == "Rows and columns":
                dendrogram_setting = 'Both rows and columns'
            elif dendro_mode == "Rows only":
                dendrogram_setting = 'Rows only'
            else:
                dendrogram_setting = 'Columns only'
        
        show_row_dendro = dendrogram_setting in ['Both rows and columns', 'Rows only']
        show_col_dendro = dendrogram_setting in ['Both rows and columns', 'Columns only']
        
        # Prepare heatmap data (clusters as rows, features as columns)
        heatmap_data = cluster_data_scaled.values
        
        # Ensure heatmap_data is numeric
        if heatmap_data.dtype == bool:
            heatmap_data = heatmap_data.astype(float)
        elif heatmap_data.dtype.name.startswith('object'):
            heatmap_data = pd.DataFrame(heatmap_data).apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        elif not np.issubdtype(heatmap_data.dtype, np.number):
            heatmap_data = heatmap_data.astype(float)
        
        # Hierarchical clustering for features (rows) - always compute for ordering, but only show if requested
        from scipy.cluster.hierarchy import linkage, leaves_list
        feature_linkage = None
        try:
            feature_linkage = linkage(heatmap_data.T, method='ward', metric='euclidean')
            feature_indices = leaves_list(feature_linkage)
            feature_cols_ordered = [feature_cols[i] for i in feature_indices]
            heatmap_data_ordered = heatmap_data[:, feature_indices]
        except Exception:
            # If clustering fails, use original order
            feature_cols_ordered = feature_cols
            heatmap_data_ordered = heatmap_data
            feature_linkage = None
        
        # Hierarchical clustering for clusters (columns) - only compute if needed
        cluster_ids_list = list(cluster_ids) if hasattr(cluster_ids, '__iter__') and not isinstance(cluster_ids, str) else [cluster_ids]
        n_clusters_total = len(cluster_ids_list)
        
        cluster_linkage = None
        cluster_order = cluster_ids_list
        cluster_order_idx = list(range(len(cluster_ids_list)))
        
        if show_col_dendro and n_clusters_total > 1:
            try:
                # Use ward linkage for cluster clustering
                cluster_linkage = linkage(heatmap_data_ordered, method='ward', metric='euclidean')
                cluster_indices = leaves_list(cluster_linkage)
                cluster_order = [cluster_ids_list[i] for i in cluster_indices]
                cluster_order_idx = cluster_indices
            except Exception:
                # If clustering fails, use sorted order
                cluster_order = self._sorted_cluster_ids(cluster_ids_list, canonical=False)
                cluster_order_idx = [cluster_ids_list.index(cid) for cid in cluster_order]
                cluster_linkage = None
        elif n_clusters_total > 1:
            # No column dendrogram, but still order clusters
            cluster_order = self._sorted_cluster_ids(cluster_ids_list, canonical=False)
            cluster_order_idx = [cluster_ids_list.index(cid) for cid in cluster_order]
        
        heatmap_data_final = heatmap_data_ordered[cluster_order_idx, :]
        
        # Transpose if landscape orientation
        if is_landscape:
            heatmap_data_final = heatmap_data_final.T
        
        if heatmap_data_final.size == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data available for cluster map.", ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_axis_off()
            self._flush_canvas()
            return

        # Enable only dendrograms that were successfully computed.
        show_row_dendro = bool(show_row_dendro and feature_linkage is not None)
        show_col_dendro = bool(show_col_dendro and cluster_linkage is not None)

        n_features = len(feature_cols_ordered)
        n_clusters = len(cluster_order)
        feature_labels_display = [self._get_feature_display_name(f) for f in feature_cols_ordered]
        cluster_labels_display = [self._get_cluster_display_name(cid) for cid in cluster_order]

        # Get font sizes (use new separate controls if available, fallback to old feature_tick_fontsize)
        x_tick_fontsize = getattr(self, 'x_tick_fontsize', getattr(self, 'feature_tick_fontsize', 10))
        y_tick_fontsize = getattr(self, 'y_tick_fontsize', getattr(self, 'feature_tick_fontsize', 8))
        x_tick_fontsize_eff = float(x_tick_fontsize)
        y_tick_fontsize_eff = float(y_tick_fontsize)

        left_dendro_visible = (is_landscape and show_row_dendro) or ((not is_landscape) and show_col_dendro)
        top_dendro_visible = (is_landscape and show_col_dendro) or ((not is_landscape) and show_row_dendro)
        show_cluster_legend = bool(getattr(self, 'cluster_map_show_legend', False))
        cbar_position = str(getattr(self, 'cluster_map_colorbar_position', 'Upper right')).strip()
        if cbar_position not in {'Upper right', 'Upper left', 'Right side'}:
            cbar_position = 'Upper right'
        cbar_orientation = str(getattr(self, 'cluster_map_colorbar_orientation', 'Vertical')).strip().title()
        if cbar_orientation not in {'Vertical', 'Horizontal'}:
            cbar_orientation = 'Vertical'
        cbar_is_horizontal = (cbar_orientation == 'Horizontal')

        # Determine desired Cluster Map sizing and keep cells square.
        n_rows_heatmap, n_cols_heatmap = heatmap_data_final.shape
        cell_size_px = float(getattr(self, 'cluster_map_cell_size', 14))
        cell_size_px = max(4.0, min(64.0, cell_size_px))
        dpi = float(self.figure.get_dpi() or 100.0)
        heatmap_width_in = max(2.0, (n_cols_heatmap * cell_size_px) / dpi)
        heatmap_height_in = max(2.0, (n_rows_heatmap * cell_size_px) / dpi)

        y_labels_for_pad = feature_labels_display if is_landscape else cluster_labels_display
        x_labels_for_pad = cluster_labels_display if is_landscape else feature_labels_display
        max_y_label_len = max([len(str(v)) for v in y_labels_for_pad], default=8)
        max_x_label_len = max([len(str(v)) for v in x_labels_for_pad], default=8)

        # Approximate space needed for labels to avoid collision with colorbar.
        y_label_pad_in = 0.25 + min(2.8, 0.05 * max_y_label_len * (max(6, y_tick_fontsize) / 8.0))
        y_label_outer_pad = min(0.10, max(0.018, 0.018 + 0.0034 * max_y_label_len * (max(6, y_tick_fontsize) / 8.0)))
        # Bottom padding: keep landscape compact (cluster labels), allow more room in portrait (feature labels).
        if is_landscape:
            x_label_bottom_base = 0.048 + 0.0025 * max_x_label_len * (max(6, x_tick_fontsize) / 9.0)
            x_label_bottom_pad = min(0.24, max(0.08, x_label_bottom_base))
        else:
            x_label_bottom_base = 0.11 + 0.0072 * max_x_label_len * (max(6, x_tick_fontsize) / 9.0)
            x_label_bottom_pad = min(0.54, max(0.14, x_label_bottom_base))

        # Keep dendrogram/annotation-strip sizing proportional to heatmap cell size.
        # This prevents dendrograms from dominating the canvas when users shrink cells.
        dendro_cell_scale = max(0.40, min(1.0, cell_size_px / 14.0))
        top_dendro_target = (0.62 * dendro_cell_scale) + min(0.12, 0.005 * n_clusters)
        left_dendro_target = (0.66 * dendro_cell_scale) + min(0.16, 0.003 * n_rows_heatmap)

        top_dendro_in = 0.0
        if top_dendro_visible:
            top_dendro_in = max(0.20, min(0.74, top_dendro_target))
            top_dendro_in = min(top_dendro_in, max(0.20, heatmap_height_in * 0.22))

        left_dendro_in = 0.0
        if left_dendro_visible:
            left_dendro_in = max(0.22, min(0.84, left_dendro_target))
            left_dendro_in = min(left_dendro_in, max(0.22, heatmap_width_in * 0.24))

        strip_in = max(0.10, min(0.20, 0.014 * cell_size_px))
        legend_in = 2.4 if show_cluster_legend else 0.0
        # Keep right-side spacer compact when colorbar is on the right; overlap is handled explicitly.
        if cbar_position in ['Upper right', 'Right side']:
            if left_dendro_visible:
                # Keep enough room for long y labels on the right; avoids truncation.
                if cbar_position == 'Upper right':
                    label_spacer_in = max(0.20, min(2.40, y_label_pad_in * 0.82))
                else:
                    label_spacer_in = max(0.10, min(1.60, y_label_pad_in * 0.58))
            else:
                # Y labels are on the left; keep right spacer compact.
                label_spacer_in = 0.02 if cbar_position == 'Upper right' else 0.015
        else:
            label_spacer_in = y_label_pad_in if left_dendro_visible else 0.08
        cbar_width_in = float(getattr(self, 'cluster_map_colorbar_width', 0.08))
        cbar_width_in = max(0.05, min(0.30, cbar_width_in))
        if cbar_is_horizontal:
            cbar_slot_in = max(0.60, min(1.20, max(cbar_width_in + 0.12, heatmap_width_in * 0.16)))
        elif cbar_position == 'Right side':
            cbar_slot_in = max(0.08, min(0.16, cbar_width_in + 0.03))
        else:
            cbar_slot_in = max(0.07, min(0.13, cbar_width_in + 0.02))
        reserve_left_cbar_slot = (cbar_position == 'Upper left')
        # Keep a dedicated right column for side bars and upper-right placements.
        reserve_right_cbar_slot = (cbar_position == 'Right side') or (cbar_position == 'Upper right')

        # Adaptive outer margins for cleaner canvas fit and less clipping.
        if left_dendro_visible:
            base_left_outer_margin = 0.02
        else:
            base_left_outer_margin = y_label_outer_pad
        base_right_outer_margin = 0.997 if reserve_right_cbar_slot else 0.992
        base_top_outer_margin = 0.988

        # Keep the current canvas/export figure size; do not force a new size here.
        # Forcing size in this renderer makes window and export resizing feel broken.

        # Precompute values reused across both layout passes.
        colormap_name = self._get_colormap_name()
        heatmap_data_clean = np.nan_to_num(heatmap_data_final, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            vmin_val = np.percentile(heatmap_data_clean, 2)
            vmax_val = np.percentile(heatmap_data_clean, 98)
        except Exception:
            vmin_val = np.nanmin(heatmap_data_clean)
            vmax_val = np.nanmax(heatmap_data_clean)
        cluster_color_map = self._build_group_color_map(cluster_order, source='Clusters', group_col='cluster')
        cluster_strip = np.array([cluster_color_map.get(cid, (0.7, 0.7, 0.7)) for cid in cluster_order])

        layout_margin_adjust = {'left': 0.0, 'right': 0.0, 'bottom': 0.0, 'top': 0.0}
        label_spacer_extra_in = 0.0
        top_band_extra_in = 0.0
        max_layout_passes = 4

        for layout_pass in range(max_layout_passes):
            self.figure.clear()

            left_outer_margin = min(0.35, max(0.01, base_left_outer_margin + layout_margin_adjust['left']))
            right_outer_margin = max(
                left_outer_margin + 0.14,
                min(0.999, base_right_outer_margin - layout_margin_adjust['right'])
            )
            bottom_outer_margin = min(0.72, max(0.04, x_label_bottom_pad + layout_margin_adjust['bottom']))
            top_outer_margin = min(0.995, max(bottom_outer_margin + 0.14, base_top_outer_margin - layout_margin_adjust['top']))
            axis_label_fontsize = max(7.0, min(10.0, x_tick_fontsize_eff))

            # Build a stable grid where each element gets dedicated space
            # (dendrogram/strip/labels/colorbar/legend).
            row_keys = []
            row_sizes = []
            if top_dendro_visible:
                row_keys.append('top_dendrogram')
                row_sizes.append(top_dendro_in + top_band_extra_in)
            if is_landscape:
                row_keys.append('cluster_strip')
                row_sizes.append(strip_in)
            row_keys.append('heatmap')
            row_sizes.append(heatmap_height_in)

            col_keys = []
            col_sizes = []
            if left_dendro_visible:
                col_keys.append('left_dendrogram')
                col_sizes.append(left_dendro_in)
            if not is_landscape:
                col_keys.append('cluster_strip')
                col_sizes.append(strip_in)
            if reserve_left_cbar_slot:
                col_keys.append('cbar_slot_left')
                col_sizes.append(cbar_slot_in)
            col_keys.append('heatmap')
            col_sizes.append(heatmap_width_in)
            col_keys.append('label_spacer')
            col_sizes.append(label_spacer_in + label_spacer_extra_in)
            if reserve_right_cbar_slot:
                col_keys.append('cbar_slot_right')
                col_sizes.append(cbar_slot_in)
            if show_cluster_legend:
                col_keys.append('legend')
                col_sizes.append(legend_in)

            gs = self.figure.add_gridspec(
                nrows=len(row_sizes),
                ncols=len(col_sizes),
                height_ratios=row_sizes,
                width_ratios=col_sizes,
                hspace=0.03,
                wspace=0.02,
                left=left_outer_margin,
                right=right_outer_margin,
                top=top_outer_margin,
                bottom=bottom_outer_margin,
            )

            row_idx = {name: i for i, name in enumerate(row_keys)}
            col_idx = {name: i for i, name in enumerate(col_keys)}

            # Main heatmap axis
            ax_heatmap = self.figure.add_subplot(gs[row_idx['heatmap'], col_idx['heatmap']])

            ax_row_dendro = None
            ax_col_dendro = None

            # Row dendrogram (feature dendrogram)
            if show_row_dendro:
                if is_landscape:
                    ax_row_dendro = self.figure.add_subplot(gs[row_idx['heatmap'], col_idx['left_dendrogram']])
                    row_orientation = 'left'
                else:
                    ax_row_dendro = self.figure.add_subplot(gs[row_idx['top_dendrogram'], col_idx['heatmap']])
                    row_orientation = 'top'
                dendrogram(
                    feature_linkage,
                    ax=ax_row_dendro,
                    orientation=row_orientation,
                    no_labels=True,
                    above_threshold_color='#333333',
                    color_threshold=0
                )
                for collection in ax_row_dendro.collections:
                    try:
                        collection.set_linewidth(0.8)
                    except Exception:
                        pass
                ax_row_dendro.set_xticks([])
                ax_row_dendro.set_yticks([])
                for spine in ax_row_dendro.spines.values():
                    spine.set_visible(False)

            # Column dendrogram (cluster dendrogram)
            if show_col_dendro:
                if is_landscape:
                    ax_col_dendro = self.figure.add_subplot(gs[row_idx['top_dendrogram'], col_idx['heatmap']])
                    col_orientation = 'top'
                else:
                    ax_col_dendro = self.figure.add_subplot(gs[row_idx['heatmap'], col_idx['left_dendrogram']])
                    col_orientation = 'left'
                dendrogram(
                    cluster_linkage,
                    ax=ax_col_dendro,
                    orientation=col_orientation,
                    no_labels=True,
                    above_threshold_color='#333333',
                    color_threshold=0
                )
                for collection in ax_col_dendro.collections:
                    try:
                        collection.set_linewidth(0.8)
                    except Exception:
                        pass
                ax_col_dendro.set_xticks([])
                ax_col_dendro.set_yticks([])
                for spine in ax_col_dendro.spines.values():
                    spine.set_visible(False)

            # Create heatmap
            im = ax_heatmap.imshow(
                heatmap_data_final,
                aspect='auto',
                cmap=colormap_name,
                interpolation='nearest',
                vmin=vmin_val,
                vmax=vmax_val
            )

            # Set labels based on orientation
            if is_landscape:
                # Landscape: clusters on x-axis, features on y-axis
                ax_heatmap.set_xticks(np.arange(n_clusters))
                ax_heatmap.set_xticklabels(
                    cluster_labels_display,
                    fontsize=x_tick_fontsize_eff,
                    rotation=90,
                    ha='right',
                    va='top'
                )
                ax_heatmap.set_xlabel('Clusters', fontsize=axis_label_fontsize, fontweight='bold', labelpad=2)

                ax_heatmap.set_yticks(np.arange(n_features))
                ax_heatmap.set_yticklabels(feature_labels_display, fontsize=y_tick_fontsize_eff, rotation=0)
                ax_heatmap.set_ylabel('Features', fontsize=axis_label_fontsize, fontweight='bold', labelpad=2)

                ax_heatmap.set_xlim(-0.5, n_clusters - 0.5)
                ax_heatmap.set_ylim(-0.5, n_features - 0.5)
            else:
                # Portrait: features on x-axis, clusters on y-axis
                ax_heatmap.set_xticks(np.arange(n_features))
                ax_heatmap.set_xticklabels(
                    feature_labels_display,
                    fontsize=x_tick_fontsize_eff,
                    rotation=90,
                    ha='right',
                    va='top'
                )
                ax_heatmap.set_xlabel('Features', fontsize=axis_label_fontsize, fontweight='bold', labelpad=2)

                ax_heatmap.set_yticks(np.arange(n_clusters))
                ax_heatmap.set_yticklabels(cluster_labels_display, fontsize=y_tick_fontsize_eff, rotation=0)
                ax_heatmap.set_ylabel('Clusters', fontsize=axis_label_fontsize, fontweight='bold', labelpad=2)

                ax_heatmap.set_xlim(-0.5, n_features - 0.5)
                ax_heatmap.set_ylim(-0.5, n_clusters - 0.5)

            # If a dendrogram sits to the left, move Y labels to the right
            # and keep a spacer column before cbar.
            if left_dendro_visible:
                ax_heatmap.yaxis.tick_right()
                ax_heatmap.yaxis.set_label_position('right')
                ax_heatmap.tick_params(axis='y', labelleft=False, labelright=True, pad=4)
                for lbl in ax_heatmap.get_yticklabels():
                    lbl.set_horizontalalignment('left')
            else:
                ax_heatmap.tick_params(axis='y', labelleft=True, labelright=False, pad=1)
                for lbl in ax_heatmap.get_yticklabels():
                    lbl.set_horizontalalignment('right')
            ax_heatmap.tick_params(axis='x', pad=1)
            # Ensure tick labels remain visible outside the axes bounds.
            for lbl in ax_heatmap.get_yticklabels():
                lbl.set_verticalalignment('center')
                lbl.set_clip_on(False)
            for lbl in ax_heatmap.get_xticklabels():
                lbl.set_clip_on(False)
                lbl.set_rotation_mode('anchor')

            # Compute rightmost rendered y-label position once to keep cbar clear of labels.
            labels_right_fig = ax_heatmap.get_position().x1
            try:
                self.figure.canvas.draw()
                renderer = self.figure.canvas.get_renderer()
                for lbl in ax_heatmap.get_yticklabels():
                    if not lbl.get_visible() or not lbl.get_text():
                        continue
                    bb = lbl.get_window_extent(renderer=renderer)
                    lbl_right = self.figure.transFigure.inverted().transform((bb.x1, bb.y1))[0]
                    labels_right_fig = max(labels_right_fig, lbl_right)
            except Exception:
                pass

            fig_w_in, fig_h_in = self.figure.get_size_inches()
            cbar_right_edge_limit = 0.999
            use_top_dendro_band_for_cbar = top_dendro_visible and cbar_position in {'Upper right', 'Upper left'}

            if use_top_dendro_band_for_cbar:
                cbar_slot_key = 'cbar_slot_right' if cbar_position == 'Upper right' else 'cbar_slot_left'
                cbar_slot_bbox = gs[row_idx['top_dendrogram'], col_idx[cbar_slot_key]].get_position(self.figure)
                slot_x_pad = max(0.002, min(0.010, 0.08 * cbar_slot_bbox.width))
                slot_y_pad = max(0.002, min(0.010, 0.12 * cbar_slot_bbox.height))
                label_gap_fig = 0.006
                cbar_right_limit = min(0.995, cbar_slot_bbox.x1 - slot_x_pad)
                cbar_right_edge_limit = cbar_right_limit
                if cbar_is_horizontal:
                    cbar_h_fig = min(
                        max(0.008, cbar_width_in / max(fig_h_in, 1e-6)),
                        max(0.010, cbar_slot_bbox.height - (2.0 * slot_y_pad))
                    )
                    cbar_w_fig = max(0.05, cbar_slot_bbox.width - (2.0 * slot_x_pad))
                    cbar_y0 = cbar_slot_bbox.y0 + (cbar_slot_bbox.height - cbar_h_fig) / 2.0
                    if cbar_position == 'Upper right':
                        cbar_x_min = max(cbar_slot_bbox.x0 + slot_x_pad, labels_right_fig + label_gap_fig)
                        cbar_w_fig = min(cbar_w_fig, max(0.002, cbar_right_limit - cbar_x_min))
                        cbar_x0 = max(cbar_x_min, cbar_right_limit - cbar_w_fig)
                    else:
                        cbar_x0 = cbar_slot_bbox.x0 + slot_x_pad
                else:
                    cbar_w_fig = min(
                        max(0.008, cbar_width_in / max(fig_w_in, 1e-6)),
                        max(0.010, cbar_slot_bbox.width - (2.0 * slot_x_pad))
                    )
                    cbar_h_fig = max(
                        0.06,
                        min(cbar_slot_bbox.height * 0.92, cbar_slot_bbox.height - (2.0 * slot_y_pad))
                    )
                    cbar_y0 = cbar_slot_bbox.y0 + (cbar_slot_bbox.height - cbar_h_fig) / 2.0
                    if cbar_position == 'Upper right':
                        cbar_x_min = max(cbar_slot_bbox.x0 + slot_x_pad, labels_right_fig + label_gap_fig)
                        cbar_w_fig = min(cbar_w_fig, max(0.002, cbar_right_limit - cbar_x_min))
                        cbar_x0 = max(cbar_x_min, cbar_right_limit - cbar_w_fig)
                    else:
                        cbar_x0 = cbar_slot_bbox.x0 + slot_x_pad
            elif cbar_position == 'Upper right' and not cbar_is_horizontal:
                # Without a top dendrogram row, fall back to a compact corner widget.
                cbar_w_fig = max(0.008, cbar_width_in / max(fig_w_in, 1e-6))
                cbar_h_fig = max(0.10, min(0.20, ax_heatmap.get_position().height * 0.28))
                cbar_right_limit = 0.982
                cbar_x_min = max(ax_heatmap.get_position().x1 + 0.006, labels_right_fig + 0.006)
                cbar_w_fig = min(cbar_w_fig, max(0.008, cbar_right_limit - cbar_x_min))
                cbar_x0 = max(cbar_x_min, cbar_right_limit - cbar_w_fig)
                cbar_right_edge_limit = cbar_right_limit
                top_anchor = ax_heatmap.get_position().y1
                cbar_y0 = top_anchor - cbar_h_fig - 0.004
            else:
                # Horizontal bars and side/upper-left vertical bars use dedicated slots.
                cbar_slot_key = 'cbar_slot_left'
                if cbar_position in {'Upper right', 'Right side'}:
                    cbar_slot_key = 'cbar_slot_right'
                if cbar_slot_key in col_idx:
                    cbar_slot_bbox = gs[row_idx['heatmap'], col_idx[cbar_slot_key]].get_position(self.figure)
                else:
                    cbar_slot_bbox = ax_heatmap.get_position()

                if cbar_is_horizontal:
                    cbar_h_fig = min(
                        max(0.008, cbar_width_in / max(fig_h_in, 1e-6)),
                        max(0.010, cbar_slot_bbox.height - 0.018)
                    )
                    slot_x_pad = max(0.002, min(0.012, 0.08 * cbar_slot_bbox.width))
                    if cbar_position == 'Right side':
                        cbar_w_fig = max(0.05, cbar_slot_bbox.width - (2.0 * slot_x_pad))
                        cbar_y0 = cbar_slot_bbox.y0 + (cbar_slot_bbox.height - cbar_h_fig) / 2.0
                        cbar_x0 = cbar_slot_bbox.x0 + slot_x_pad
                    else:
                        cbar_w_fig = max(0.05, cbar_slot_bbox.width - (2.0 * slot_x_pad))
                        cbar_y0 = cbar_slot_bbox.y1 - cbar_h_fig - 0.004
                        if cbar_position == 'Upper right':
                            cbar_x0 = cbar_slot_bbox.x1 - slot_x_pad - cbar_w_fig
                        else:
                            cbar_x0 = cbar_slot_bbox.x0 + slot_x_pad
                    cbar_right_limit = min(0.995, cbar_slot_bbox.x1 - 0.002)
                    cbar_w_fig = min(cbar_w_fig, max(0.05, cbar_right_limit - cbar_x0))
                    cbar_right_edge_limit = cbar_right_limit
                else:
                    cbar_w_fig = min(cbar_slot_bbox.width * 0.74, max(0.008, cbar_width_in / max(fig_w_in, 1e-6)))
                    if cbar_position == 'Right side':
                        cbar_h_fig = max(0.16, min(cbar_slot_bbox.height * 0.82, cbar_slot_bbox.height - 0.02))
                        cbar_y0 = cbar_slot_bbox.y0 + (cbar_slot_bbox.height - cbar_h_fig) / 2.0
                        gap = 0.006
                        cbar_x0 = max(cbar_slot_bbox.x0 + 0.002, labels_right_fig + gap)
                        cbar_right_limit = min(0.982, cbar_slot_bbox.x1 - 0.002)
                        cbar_w_fig = min(cbar_w_fig, max(0.008, cbar_right_limit - cbar_x0))
                        cbar_right_edge_limit = cbar_right_limit
                    else:
                        cbar_h_fig = max(0.10, min(cbar_slot_bbox.height * 0.28, cbar_slot_bbox.height - 0.015))
                        cbar_y0 = cbar_slot_bbox.y1 - cbar_h_fig - 0.004
                        cbar_x0 = cbar_slot_bbox.x0 + max(0.0015, 0.06 * cbar_slot_bbox.width)
                        cbar_right_limit = min(0.995, cbar_slot_bbox.x1 - 0.002)
                        cbar_w_fig = min(cbar_w_fig, max(0.008, cbar_right_limit - cbar_x0))

            cbar_x0 = max(0.001, min(cbar_x0, cbar_right_edge_limit - cbar_w_fig))
            cbar_y0 = max(0.001, min(cbar_y0, 0.999 - cbar_h_fig))

            ax_cbar = self.figure.add_axes([cbar_x0, cbar_y0, cbar_w_fig, cbar_h_fig], zorder=5)
            ax_cbar.set_facecolor((1.0, 1.0, 1.0, 0.85))
            for spine in ax_cbar.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor('#666666')
            if cbar_is_horizontal:
                cbar = self.figure.colorbar(im, cax=ax_cbar, orientation='horizontal')
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')
                cbar_tick_fontsize = max(6, min(7, int(x_tick_fontsize_eff)))
                cbar.ax.tick_params(
                    labelsize=cbar_tick_fontsize,
                    length=2,
                    width=0.6,
                    top=True,
                    labeltop=True,
                    bottom=False,
                    labelbottom=False
                )
                try:
                    cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                except Exception:
                    pass
            else:
                cbar = self.figure.colorbar(im, cax=ax_cbar, orientation='vertical')
                cbar.ax.yaxis.set_ticks_position('right')
                cbar.ax.yaxis.set_label_position('right')
                cbar_tick_fontsize = max(6, min(7, int(y_tick_fontsize_eff)))
                cbar.ax.tick_params(
                    labelsize=cbar_tick_fontsize,
                    length=2,
                    width=0.6,
                    right=True,
                    labelright=True,
                    left=False,
                    labelleft=False
                )
                try:
                    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4 if cbar_position == 'Right side' else 3))
                except Exception:
                    pass
            # Keep Cluster Map colorbar unlabeled by request; show only the scale bar itself.

            # Add clustermap-style cluster annotation strip in dedicated axis.
            if is_landscape:
                ax_cluster_strip = self.figure.add_subplot(gs[row_idx['cluster_strip'], col_idx['heatmap']])
                ax_cluster_strip.imshow(cluster_strip.reshape(1, -1, 3), aspect='auto', interpolation='nearest')
                ax_cluster_strip.set_xlim(-0.5, n_clusters - 0.5)
            else:
                ax_cluster_strip = self.figure.add_subplot(gs[row_idx['heatmap'], col_idx['cluster_strip']])
                ax_cluster_strip.imshow(cluster_strip.reshape(-1, 1, 3), aspect='auto', interpolation='nearest')
                ax_cluster_strip.set_ylim(-0.5, n_clusters - 0.5)

            ax_cluster_strip.set_xticks([])
            ax_cluster_strip.set_yticks([])
            for spine in ax_cluster_strip.spines.values():
                spine.set_visible(False)

            # Legend for cluster strip in dedicated axis (when enabled).
            if show_cluster_legend and 'legend' in col_idx:
                legend_items = []
                for cid in cluster_order:
                    legend_items.append(
                        plt.Rectangle(
                            (0, 0), 1, 1,
                            facecolor=cluster_color_map[cid],
                            edgecolor='black',
                            linewidth=0.4,
                            label=self._get_cluster_display_name(cid),
                        )
                    )

                if legend_items:
                    legend_fontsize = getattr(self, 'legend_fontsize', 8)
                    legend_ncols = getattr(self, 'legend_ncols', 1)
                    legend_nrows = getattr(self, 'legend_nrows', 1)
                    n_items = len(legend_items)
                    if legend_nrows > 1:
                        ncol = max(1, (n_items + legend_nrows - 1) // legend_nrows)
                    else:
                        ncol = max(1, min(legend_ncols, n_items))

                    ax_legend = self.figure.add_subplot(gs[row_idx['heatmap'], col_idx['legend']])
                    ax_legend.set_axis_off()
                    ax_legend.legend(
                        handles=legend_items,
                        loc='center left',
                        frameon=True,
                        fontsize=legend_fontsize,
                        title='Clusters',
                        title_fontsize=legend_fontsize + 1,
                        ncol=ncol,
                    )

            # Remove spines for cleaner look
            for spine in ax_heatmap.spines.values():
                spine.set_visible(False)

            # First pass: measure real text extents, then reserve just enough margin so
            # the live GUI view keeps labels inside the window without hurting export.
            if layout_pass + 1 < max_layout_passes:
                collision_needs_retry = False
                try:
                    self.figure.canvas.draw()
                    renderer = self.figure.canvas.get_renderer()
                    rightmost_ylabel_px = ax_heatmap.bbox.x1
                    for lbl in ax_heatmap.get_yticklabels():
                        if not lbl.get_visible() or not lbl.get_text():
                            continue
                        rightmost_ylabel_px = max(rightmost_ylabel_px, lbl.get_window_extent(renderer=renderer).x1)

                    cbar_bbox = ax_cbar.bbox
                    min_gap_px = 2.0
                    horizontal_shortfall_px = max(0.0, (rightmost_ylabel_px + min_gap_px) - cbar_bbox.x0)
                    vertical_shortfall_px = max(0.0, (ax_heatmap.bbox.y1 + min_gap_px) - cbar_bbox.y0)

                    if horizontal_shortfall_px > 0.0:
                        label_spacer_extra_in = min(
                            3.0,
                            label_spacer_extra_in + (horizontal_shortfall_px / max(dpi, 1.0)) + 0.05,
                        )
                        collision_needs_retry = True

                    if cbar_position in {'Upper right', 'Upper left'} and vertical_shortfall_px > 0.0:
                        top_band_extra_in = min(
                            1.5,
                            top_band_extra_in + (vertical_shortfall_px / max(dpi, 1.0)) + 0.04,
                        )
                        collision_needs_retry = True
                except Exception:
                    pass

                if collision_needs_retry:
                    continue

                overflow = self._measure_figure_text_overflow()
                max_overflow = max(overflow.values())
                if max_overflow > 0.0035:
                    layout_margin_adjust['left'] = min(0.22, layout_margin_adjust['left'] + overflow['left'] + 0.010)
                    layout_margin_adjust['right'] = min(0.24, layout_margin_adjust['right'] + overflow['right'] + 0.010)
                    layout_margin_adjust['bottom'] = min(0.30, layout_margin_adjust['bottom'] + overflow['bottom'] + 0.012)
                    layout_margin_adjust['top'] = min(0.08, layout_margin_adjust['top'] + overflow['top'] + 0.008)
                    side_overflow = max(overflow['left'], overflow['right'])
                    if overflow['bottom'] > 0.006 and x_tick_fontsize_eff > 6.0:
                        x_tick_fontsize_eff = max(6.0, x_tick_fontsize_eff * (1.0 - min(0.35, overflow['bottom'] * 1.8)))
                    if side_overflow > 0.006 and y_tick_fontsize_eff > 6.0:
                        y_tick_fontsize_eff = max(6.0, y_tick_fontsize_eff * (1.0 - min(0.35, side_overflow * 2.0)))
                    continue

            break

        self._finalize_standard_plot(
            pad=0.6,
            allow_text_compaction=True,
            force_layout_refresh=True,
        )
    
    def _create_seaborn_heatmap(self):
        """Create heatmap using seaborn clustermap with color bars."""
        try:
            # Check if clustered_data exists
            if self.clustered_data is None:
                self._create_matplotlib_heatmap()
                return
            scaling_method = self._get_heatmap_scaling_method()
            
            # Use unscaled data if available, otherwise use clustered_data
            base_data = self.clustered_data_unscaled if self.clustered_data_unscaled is not None else self.clustered_data
            
            # Prepare data considering source/filter
            source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
            data_to_plot = base_data.copy()
            group_col = 'cluster'
            if source == 'Manual Gates' and 'manual_phenotype' in data_to_plot.columns:
                groups = self._get_manual_groups_series()
                if groups is not None:
                    data_to_plot['__group__'] = groups.values
                    group_col = '__group__'
                    if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                        data_to_plot = self._apply_heatmap_filter(data_to_plot, group_col)
                    data_to_plot = data_to_plot.sort_values(group_col)
            else:
                if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                    wanted_ids = set()
                    for cid in self._sorted_cluster_ids(base_data['cluster'].unique(), canonical=False):
                        name = self._get_cluster_display_name(cid)
                        if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                            wanted_ids.add(cid)
                    if wanted_ids:
                        data_to_plot = data_to_plot[data_to_plot['cluster'].isin(sorted(wanted_ids))]
                data_to_plot = data_to_plot.sort_values('cluster')

            feature_cols = self._select_feature_columns(data_to_plot)
            
            # Apply selected scaling to feature data
            feature_data = data_to_plot[feature_cols].copy()
            feature_data_scaled = self._apply_scaling(feature_data, scaling_method)
            feature_data_scaled = feature_data_scaled.fillna(0)  # Handle any NaN from scaling
            
            heatmap_data = feature_data_scaled
            
            # Store original feature order for y-tick labels
            original_feature_order = list(feature_cols)
            
            # Create group color mapping
            unique_groups = sorted(data_to_plot[group_col].unique())
            # Use vivid colors instead of Set3
            cluster_colors_raw = _get_vivid_colors(len(unique_groups))
            # Convert to RGB tuples for seaborn (remove alpha channel)
            cluster_colors = [tuple(c[:3]) for c in cluster_colors_raw]
            cluster_color_map = {gid: cluster_colors[i] for i, gid in enumerate(unique_groups)}
            
            # Create color series for color bar
            cluster_colors_series = data_to_plot[group_col].map(cluster_color_map)
            
            # Determine clustering settings based on method
            # Use the actual clustering method that was used when clustering was run, not the current UI selection
            clustering_type = self.actual_clustering_method if hasattr(self, 'actual_clustering_method') and self.actual_clustering_method is not None else (self.clustering_type.currentText() if hasattr(self, 'clustering_type') else 'Hierarchical')
            is_leiden = clustering_type == "Leiden"
            is_louvain = clustering_type == "Louvain"
            is_hdbscan = clustering_type == "HDBSCAN"
            
            if is_leiden or is_louvain or is_hdbscan:
                # For Leiden, Louvain, and HDBSCAN clustering, disable dendrograms
                row_cluster = False
                col_cluster = False
                linkage_method = None
            else:
                # For hierarchical clustering, use dendrograms
                row_cluster = True
                # Use the actual dendrogram mode that was used when clustering was run, not the current UI selection
                dendro_mode = self.actual_dendrogram_mode if hasattr(self, 'actual_dendrogram_mode') and self.actual_dendrogram_mode is not None else (self.dendro_mode.currentText() if hasattr(self, 'dendro_mode') else "Rows and columns")
                col_cluster = (dendro_mode == "Rows and columns")
                linkage_method = self.hierarchical_method.currentText()
            
            # Get canvas size to determine appropriate figure size
            canvas_width = self.canvas.width()
            canvas_height = self.canvas.height()
            # Convert pixels to inches (assuming 100 DPI)
            fig_width = max(8, canvas_width / 100)
            fig_height = max(6, canvas_height / 100)
            
            # Create clustermap with appropriate parameters
            colormap_name = self._get_colormap_name()
            g = sns.clustermap(
                heatmap_data.T,  # Transpose for features as rows, cells as columns
                cmap=colormap_name,
                row_cluster=row_cluster,
                col_cluster=col_cluster,
                method=linkage_method,
                metric='euclidean',
                cbar_kws={'label': 'Normalized Feature Value'},
                figsize=(fig_width, fig_height),  # Dynamic figure size based on canvas
                col_colors=cluster_colors_series  # This creates the color bar
            )
            
            # Labels – let seaborn manage tick order after clustering; just style
            g.ax_heatmap.set_xlabel('Cells')
            g.ax_heatmap.set_ylabel('Features')
            
            # Force all row tick labels to show (features)
            # Use reordered features if row clustering is enabled, otherwise use original order
            if row_cluster:
                # When row clustering is enabled, use the reordered feature names
                feature_labels = g.ax_heatmap.get_yticklabels()
                feature_names = [label.get_text() for label in feature_labels]
            else:
                # When row clustering is disabled, use original feature order
                feature_names = original_feature_order
            
            # Ensure all feature labels are displayed - disable automatic tick limiting
            g.ax_heatmap.set_yticks(range(len(feature_names)))
            # Use custom feature labels if available
            feature_labels_display = [self._get_feature_display_name(f) for f in feature_names]
            g.ax_heatmap.set_yticklabels(feature_labels_display, fontsize=self.feature_tick_fontsize, minor=False)
            # Prevent matplotlib from automatically hiding overlapping labels
            g.ax_heatmap.tick_params(axis='y', which='major', labelsize=self.feature_tick_fontsize)
            # Ensure y-axis limits show all features
            g.ax_heatmap.set_ylim(-0.5, len(feature_names) - 0.5)
            # Force all labels to be visible
            for label in g.ax_heatmap.get_yticklabels():
                label.set_visible(True)
            
            # Remove column tick labels (cells)
            g.ax_heatmap.set_xticks([])
            g.ax_heatmap.set_xticklabels([])
            
            # Add legend
            self._add_cluster_legend(g, cluster_color_map, source=source)
            
            # Replace the figure with the seaborn figure
            old_figure = self.figure
            self.figure = g.fig
            self.canvas.figure = self.figure
            
            # Use tight layout to maximize plot area
            # Avoid tight_layout on clustermap to prevent warnings
            
            # Close the old figure to free memory
            plt.close(old_figure)
            
            # Force canvas update
            self.canvas.draw()
            
        except Exception as e:
            # Fall back to matplotlib implementation
            self._create_matplotlib_heatmap()
    
    def _create_matplotlib_heatmap(self):
        """Fallback heatmap using matplotlib (original implementation)."""
        # Check if clustered_data exists
        if self.clustered_data is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No clustered data available.\nPlease run clustering first.", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Heatmap")
            self.canvas.draw()
            return
        
        scaling_method = self._get_heatmap_scaling_method()
        
        # Use unscaled data if available, otherwise use clustered_data
        base_data = self.clustered_data_unscaled if self.clustered_data_unscaled is not None else self.clustered_data
        
        # Filter out dropped clusters (cluster 0)
        try:
            # Ensure cluster column is integer before comparison
            if base_data['cluster'].dtype == bool:
                base_data['cluster'] = base_data['cluster'].astype(int)
            elif base_data['cluster'].dtype.name.startswith('object'):
                base_data['cluster'] = pd.to_numeric(base_data['cluster'], errors='coerce').fillna(0).astype(int)
            else:
                base_data['cluster'] = base_data['cluster'].astype(int)
            # Now do the comparison - filter out cluster 0 (excluded/filtered cells)
            mask = base_data['cluster'] != 0
            base_data = base_data[mask].copy()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
        # Determine source and prepare data ordering and optional grouping
        source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
        data_to_plot = base_data.copy()
        group_col = 'cluster'
        if source == 'Manual Gates' and 'manual_phenotype' in data_to_plot.columns:
            groups = self._get_manual_groups_series()
            if groups is not None:
                data_to_plot = data_to_plot.copy()
                data_to_plot['__group__'] = groups.values
                group_col = '__group__'
                # Apply filter by names if set
                if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                    data_to_plot = self._apply_heatmap_filter(data_to_plot, group_col)
                # Sort by group label
                data_to_plot = data_to_plot.sort_values(group_col)
            else:
                group_col = 'cluster'
        else:
            # Clusters source: optionally filter by selected clusters (by display name or id)
            if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                wanted_ids = set()
                for cid in self._sorted_cluster_ids(base_data['cluster'].unique(), canonical=False):
                    name = self._get_cluster_display_name(cid)
                    if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                        wanted_ids.add(cid)
                if wanted_ids:
                    data_to_plot = data_to_plot[data_to_plot['cluster'].isin(sorted(wanted_ids))]
            data_to_plot = data_to_plot.sort_values('cluster')

        # Create subplots - simplified layout without cluster size bar
        gs = self.figure.add_gridspec(1, 1, hspace=0.1, wspace=0.1)
        
        # Main heatmap - use full figure area
        ax_heatmap = self.figure.add_subplot(gs[0])
        
        # Prepare feature columns before scaling
        feature_cols = self._select_feature_columns(data_to_plot)
        
        # Apply selected scaling to feature data
        feature_data = data_to_plot[feature_cols].copy()
        feature_data_scaled = self._apply_scaling(feature_data, scaling_method)
        feature_data_scaled = feature_data_scaled.fillna(0)  # Handle any NaN from scaling
        
        heatmap_data = feature_data_scaled.values

        # No dendrograms - just show the heatmap data as-is
        
        # Create heatmap with user-selected colormap
        colormap_name = self._get_colormap_name()
        im = ax_heatmap.imshow(heatmap_data.T, aspect='auto', cmap=colormap_name, interpolation='nearest')
        
        # Set labels and ticks
        ax_heatmap.set_xlabel('Cells')
        ax_heatmap.set_ylabel('Features')
        # Ensure all feature labels are displayed - disable automatic tick limiting
        ax_heatmap.set_yticks(np.arange(len(feature_cols)))
        # Use custom feature labels if available
        feature_labels_display = [self._get_feature_display_name(f) for f in feature_cols]
        ax_heatmap.set_yticklabels(feature_labels_display, fontsize=self.feature_tick_fontsize, rotation=0)
        # Prevent matplotlib from automatically hiding overlapping labels
        ax_heatmap.tick_params(axis='y', which='major', labelsize=self.feature_tick_fontsize)
        # Force all labels to be visible
        for label in ax_heatmap.get_yticklabels():
            label.set_visible(True)
        
        # Remove x-axis tick labels (cluster identity shown via color bar instead)
        ax_heatmap.set_xticks([])
        
        
        # Add group color bars along x-axis
        unique_groups = sorted(data_to_plot[group_col].unique())
        cluster_colors = _get_vivid_colors(len(unique_groups))
        cluster_color_map = {gid: cluster_colors[i] for i, gid in enumerate(unique_groups)}
        
        # Create color bar for each cell
        cell_colors = [cluster_color_map[val] for val in data_to_plot[group_col]]
        
        # Add color bar below the heatmap
        for i, color in enumerate(cell_colors):
            ax_heatmap.axvline(x=i, ymin=-0.05, ymax=0, color=color, linewidth=1, solid_capstyle='butt')
        
        # Adjust y-axis to make room for color bar
        ax_heatmap.set_ylim(-0.5, len(feature_cols) - 0.5)
        
        # Colorbar
        cbar = self.figure.colorbar(im, ax=ax_heatmap, shrink=0.8)
        cbar.set_label('Normalized Feature Value')
        
        # Row dendrogram (top-left)
        # No row dendrogram - just the heatmap
        
        # Cluster size bar removed - using color bars for cluster identity instead

        # Add legend for groups/clusters - horizontal at the top
        legend_elements = []
        if source == 'Manual Gates':
            for key, color in cluster_color_map.items():
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=str(key)))
        else:
            for key, color in cluster_color_map.items():
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=self._get_cluster_display_name(key)))
        # Place legend horizontally at the top of the figure
        self.figure.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                          ncol=min(len(legend_elements), 10), frameon=True, fontsize=8)
        
        # Adjust layout to account for legend at top - add extra top padding
        self._finalize_standard_plot(rect=[0, 0, 1, 0.95], pad=0.9, allow_text_compaction=True)
    
    def _add_cluster_legend(self, g, cluster_color_map, source='Clusters'):
        """Add legend to seaborn clustermap using cluster names or manual group labels."""
        legend_elements = []
        for key, color in cluster_color_map.items():
            if source == 'Manual Gates':
                label = str(key)
            else:
                label = self._get_cluster_display_name(key)
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label))
        # Place legend horizontally at the top of the figure
        g.fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                     ncol=min(len(legend_elements), 10), frameon=True, fontsize=8)

    def _confirm_large_umap_run(self, n_cells: int) -> bool:
        """Warn before running UMAP on very large datasets."""
        threshold = int(getattr(self, 'umap_large_dataset_threshold', 250000))
        if n_cells < threshold:
            return True

        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Large UMAP Dataset")
        msg.setText(f"You are about to run UMAP on {n_cells:,} cells.")
        msg.setInformativeText(
            "This can take a long time, and rendering may appear unresponsive for very large datasets.\n\n"
            "Consider filtering or downsampling cells before running UMAP."
        )
        continue_btn = msg.addButton("Continue Anyway", QtWidgets.QMessageBox.AcceptRole)
        msg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        msg.setDefaultButton(continue_btn)
        msg.exec_()
        return msg.clickedButton() is continue_btn

    def _warn_large_umap_plot_once(self, n_cells: int):
        """Show a one-time warning when rendering very large UMAP scatter plots."""
        threshold = int(getattr(self, 'umap_large_dataset_threshold', 250000))
        if n_cells < threshold:
            return
        if getattr(self, '_last_umap_plot_warning_n', None) == int(n_cells):
            return
        self._last_umap_plot_warning_n = int(n_cells)
        QtWidgets.QMessageBox.information(
            self,
            "Large UMAP Plot",
            (
                f"Rendering {n_cells:,} UMAP points may be slow.\n\n"
                "If interaction is sluggish, reduce cell count with filters before UMAP."
            ),
        )

    def _run_umap(self):
        """Run UMAP dimensionality reduction analysis."""
        try:
            if not _HAVE_UMAP:
                QtWidgets.QMessageBox.warning(self, "UMAP Not Available", 
                    "UMAP is not installed. Please install umap-learn to use this feature.")
                return
            
            # Get feature selection from user
            available_cols = self._list_available_feature_columns(True)  # Include morphometric features
            from openimc.ui.dialogs.feature_selector_dialog import FeatureSelectorDialog
            selector = FeatureSelectorDialog(available_cols, self)
            # Pre-populate filter settings if available
            if self.filter_settings is not None:
                selector.set_filter_settings(self.filter_settings)
            if selector.exec_() != QtWidgets.QDialog.Accepted:
                return
            selected_columns = selector.get_selected_columns()
            
            if not selected_columns:
                QtWidgets.QMessageBox.warning(self, "No Features", "Please select at least one feature for UMAP analysis.")
                return
            
            # Get filter settings (use stored settings if available, otherwise get from dialog)
            filter_settings = self.filter_settings
            if filter_settings is None:
                filter_settings = selector.get_filter_settings()
            
            # Apply filters to feature dataframe
            filtered_df = self._apply_filters(self.feature_dataframe.copy(), filter_settings)
            if filtered_df.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No cells remain after applying filters.")
                return
            
            # Prepare data for UMAP, align with clustered order if available
            if self.clustered_data is not None:
                # Use intersection of filtered data and clustered data indices
                ordered_index = self.clustered_data.index.intersection(filtered_df.index)
                data = filtered_df.loc[ordered_index, selected_columns].copy()
            else:
                data = filtered_df[selected_columns].copy()
            
            # Handle missing values and infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.median())
            
            if data.empty or data.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for UMAP analysis.")
                return
            
            # Apply percentile censoring if enabled (before scaling)
            data = self._apply_percentile_censoring(data, filter_settings)
            
            # Ensure all columns are numeric (float64) to avoid boolean subtraction issues
            for col in data.columns:
                if data[col].dtype == bool:
                    data[col] = data[col].astype(int).astype(np.float64)
                elif not np.issubdtype(data[col].dtype, np.number):
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)
                    except (ValueError, TypeError):
                        data = data.drop(columns=[col])
            
            # Clear canvas before UMAP
            self.figure.clear()
            self.canvas.draw()
            
            # Allow user to choose scaling method
            scaling_options = ["None (no scaling)", "Z-score", "MAD (Median Absolute Deviation)"]
            # Default to clustering scaling method if available
            default_index = 0
            if (hasattr(self, 'clustering_scaling_method') and 
                self.clustering_scaling_method is not None and 
                self.clustering_scaling_method in scaling_options):
                default_index = scaling_options.index(self.clustering_scaling_method)
            
            scaling_method, ok = QtWidgets.QInputDialog.getItem(
                self,
                "UMAP Feature Scaling",
                "Select scaling method for features:",
                scaling_options,
                current=default_index,  # Default to clustering scaling method
                editable=False
            )
            if not ok:
                return
            
            # Map selection to method string
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            selected_scaling = scaling_map[scaling_method]
            
            # Apply scaling
            data_scaled = self._apply_scaling(data, selected_scaling)
            
            # Handle any NaN values that might have been introduced
            data_scaled = data_scaled.fillna(0)
            
            if data_scaled.empty or data_scaled.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for UMAP analysis after scaling.")
                return

            # Warn users before expensive UMAP runs on very large datasets.
            if not self._confirm_large_umap_run(int(data_scaled.shape[0])):
                return
            
            max_n = max(2, data_scaled.shape[0] - 1)
            default_n = self.n_neighbors_spinbox.value() if hasattr(self, "n_neighbors_spinbox") else 15
            n_neighbors, ok = self._prompt_embedding_parameter(
                title="UMAP n_neighbors",
                method_name="UMAP",
                parameter_name="n_neighbors",
                minimum=2,
                maximum=max_n,
                default_value=default_n,
            )
            if not ok:
                return
            seed, ok = self._prompt_embedding_seed(method_name="UMAP")
            if not ok:
                return

            def _umap_task():
                reducer = umap.UMAP(
                    n_components=2,
                    random_state=seed,
                    n_neighbors=int(n_neighbors),
                    min_dist=0.1,
                )
                return reducer.fit_transform(data_scaled.values)

            self.umap_embedding = run_blocking_task_with_progress(
                parent=self,
                window_title="UMAP In Progress",
                initial_message="Running UMAP embedding",
                detail_text=(
                    f"Cells: {data_scaled.shape[0]:,} | Features: {data_scaled.shape[1]}\n"
                    "Rendering the plot may still take additional time."
                ),
                task=_umap_task,
            )
            # Persist for coloring
            self.umap_index = data.index.to_list()
            self.umap_selected_columns = list(selected_columns)
            self.umap_raw_data = data.copy()
            
            # Create UMAP plot
            self._create_umap_plot()
            
            # Force canvas refresh
            self.canvas.draw()
            
            # Populate color-by options
            self._populate_color_by_options()
            # Enable save button since a plot is shown
            self.save_plot_btn.setEnabled(True)
            self.save_output_btn.setEnabled(True)
            
            # Update statistical cluster combo if it exists
            if hasattr(self, 'stats_cluster_combo'):
                self._update_stats_cluster_combo()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UMAP Error", f"Error during UMAP analysis: {str(e)}")
    
    def _remake_umap(self):
        """Remake UMAP with new parameters. This allows users to easily regenerate UMAP."""
        # Simply call _run_umap which will prompt for new parameters
        self._run_umap()

    def _prompt_embedding_parameter(
        self,
        *,
        title: str,
        method_name: str,
        parameter_name: str,
        minimum: int,
        maximum: int,
        default_value: int,
    ):
        """Prompt for an embedding integer parameter with consistent wording."""
        clamped_default = max(minimum, min(int(default_value), maximum))
        return QtWidgets.QInputDialog.getInt(
            self,
            title,
            f"Set {method_name} {parameter_name} ({minimum}-{maximum}).",
            value=clamped_default,
            min=minimum,
            max=maximum,
        )

    def _prompt_embedding_seed(self, *, method_name: str):
        """Prompt for the random seed used by an embedding method."""
        return QtWidgets.QInputDialog.getInt(
            self,
            f"{method_name} Random Seed",
            (
                f"Set the random seed for {method_name}.\n\n"
                "This controls reproducibility for the embedding run."
            ),
            value=self.seed_spinbox.value(),
            min=0,
            max=2**31 - 1,
        )
    
    def _run_tsne(self):
        """Run t-SNE dimensionality reduction."""
        if not _HAVE_TSNE:
            QtWidgets.QMessageBox.warning(self, "t-SNE Not Available", "scikit-learn is required for t-SNE.")
            return
        
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No feature data available for t-SNE.")
            return
        
        try:
            # Get feature selection from user (same as UMAP)
            available_cols = self._list_available_feature_columns(True)  # Include morphometric features
            from openimc.ui.dialogs.feature_selector_dialog import FeatureSelectorDialog
            selector = FeatureSelectorDialog(available_cols, self)
            # Pre-populate filter settings if available
            if self.filter_settings is not None:
                selector.set_filter_settings(self.filter_settings)
            if selector.exec_() != QtWidgets.QDialog.Accepted:
                return
            selected_columns = selector.get_selected_columns()
            
            if not selected_columns:
                QtWidgets.QMessageBox.warning(self, "No Features", "Please select at least one feature for t-SNE analysis.")
                return
            
            # Get filter settings (use stored settings if available, otherwise get from dialog)
            filter_settings = self.filter_settings
            if filter_settings is None:
                filter_settings = selector.get_filter_settings()
            
            # Apply filters to feature dataframe
            filtered_df = self._apply_filters(self.feature_dataframe.copy(), filter_settings)
            if filtered_df.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No cells remain after applying filters.")
                return
            
            # Prepare data for t-SNE, align with clustered order if available
            if self.clustered_data is not None:
                # Use intersection of filtered data and clustered data indices
                ordered_index = self.clustered_data.index.intersection(filtered_df.index)
                data = filtered_df.loc[ordered_index, selected_columns].copy()
            else:
                data = filtered_df[selected_columns].copy()
            
            # Handle missing values and infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.median())
            
            if data.empty or data.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for t-SNE analysis.")
                return
            
            # Apply percentile censoring if enabled (before scaling)
            data = self._apply_percentile_censoring(data, filter_settings)
            
            # Ensure all columns are numeric (float64) to avoid boolean subtraction issues
            for col in data.columns:
                if data[col].dtype == bool:
                    data[col] = data[col].astype(int).astype(np.float64)
                elif not np.issubdtype(data[col].dtype, np.number):
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)
                    except (ValueError, TypeError):
                        data = data.drop(columns=[col])
            
            # Clear canvas before t-SNE
            self.figure.clear()
            self.canvas.draw()
            
            # Allow user to choose scaling method
            scaling_options = ["None (no scaling)", "Z-score", "MAD (Median Absolute Deviation)"]
            # Default to clustering scaling method if available
            default_index = 0
            if (hasattr(self, 'clustering_scaling_method') and 
                self.clustering_scaling_method is not None and 
                self.clustering_scaling_method in scaling_options):
                default_index = scaling_options.index(self.clustering_scaling_method)
            
            scaling_method, ok = QtWidgets.QInputDialog.getItem(
                self,
                "t-SNE Feature Scaling",
                "Select scaling method for features:",
                scaling_options,
                current=default_index,  # Default to clustering scaling method
                editable=False
            )
            if not ok:
                return
            
            # Map selection to method string
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            selected_scaling = scaling_map[scaling_method]
            
            # Apply scaling
            data_scaled = self._apply_scaling(data, selected_scaling)
            
            # Handle any NaN values that might have been introduced
            data_scaled = data_scaled.fillna(0)
            
            if data_scaled.empty or data_scaled.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for t-SNE analysis after scaling.")
                return
            
            max_perplexity = min(30, data_scaled.shape[0] - 1)
            perplexity, ok = self._prompt_embedding_parameter(
                title="t-SNE Perplexity",
                method_name="t-SNE",
                parameter_name="perplexity",
                minimum=5,
                maximum=max_perplexity,
                default_value=getattr(self, "_last_tsne_perplexity", min(30, max_perplexity)),
            )
            if not ok:
                return
            self._last_tsne_perplexity = perplexity
            seed, ok = self._prompt_embedding_seed(method_name="t-SNE")
            if not ok:
                return
            
            def _tsne_task():
                return TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=seed,
                    max_iter=1000,
                ).fit_transform(data_scaled.values)

            self.tsne_embedding = run_blocking_task_with_progress(
                parent=self,
                window_title="t-SNE In Progress",
                initial_message="Running t-SNE embedding",
                detail_text=(
                    f"Cells: {data_scaled.shape[0]:,} | Features: {data_scaled.shape[1]}\n"
                    "Rendering the plot may still take additional time."
                ),
                task=_tsne_task,
            )
            
            # Persist for coloring
            self.tsne_index = data.index.to_list()
            self.tsne_selected_columns = list(selected_columns)
            self.tsne_raw_data = data.copy()
            
            # Create plot
            self._create_tsne_plot()
            
            # Populate color-by options
            self._populate_color_by_options()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "t-SNE Error", f"Error during t-SNE analysis: {str(e)}")
    
    def _plot_tsne_single(self, ax, color_by, point_size, point_alpha, show_legend=True, title=None):
        """Plot a single t-SNE subplot with specified coloring."""
        if color_by == 'Cluster' and self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
            # Align cluster labels to t-SNE order
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            
            # Filter out dropped clusters (cluster 0)
            valid_mask = cluster_labels != 0
            cluster_labels = cluster_labels[valid_mask]
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            unique_clusters = self._sorted_cluster_ids(np.unique(cluster_labels), canonical=False)
            colors = _get_vivid_colors(len(unique_clusters))
            cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
            handles = []
            labels = []
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                c=[cluster_color_map[cluster_id]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size (18)
                color = cluster_color_map[cluster_id]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(self._get_cluster_display_name(cluster_id))
            # Use multiple columns if there are more than 10 clusters
            n_clusters = len(handles)
            ncol = max(1, (n_clusters + 9) // 10) if n_clusters > 10 else 1
            if show_legend:
                ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, ncol=ncol)
        elif color_by == 'Source File' and 'source_file' in self.feature_dataframe.columns:
            # Filter out dropped clusters (cluster 0) for source file coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Color by source file to visualize batch effects (using custom patient labels)
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                source_file_series = self.feature_dataframe.loc[self.tsne_index, 'source_file']
                source_files = source_file_series.values[valid_mask]
            else:
                source_files = self.feature_dataframe['source_file'].values[valid_mask]
            unique_files = sorted([f for f in np.unique(source_files) if pd.notna(f)])
            if len(unique_files) > 0:
                # Check if cohorts are defined and cohort coloring is enabled
                cohorts_used = set()
                if self.use_cohort_coloring:
                    for file_name in unique_files:
                        if file_name in self.patient_cohort_map:
                            cohorts_used.add(self.patient_cohort_map[file_name])
                
                if cohorts_used:
                    # Use cohort-based coloring
                    unique_cohorts = sorted(cohorts_used)
                    cohort_colors_raw = _get_patient_colors(len(unique_cohorts))
                    cohort_color_map = {}
                    for i, cohort_name in enumerate(unique_cohorts):
                        color = cohort_colors_raw[i]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        cohort_color_map[cohort_name] = rgb
                        self.cohort_colors[cohort_name] = rgb
                    
                    # Map files to cohort colors
                    file_color_map = {}
                    unassigned_files = [f for f in unique_files if f not in self.patient_cohort_map]
                    if unassigned_files:
                        unassigned_colors_raw = _get_patient_colors(len(unassigned_files))
                        for i, file_name in enumerate(unassigned_files):
                            color = unassigned_colors_raw[i]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            file_color_map[file_name] = rgb
                    
                    for file_name in unique_files:
                        if file_name in self.patient_cohort_map:
                            cohort = self.patient_cohort_map[file_name]
                            file_color_map[file_name] = cohort_color_map[cohort]
                    
                    # Plot by cohort
                    handles = []
                    labels = []
                    # Plot cohorts first
                    for cohort_name in unique_cohorts:
                        cohort_files = [f for f in unique_files if f in self.patient_cohort_map and self.patient_cohort_map[f] == cohort_name]
                        if cohort_files:
                            combined_mask = np.zeros(len(source_files), dtype=bool)
                            for file_name in cohort_files:
                                combined_mask |= (source_files == file_name)
                            combined_mask = combined_mask[valid_mask]
                            if np.any(combined_mask):
                                color = cohort_color_map[cohort_name]
                                sc = ax.scatter(tsne_embedding_filtered[combined_mask, 0], tsne_embedding_filtered[combined_mask, 1],
                                                c=[color], alpha=point_alpha, s=point_size, edgecolors='none')
                                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                             markeredgecolor='none', markersize=6, alpha=point_alpha)
                                handles.append(handle)
                                if len(cohort_files) <= 3:
                                    patient_labels = [self._get_patient_display_name(f) for f in cohort_files]
                                    labels.append(f"{cohort_name} ({', '.join(patient_labels)})")
                                else:
                                    labels.append(f"{cohort_name} ({len(cohort_files)} patients)")
                    
                    # Plot unassigned patients
                    for file_name in unassigned_files:
                        mask = (source_files == file_name)
                        mask = mask[valid_mask]
                        if np.any(mask):
                            color = file_color_map[file_name]
                            sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                            c=[color], alpha=point_alpha, s=point_size, edgecolors='none')
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                         markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(self._get_patient_display_name(file_name))
                    
                    legend_title = 'Cohorts'
                else:
                    # Use individual patient colors (original behavior)
                    patient_colors_raw = _get_patient_colors(len(unique_files))
                    file_color_map = {file_name: patient_colors_raw[i] for i, file_name in enumerate(unique_files)}
                    handles = []
                    labels = []
                    for file_name in unique_files:
                        mask = source_files == file_name
                        mask = mask[valid_mask]
                        if np.any(mask):  # Only add if there are points for this file
                            sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                            c=[file_color_map[file_name]],
                                            alpha=point_alpha, s=point_size, edgecolors='none')
                            # Create custom legend handle with fixed size
                            color = file_color_map[file_name]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                           markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            # Use custom patient label if available
                            labels.append(self._get_patient_display_name(file_name))
                    legend_title = self.patient_legend_label
                
                # Place legend inside axes to avoid clipping - ensure it's visible
                if show_legend and handles and labels:
                    legend = ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=legend_title)
                    legend.set_visible(True)
            else:
                # Fallback if no source files
                ax.scatter(tsne_embedding_filtered[:, 0], tsne_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        elif color_by == 'Batch Group' and 'batch_group' in self.feature_dataframe.columns:
            # Filter out dropped clusters (cluster 0) for batch group coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Color by batch group to visualize batch effects (using custom patient labels and cohorts)
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                batch_group_series = self.feature_dataframe.loc[self.tsne_index, 'batch_group']
                batch_groups = batch_group_series.values[valid_mask]
            else:
                batch_groups = self.feature_dataframe['batch_group'].values[valid_mask]
            unique_groups = sorted([f for f in np.unique(batch_groups) if pd.notna(f)])
            if len(unique_groups) > 0:
                # Check if cohorts are defined for batch groups and cohort coloring is enabled
                cohorts_used = set()
                if self.use_cohort_coloring:
                    for group_name in unique_groups:
                        if group_name in self.patient_cohort_map:
                            cohorts_used.add(self.patient_cohort_map[group_name])
                
                if cohorts_used:
                    # Use cohort-based coloring
                    unique_cohorts = sorted(cohorts_used)
                    cohort_colors_raw = _get_patient_colors(len(unique_cohorts))
                    cohort_color_map = {}
                    for i, cohort_name in enumerate(unique_cohorts):
                        color = cohort_colors_raw[i]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        cohort_color_map[cohort_name] = rgb
                        self.cohort_colors[cohort_name] = rgb
                    
                    # Map batch groups to cohort colors
                    group_color_map = {}
                    unassigned_groups = [g for g in unique_groups if g not in self.patient_cohort_map]
                    if unassigned_groups:
                        unassigned_colors_raw = _get_patient_colors(len(unassigned_groups))
                        for i, group_name in enumerate(unassigned_groups):
                            color = unassigned_colors_raw[i]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            group_color_map[group_name] = rgb
                    
                    for group_name in unique_groups:
                        if group_name in self.patient_cohort_map:
                            cohort = self.patient_cohort_map[group_name]
                            group_color_map[group_name] = cohort_color_map[cohort]
                    
                    # Plot by cohort
                    handles = []
                    labels = []
                    # Plot cohorts first
                    for cohort_name in unique_cohorts:
                        cohort_groups = [g for g in unique_groups if g in self.patient_cohort_map and self.patient_cohort_map[g] == cohort_name]
                        if cohort_groups:
                            combined_mask = np.zeros(len(batch_groups), dtype=bool)
                            for group_name in cohort_groups:
                                combined_mask |= (batch_groups == group_name)
                            if np.any(combined_mask):
                                color = cohort_color_map[cohort_name]
                                sc = ax.scatter(tsne_embedding_filtered[combined_mask, 0], tsne_embedding_filtered[combined_mask, 1],
                                                c=[color], alpha=point_alpha, s=point_size, edgecolors='none')
                                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                             markeredgecolor='none', markersize=6, alpha=point_alpha)
                                handles.append(handle)
                                if len(cohort_groups) <= 3:
                                    labels.append(f"{cohort_name} ({', '.join(cohort_groups)})")
                                else:
                                    labels.append(f"{cohort_name} ({len(cohort_groups)} groups)")
                    
                    # Plot unassigned groups
                    for group_name in unassigned_groups:
                        mask = (batch_groups == group_name)
                        if np.any(mask):
                            color = group_color_map[group_name]
                            sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                            c=[color], alpha=point_alpha, s=point_size, edgecolors='none')
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                         markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(str(group_name))
                    
                    legend_title = 'Cohorts'
                else:
                    # Use individual batch group colors (original behavior)
                    group_colors_raw = _get_patient_colors(len(unique_groups))
                    group_color_map = {group_name: group_colors_raw[i] for i, group_name in enumerate(unique_groups)}
                    handles = []
                    labels = []
                    for group_name in unique_groups:
                        mask = batch_groups == group_name
                        if np.any(mask):  # Only add if there are points for this group
                            sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                            c=[group_color_map[group_name]],
                                            alpha=point_alpha, s=point_size, edgecolors='none')
                            # Create custom legend handle with fixed size
                            color = group_color_map[group_name]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                           markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(str(group_name))
                    legend_title = 'Batch Group'
                
                # Place legend inside axes to avoid clipping - ensure it's visible
                if show_legend and handles and labels:
                    legend = ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=legend_title)
                    legend.set_visible(True)
            else:
                # Fallback if no batch groups
                ax.scatter(tsne_embedding_filtered[:, 0], tsne_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        elif color_by == 'Phenotype' and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
            # Filter out dropped clusters (cluster 0) for phenotype coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Color by cluster phenotype
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                phenotype_series = self.clustered_data['cluster_phenotype'].reindex(self.tsne_index)
                phenotypes = phenotype_series.fillna('Unassigned').values[valid_mask]
            else:
                phenotypes = self.clustered_data['cluster_phenotype'].fillna('Unassigned').values[valid_mask]
            unique_phenotypes = sorted([p for p in np.unique(phenotypes) if pd.notna(p)])
            colors = _get_vivid_colors(len(unique_phenotypes))
            phenotype_color_map = {p: colors[i] for i, p in enumerate(unique_phenotypes)}
            handles = []
            labels = []
            for phenotype in unique_phenotypes:
                mask = phenotypes == phenotype
                sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                c=[phenotype_color_map[phenotype]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size
                color = phenotype_color_map[phenotype]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(str(phenotype))
            if show_legend:
                ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Phenotype')
        elif color_by == 'Manual Phenotype' and 'manual_phenotype' in self.feature_dataframe.columns:
            # Filter out dropped clusters (cluster 0) for manual phenotype coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Color by manual phenotype
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                manual_phenotype_series = self.feature_dataframe.loc[self.tsne_index, 'manual_phenotype']
                manual_phenotypes = manual_phenotype_series.fillna('Unassigned').values[valid_mask]
            else:
                manual_phenotypes = self.feature_dataframe['manual_phenotype'].fillna('Unassigned').values[valid_mask]
            unique_phenotypes = sorted([p for p in np.unique(manual_phenotypes) if pd.notna(p)])
            colors = _get_vivid_colors(len(unique_phenotypes))
            phenotype_color_map = {p: colors[i] for i, p in enumerate(unique_phenotypes)}
            handles = []
            labels = []
            for phenotype in unique_phenotypes:
                mask = manual_phenotypes == phenotype
                sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                c=[phenotype_color_map[phenotype]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size
                color = phenotype_color_map[phenotype]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(str(phenotype))
            if show_legend:
                ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Manual Phenotype')
        elif color_by in self.feature_dataframe.columns:
            # Handle metadata columns or other dataframe columns as categorical
            metadata_cols = self._get_metadata_columns(self.feature_dataframe)
            if color_by in metadata_cols or color_by == 'batch_group':
                # Filter out dropped clusters (cluster 0) for metadata column coloring
                if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                    cluster_labels_series = self.clustered_data['cluster']
                    cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
                else:
                    cluster_labels = self.clustered_data['cluster'].values
                valid_mask = cluster_labels != 0
                tsne_embedding_filtered = self.tsne_embedding[valid_mask]
                
                # Color by metadata column (categorical)
                if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                    col_series = self.feature_dataframe.loc[self.tsne_index, color_by]
                    col_values = col_series.fillna('Unknown').values[valid_mask]
                else:
                    col_values = self.feature_dataframe[color_by].fillna('Unknown').values[valid_mask]
                unique_values = sorted([v for v in np.unique(col_values) if pd.notna(v)])
                if len(unique_values) > 0:
                    colors = _get_vivid_colors(len(unique_values))
                    value_color_map = {v: colors[i] for i, v in enumerate(unique_values)}
                    handles = []
                    labels = []
                    for value in unique_values:
                        mask = col_values == value
                        if np.any(mask):
                            sc = ax.scatter(tsne_embedding_filtered[mask, 0], tsne_embedding_filtered[mask, 1],
                                            c=[value_color_map[value]],
                                            alpha=point_alpha, s=point_size, edgecolors='none')
                            color = value_color_map[value]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                           markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(str(value))
                    if show_legend and handles and labels:
                        ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=color_by)
        elif hasattr(self, 'tsne_raw_data') and color_by in getattr(self, 'tsne_selected_columns', []):
            # Filter out dropped clusters (cluster 0) for feature coloring
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Continuous coloring by selected feature (aligned to t-SNE order)
            vals = self.tsne_raw_data[color_by].values[valid_mask]
            sc = ax.scatter(tsne_embedding_filtered[:, 0], tsne_embedding_filtered[:, 1], c=vals,
                            cmap='viridis', alpha=point_alpha, s=point_size, edgecolors='none')
            cbar = self.figure.colorbar(sc, ax=ax)
            cbar.set_label(color_by)
        else:
            # Filter out dropped clusters (cluster 0) for fallback
            if hasattr(self, 'tsne_index') and self.tsne_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.tsne_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            valid_mask = cluster_labels != 0
            tsne_embedding_filtered = self.tsne_embedding[valid_mask]
            
            # Fallback single color
            ax.scatter(tsne_embedding_filtered[:, 0], tsne_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        
        ax.set_xlabel('t-SNE 1', fontsize=10)
        ax.set_ylabel('t-SNE 2', fontsize=10)
        if title is None:
            title = f't-SNE: {color_by}'
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    def _create_tsne_plot(self):
        """Create t-SNE scatter plot(s) with faceted plotting support."""
        self._active_view_name = 't-SNE'
        if self.tsne_embedding is None:
            return
        
        self.figure.clear()
        
        # Get selected color-by options (use stored data which contains actual column names)
        if hasattr(self, 'color_by_listwidget'):
            selected_items = []
            for item in self.color_by_listwidget.selectedItems():
                # Use stored data (actual column name) if available, otherwise use text
                actual_name = item.data(QtCore.Qt.UserRole)
                if actual_name:
                    selected_items.append(actual_name)
                else:
                    selected_items.append(item.text())
        else:
            # Fallback to combo box if list widget doesn't exist
            selected_items = [self.color_by_combo.currentText()] if hasattr(self, 'color_by_combo') else ['Cluster']
        
        # Ensure at least one selection
        if not selected_items:
            selected_items = ['Cluster']
        
        # Limit to max 3 plots (3 columns in single row)
        selected_items = selected_items[:3]
        
        # Get point size and alpha from controls
        point_size = self.point_size_spinbox.value() if hasattr(self, 'point_size_spinbox') else 18
        point_alpha = self.point_alpha_spinbox.value() if hasattr(self, 'point_alpha_spinbox') else 0.8
        show_legend = self.show_legend_checkbox.isChecked() if hasattr(self, 'show_legend_checkbox') else True
        
        n_plots = len(selected_items)
        
        if n_plots == 1:
            # Single plot - use full figure
            ax = self.figure.add_subplot(111)
            self._plot_tsne_single(ax, selected_items[0], point_size, point_alpha, show_legend=show_legend)
            pass
        else:
            # Multiple plots - create subplots in a single row with max 3 columns
            n_cols = n_plots
            n_rows = 1
            
            for idx, color_by in enumerate(selected_items):
                ax = self.figure.add_subplot(n_rows, n_cols, idx + 1)
                self._plot_tsne_single(ax, color_by, point_size, point_alpha, show_legend=show_legend)
            
            pass
        
        self._finalize_standard_plot(pad=0.9, allow_text_compaction=True)
    
    def _show_heatmap(self):
        """Switch back to heatmap view."""
        self._active_view_name = 'Heatmap'
        if hasattr(self, 'view_combo') and self.view_combo.currentText() != 'Heatmap':
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentText('Heatmap')
            self.view_combo.blockSignals(False)
            self._update_viz_controls_visibility()
        if self.clustered_data is not None:
            self._create_heatmap()
        else:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view the heatmap.")
        self._update_cluster_action_buttons()

    def _show_cluster_map(self):
        """Switch to cluster map view."""
        self._active_view_name = 'Cluster Map'
        if hasattr(self, 'view_combo') and self.view_combo.currentText() != 'Cluster Map':
            self.view_combo.blockSignals(True)
            self.view_combo.setCurrentText('Cluster Map')
            self.view_combo.blockSignals(False)
            self._update_viz_controls_visibility()
        if self.clustered_data is not None:
            self._create_cluster_map()
        else:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view the cluster map.")

    def _on_view_changed(self, view: str):
        """Switch visualization based on selected view and manage dependencies."""
        # Hard reset before switching views to avoid stale artists across modes.
        self._hard_reset_canvas()
        
        self._update_viz_controls_visibility()
        if view == 'Heatmap':
            self._show_heatmap()
        elif view == 'Cluster Map':
            self._show_cluster_map()
        elif view == 'UMAP':
            if getattr(self, 'umap_embedding', None) is None:
                self._run_umap()
            else:
                self._create_umap_plot()
        elif view == 't-SNE':
            if getattr(self, 'tsne_embedding', None) is None:
                self._run_tsne()
            else:
                self._create_tsne_plot()
        elif view == 'Stacked Bars':
            self._show_stacked_bars()
        elif view == 'Differential Expression':
            self._show_differential_expression()
        elif view == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()
        
        # Enable save if there is content
        self.save_plot_btn.setEnabled(True)
        self.save_output_btn.setEnabled(True)
        self._update_cluster_action_buttons()

    def _update_viz_controls_visibility(self):
        """Show/hide controls depending on selected view."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        # Color-by visible only for UMAP and t-SNE
        for i in range(self.color_by_combo.count()):
            pass
        if hasattr(self, 'color_by_label'):
            self.color_by_label.setVisible(view in ['UMAP', 't-SNE'])
        if hasattr(self, 'color_by_search'):
            self.color_by_search.setVisible(view in ['UMAP', 't-SNE'])
        if hasattr(self, 'color_by_listwidget'):
            self.color_by_listwidget.setVisible(view in ['UMAP', 't-SNE'])
        self.color_by_combo.setVisible(False)  # Keep hidden for backward compatibility
        # Show cohort checkbox only for UMAP/t-SNE and if cohorts exist
        if hasattr(self, 'use_cohort_checkbox'):
            has_cohorts = bool(self.patient_cohort_map)
            self.use_cohort_checkbox.setVisible(view in ['UMAP', 't-SNE'] and has_cohorts)
        # Point size and alpha visible only for UMAP and t-SNE
        if hasattr(self, 'point_size_label'):
            self.point_size_label.setVisible(view in ['UMAP', 't-SNE'])
            if hasattr(self, 'point_size_spinbox'):
                self.point_size_spinbox.setVisible(view in ['UMAP', 't-SNE'])
        if hasattr(self, 'point_alpha_label'):
            self.point_alpha_label.setVisible(view in ['UMAP', 't-SNE'])
            if hasattr(self, 'point_alpha_spinbox'):
                self.point_alpha_spinbox.setVisible(view in ['UMAP', 't-SNE'])
        # Show legend checkbox visible for all views that have legends
        if hasattr(self, 'show_legend_checkbox'):
            # Cluster Map legend is intentionally disabled by default to maximize plot space.
            self.show_legend_checkbox.setVisible(view in ['UMAP', 't-SNE', 'Stacked Bars', 'Heatmap'])
        # Remake UMAP button visible only for UMAP
        if hasattr(self, 'remake_umap_btn'):
            self.remake_umap_btn.setVisible(view == 'UMAP')
        # Group-by visible only for Stacked Bars
        if hasattr(self, 'group_by_label'):
            self.group_by_label.setVisible(view == 'Stacked Bars')
        self.group_by_combo.setVisible(view == 'Stacked Bars')
        # View type, normalization, and filter controls visible only for Stacked Bars
        if hasattr(self, 'stacked_bars_view_type_label'):
            self.stacked_bars_view_type_label.setVisible(view == 'Stacked Bars')
        if hasattr(self, 'stacked_bars_view_type_combo'):
            self.stacked_bars_view_type_combo.setVisible(view == 'Stacked Bars')
        if hasattr(self, 'stacked_bars_filter_btn'):
            self.stacked_bars_filter_btn.setVisible(view == 'Stacked Bars')
        # Colormap visible only for Heatmap, Cluster Map, and Differential Expression; hidden for UMAP and Stacked Bars
        if hasattr(self, 'colormap_label'):
            self.colormap_label.setVisible(view in ['Heatmap', 'Cluster Map', 'Differential Expression'])
        self.colormap_combo.setVisible(view in ['Heatmap', 'Cluster Map', 'Differential Expression'])
        # Top N visible only for Differential Expression
        if hasattr(self, 'top_n_label'):
            self.top_n_label.setVisible(view == 'Differential Expression')
        self.top_n_spinbox.setVisible(view == 'Differential Expression')
        if hasattr(self, 'de_filter_btn'):
            self.de_filter_btn.setVisible(view == 'Differential Expression')
        if hasattr(self, 'de_flip_axes_checkbox'):
            self.de_flip_axes_checkbox.setVisible(view == 'Differential Expression')
        if hasattr(self, 'de_show_values_checkbox'):
            self.de_show_values_checkbox.setVisible(view == 'Differential Expression')
        if hasattr(self, 'de_show_boxes_checkbox'):
            self.de_show_boxes_checkbox.setVisible(view == 'Differential Expression')
        # Feature labels button visible for Differential Expression, Stacked Bars, and Boxplot/Violin Plot
        if hasattr(self, 'feature_labels_btn'):
            self.feature_labels_btn.setVisible(view in ['Differential Expression', 'Stacked Bars', 'Boxplot/Violin Plot'])
        # Marker selection and plot type visible only for Boxplot/Violin Plot
        is_boxplot_violin = view == 'Boxplot/Violin Plot'
        if hasattr(self, 'marker_select_label'):
            self.marker_select_label.setVisible(is_boxplot_violin)
        if hasattr(self, 'marker_select_btn'):
            self.marker_select_btn.setVisible(is_boxplot_violin)
        if hasattr(self, 'plot_type_label'):
            self.plot_type_label.setVisible(is_boxplot_violin)
        if hasattr(self, 'plot_type_combo'):
            self.plot_type_combo.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_test_checkbox'):
            self.stats_test_checkbox.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_mode_label'):
            self.stats_mode_label.setVisible(is_boxplot_violin)
        if hasattr(self, 'stats_mode_combo'):
            self.stats_mode_combo.setVisible(is_boxplot_violin)
        show_stats_reference = (
            is_boxplot_violin
            and hasattr(self, 'stats_mode_combo')
            and self.stats_mode_combo.currentText() == "One vs Others"
        )
        if hasattr(self, 'stats_cluster_label'):
            self.stats_cluster_label.setVisible(show_stats_reference)
        if hasattr(self, 'stats_cluster_combo'):
            self.stats_cluster_combo.setVisible(show_stats_reference)
        if hasattr(self, 'stats_export_btn'):
            self.stats_export_btn.setVisible(is_boxplot_violin)
        # Heatmap and Cluster Map controls
        is_heatmap = view == 'Heatmap'
        is_cluster_map = view == 'Cluster Map'
        is_heatmap_or_cluster_map = is_heatmap or is_cluster_map
        if hasattr(self, 'heatmap_source_combo'):
            self.heatmap_source_combo.setVisible(is_heatmap)  # Only for Heatmap, not Cluster Map
        if hasattr(self, 'heatmap_source_label'):
            self.heatmap_source_label.setVisible(is_heatmap)  # Only for Heatmap, not Cluster Map
        if hasattr(self, 'heatmap_filter_btn'):
            self.heatmap_filter_btn.setVisible(is_heatmap_or_cluster_map)
        if hasattr(self, 'heatmap_scaling_combo'):
            self.heatmap_scaling_combo.setVisible(is_heatmap_or_cluster_map)
        if hasattr(self, 'heatmap_scaling_label'):
            self.heatmap_scaling_label.setVisible(is_heatmap_or_cluster_map)
        if hasattr(self, 'patient_annotation_checkbox'):
            self.patient_annotation_checkbox.setVisible(is_heatmap)  # Only for Heatmap, not Cluster Map
        # Configure plot button visible for Heatmap and Cluster Map
        if hasattr(self, 'configure_plot_btn'):
            self.configure_plot_btn.setVisible(is_heatmap_or_cluster_map)
    
    def _on_colormap_changed(self, _text: str):
        """Handle colormap selection change."""
        # Refresh the current view if it uses colormaps
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view in ['Heatmap', 'Cluster Map', 'Differential Expression']:
            if view == 'Heatmap':
                self._show_heatmap()
            elif view == 'Cluster Map':
                self._show_cluster_map()
            elif view == 'Differential Expression':
                self._show_differential_expression()
    
    def _on_group_by_changed(self, _text: str):
        """Handle group by selection change for stacked bars."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Stacked Bars':
            self._show_stacked_bars()
    
    def _on_stacked_bars_view_type_changed(self, _text: str):
        """Handle view type change for stacked bars (Fraction vs Total enumeration)."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Stacked Bars':
            self._show_stacked_bars()
    
    def _open_stacked_bars_filter_dialog(self):
        """Open a dialog to choose which clusters to show in stacked bars."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Run clustering first to filter stacked bars.")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Clusters to Display")
        v = QtWidgets.QVBoxLayout(dlg)
        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        # Build items from clusters (excluding cluster 0)
        options = self._sorted_cluster_ids([c for c in self.clustered_data['cluster'].unique() if c != 0], canonical=False)
        items = [self._get_cluster_display_name(cid) for cid in options]
        for label in items:
            it = QtWidgets.QListWidgetItem(label)
            # If no filter selection exists, select all by default
            if not getattr(self, 'stacked_bars_filter_selection', None):
                it.setSelected(True)
            else:
                it.setSelected(label in self.stacked_bars_filter_selection)
            listw.addItem(it)
        v.addWidget(listw)
        # Action buttons
        btns = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch()
        btns.addWidget(select_all_btn)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        def do_select_all():
            for i in range(listw.count()):
                listw.item(i).setSelected(True)
        select_all_btn.clicked.connect(do_select_all)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            selected_labels = [i.text() for i in listw.selectedItems()]
            if selected_labels:
                self.stacked_bars_filter_selection = set(selected_labels)
            else:
                # If nothing selected, show all (set to None)
                self.stacked_bars_filter_selection = None
            self._show_stacked_bars()

    def _open_de_cluster_filter_dialog(self):
        """Open a dialog to choose which clusters to compare in differential expression."""
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Run clustering first to filter differential expression.")
            return

        cluster_ids = self._sorted_cluster_ids(
            pd.to_numeric(self.clustered_data['cluster'], errors='coerce')
            .dropna()
            .astype(int)
            .loc[lambda s: s != 0]
            .unique()
            .tolist(),
            canonical=True,
        )
        if not cluster_ids:
            QtWidgets.QMessageBox.warning(self, "No Clusters", "No clusters are available for differential expression.")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Clusters for Differential Expression")
        v = QtWidgets.QVBoxLayout(dlg)
        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)

        selected_clusters = getattr(self, 'de_cluster_filter_selection', None)
        for cid in cluster_ids:
            label = self._get_cluster_display_name(cid)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, cid)
            if not selected_clusters:
                item.setSelected(True)
            else:
                item.setSelected(cid in selected_clusters)
            listw.addItem(item)
        v.addWidget(listw)

        btns = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        clear_all_btn = QtWidgets.QPushButton("Clear All")
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch()
        btns.addWidget(select_all_btn)
        btns.addWidget(clear_all_btn)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        def do_select_all():
            for i in range(listw.count()):
                listw.item(i).setSelected(True)

        def do_clear_all():
            for i in range(listw.count()):
                listw.item(i).setSelected(False)

        select_all_btn.clicked.connect(do_select_all)
        clear_all_btn.clicked.connect(do_clear_all)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            selected_ids = [int(i.data(QtCore.Qt.UserRole)) for i in listw.selectedItems()]
            if selected_ids and len(selected_ids) < len(cluster_ids):
                self.de_cluster_filter_selection = set(selected_ids)
            else:
                # No selection or full selection => use all clusters
                self.de_cluster_filter_selection = None

            if self.view_combo.currentText() == 'Differential Expression':
                self._show_differential_expression()
    
    def _on_top_n_changed(self, _value: int):
        """Handle top N markers selection change."""
        # Refresh the differential expression view
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Differential Expression':
            self._show_differential_expression()

    def _on_de_axis_orientation_changed(self, _state: int):
        """Handle differential expression axis flip toggle."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Differential Expression':
            self._show_differential_expression()
    
    def _get_colormap_name(self):
        """Get the matplotlib colormap name from the combo box selection."""
        colormap_text = self.colormap_combo.currentText()
        # Extract the colormap name (part before the parenthesis)
        colormap_name = colormap_text.split(' (')[0]
        return colormap_name

    def _select_feature_columns(self, df: pd.DataFrame):
        """Return numeric feature columns to plot, excluding non-numeric/meta columns.
        
        If selected_display_features is set, only return those features.
        Otherwise, return all numeric features (excluding metadata and cluster columns).
        """
        # If display features are explicitly selected, use only those
        if self.selected_display_features is not None and len(self.selected_display_features) > 0:
            # Filter to only include features that exist in the dataframe
            available_features = [f for f in self.selected_display_features if f in df.columns]
            if available_features:
                return available_features
        
        # Standard columns to exclude
        exclude_cols = { 'cluster', '__group__', 'cluster_phenotype', 'manual_phenotype' }
        
        # Get all metadata columns (including those added during batch correction)
        metadata_cols = set(self._get_metadata_columns(df))
        
        # Combine exclusions
        all_exclude_cols = exclude_cols | metadata_cols
        
        feature_cols = []
        for col in df.columns:
            if col in all_exclude_cols:
                continue
            try:
                # Check if numeric but exclude boolean columns (they cause issues with numpy operations)
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].dtype != bool:
                    feature_cols.append(col)
            except Exception:
                continue
        return feature_cols

    def _on_heatmap_source_changed(self, _text: str):
        """Refresh heatmap when the source (Clusters vs Manual Gates) changes."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Heatmap':
            self._show_heatmap()
    
    def _on_heatmap_scaling_changed(self, _text: str):
        """Refresh heatmap or cluster map when the scaling method changes."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Heatmap':
            self._show_heatmap()
        elif view == 'Cluster Map':
            self._show_cluster_map()
    
    def _on_patient_annotation_changed(self, state: int):
        """Handle patient annotation checkbox state change."""
        # Update enabled flag
        self.patient_annotation_enabled = (state == 2)  # 2 = checked
        
        # Enable/disable the customize button (check if patient annotation column is available)
        has_patient_col = False
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            has_patient_col = self.patient_annotation_column in self.feature_dataframe.columns
        else:
            # Check default priority columns
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in self.feature_dataframe.columns:
                    has_patient_col = True
                    break
            # Also check metadata columns
            if not has_patient_col:
                metadata_cols = self._get_metadata_columns(self.feature_dataframe)
                has_patient_col = len(metadata_cols) > 0
        
        # Refresh heatmap if it's the current view
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Heatmap':
            self._show_heatmap()

    def _get_cluster_display_name(self, cluster_id):
        """Return display label for a cluster id, using annotation if available."""
        return get_cluster_display_name(cluster_id, annotation_map=self.cluster_annotation_map)

    def _sorted_cluster_ids(self, values, *, canonical: bool = False):
        """Return cluster ids in consistent numeric order."""
        return sort_cluster_values(values, annotation_map=self.cluster_annotation_map, canonical=canonical)

    def _sort_group_values(self, values):
        """Sort group identifiers deterministically, handling mixed numeric/string values."""
        unique_values = []
        seen = set()
        for val in values:
            if pd.isna(val):
                continue
            if isinstance(val, np.generic):
                val = val.item()
            try:
                if val in seen:
                    continue
                seen.add(val)
            except Exception:
                # Fallback for unusual unhashable values.
                if val in unique_values:
                    continue
            unique_values.append(val)

        def _sort_key(v):
            parsed = canonicalize_cluster_id(v)
            if isinstance(parsed, int):
                return (0, parsed)
            if isinstance(parsed, float):
                return (1, parsed)
            if isinstance(v, (bool, np.bool_)):
                return (3, str(v))
            if isinstance(v, (int, np.integer)):
                return (0, int(v))
            if isinstance(v, (float, np.floating)):
                fv = float(v)
                if fv.is_integer():
                    return (0, int(fv))
                return (1, fv)
            return (2, str(v))

        try:
            return sorted(unique_values, key=_sort_key)
        except Exception:
            return unique_values

    def _build_group_color_map(self, groups, source='Clusters', group_col='cluster'):
        """Build a stable color map for groups/clusters across views."""
        display_groups = self._sort_group_values(groups)
        palette_groups = list(display_groups)

        # For clusters, anchor colors to the full cluster domain so filtered views keep identical colors.
        if source == 'Clusters':
            base_data = self.clustered_data_unscaled if self.clustered_data_unscaled is not None else self.clustered_data
            if isinstance(base_data, pd.DataFrame) and group_col in base_data.columns:
                full_groups = self._sort_group_values(base_data[group_col].values)
                palette_groups = []
                for gid in full_groups + display_groups:
                    if gid not in palette_groups:
                        palette_groups.append(gid)

        colors_raw = _get_vivid_colors(max(1, len(palette_groups)))
        color_map = {}
        for i, gid in enumerate(palette_groups):
            color = colors_raw[i]
            if len(color) >= 3:
                rgb = tuple(color[:3])
            else:
                rgb = (0.7, 0.7, 0.7)
            color_map[gid] = rgb
        return color_map
    
    def _get_patient_display_name(self, source_file):
        """Return display label for a source file/patient, using custom annotation if available."""
        if pd.isna(source_file):
            return "Unknown"
        if isinstance(self.patient_annotation_map, dict) and source_file in self.patient_annotation_map and self.patient_annotation_map[source_file]:
            return self.patient_annotation_map[source_file]
        # For source_file column, use basename of file as default
        # For other columns (batch_group, source_well), use the value as-is
        import os
        # Check if it looks like a file path (contains path separators)
        source_str = str(source_file)
        if os.sep in source_str or '/' in source_str or '\\' in source_str:
            return os.path.basename(source_str)
        # Otherwise, return the value as-is (for batch_group, source_well, etc.)
        return source_str

    def _get_heatmap_annotation_bar_label(self, source: str, *, patient: bool = False) -> str:
        """Return the label shown beside a heatmap annotation bar."""
        if patient:
            patient_label = str(getattr(self, 'patient_legend_label', '') or '').strip()
            return patient_label or 'Patient/Source'

        source_label = str(source or '').strip()
        if source_label == 'Clusters':
            return 'Cluster'
        if source_label == 'Manual Gates':
            return 'Manual Gate'
        if source_label.endswith('s') and len(source_label) > 1:
            return source_label[:-1]
        return source_label or 'Group'

    def _annotate_heatmap_bar_axis(self, ax, label: str) -> None:
        """Place a compact horizontal label just left of a heatmap annotation bar."""
        if ax is None:
            return

        label_text = str(label or '').strip()
        if not label_text:
            return

        label_fontsize = max(7, min(9, int(getattr(self, 'legend_fontsize', 8))))
        ax.text(
            -0.012,
            0.5,
            label_text,
            transform=ax.transAxes,
            rotation=0,
            ha='right',
            va='center',
            fontsize=label_fontsize,
            fontweight='bold',
            clip_on=False,
        )

    def _get_manual_groups_series(self):
        """Compute grouping series for manual gates. Single named phenotype -> name vs Other; otherwise names with Unassigned for blanks."""
        if self.clustered_data is None:
            return None
        if 'manual_phenotype' not in self.clustered_data.columns:
            return None
        series = self.clustered_data['manual_phenotype'].fillna('').astype(str)
        unique_named = sorted([s for s in series.unique() if s.strip() != ''])
        if len(unique_named) == 1:
            name = unique_named[0]
            return series.apply(lambda s: name if s == name else 'Other')
        return series.apply(lambda s: s if s.strip() != '' else 'Unassigned')

    def _apply_heatmap_filter(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        """Apply heatmap filter selection to the dataframe, if any selection present."""
        selected = getattr(self, 'heatmap_filter_selection', None)
        if not selected:
            return df
        mask = df[group_col].isin(list(selected))
        filtered = df.loc[mask]
        return filtered

    def _open_heatmap_filter_dialog(self):
        """Open a dialog to choose which clusters/phenotypes to show in heatmap."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Run clustering first to filter heatmap.")
            return
        source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select groups to display")
        v = QtWidgets.QVBoxLayout(dlg)
        listw = QtWidgets.QListWidget()
        listw.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        # Build items
        items = []
        if source == 'Manual Gates' and 'manual_phenotype' in self.clustered_data.columns:
            groups = self._get_manual_groups_series()
            options = sorted(groups.unique()) if groups is not None else []
            items = options
        else:
            options = self._sorted_cluster_ids(self.clustered_data['cluster'].unique(), canonical=False)
            items = [self._get_cluster_display_name(cid) for cid in options]
        for label in items:
            it = QtWidgets.QListWidgetItem(label)
            it.setSelected(True if not getattr(self, 'heatmap_filter_selection', None) else (label in self.heatmap_filter_selection))
            listw.addItem(it)
        v.addWidget(listw)
        # Action buttons
        btns = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        btns.addStretch()
        btns.addWidget(select_all_btn)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        def do_select_all():
            for i in range(listw.count()):
                listw.item(i).setSelected(True)
        select_all_btn.clicked.connect(do_select_all)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.heatmap_filter_selection = set([i.text() for i in listw.selectedItems()])
            self._show_heatmap()
            # Also update UMAP if it's currently visible
            view = self.view_combo.currentText() if hasattr(self, 'view_combo') else ''
            if view == 'UMAP' and getattr(self, 'umap_embedding', None) is not None:
                self._create_umap_plot()
    
    def _plot_umap_single(self, ax, color_by, point_size, point_alpha, title=None, show_legend=True):
        """Plot a single UMAP subplot with specified coloring."""
        if hasattr(self, 'umap_selected_columns'):
            pass
        if hasattr(self, 'umap_raw_data'):
            pass
        
        # Ensure UMAP embedding size matches the data we're trying to plot
        # This is critical when cells have been filtered (e.g., edge cells excluded)
        if hasattr(self, 'umap_index') and self.umap_index is not None:
            # UMAP was created from specific indices - use those
            cluster_labels_series = self.clustered_data['cluster']
            # Only reindex with indices that exist in both umap_index and clustered_data
            valid_indices = [idx for idx in self.umap_index if idx in cluster_labels_series.index]
            cluster_labels = cluster_labels_series.loc[valid_indices].values
            
            # Verify that umap_embedding length matches the cluster_labels we extracted
            if len(cluster_labels) != len(self.umap_embedding):
                # Mismatch - use direct clustered_data values instead
                cluster_labels = self.clustered_data['cluster'].values
        else:
            cluster_labels = self.clustered_data['cluster'].values
        
        # Final safety check: ensure mask size matches embedding size
        if len(cluster_labels) != len(self.umap_embedding):
            # If still mismatched, truncate or pad to match
            min_len = min(len(cluster_labels), len(self.umap_embedding))
            cluster_labels = cluster_labels[:min_len]
            self.umap_embedding = self.umap_embedding[:min_len]
        
        # Filter out cluster 0 (noise/unassigned)
        valid_mask = cluster_labels != 0
        
        # Apply heatmap filter if it exists (filter cells from UMAP that are filtered in heatmap)
        if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
            # Convert display names to cluster IDs (same logic as heatmap)
            wanted_ids = set()
            for cid in self._sorted_cluster_ids(self.clustered_data['cluster'].unique(), canonical=False):
                if pd.notna(cid) and cid != 0:
                    name = self._get_cluster_display_name(cid)
                    if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                        wanted_ids.add(int(cid))
            if wanted_ids:
                # Apply filter: only keep clusters that are in the heatmap filter
                heatmap_filter_mask = np.isin(cluster_labels, list(wanted_ids))
                valid_mask = valid_mask & heatmap_filter_mask
        
        # Filter out NaN clusters
        valid_mask = valid_mask & pd.notna(cluster_labels)
        
        umap_embedding_filtered = self.umap_embedding[valid_mask]
        
        if self.clustered_data is not None:
            pass
        
        if color_by == 'Cluster' and self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
            # Use pre-computed filtered data
            cluster_labels_filtered = cluster_labels[valid_mask]
            
            # Filter out NaN clusters and ensure all are valid integers
            valid_cluster_mask = pd.notna(cluster_labels_filtered)
            cluster_labels_filtered = cluster_labels_filtered[valid_cluster_mask]
            umap_embedding_filtered = umap_embedding_filtered[valid_cluster_mask]
            
            unique_clusters = self._sorted_cluster_ids(
                [c for c in np.unique(cluster_labels_filtered) if pd.notna(c) and c != 0],
                canonical=False,
            )
            if len(unique_clusters) == 0:
                # No valid clusters, just plot without legend
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], 
                           c='gray', alpha=point_alpha, s=point_size, edgecolors='none')
            else:
                colors = _get_vivid_colors(len(unique_clusters))
                cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
                handles = []
                labels = []
                for cluster_id in unique_clusters:
                    mask = cluster_labels_filtered == cluster_id
                    if np.any(mask):  # Only plot if there are points
                        sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                        c=[cluster_color_map[cluster_id]],
                                        alpha=point_alpha, s=point_size, edgecolors='none')
                        # Create custom legend handle with fixed size (18)
                        color = cluster_color_map[cluster_id]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                       markeredgecolor='none', markersize=6, alpha=point_alpha)
                        handles.append(handle)
                        labels.append(self._get_cluster_display_name(cluster_id))
                # Place legend inside axes to avoid clipping (only if show_legend is True)
                if show_legend and handles and labels:
                    n_clusters = len(handles)
                    ncol = max(1, (n_clusters + 9) // 10) if n_clusters > 10 else 1
                    ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, ncol=ncol)
        elif color_by == 'Source File' and 'source_file' in self.feature_dataframe.columns:
            # Use pre-computed filtered data
            # Color by source file to visualize batch effects (using custom patient labels)
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                source_file_series = self.feature_dataframe.loc[self.umap_index, 'source_file']
                source_files = source_file_series.values[valid_mask]
            else:
                source_files = self.feature_dataframe['source_file'].values[valid_mask]
            unique_files = sorted([f for f in np.unique(source_files) if pd.notna(f)])
            if len(unique_files) > 0:
                # Check if cohorts are defined and cohort coloring is enabled
                cohorts_used = set()
                if self.use_cohort_coloring:
                    for file_name in unique_files:
                        if file_name in self.patient_cohort_map:
                            cohorts_used.add(self.patient_cohort_map[file_name])
                
                if cohorts_used:
                    # Use cohort-based coloring
                    unique_cohorts = sorted(cohorts_used)
                    cohort_colors_raw = _get_patient_colors(len(unique_cohorts))
                    cohort_color_map = {}
                    for i, cohort_name in enumerate(unique_cohorts):
                        color = cohort_colors_raw[i]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        cohort_color_map[cohort_name] = rgb
                        self.cohort_colors[cohort_name] = rgb
                    
                    # Map files to cohort colors
                    file_color_map = {}
                    unassigned_files = [f for f in unique_files if f not in self.patient_cohort_map]
                    if unassigned_files:
                        unassigned_colors_raw = _get_patient_colors(len(unassigned_files))
                        for i, file_name in enumerate(unassigned_files):
                            color = unassigned_colors_raw[i]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            file_color_map[file_name] = rgb
                    
                    for file_name in unique_files:
                        if file_name in self.patient_cohort_map:
                            cohort = self.patient_cohort_map[file_name]
                            file_color_map[file_name] = cohort_color_map[cohort]
                    
                    # Plot by cohort
                    handles = []
                    labels = []
                    # Plot cohorts first
                    for cohort_name in unique_cohorts:
                        cohort_files = [f for f in unique_files if f in self.patient_cohort_map and self.patient_cohort_map[f] == cohort_name]
                        if cohort_files:
                            combined_mask = np.zeros(len(source_files), dtype=bool)
                            for file_name in cohort_files:
                                combined_mask |= (source_files == file_name)
                            combined_mask = combined_mask[valid_mask]
                            if np.any(combined_mask):
                                color = cohort_color_map[cohort_name]
                                sc = ax.scatter(umap_embedding_filtered[combined_mask, 0], umap_embedding_filtered[combined_mask, 1],
                                                c=[color], alpha=point_alpha, s=point_size, edgecolors='none')
                                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                             markeredgecolor='none', markersize=6, alpha=point_alpha)
                                handles.append(handle)
                                if len(cohort_files) <= 3:
                                    patient_labels = [self._get_patient_display_name(f) for f in cohort_files]
                                    labels.append(f"{cohort_name} ({', '.join(patient_labels)})")
                                else:
                                    labels.append(f"{cohort_name} ({len(cohort_files)} patients)")
                    
                    # Plot unassigned patients
                    for file_name in unassigned_files:
                        mask = (source_files == file_name)
                        mask = mask[valid_mask]
                        if np.any(mask):
                            color = file_color_map[file_name]
                            sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                            c=[color], alpha=point_alpha, s=point_size, edgecolors='none')
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                         markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(self._get_patient_display_name(file_name))
                    
                    legend_title = 'Cohorts'
                else:
                    # Use individual patient colors (original behavior)
                    patient_colors_raw = _get_patient_colors(len(unique_files))
                    file_color_map = {file_name: patient_colors_raw[i] for i, file_name in enumerate(unique_files)}
                    handles = []
                    labels = []
                    for file_name in unique_files:
                        mask = source_files == file_name
                        mask = mask[valid_mask]
                        if np.any(mask):  # Only add if there are points for this file
                            sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                            c=[file_color_map[file_name]],
                                            alpha=point_alpha, s=point_size, edgecolors='none')
                            # Create custom legend handle with fixed size
                            color = file_color_map[file_name]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                           markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            # Use custom patient label if available
                            labels.append(self._get_patient_display_name(file_name))
                    legend_title = self.patient_legend_label
                
                # Place legend inside axes to avoid clipping - ensure it's visible
                if show_legend and handles and labels:
                    legend = ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=legend_title)
                    legend.set_visible(True)
            else:
                # Fallback if no source files
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        elif color_by == 'Batch Group' and 'batch_group' in self.feature_dataframe.columns:
            # Use pre-computed filtered data
            # Color by batch group to visualize batch effects (using custom patient labels and cohorts)
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                batch_group_series = self.feature_dataframe.loc[self.umap_index, 'batch_group']
                batch_groups = batch_group_series.values[valid_mask]
            else:
                batch_groups = self.feature_dataframe['batch_group'].values[valid_mask]
            unique_groups = sorted([f for f in np.unique(batch_groups) if pd.notna(f)])
            if len(unique_groups) > 0:
                # Check if cohorts are defined for batch groups and cohort coloring is enabled
                cohorts_used = set()
                if self.use_cohort_coloring:
                    for group_name in unique_groups:
                        if group_name in self.patient_cohort_map:
                            cohorts_used.add(self.patient_cohort_map[group_name])
                
                if cohorts_used:
                    # Use cohort-based coloring
                    unique_cohorts = sorted(cohorts_used)
                    cohort_colors_raw = _get_patient_colors(len(unique_cohorts))
                    cohort_color_map = {}
                    for i, cohort_name in enumerate(unique_cohorts):
                        color = cohort_colors_raw[i]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        cohort_color_map[cohort_name] = rgb
                        self.cohort_colors[cohort_name] = rgb
                    
                    # Map batch groups to cohort colors
                    group_color_map = {}
                    unassigned_groups = [g for g in unique_groups if g not in self.patient_cohort_map]
                    if unassigned_groups:
                        unassigned_colors_raw = _get_patient_colors(len(unassigned_groups))
                        for i, group_name in enumerate(unassigned_groups):
                            color = unassigned_colors_raw[i]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            group_color_map[group_name] = rgb
                    
                    for group_name in unique_groups:
                        if group_name in self.patient_cohort_map:
                            cohort = self.patient_cohort_map[group_name]
                            group_color_map[group_name] = cohort_color_map[cohort]
                    
                    # Plot by cohort
                    handles = []
                    labels = []
                    # Plot cohorts first
                    for cohort_name in unique_cohorts:
                        cohort_groups = [g for g in unique_groups if g in self.patient_cohort_map and self.patient_cohort_map[g] == cohort_name]
                        if cohort_groups:
                            combined_mask = np.zeros(len(batch_groups), dtype=bool)
                            for group_name in cohort_groups:
                                combined_mask |= (batch_groups == group_name)
                            if np.any(combined_mask):
                                color = cohort_color_map[cohort_name]
                                sc = ax.scatter(umap_embedding_filtered[combined_mask, 0], umap_embedding_filtered[combined_mask, 1],
                                                c=[color], alpha=point_alpha, s=point_size, edgecolors='none')
                                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                             markeredgecolor='none', markersize=6, alpha=point_alpha)
                                handles.append(handle)
                                if len(cohort_groups) <= 3:
                                    labels.append(f"{cohort_name} ({', '.join(cohort_groups)})")
                                else:
                                    labels.append(f"{cohort_name} ({len(cohort_groups)} groups)")
                    
                    # Plot unassigned groups
                    for group_name in unassigned_groups:
                        mask = (batch_groups == group_name)
                        if np.any(mask):
                            color = group_color_map[group_name]
                            sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                            c=[color], alpha=point_alpha, s=point_size, edgecolors='none')
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=color,
                                         markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(str(group_name))
                    
                    legend_title = 'Cohorts'
                else:
                    # Use individual batch group colors (original behavior)
                    group_colors_raw = _get_patient_colors(len(unique_groups))
                    group_color_map = {group_name: group_colors_raw[i] for i, group_name in enumerate(unique_groups)}
                    handles = []
                    labels = []
                    for group_name in unique_groups:
                        mask = batch_groups == group_name
                        if np.any(mask):  # Only add if there are points for this group
                            sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                            c=[group_color_map[group_name]],
                                            alpha=point_alpha, s=point_size, edgecolors='none')
                            # Create custom legend handle with fixed size
                            color = group_color_map[group_name]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                           markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(str(group_name))
                    legend_title = 'Batch Group'
                
                # Place legend inside axes to avoid clipping - ensure it's visible
                if show_legend and handles and labels:
                    legend = ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=legend_title)
                    legend.set_visible(True)
            else:
                # Fallback if no batch groups
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        elif color_by == 'Phenotype' and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
            # Use pre-computed filtered data
            # Color by cluster phenotype
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                phenotype_series = self.clustered_data['cluster_phenotype'].reindex(self.umap_index)
                phenotypes = phenotype_series.fillna('Unassigned').values[valid_mask]
            else:
                phenotypes = self.clustered_data['cluster_phenotype'].fillna('Unassigned').values[valid_mask]
            unique_phenotypes = sorted([p for p in np.unique(phenotypes) if pd.notna(p)])
            colors = _get_vivid_colors(len(unique_phenotypes))
            phenotype_color_map = {p: colors[i] for i, p in enumerate(unique_phenotypes)}
            handles = []
            labels = []
            for phenotype in unique_phenotypes:
                mask = phenotypes == phenotype
                sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                c=[phenotype_color_map[phenotype]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size
                color = phenotype_color_map[phenotype]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(str(phenotype))
            if show_legend:
                ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Phenotype')
        elif color_by == 'Cohorts' and 'cohort' in self.feature_dataframe.columns:
            # Use pre-computed filtered data
            # Color by cohort
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                cohort_series = self.feature_dataframe.loc[self.umap_index, 'cohort']
                cohorts = cohort_series.fillna('Unassigned').values[valid_mask]
            else:
                cohorts = self.feature_dataframe['cohort'].fillna('Unassigned').values[valid_mask]
            unique_cohorts = sorted([c for c in np.unique(cohorts) if pd.notna(c) and c != ''])
            if len(unique_cohorts) > 0:
                cohort_colors_raw = _get_patient_colors(len(unique_cohorts))
                cohort_color_map = {}
                for i, cohort_name in enumerate(unique_cohorts):
                    color = cohort_colors_raw[i]
                    if len(color) == 4:
                        rgb = tuple(color[:3])
                    elif len(color) == 3:
                        rgb = tuple(color)
                    else:
                        rgb = (color[0], color[1], color[2])
                    cohort_color_map[cohort_name] = rgb
                    self.cohort_colors[cohort_name] = rgb
                
                handles = []
                labels = []
                for cohort_name in unique_cohorts:
                    mask = cohorts == cohort_name
                    if np.any(mask):
                        sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                        c=[cohort_color_map[cohort_name]],
                                        alpha=point_alpha, s=point_size, edgecolors='none')
                        color = cohort_color_map[cohort_name]
                        if len(color) == 4:
                            rgb = tuple(color[:3])
                        elif len(color) == 3:
                            rgb = tuple(color)
                        else:
                            rgb = (color[0], color[1], color[2])
                        handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                       markeredgecolor='none', markersize=6, alpha=point_alpha)
                        handles.append(handle)
                        labels.append(str(cohort_name) if cohort_name != 'Unassigned' else 'Unassigned')
                if show_legend and handles and labels:
                    legend = ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Cohorts')
                    legend.set_visible(True)
            else:
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        elif color_by == 'Manual Phenotype' and 'manual_phenotype' in self.feature_dataframe.columns:
            # Use pre-computed filtered data
            # Color by manual phenotype
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                manual_phenotype_series = self.feature_dataframe.loc[self.umap_index, 'manual_phenotype']
                manual_phenotypes = manual_phenotype_series.fillna('Unassigned').values[valid_mask]
            else:
                manual_phenotypes = self.feature_dataframe['manual_phenotype'].fillna('Unassigned').values[valid_mask]
            unique_phenotypes = sorted([p for p in np.unique(manual_phenotypes) if pd.notna(p)])
            colors = _get_vivid_colors(len(unique_phenotypes))
            phenotype_color_map = {p: colors[i] for i, p in enumerate(unique_phenotypes)}
            handles = []
            labels = []
            for phenotype in unique_phenotypes:
                mask = manual_phenotypes == phenotype
                sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                c=[phenotype_color_map[phenotype]],
                                alpha=point_alpha, s=point_size, edgecolors='none')
                # Create custom legend handle with fixed size
                color = phenotype_color_map[phenotype]
                if len(color) == 4:
                    rgb = tuple(color[:3])
                elif len(color) == 3:
                    rgb = tuple(color)
                else:
                    rgb = (color[0], color[1], color[2])
                handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                               markeredgecolor='none', markersize=6, alpha=point_alpha)
                handles.append(handle)
                labels.append(str(phenotype))
            if show_legend:
                ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title='Manual Phenotype')
        elif hasattr(self, 'umap_raw_data') and color_by in getattr(self, 'umap_selected_columns', []):
            # Check all conditions before processing
            if hasattr(self, 'umap_index'):
                pass
            
            # Use pre-computed filtered data
            # Align umap_raw_data with umap_index order
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                # Align umap_raw_data with umap_index order
                
                try:
                    feature_series = self.umap_raw_data.loc[self.umap_index, color_by]
                    vals = feature_series.values
                except Exception as e:
                    vals = self.umap_raw_data[color_by].values
            else:
                vals = self.umap_raw_data[color_by].values
            
            # Continuous coloring by selected feature (aligned to UMAP order)
            vals_filtered = vals[valid_mask]
            
            # Check for valid values
            if len(vals_filtered) == 0:
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', 
                          alpha=point_alpha, s=point_size, edgecolors='none')
                ax.text(0.5, 0.5, 'No valid values for coloring', transform=ax.transAxes, 
                       ha='center', va='center')
            elif np.all(np.isnan(vals_filtered)):
                ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', 
                          alpha=point_alpha, s=point_size, edgecolors='none')
                ax.text(0.5, 0.5, 'All values are NaN', transform=ax.transAxes, 
                       ha='center', va='center')
            else:
                # Remove NaN values for plotting
                valid_vals_mask = ~np.isnan(vals_filtered)
                if np.any(valid_vals_mask):
                    
                    try:
                        sc = ax.scatter(umap_embedding_filtered[valid_vals_mask, 0], 
                                      umap_embedding_filtered[valid_vals_mask, 1], 
                                      c=vals_filtered[valid_vals_mask],
                                      cmap='viridis', alpha=point_alpha, s=point_size, edgecolors='none')
                        cbar = self.figure.colorbar(sc, ax=ax)
                        cbar.set_label(color_by)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', 
                                  alpha=point_alpha, s=point_size, edgecolors='none')
                        ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, 
                               ha='center', va='center', fontsize=8)
                else:
                    ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', 
                              alpha=point_alpha, s=point_size, edgecolors='none')
                    ax.text(0.5, 0.5, 'No valid values for coloring', transform=ax.transAxes, 
                           ha='center', va='center')
        elif color_by in self.feature_dataframe.columns:
            # Handle metadata columns or other dataframe columns as categorical
            # (This comes after intensity feature check to avoid conflicts)
            metadata_cols = self._get_metadata_columns(self.feature_dataframe)
            if color_by in metadata_cols or color_by == 'batch_group':
                # Use pre-computed filtered data
                # Color by metadata column (categorical)
                if hasattr(self, 'umap_index') and self.umap_index is not None:
                    col_series = self.feature_dataframe.loc[self.umap_index, color_by]
                    col_values = col_series.fillna('Unknown').values[valid_mask]
                else:
                    col_values = self.feature_dataframe[color_by].fillna('Unknown').values[valid_mask]
                unique_values = sorted([v for v in np.unique(col_values) if pd.notna(v)])
                if len(unique_values) > 0:
                    colors = _get_vivid_colors(len(unique_values))
                    value_color_map = {v: colors[i] for i, v in enumerate(unique_values)}
                    handles = []
                    labels = []
                    for value in unique_values:
                        mask = col_values == value
                        if np.any(mask):
                            sc = ax.scatter(umap_embedding_filtered[mask, 0], umap_embedding_filtered[mask, 1],
                                            c=[value_color_map[value]],
                                            alpha=point_alpha, s=point_size, edgecolors='none')
                            color = value_color_map[value]
                            if len(color) == 4:
                                rgb = tuple(color[:3])
                            elif len(color) == 3:
                                rgb = tuple(color)
                            else:
                                rgb = (color[0], color[1], color[2])
                            handle = Line2D([0], [0], marker='o', color='none', markerfacecolor=rgb,
                                           markeredgecolor='none', markersize=6, alpha=point_alpha)
                            handles.append(handle)
                            labels.append(str(value))
                    if show_legend and handles and labels:
                        ax.legend(handles, labels, loc='best', frameon=True, fontsize=8, title=color_by)
        else:
            # Use pre-computed filtered data
            # Fallback single color
            ax.scatter(umap_embedding_filtered[:, 0], umap_embedding_filtered[:, 1], c='blue', alpha=point_alpha, s=point_size, edgecolors='none')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        if title is None:
            title = f'UMAP: {color_by}'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _create_umap_plot(self):
        """Create UMAP scatter plot(s) with faceted plotting support."""
        self._active_view_name = 'UMAP'
        if self.umap_embedding is None:
            return

        self._warn_large_umap_plot_once(len(self.umap_embedding))
        
        self.figure.clear()
        
        # Get selected color-by options (use stored data which contains actual column names)
        if hasattr(self, 'color_by_listwidget'):
            selected_items = []
            for item in self.color_by_listwidget.selectedItems():
                # Use stored data (actual column name) if available, otherwise use text
                actual_name = item.data(QtCore.Qt.UserRole)
                if actual_name:
                    selected_items.append(actual_name)
                else:
                    selected_items.append(item.text())
        else:
            # Fallback to combo box if list widget doesn't exist
            selected_items = [self.color_by_combo.currentText()] if hasattr(self, 'color_by_combo') else ['Cluster']
        
        # Ensure at least one selection
        if not selected_items:
            selected_items = ['Cluster']
        
        # Limit to max 3 plots (3 columns in single row)
        selected_items = selected_items[:3]
        
        # Get point size and alpha from controls
        point_size = self.point_size_spinbox.value() if hasattr(self, 'point_size_spinbox') else 18
        point_alpha = self.point_alpha_spinbox.value() if hasattr(self, 'point_alpha_spinbox') else 0.8
        show_legend = self.show_legend_checkbox.isChecked() if hasattr(self, 'show_legend_checkbox') else True
        
        n_plots = len(selected_items)
        
        if n_plots == 1:
            # Single plot - use full figure
            ax = self.figure.add_subplot(111)
            self._plot_umap_single(ax, selected_items[0], point_size, point_alpha, show_legend=show_legend)
            pass
        else:
            # Multiple plots - create subplots in a single row with max 3 columns
            n_cols = n_plots
            n_rows = 1
            
            for idx, color_by in enumerate(selected_items):
                ax = self.figure.add_subplot(n_rows, n_cols, idx + 1)
                self._plot_umap_single(ax, color_by, point_size, point_alpha, show_legend=show_legend)
            
            pass
        
        self._finalize_standard_plot(pad=0.9, allow_text_compaction=True)

    def _populate_color_by_options(self):
        """Populate the color-by list widget with Cluster + used features."""
        if not hasattr(self, 'color_by_listwidget'):
            return
        # Get currently selected items (use stored data for actual column names)
        selected_items = []
        for item in self.color_by_listwidget.selectedItems():
            actual_name = item.data(QtCore.Qt.UserRole)
            if actual_name:
                selected_items.append(actual_name)
            else:
                selected_items.append(item.text())
        if not selected_items:
            selected_items = ['Cluster']  # Default selection
        
        self.color_by_listwidget.blockSignals(True)
        self.color_by_listwidget.clear()
        
        # Add all available options
        options = ['Cluster']
        # Add source_file if available
        if 'source_file' in self.feature_dataframe.columns:
            options.append('Source File')
        # Add batch_group if available
        if 'batch_group' in self.feature_dataframe.columns:
            options.append('Batch Group')
        # Add metadata columns for coloring
        metadata_cols = self._get_metadata_columns(self.feature_dataframe)
        for col in metadata_cols:
            if col not in options:
                options.append(col)
        
        # Prioritize features used for clustering (from last_features_used)
        # These are the most relevant features to display
        features_to_show = []
        if hasattr(self, 'last_features_used') and self.last_features_used:
            features_to_show = list(self.last_features_used)
        elif hasattr(self, 'umap_selected_columns') and self.umap_selected_columns:
            features_to_show = list(self.umap_selected_columns)
        elif hasattr(self, 'tsne_selected_columns') and self.tsne_selected_columns:
            features_to_show = list(self.tsne_selected_columns)
        
        # Add feature columns (prioritize clustering features)
        for col in features_to_show:
            # Check if column is already in options (either as string or as tuple)
            already_in = col in options or any(isinstance(opt, tuple) and opt[1] == col for opt in options)
            if not already_in:
                # Use display name for UI, but we'll store the actual column name
                display_name = self._get_feature_display_name(col)
                options.append((display_name, col))  # Store as (display, actual) tuple
        # Add phenotype if available
        if hasattr(self, 'clustered_data') and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
            if 'Phenotype' not in options:
                options.append('Phenotype')
        # Add manual phenotype if available
        if 'manual_phenotype' in self.feature_dataframe.columns:
            if 'Manual Phenotype' not in options:
                options.append('Manual Phenotype')
        # Add Cohorts if cohorts exist
        if 'cohort' in self.feature_dataframe.columns and self.feature_dataframe['cohort'].notna().any():
            if 'Cohorts' not in options:
                options.append('Cohorts')
        
        # Add items to list widget
        for option in options:
            if isinstance(option, tuple):
                # Feature column: (display_name, actual_column_name)
                display_name, actual_name = option
                item = QtWidgets.QListWidgetItem(display_name)
                item.setData(QtCore.Qt.UserRole, actual_name)  # Store actual column name
            else:
                # Standard option (Cluster, Source File, etc.)
                item = QtWidgets.QListWidgetItem(option)
                item.setData(QtCore.Qt.UserRole, option)  # Store same value
            self.color_by_listwidget.addItem(item)
            # Check if this item should be selected (compare display name or actual name)
            item_text = item.text()
            item_data = item.data(QtCore.Qt.UserRole)
            if item_text in selected_items or item_data in selected_items:
                item.setSelected(True)
        
        # Ensure at least "Cluster" is selected if nothing was selected
        if not self.color_by_listwidget.selectedItems():
            for i in range(self.color_by_listwidget.count()):
                item = self.color_by_listwidget.item(i)
                if item.text() == 'Cluster':
                    item.setSelected(True)
                    break
        
        self.color_by_listwidget.blockSignals(False)

    def _filter_color_by_options(self, search_text: str):
        """Filter the color-by list widget items based on search text."""
        if not hasattr(self, 'color_by_listwidget'):
            return
        
        search_text = search_text.lower()
        for i in range(self.color_by_listwidget.count()):
            item = self.color_by_listwidget.item(i)
            item_text = item.text().lower()
            # Show item if search text is empty or matches item text
            item.setHidden(bool(search_text) and search_text not in item_text)
    
    def _on_color_by_changed(self, _text: str = None):
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else ''
        if view == 'UMAP' and getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()
        elif view == 't-SNE' and getattr(self, 'tsne_embedding', None) is not None:
            self._create_tsne_plot()

    def _on_point_style_changed(self):
        """Update plot when point size or alpha changes."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else ''
        if view == 'UMAP' and getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()
        elif view == 't-SNE' and getattr(self, 'tsne_embedding', None) is not None:
            self._create_tsne_plot()
    
    def _on_legend_changed(self):
        """Handle legend visibility changes for all plot types."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else ''
        if view == 'UMAP' and getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()
        elif view == 't-SNE' and getattr(self, 'tsne_embedding', None) is not None:
            self._create_tsne_plot()
        elif view == 'Stacked Bars':
            self._show_stacked_bars()
        elif view == 'Heatmap':
            # Recreate heatmap with updated legend visibility
            self._create_heatmap()
        elif view == 'Cluster Map':
            self._create_cluster_map()

    def _show_stacked_bars(self):
        """Show stacked bar plots of cluster frequencies per selected group (ROI/condition/slide)."""
        self._active_view_name = 'Stacked Bars'
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view stacked bars.")
            return
        
        # Use patient_annotation_column if set and available, otherwise use group_by_combo
        group_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            # Check if patient_annotation_column exists in feature_dataframe
            if self.patient_annotation_column in self.feature_dataframe.columns:
                group_col = self.patient_annotation_column
                # Also update group_by_combo to match if it exists
                if hasattr(self, 'group_by_combo'):
                    for i in range(self.group_by_combo.count()):
                        if self.group_by_combo.itemText(i) == group_col:
                            self.group_by_combo.setCurrentIndex(i)
                            break
        
        # Fall back to group_by_combo if patient_annotation_column not set or not available
        if not group_col:
            group_col = self.group_by_combo.currentText() if hasattr(self, 'group_by_combo') and self.group_by_combo.count() > 0 else None
        
        if not group_col:
            QtWidgets.QMessageBox.warning(self, "No Grouping", "No valid grouping column is available.")
            return
        
        # Handle acquisition_id: merge with source_file to create unique identifier
        if group_col == 'acquisition_id' and 'source_file' in self.feature_dataframe.columns:
            # Create merged column if it doesn't exist
            merged_col = 'source_file_acquisition_id'
            if merged_col not in self.feature_dataframe.columns:
                self.feature_dataframe[merged_col] = (
                    self.feature_dataframe['source_file'].astype(str) + '_' + 
                    self.feature_dataframe['acquisition_id'].astype(str)
                )
            group_col = merged_col
        
        if group_col not in self.feature_dataframe.columns:
            QtWidgets.QMessageBox.warning(self, "No Grouping", "No valid grouping column is available.")
            return

        try:
            # Filter out dropped clusters (cluster 0)
            filtered_clustered_data = self.clustered_data[self.clustered_data['cluster'] != 0].copy()
            
            # Apply cluster filter if specified
            filter_selection = getattr(self, 'stacked_bars_filter_selection', None)
            if filter_selection:
                # Convert display names to cluster IDs
                cluster_id_map = {}
                for cid in filtered_clustered_data['cluster'].unique():
                    display_name = self._get_cluster_display_name(cid)
                    cluster_id_map[display_name] = cid
                # Get cluster IDs that match the selected display names
                selected_cluster_ids = [cluster_id_map[name] for name in filter_selection if name in cluster_id_map]
                if selected_cluster_ids:
                    filtered_clustered_data = filtered_clustered_data[filtered_clustered_data['cluster'].isin(selected_cluster_ids)]
            
            # Align metadata to clustered_data order
            meta_series = self.feature_dataframe.loc[filtered_clustered_data.index, group_col]
            clusters = filtered_clustered_data['cluster']
            
            # Apply custom patient labels if available and using patient annotation column
            if (hasattr(self, 'patient_annotation_column') and 
                self.patient_annotation_column == group_col and 
                hasattr(self, 'patient_annotation_map') and 
                self.patient_annotation_map):
                # Map values using custom labels
                meta_series = meta_series.map(lambda x: self.patient_annotation_map.get(x, x) if pd.notna(x) else x)
            
            # Build counts per group and cluster
            ct = pd.crosstab(meta_series, clusters).sort_index()
            
            # Get view type (Fraction or Total enumeration)
            view_type = self.stacked_bars_view_type_combo.currentText() if hasattr(self, 'stacked_bars_view_type_combo') else 'Fraction'
            
            if view_type == 'Fraction':
                # Convert to frequencies
                data_to_plot = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
                ylabel = 'Fraction of cells'
                ylim = (0, 1)
            else:
                # Use raw counts
                data_to_plot = ct.copy()
                ylabel = 'Number of cells'
                ylim = None

            # Prepare colors consistent with other views
            unique_clusters = self._sorted_cluster_ids(clusters.unique(), canonical=False)
            colors = _get_vivid_colors(len(unique_clusters))
            cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}

            # Plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            bottom = np.zeros(len(data_to_plot))
            x = np.arange(len(data_to_plot))
            for cluster_id in unique_clusters:
                vals = data_to_plot.get(cluster_id, pd.Series(0, index=data_to_plot.index)).values
                ax.bar(x, vals, bottom=bottom, color=cluster_color_map[cluster_id], label=self._get_cluster_display_name(cluster_id))
                bottom = bottom + vals

            ax.set_xticks(x)
            # Use custom legend label if available and using patient annotation column
            xlabel = group_col
            if (hasattr(self, 'patient_annotation_column') and 
                self.patient_annotation_column == group_col and 
                hasattr(self, 'patient_legend_label')):
                xlabel = self.patient_legend_label
            ax.set_xticklabels([str(i) for i in data_to_plot.index], rotation=45, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(f'Cluster composition by {xlabel}')
            if ylim:
                ax.set_ylim(ylim)
            # Show legend only if checkbox is checked
            show_legend = self.show_legend_checkbox.isChecked() if hasattr(self, 'show_legend_checkbox') else True
            if show_legend:
                # Use multiple columns if there are many clusters to make legend more compact
                n_clusters = len(unique_clusters)
                # Calculate number of columns: use 2 columns if >10 clusters, 3 if >20, 4 if >30, etc.
                if n_clusters > 30:
                    ncol = 4
                elif n_clusters > 20:
                    ncol = 3
                elif n_clusters > 10:
                    ncol = 2
                else:
                    ncol = 1
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=ncol)
            self._finalize_standard_plot(pad=0.9, allow_text_compaction=True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating stacked bars: {str(e)}")

    def _show_differential_expression(self):
        """Show differential expression heatmap with top-N markers per cluster."""
        self._active_view_name = 'Differential Expression'
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view differential expression.")
            return
        
        try:
            # Filter out dropped clusters (cluster 0)
            filtered_clustered_data = self.clustered_data[self.clustered_data['cluster'] != 0].copy()
            
            # Get numeric feature columns only (exclude cluster/phenotype/text metadata)
            feature_cols = self._select_feature_columns(filtered_clustered_data)
            
            if not feature_cols:
                QtWidgets.QMessageBox.warning(self, "No Features", "No features available for differential expression analysis.")
                return
            
            # Calculate mean expression per cluster for each feature
            cluster_means = filtered_clustered_data.groupby('cluster')[feature_cols].mean()
            all_clusters = self._sorted_cluster_ids(cluster_means.index.tolist(), canonical=False)

            # Optionally restrict DE analysis to selected clusters
            selected_clusters = getattr(self, 'de_cluster_filter_selection', None)
            if selected_clusters:
                selected_cluster_ids = [cid for cid in all_clusters if int(cid) in selected_clusters]
                if not selected_cluster_ids:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "No Matching Clusters",
                        "None of the selected clusters are present in the current clustering result."
                    )
                    self.de_cluster_filter_selection = None
                    return
                cluster_means = cluster_means.loc[selected_cluster_ids]
            
            # Calculate differential expression (z-score across clusters for each feature)
            # This shows which features are most variable across clusters
            feature_means = cluster_means.mean(axis=0)  # Mean across clusters
            feature_stds = cluster_means.std(axis=0)    # Std across clusters
            
            # Avoid division by zero
            feature_stds = feature_stds.replace(0, 1)
            
            # Z-score normalization: (value - mean) / std
            differential_scores = (cluster_means - feature_means) / feature_stds
            
            # Find top N markers FOR EACH cluster individually
            # Get the user-selected number of top markers
            top_n = self.top_n_spinbox.value()
            
            # For each cluster, find the top N features with highest z-scores
            cluster_top_features = {}
            top_features = []
            seen_top_features = set()
            
            # Sort clusters for consistent ordering
            sorted_clusters = self._sorted_cluster_ids(differential_scores.index, canonical=False)
            
            for cluster_id in sorted_clusters:
                # Get z-scores for this cluster
                cluster_scores = differential_scores.loc[cluster_id]
                # Sort by z-score (descending) and take top N
                top_n_for_cluster = cluster_scores.nlargest(top_n).index.tolist()
                cluster_top_features[cluster_id] = top_n_for_cluster
                # Keep marker order stable, but only show each feature once.
                for feature_name in top_n_for_cluster:
                    if feature_name in seen_top_features:
                        continue
                    seen_top_features.add(feature_name)
                    top_features.append(feature_name)
            
            if not top_features:
                QtWidgets.QMessageBox.warning(self, "No Features", "No features found for differential expression analysis.")
                return
            
            # Create heatmap data with all top features
            heatmap_data = differential_scores[top_features]
            
            # Create the plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            flip_axes = bool(
                hasattr(self, 'de_flip_axes_checkbox')
                and self.de_flip_axes_checkbox.isChecked()
            )
            show_values = bool(
                hasattr(self, 'de_show_values_checkbox')
                and self.de_show_values_checkbox.isChecked()
            )
            show_highlights = bool(
                hasattr(self, 'de_show_boxes_checkbox')
                and self.de_show_boxes_checkbox.isChecked()
            )
            show_guidance = getattr(self, '_de_render_context', 'gui') == 'gui'

            # Create heatmap with user-selected colormap
            colormap_name = self._get_colormap_name()
            plot_matrix = heatmap_data if flip_axes else heatmap_data.T
            im = ax.imshow(
                plot_matrix,
                cmap=colormap_name,
                aspect='auto',
                vmin=-3,
                vmax=3,
            )  # Limit color scale to ±3 z-scores

            cluster_labels_display = [self._get_cluster_display_name(i) for i in heatmap_data.index]
            feature_labels_display = [self._get_feature_display_name(f) for f in heatmap_data.columns]

            if flip_axes:
                ax.set_xticks(range(len(heatmap_data.columns)))
                ax.set_xticklabels(feature_labels_display)
                ax.set_yticks(range(len(heatmap_data.index)))
                ax.set_yticklabels(cluster_labels_display, rotation=0)
            else:
                ax.set_xticks(range(len(heatmap_data.index)))
                ax.set_xticklabels(cluster_labels_display)
                ax.set_yticks(range(len(heatmap_data.columns)))
                ax.set_yticklabels(feature_labels_display, rotation=0)
            
            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Z-score (Differential Expression)', rotation=270, labelpad=20)
            
            # Add title and labels
            clusters_used = len(differential_scores.index)
            total_clusters = len(all_clusters)
            title_suffix = ""
            if selected_clusters and clusters_used < total_clusters:
                title_suffix = f" ({clusters_used}/{total_clusters} clusters selected)"
            ax.set_title(f'Top {top_n} Differential Expression Markers per Cluster{title_suffix}')
            ax.set_xlabel('Features' if flip_axes else 'Clusters')
            ax.set_ylabel('Clusters' if flip_axes else 'Features')
            
            # Add top-marker highlights and optional z-score annotations.
            for i in range(len(heatmap_data.index)):
                cluster_id = heatmap_data.index[i]
                top_n_for_this_cluster = cluster_top_features[cluster_id]
                
                for j in range(len(heatmap_data.columns)):
                    feature_name = heatmap_data.columns[j]
                    value = heatmap_data.iloc[i, j]
                    
                    if show_highlights and feature_name in top_n_for_this_cluster:
                        # Add a subtle highlight around the per-cluster top markers.
                        rect_x = j - 0.4 if flip_axes else i - 0.4
                        rect_y = i - 0.4 if flip_axes else j - 0.4
                        ax.add_patch(
                            plt.Rectangle(
                                (rect_x, rect_y),
                                0.8,
                                0.8,
                                fill=False,
                                edgecolor='black',
                                linewidth=2,
                                alpha=0.7,
                            )
                        )

                    if show_values:
                        text_color = 'white' if abs(value) > 1.5 else 'black'
                        text_x = j if flip_axes else i
                        text_y = i if flip_axes else j
                        ax.text(
                            text_x,
                            text_y,
                            f'{value:.2f}',
                            ha='center',
                            va='center',
                            color=text_color,
                            fontsize=10 if feature_name in top_n_for_this_cluster else 9,
                            fontweight='bold',
                        )
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Keep the explanatory note in the GUI only so exports stay manuscript-ready.
            if show_guidance:
                guidance_lines = []
                if show_highlights:
                    guidance_lines.append(
                        f"Black boxes highlight the top {top_n} markers for each cluster."
                    )
                if show_values:
                    guidance_lines.append(
                        "Z-scores show how much each cluster differs from the overall mean."
                    )
                if guidance_lines:
                    ax.text(
                        1.02,
                        -0.15,
                        "\n".join(guidance_lines),
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment='top',
                        horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    )
            
            self._finalize_standard_plot(pad=0.9, allow_text_compaction=True)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating differential expression heatmap: {str(e)}")

    def _open_marker_selection_dialog(self):
        """Open a dialog to select markers for boxplot/violin plot visualization."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to select markers.")
            return
        
        # Get available marker columns
        marker_cols = self._select_feature_columns(self.clustered_data)
        
        if not marker_cols:
            QtWidgets.QMessageBox.warning(self, "No Markers", "No markers available for visualization.")
            return
        
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Select Markers")
        dlg.setMinimumSize(400, 500)
        
        layout = QtWidgets.QVBoxLayout(dlg)
        
        # Instructions
        instructions = QtWidgets.QLabel("Select markers to visualize (multiple selection allowed):")
        layout.addWidget(instructions)
        
        # List widget with multi-selection
        list_widget = QtWidgets.QListWidget()
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        
        # Add markers to list
        for marker in sorted(marker_cols):
            item = QtWidgets.QListWidgetItem(marker)
            # Pre-select if already in selected_markers
            if marker in self.selected_markers:
                item.setSelected(True)
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        clear_all_btn = QtWidgets.QPushButton("Clear All")
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        
        def select_all():
            for i in range(list_widget.count()):
                list_widget.item(i).setSelected(True)
        
        def clear_all():
            for i in range(list_widget.count()):
                list_widget.item(i).setSelected(False)
        
        select_all_btn.clicked.connect(select_all)
        clear_all_btn.clicked.connect(clear_all)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(clear_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Get selected markers
            selected_items = list_widget.selectedItems()
            self.selected_markers = [item.text() for item in selected_items]
            
            # Refresh the plot if we're in boxplot/violin view
            if self.view_combo.currentText() == 'Boxplot/Violin Plot':
                self._show_boxplot_violin()

    def _on_plot_type_changed(self, _text: str):
        """Handle plot type change (boxplot vs violin plot)."""
        # Refresh the plot if we're in boxplot/violin view and have markers selected
        if self.view_combo.currentText() == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()

    def _on_stats_test_changed(self, _state: int):
        """Handle statistical testing checkbox change."""
        # Update cluster combo visibility and enable export button
        if hasattr(self, 'stats_cluster_combo') and hasattr(self, 'stats_cluster_label'):
            is_enabled = self.stats_test_checkbox.isChecked()
            self.stats_cluster_combo.setEnabled(is_enabled)
            self.stats_cluster_label.setEnabled(is_enabled)
            if is_enabled:
                self._update_stats_cluster_combo()
        if hasattr(self, 'stats_export_btn'):
            self.stats_export_btn.setEnabled(
                self.stats_test_checkbox.isChecked()
                and (len(self.statistical_results) > 0 or len(getattr(self, 'statistical_results_summary', {})) > 0)
            )
        # Refresh the plot if we're in boxplot/violin view
        if self.view_combo.currentText() == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()

    def _on_stats_mode_changed(self, _text: str):
        """Handle statistical test mode change."""
        # Update cluster combo visibility based on mode
        if hasattr(self, 'stats_mode_combo') and hasattr(self, 'stats_cluster_combo'):
            is_one_vs_others = self.stats_mode_combo.currentText() == "One vs Others"
            self.stats_cluster_combo.setVisible(is_one_vs_others)
            if hasattr(self, 'stats_cluster_label'):
                self.stats_cluster_label.setVisible(is_one_vs_others)
            if is_one_vs_others:
                self._update_stats_cluster_combo()
        # Refresh the plot if we're in boxplot/violin view
        if self.view_combo.currentText() == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()

    def _on_stats_cluster_changed(self, _text: str):
        """Handle reference cluster selection change for one-vs-others mode."""
        # Refresh the plot if we're in boxplot/violin view
        if self.view_combo.currentText() == 'Boxplot/Violin Plot':
            self._show_boxplot_violin()

    def _update_stats_cluster_combo(self):
        """Update the cluster combo box with available clusters."""
        if not hasattr(self, 'stats_cluster_combo'):
            return
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            return
        
        previous_cluster_id = self.stats_cluster_combo.currentData()
        self.stats_cluster_combo.clear()
        # Filter out dropped clusters (cluster 0)
        unique_cluster_ids = self._sorted_cluster_ids(
            [cid for cid in self.clustered_data['cluster'].unique() if cid != 0],
            canonical=False,
        )
        selected_index = 0
        for idx, cluster_id in enumerate(unique_cluster_ids):
            cluster_name = self._get_cluster_display_name(cluster_id)
            self.stats_cluster_combo.addItem(cluster_name, cluster_id)
            if cluster_id == previous_cluster_id:
                selected_index = idx
        if unique_cluster_ids:
            self.stats_cluster_combo.setCurrentIndex(selected_index)

    def _bh_correction(self, p_values):
        """Apply Benjamini-Hochberg correction for multiple testing.
        
        Args:
            p_values: List or array of p-values
            
        Returns:
            List of adjusted p-values
        """
        p_values = np.array(p_values)
        n = len(p_values)
        if n == 0:
            return []
        
        # Sort p-values with their original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Apply BH correction
        adjusted_p = np.zeros(n)
        for i in range(n-1, -1, -1):
            if i == n-1:
                adjusted_p[i] = min(sorted_p[i], 1.0)
            else:
                adjusted_p[i] = min(min(adjusted_p[i+1], sorted_p[i] * n / (i+1)), 1.0)
        
        # Restore original order
        result = np.zeros(n)
        result[sorted_indices] = adjusted_p
        return result.tolist()

    def _perform_pairwise_tests(self, data_dict, cluster_ids, mode='pairwise', reference_cluster=None):
        """Perform pairwise Mann-Whitney U tests.
        
        Args:
            data_dict: Dictionary mapping cluster_id to array of values
            cluster_ids: List of cluster IDs to test
            mode: 'pairwise' for all pairs, 'one_vs_others' for one vs all others
            reference_cluster: Cluster to compare against others (for one_vs_others mode)
            
        Returns:
            List of tuples: (cluster1, cluster2, p_value, adjusted_p_value)
        """
        results = []
        p_values = []
        pairs = []
        
        if mode == 'one_vs_others' and reference_cluster is not None:
            # One cluster vs all others
            if reference_cluster not in cluster_ids:
                return results
            
            data1 = data_dict[reference_cluster]
            if len(data1) < 2:
                return results
            
            for cluster2 in cluster_ids:
                if cluster2 == reference_cluster:
                    continue
                data2 = data_dict[cluster2]
                
                # Skip if insufficient data
                if len(data2) < 2:
                    continue
                
                try:
                    # Mann-Whitney U test (two-sided)
                    statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    p_values.append(p_value)
                    pairs.append((reference_cluster, cluster2))
                except Exception:
                    # If test fails, skip this pair
                    continue
        else:
            # All pairwise comparisons
            for i, cluster1 in enumerate(cluster_ids):
                for cluster2 in cluster_ids[i+1:]:
                    data1 = data_dict[cluster1]
                    data2 = data_dict[cluster2]
                    
                    # Skip if insufficient data
                    if len(data1) < 2 or len(data2) < 2:
                        continue
                    
                    try:
                        # Mann-Whitney U test (two-sided)
                        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                        p_values.append(p_value)
                        pairs.append((cluster1, cluster2))
                    except Exception:
                        # If test fails, skip this pair
                        continue
        
        # Apply BH correction
        if p_values:
            adjusted_p_values = self._bh_correction(p_values)
            
            # Build results list
            for (cluster1, cluster2), p_val, adj_p_val in zip(pairs, p_values, adjusted_p_values):
                results.append((cluster1, cluster2, p_val, adj_p_val))
        
        return results

    def _prepare_statistical_results(
        self,
        plot_data_source: pd.DataFrame,
        markers,
        marker_cluster_orders,
        cluster_name_to_id,
        *,
        perform_stats: bool,
        test_mode: str,
        reference_cluster=None,
    ):
        """Compute omnibus and pairwise non-parametric tests for the current marker set."""
        results_by_marker = {}
        summaries_by_marker = {}
        overall_entries = []
        pairwise_entries = []

        if not perform_stats:
            return results_by_marker, summaries_by_marker

        for marker in markers:
            cluster_order_names = marker_cluster_orders.get(marker, [])
            cluster_data_dict = {}
            valid_cluster_order = []

            for cluster_name in cluster_order_names:
                cluster_id = cluster_name_to_id.get(cluster_name)
                if cluster_id is None:
                    continue
                values = pd.to_numeric(
                    plot_data_source.loc[plot_data_source['cluster'] == cluster_id, marker],
                    errors='coerce',
                ).dropna().to_numpy(dtype=float)
                if len(values) < 2:
                    continue
                cluster_data_dict[cluster_name] = values
                valid_cluster_order.append(cluster_name)

            summary = {
                'overall_test': "Kruskal-Wallis",
                'pairwise_test': "Mann-Whitney U",
                'pairwise_correction': "Benjamini-Hochberg across displayed comparisons",
                'overall_correction': "Benjamini-Hochberg across displayed markers",
                'overall_p_value': None,
                'overall_adj_p_value': None,
                'n_comparisons': 0,
                'n_significant_pairs': 0,
                'significant_pairs': [],
                'display_pairs': [],
                'note': None,
            }

            results_by_marker[marker] = []
            summaries_by_marker[marker] = summary

            if len(valid_cluster_order) < 2:
                summary['note'] = "Insufficient data for multi-group testing."
                continue

            try:
                overall_result = kruskal(*[cluster_data_dict[name] for name in valid_cluster_order], nan_policy='omit')
                overall_p_value = float(overall_result.pvalue)
                if np.isfinite(overall_p_value):
                    summary['overall_p_value'] = overall_p_value
                    overall_entries.append((marker, overall_p_value))
            except Exception:
                summary['note'] = "Omnibus test could not be computed."

            if test_mode == 'one_vs_others':
                if reference_cluster not in valid_cluster_order:
                    continue
                pairs_to_test = [
                    (reference_cluster, cluster_name)
                    for cluster_name in valid_cluster_order
                    if cluster_name != reference_cluster
                ]
            else:
                pairs_to_test = []
                for idx, cluster1 in enumerate(valid_cluster_order):
                    for cluster2 in valid_cluster_order[idx + 1:]:
                        pairs_to_test.append((cluster1, cluster2))

            for cluster1, cluster2 in pairs_to_test:
                data1 = cluster_data_dict.get(cluster1)
                data2 = cluster_data_dict.get(cluster2)
                if data1 is None or data2 is None or len(data1) < 2 or len(data2) < 2:
                    continue
                try:
                    test_result = mannwhitneyu(data1, data2, alternative='two-sided')
                    raw_p_value = float(test_result.pvalue)
                except Exception:
                    continue
                pair_idx = len(results_by_marker[marker])
                results_by_marker[marker].append((cluster1, cluster2, raw_p_value, raw_p_value))
                pairwise_entries.append((marker, pair_idx, raw_p_value))

            summary['n_comparisons'] = len(results_by_marker[marker])

        if overall_entries:
            overall_adj_values = self._bh_correction([entry[1] for entry in overall_entries])
            for (marker, _raw_p_value), adj_p_value in zip(overall_entries, overall_adj_values):
                summaries_by_marker[marker]['overall_adj_p_value'] = float(adj_p_value)

        if pairwise_entries:
            pairwise_adj_values = self._bh_correction([entry[2] for entry in pairwise_entries])
            for (marker, pair_idx, _raw_p_value), adj_p_value in zip(pairwise_entries, pairwise_adj_values):
                cluster1, cluster2, raw_p_value, _ = results_by_marker[marker][pair_idx]
                results_by_marker[marker][pair_idx] = (cluster1, cluster2, raw_p_value, float(adj_p_value))

        for marker, marker_results in results_by_marker.items():
            summary = summaries_by_marker[marker]
            significant_pairs = [result for result in marker_results if result[3] < 0.05]
            summary['significant_pairs'] = significant_pairs
            summary['n_significant_pairs'] = len(significant_pairs)

            overall_adj = summary.get('overall_adj_p_value')
            if overall_adj is not None and overall_adj >= 0.05 and significant_pairs:
                summary['note'] = (
                    "Overall Kruskal-Wallis test was not significant after BH correction; "
                    "pairwise results are available for export but are hidden on the plot."
                )
            elif not significant_pairs and summary['n_comparisons'] > 0 and summary['note'] is None:
                summary['note'] = "No pairwise comparisons remained significant after BH correction."

            if overall_adj is not None and overall_adj < 0.05:
                summary['display_pairs'] = significant_pairs
            else:
                summary['display_pairs'] = []

        return results_by_marker, summaries_by_marker

    def _format_adjusted_p_value(self, p_value):
        """Format an adjusted p-value for compact on-plot display."""
        if p_value is None:
            return "NA"
        try:
            p_value = float(p_value)
        except Exception:
            return "NA"
        if not np.isfinite(p_value):
            return "NA"
        if p_value < 0.001:
            return "<0.001"
        if p_value < 0.01:
            return f"{p_value:.3f}"
        return f"{p_value:.2f}"

    def _build_statistical_summary_text(self, marker_name):
        """Create a compact statistical summary suitable for plot annotation."""
        summary = self.statistical_results_summary.get(marker_name)
        if not summary:
            return None

        lines = [
            "Stats: Kruskal-Wallis + pairwise Mann-Whitney U",
        ]
        overall_adj = summary.get('overall_adj_p_value')
        if overall_adj is not None:
            lines.append(f"Overall q={self._format_adjusted_p_value(overall_adj)}")
        elif summary.get('overall_p_value') is not None:
            lines.append(f"Overall p={self._format_adjusted_p_value(summary['overall_p_value'])}")

        display_pairs = summary.get('display_pairs') or []
        if display_pairs:
            max_pairs = 4
            lines.append(
                f"{summary['n_significant_pairs']} significant pairwise comparison"
                f"{'' if summary['n_significant_pairs'] == 1 else 's'}"
            )
            for cluster1, cluster2, _p_value, adj_p_value in display_pairs[:max_pairs]:
                lines.append(
                    f"{cluster1} vs {cluster2}: q={self._format_adjusted_p_value(adj_p_value)}"
                )
            if len(display_pairs) > max_pairs:
                lines.append(f"+{len(display_pairs) - max_pairs} more in export")
        elif summary.get('note'):
            lines.append(summary['note'])
        elif summary.get('n_comparisons', 0) == 0:
            lines.append("Insufficient pairwise comparisons.")

        return "\n".join(lines)

    def _annotate_statistical_summary(self, ax, marker_name):
        """Render the compact statistical summary box inside a plot."""
        summary_text = self._build_statistical_summary_text(marker_name)
        if not summary_text:
            return
        ax.text(
            0.02,
            0.98,
            summary_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.88, edgecolor='gray'),
        )

    def _get_significance_stars(self, p_value):
        """Convert p-value to significance stars.
        
        Args:
            p_value: Adjusted p-value
            
        Returns:
            String with asterisks representing significance level
        """
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'

    def _draw_significance_bar(self, ax, x1, x2, y, text, line_height=0.02, text_offset=0.03):
        """Draw a significance bar between two positions.
        
        Args:
            ax: Matplotlib axis
            x1, x2: X positions of the two groups
            y: Y position for the bar
            text: Text to display (e.g., '***')
            line_height: Height of the bar line
            text_offset: Offset for text above the bar (in data coordinates)
        """
        # Get y-axis range for proper scaling
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        
        # Draw horizontal line
        ax.plot([x1, x2], [y, y], 'k', linewidth=1)
        # Draw vertical lines at ends
        ax.plot([x1, x1], [y - line_height * y_range / 2, y], 'k', linewidth=1)
        ax.plot([x2, x2], [y - line_height * y_range / 2, y], 'k', linewidth=1)
        # Add text with proper spacing
        ax.text((x1 + x2) / 2, y + text_offset * y_range, text, ha='center', va='bottom', fontsize=9, fontweight='bold')

    def _show_boxplot_violin(self):
        """Show boxplot or violin plot of marker expressions by cluster."""
        self._active_view_name = 'Boxplot/Violin Plot'
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Please run clustering first to view boxplot/violin plots", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return
        
        if not self.selected_markers:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Please select markers to visualize\n(Click 'Select Markers...' button)", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return
        
        try:
            # Use raw (unscaled) data for boxplot/violin plots instead of z-scored data
            # Fall back to clustered_data if unscaled data is not available
            plot_data_source = self.clustered_data_unscaled if (
                hasattr(self, 'clustered_data_unscaled') and 
                self.clustered_data_unscaled is not None and 
                'cluster' in self.clustered_data_unscaled.columns
            ) else self.clustered_data
            
            # Filter out dropped clusters (cluster 0)
            plot_data_source = plot_data_source[plot_data_source['cluster'] != 0].copy()
            
            # Filter markers to only those available in the data
            available_markers = [m for m in self.selected_markers if m in plot_data_source.columns]
            
            if not available_markers:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "Selected markers not found in data", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas.draw()
                return
            
            self.figure.clear()
            
            # Prepare data for plotting
            # Store both original marker name (for filtering) and display name (for labels)
            marker_display_map = {marker: self._get_feature_display_name(marker) for marker in available_markers}
            plot_data = []
            for marker in available_markers:
                for cluster_id in self._sorted_cluster_ids(plot_data_source['cluster'].unique(), canonical=False):
                    cluster_data = plot_data_source[
                        plot_data_source['cluster'] == cluster_id
                    ][marker].dropna()
                    
                    for value in cluster_data:
                        plot_data.append({
                            'Marker': marker,  # Keep original for filtering
                            'MarkerDisplay': marker_display_map[marker],  # Custom label for display
                            'Cluster': self._get_cluster_display_name(cluster_id),
                            'Value': value
                        })
            
            if not plot_data:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No data available for selected markers", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                self.canvas.draw()
                return
            
            df_plot = pd.DataFrame(plot_data)
            
            # Get cluster colors using vivid colormap (same as UMAP)
            unique_cluster_ids = self._sorted_cluster_ids(self.clustered_data['cluster'].unique(), canonical=False)
            unique_cluster_names = [self._get_cluster_display_name(cid) for cid in unique_cluster_ids]
            cluster_name_to_id = {
                self._get_cluster_display_name(cid): cid for cid in unique_cluster_ids
            }
            colors = _get_vivid_colors(len(unique_cluster_ids))
            cluster_name_color_map = {name: colors[i] for i, name in enumerate(unique_cluster_names)}
            
            # Check if statistical testing is enabled
            perform_stats = self.stats_test_checkbox.isChecked() if hasattr(self, 'stats_test_checkbox') else False
            
            # Get statistical test mode and reference cluster
            test_mode = 'pairwise'
            reference_cluster = None
            stats_warning_text = None
            if perform_stats and hasattr(self, 'stats_mode_combo'):
                mode_text = self.stats_mode_combo.currentText()
                if mode_text == "One vs Others":
                    test_mode = 'one_vs_others'
                    reference_cluster_id = (
                        self.stats_cluster_combo.currentData()
                        if hasattr(self, 'stats_cluster_combo') and self.stats_cluster_combo.count() > 0
                        else None
                    )
                    if reference_cluster_id in unique_cluster_ids:
                        reference_cluster = self._get_cluster_display_name(reference_cluster_id)
                    else:
                        stats_warning_text = (
                            "Reference cluster is unavailable in the current clustering. "
                            "One-vs-others statistics were skipped."
                        )
            
            # Clear previous results
            self.statistical_results = {}
            self.statistical_results_summary = {}
            
            # Determine plot type
            plot_type = self.plot_type_combo.currentText()
            use_violin = plot_type == 'Violin Plot'

            marker_cluster_orders = {
                marker: sorted(
                    df_plot[df_plot['Marker'] == marker]['Cluster'].unique(),
                    key=lambda label: cluster_sort_key(
                        cluster_name_to_id.get(label, label),
                        annotation_map=self.cluster_annotation_map,
                    ),
                )
                for marker in available_markers
            }

            if perform_stats and stats_warning_text is None:
                self.statistical_results, self.statistical_results_summary = self._prepare_statistical_results(
                    plot_data_source,
                    available_markers,
                    marker_cluster_orders,
                    cluster_name_to_id,
                    perform_stats=perform_stats,
                    test_mode=test_mode,
                    reference_cluster=reference_cluster,
                )
            elif perform_stats:
                self.statistical_results = {marker: [] for marker in available_markers}
                self.statistical_results_summary = {
                    marker: {
                        'overall_test': "Kruskal-Wallis",
                        'pairwise_test': "Mann-Whitney U",
                        'pairwise_correction': "Benjamini-Hochberg across displayed comparisons",
                        'overall_correction': "Benjamini-Hochberg across displayed markers",
                        'overall_p_value': None,
                        'overall_adj_p_value': None,
                        'n_comparisons': 0,
                        'n_significant_pairs': 0,
                        'significant_pairs': [],
                        'display_pairs': [],
                        'note': stats_warning_text,
                    }
                    for marker in available_markers
                }
            
            # Use seaborn if available, otherwise matplotlib
            if _HAVE_SEABORN and len(available_markers) > 1:
                # Faceted plot for multiple markers
                n_markers = len(available_markers)
                n_cols = min(2, n_markers)
                n_rows = (n_markers + n_cols - 1) // n_cols
                
                # Create faceted plot with shared x-axis but not shared y-axis
                self.figure.clear()
                axes = []
                first_ax_per_col = {}  # Store first axis for each column
                for idx, marker in enumerate(available_markers):
                    row = idx // n_cols
                    col = idx % n_cols
                    pos = row * n_cols + col + 1
                    
                    # Create subplot with appropriate sharing
                    if row == 0:
                        # First row - no sharing needed
                        ax = self.figure.add_subplot(n_rows, n_cols, pos)
                        first_ax_per_col[col] = ax
                    else:
                        # Share x-axis with first subplot in same column
                        share_ax = first_ax_per_col[col]
                        ax = self.figure.add_subplot(n_rows, n_cols, pos, sharex=share_ax)
                    
                    axes.append(ax)
                    marker_data = df_plot[df_plot['Marker'] == marker]
                    
                    # Get display name for this marker
                    marker_display = marker_display_map[marker]
                    
                    # Create color palette ordered by cluster names in the plot
                    cluster_order = sorted(
                        marker_data['Cluster'].unique(),
                        key=lambda label: cluster_sort_key(
                            cluster_name_to_id.get(label, label),
                            annotation_map=self.cluster_annotation_map,
                        ),
                    )
                    palette = [cluster_name_color_map.get(cluster, 'gray') for cluster in cluster_order]
                    
                    if use_violin:
                        sns.violinplot(data=marker_data, x='Cluster', y='Value', ax=ax, hue='Cluster', palette=palette, order=cluster_order, legend=False)
                    else:
                        sns.boxplot(data=marker_data, x='Cluster', y='Value', ax=ax, hue='Cluster', palette=palette, order=cluster_order, legend=False)
                    
                    if perform_stats:
                        self._annotate_statistical_summary(ax, marker)
                    
                    ax.set_title(marker_display, fontsize=10)
                    if row == n_rows - 1:  # Only show xlabel on bottom row
                        ax.set_xlabel('Cluster', fontsize=9)
                    else:
                        ax.set_xlabel('')
                        # Hide x-axis tick labels for non-bottom rows (sharex handles this, but ensure it)
                        plt.setp(ax.get_xticklabels(), visible=False)
                    ax.set_ylabel('Expression Value', fontsize=9)
                    if row == n_rows - 1:  # Only rotate x-axis labels on bottom row
                        ax.tick_params(axis='x', rotation=45, labelsize=8)
                    else:
                        ax.tick_params(axis='x', labelsize=8)
                    ax.tick_params(axis='y', labelsize=8)
                
            elif _HAVE_SEABORN and len(available_markers) == 1:
                # Single marker with seaborn
                marker = available_markers[0]
                marker_display = marker_display_map[marker]
                marker_data = df_plot[df_plot['Marker'] == marker]
                
                ax = self.figure.add_subplot(111)
                
                # Create color palette ordered by cluster names in the plot
                cluster_order = sorted(
                    marker_data['Cluster'].unique(),
                    key=lambda label: cluster_sort_key(
                        cluster_name_to_id.get(label, label),
                        annotation_map=self.cluster_annotation_map,
                    ),
                )
                palette = [cluster_name_color_map.get(cluster, 'gray') for cluster in cluster_order]
                
                if use_violin:
                    sns.violinplot(
                        data=marker_data,
                        x='Cluster',
                        y='Value',
                        ax=ax,
                        hue='Cluster',
                        palette=palette,
                        order=cluster_order,
                        legend=False,
                    )
                else:
                    sns.boxplot(
                        data=marker_data,
                        x='Cluster',
                        y='Value',
                        ax=ax,
                        hue='Cluster',
                        palette=palette,
                        order=cluster_order,
                        legend=False,
                    )
                
                if perform_stats:
                    self._annotate_statistical_summary(ax, marker)
                
                ax.set_title(marker_display, fontsize=12)
                ax.set_xlabel('Cluster', fontsize=10)
                ax.set_ylabel('Expression Value', fontsize=10)
                ax.tick_params(axis='x', rotation=45, labelsize=9)
            else:
                # Fallback to matplotlib if seaborn not available
                n_markers = len(available_markers)
                n_cols = min(2, n_markers)
                n_rows = (n_markers + n_cols - 1) // n_cols
                
                for idx, marker in enumerate(available_markers):
                    ax = self.figure.add_subplot(n_rows, n_cols, idx + 1)
                    marker_display = marker_display_map[marker]
                    marker_data = df_plot[df_plot['Marker'] == marker]
                    
                    # Group data by cluster
                    cluster_values = {}
                    cluster_order = sorted(
                        marker_data['Cluster'].unique(),
                        key=lambda label: cluster_sort_key(
                            cluster_name_to_id.get(label, label),
                            annotation_map=self.cluster_annotation_map,
                        ),
                    )
                    for cluster in cluster_order:
                        cluster_values[cluster] = marker_data[
                            marker_data['Cluster'] == cluster
                        ]['Value'].values
                    
                    # Get colors for clusters
                    cluster_colors = [cluster_name_color_map.get(cluster, 'gray') for cluster in cluster_order]
                    
                    # Create boxplot or violin-like plot
                    if use_violin:
                        # Simple violin plot approximation with KDE
                        positions = range(len(cluster_values))
                        cluster_names = list(cluster_values.keys())
                        
                        for i, (cluster, values) in enumerate(cluster_values.items()):
                            if len(values) > 0:
                                # Use kde for violin shape approximation
                                from scipy.stats import gaussian_kde
                                try:
                                    kde = gaussian_kde(values)
                                    y_range = np.linspace(values.min(), values.max(), 100)
                                    density = kde(y_range)
                                    # Normalize density for width
                                    density = density / density.max() * 0.3
                                    ax.fill_betweenx(y_range, i - density, i + density, 
                                                     alpha=0.6, color=cluster_colors[i])
                                except:
                                    # Fallback to histogram if kde fails
                                    parts = ax.violinplot([values], positions=[i], widths=0.6, showmeans=True)
                                    for pc in parts['bodies']:
                                        pc.set_facecolor(cluster_colors[i])
                    else:
                        # Boxplot
                        positions = range(len(cluster_values))
                        cluster_names = list(cluster_values.keys())
                        bp = ax.boxplot(list(cluster_values.values()), positions=positions, widths=0.6)
                        # Color the boxplot elements
                        for i, patch in enumerate(bp['boxes']):
                            patch.set_facecolor(cluster_colors[i])
                            patch.set_alpha(0.7)
                        for median in bp['medians']:
                            median.set_color('black')
                        for whisker in bp['whiskers']:
                            whisker.set_color('black')
                        for cap in bp['caps']:
                            cap.set_color('black')
                    
                    if perform_stats:
                        self._annotate_statistical_summary(ax, marker)
                    
                    ax.set_xticks(range(len(cluster_values)))
                    ax.set_xticklabels(cluster_names, rotation=45, ha='right')
                    ax.set_title(marker_display, fontsize=10)
                    ax.set_xlabel('Cluster', fontsize=9)
                    ax.set_ylabel('Expression Value', fontsize=9)
                
            self._finalize_standard_plot(pad=0.95, allow_text_compaction=True)
            
            # Enable export button if statistical results are available
            if hasattr(self, 'stats_export_btn'):
                self.stats_export_btn.setEnabled(
                    perform_stats
                    and (len(self.statistical_results) > 0 or len(self.statistical_results_summary) > 0)
                )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating boxplot/violin plot: {str(e)}")
            import traceback
            traceback.print_exc()

    def _export_statistical_results(self):
        """Export statistical test results to CSV file."""
        if not self.statistical_results:
            QtWidgets.QMessageBox.warning(self, "No Results", "No statistical test results available to export.")
            return
        
        # Get default filename
        default = "statistical_test_results.csv"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Statistical Test Results", default,
            "CSV Files (*.csv)"
        )
        if not file_path:
            return
        
        try:
            # Prepare data for export
            export_data = []
            for marker, results in self.statistical_results.items():
                summary = self.statistical_results_summary.get(marker, {})
                if results:
                    for cluster1, cluster2, p_value, adj_p_value in results:
                        export_data.append({
                            'Marker': marker,
                            'Overall_Test': summary.get('overall_test'),
                            'Overall_P_value': summary.get('overall_p_value'),
                            'Overall_Adjusted_P_value_BH': summary.get('overall_adj_p_value'),
                            'Pairwise_Test': summary.get('pairwise_test'),
                            'Cluster_1': cluster1,
                            'Cluster_2': cluster2,
                            'P_value': p_value,
                            'Adjusted_P_value_BH': adj_p_value,
                            'Significant': 'Yes' if adj_p_value < 0.05 else 'No',
                            'Significance_level': self._get_significance_stars(adj_p_value),
                            'Note': summary.get('note'),
                        })
                elif summary:
                    export_data.append({
                        'Marker': marker,
                        'Overall_Test': summary.get('overall_test'),
                        'Overall_P_value': summary.get('overall_p_value'),
                        'Overall_Adjusted_P_value_BH': summary.get('overall_adj_p_value'),
                        'Pairwise_Test': summary.get('pairwise_test'),
                        'Cluster_1': '',
                        'Cluster_2': '',
                        'P_value': None,
                        'Adjusted_P_value_BH': None,
                        'Significant': 'No',
                        'Significance_level': '',
                        'Note': summary.get('note'),
                    })
            
            if not export_data:
                QtWidgets.QMessageBox.warning(self, "No Results", "No statistical test results to export.")
                return
            
            # Create DataFrame and save
            df_export = pd.DataFrame(export_data)
            df_export.to_csv(file_path, index=False)
            
            # Show success message
            n_tests = len(export_data)
            n_significant = sum(1 for r in export_data if r['Significant'] == 'Yes')
            summary = f"Exported {n_tests} statistical test results"
            if n_significant > 0:
                summary += f"\n{n_significant} significant comparisons (adjusted p < 0.05)"
            
            QtWidgets.QMessageBox.information(self, "Export Success", 
                                            f"Statistical test results saved to:\n{file_path}\n\n{summary}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting statistical results: {str(e)}")

    def _open_k_range_dialog(self):
        """Open dialog to search over k range and plot elbow/silhouette scores."""
        clustering_type = self.clustering_type.currentText()
        if clustering_type not in ["Hierarchical", "K-means"]:
            QtWidgets.QMessageBox.warning(self, "Invalid Method", 
                                         "K-range search is only available for Hierarchical and K-means clustering.")
            return
        
        # Check if we have prepared data
        if not hasattr(self, 'feature_dataframe') or self.feature_dataframe is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please select features first.")
            return
        
        # Get feature columns
        feature_cols = self._select_feature_columns(self.feature_dataframe)
        if not feature_cols:
            QtWidgets.QMessageBox.warning(self, "No Features", "No numeric features available.")
            return
        
        # Create dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Find Optimal K")
        dlg.setMinimumSize(600, 400)
        layout = QtWidgets.QVBoxLayout(dlg)
        
        # Input parameters
        params_layout = QtWidgets.QFormLayout()
        
        # K range
        k_min_spin = QtWidgets.QSpinBox()
        k_min_spin.setRange(2, 20)
        k_min_spin.setValue(2)
        k_max_spin = QtWidgets.QSpinBox()
        k_max_spin.setRange(2, 30)
        k_max_spin.setValue(10)
        k_range_layout = QtWidgets.QHBoxLayout()
        k_range_layout.addWidget(k_min_spin)
        k_range_layout.addWidget(QtWidgets.QLabel("to"))
        k_range_layout.addWidget(k_max_spin)
        params_layout.addRow("K range:", k_range_layout)
        
        # Linkage method (for hierarchical only)
        linkage_combo = None
        if clustering_type == "Hierarchical":
            linkage_combo = QtWidgets.QComboBox()
            linkage_combo.addItems(["ward", "complete", "average", "single"])
            linkage_combo.setCurrentText(self.hierarchical_method.currentText())
            params_layout.addRow("Linkage method:", linkage_combo)
        
        layout.addLayout(params_layout)
        
        # Progress bar
        progress = QtWidgets.QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        layout.addWidget(progress)
        
        # Results label
        results_label = QtWidgets.QLabel("")
        layout.addWidget(results_label)
        
        # Plot area
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        run_btn = QtWidgets.QPushButton("Run Analysis")
        close_btn = QtWidgets.QPushButton("Close")
        button_layout.addWidget(run_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        def run_analysis():
            from openimc.core import cluster, _prepare_clustering_matrix
            
            k_min = k_min_spin.value()
            k_max = k_max_spin.value()
            if k_min >= k_max:
                QtWidgets.QMessageBox.warning(dlg, "Invalid Range", "K min must be less than K max.")
                return
            
            k_values = list(range(k_min, k_max + 1))
            progress.setRange(0, len(k_values))
            
            # Get scaling method (matching main clustering)
            scaling_method = self.clustering_scaling_combo.currentText()
            scaling_map = {
                "None (no scaling)": "none",
                "Z-score": "zscore",
                "MAD (Median Absolute Deviation)": "mad"
            }
            selected_scaling = scaling_map.get(scaling_method, "zscore")
            use_pca, pca_mode, pca_variance, pca_n_components = self._get_pca_settings()
            
            # Get full dataframe (core.cluster will handle column selection and scaling)
            full_data = self.feature_dataframe.copy()
            
            if full_data.empty:
                QtWidgets.QMessageBox.warning(dlg, "No Data", "No data available.")
                return
            
            seed = self.seed_spinbox.value()
            inertias = []
            silhouette_scores = []
            
            try:
                for idx, k in enumerate(k_values):
                    progress.setValue(idx + 1)
                    QtWidgets.QApplication.processEvents()
                    
                    # Use core.cluster for consistency with main clustering
                    if clustering_type == "K-means":
                        # Use core.cluster for K-means
                        clustered_df = cluster(
                            features_df=full_data,
                            method="kmeans",
                            columns=feature_cols,
                            scaling=selected_scaling,
                            output_path=None,
                            n_clusters=k,
                            seed=seed,
                            n_init=10,
                            use_pca=use_pca,
                            pca_mode=pca_mode,
                            pca_variance=pca_variance,
                            pca_n_components=pca_n_components
                        )
                        labels = clustered_df['cluster'].values - 1  # Convert back to 0-based for calculations
                        
                    else:  # Hierarchical
                        linkage_method = linkage_combo.currentText() if linkage_combo else "ward"
                        # Use core.cluster for hierarchical
                        clustered_df = cluster(
                            features_df=full_data,
                            method="hierarchical",
                            columns=feature_cols,
                            scaling=selected_scaling,
                            output_path=None,
                            n_clusters=k,
                            linkage=linkage_method,
                            seed=seed,
                            use_pca=use_pca,
                            pca_mode=pca_mode,
                            pca_variance=pca_variance,
                            pca_n_components=pca_n_components
                        )
                        labels = clustered_df['cluster'].values - 1  # Convert back to 0-based for calculations
                    
                    # Rebuild the same matrix used by core.cluster for scoring.
                    data_scaled, _cluster_columns, _pca_metadata = _prepare_clustering_matrix(
                        full_data,
                        columns=feature_cols,
                        scaling=selected_scaling,
                        use_pca=use_pca,
                        pca_mode=pca_mode,
                        pca_variance=pca_variance,
                        pca_n_components=pca_n_components,
                        seed=seed,
                    )
                    data_scaled = data_scaled.fillna(0)
                    
                    # Filter to only rows that have valid cluster labels (not NaN/0 from dropped rows)
                    # core.cluster maps labels back, so rows that were dropped will have cluster=0 or NaN
                    valid_cluster_mask = clustered_df['cluster'].notna() & (clustered_df['cluster'] > 0)
                    valid_indices = clustered_df.index[valid_cluster_mask]
                    
                    # Align data_scaled with valid indices (some rows may have been dropped)
                    data_scaled_valid = data_scaled.reindex(valid_indices).dropna()
                    labels_valid = clustered_df.loc[data_scaled_valid.index, 'cluster'].values - 1
                    
                    # Calculate WCSS/inertia from cluster labels and scaled data
                    wcss = 0
                    for cluster_id in np.unique(labels_valid):
                        if cluster_id < 0:  # Skip noise/unassigned
                            continue
                        cluster_mask = labels_valid == cluster_id
                        cluster_data = data_scaled_valid.values[cluster_mask]
                        if len(cluster_data) > 0:
                            centroid = cluster_data.mean(axis=0)
                            wcss += np.sum((cluster_data - centroid) ** 2)
                    inertias.append(wcss)
                    
                    # Calculate silhouette score (using scaled data and 0-based labels)
                    if not _HAVE_SKLEARN:
                        silhouette_scores.append(0)
                    else:
                        # Filter out noise points (label < 0) for silhouette calculation
                        valid_mask = labels_valid >= 0
                        if valid_mask.sum() > 1 and len(np.unique(labels_valid[valid_mask])) > 1:
                            data_for_silhouette = data_scaled_valid.values[valid_mask]
                            labels_for_silhouette = labels_valid[valid_mask]
                            sil_score = silhouette_score(data_for_silhouette, labels_for_silhouette)
                        else:
                            sil_score = 0
                        silhouette_scores.append(sil_score)
                
                # Find optimal k
                # Elbow: find point with maximum curvature (second derivative)
                if len(inertias) > 2:
                    # Calculate rate of change
                    deltas = np.diff(inertias)
                    deltas2 = np.diff(deltas)
                    if len(deltas2) > 0:
                        elbow_idx = np.argmax(np.abs(deltas2)) + 1
                        optimal_k_elbow = k_values[elbow_idx]
                    else:
                        optimal_k_elbow = k_values[np.argmin(inertias)]
                else:
                    optimal_k_elbow = k_values[0]
                
                # Silhouette: maximum score
                optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
                
                # Plot results
                fig.clear()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                
                # Elbow plot
                ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
                ax1.axvline(x=optimal_k_elbow, color='r', linestyle='--', alpha=0.7, label=f'Suggested (elbow): k={optimal_k_elbow}')
                ax1.set_xlabel('Number of clusters (k)', fontsize=10)
                ax1.set_ylabel('WCSS / Inertia', fontsize=10)
                ax1.set_title('Elbow Method', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Silhouette plot
                ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
                ax2.axvline(x=optimal_k_silhouette, color='r', linestyle='--', alpha=0.7, label=f'Suggested (silhouette): k={optimal_k_silhouette}')
                ax2.set_xlabel('Number of clusters (k)', fontsize=10)
                ax2.set_ylabel('Silhouette Score', fontsize=10)
                ax2.set_title('Silhouette Score', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                fig.tight_layout()
                canvas.draw()
                
                # Update results label
                results_text = (f"Optimal k (elbow method): {optimal_k_elbow}\n"
                              f"Optimal k (silhouette): {optimal_k_silhouette}\n"
                              f"Max silhouette score: {max(silhouette_scores):.3f}")
                results_label.setText(results_text)
                
                # Update n_clusters spinbox with optimal value
                if optimal_k_silhouette >= 2:
                    self.n_clusters.setValue(optimal_k_silhouette)
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(dlg, "Error", f"Error during k-range analysis: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                progress.setValue(len(k_values))
        
        run_btn.clicked.connect(run_analysis)
        close_btn.clicked.connect(dlg.accept)
        
        dlg.exec_()
    
    def _open_gating_dialog(self):
        """Open gating rules editor and apply on save."""
        # Allow selection among intensity features by default
        marker_cols = [col for col in self.feature_dataframe.columns
                       if any(col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        dlg = GatingRulesDialog(self.gating_rules, marker_cols, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.gating_rules = dlg.get_rules()
            self._apply_gating_rules()
            
            # Log gating operation
            logger = get_logger()
            logger.log_gating(
                gating_rules=self.gating_rules,
                acquisitions=self._get_logging_acquisitions(),
                notes=f"Applied {len(self.gating_rules)} gating rules",
                source_file=self._get_source_file_summary_for_logging()
            )
            
            QtWidgets.QMessageBox.information(self, "Gating Applied", "Manual phenotypes assigned using gating rules.")
            # If user just applied manual gates, default heatmap source to Manual Gates for immediate view
            if hasattr(self, 'heatmap_source_combo'):
                self.heatmap_source_combo.setCurrentText('Manual Gates')

    def _apply_gating_rules(self):
        """Evaluate gating rules and create/update 'manual_phenotype' column on feature_dataframe."""
        if not self.gating_rules:
            return
        # Initialize column
        if 'manual_phenotype' not in self.feature_dataframe.columns:
            self.feature_dataframe['manual_phenotype'] = ''
        assigned = pd.Series(self.feature_dataframe['manual_phenotype'] != '', index=self.feature_dataframe.index)
        # Evaluate rules in order
        for rule in self.gating_rules:
            name = rule.get('name', '').strip()
            logic = rule.get('logic', 'AND').upper()
            conditions = rule.get('conditions', [])
            if not name or not conditions:
                continue
            masks = []
            for cond in conditions:
                col = cond.get('column')
                op = cond.get('op', '>')
                thr = cond.get('threshold', 0)
                if col not in self.feature_dataframe.columns:
                    continue
                series = self.feature_dataframe[col]
                if op == '>':
                    mask = series > thr
                elif op == '>=':
                    mask = series >= thr
                elif op == '<':
                    mask = series < thr
                elif op == '<=':
                    mask = series <= thr
                elif op == '==':
                    mask = series == thr
                elif op == '!=':
                    mask = series != thr
                else:
                    continue
                masks.append(mask.fillna(False))
            if not masks:
                continue
            if logic == 'OR':
                rule_mask = masks[0]
                for m in masks[1:]:
                    rule_mask = rule_mask | m
            else:
                rule_mask = masks[0]
                for m in masks[1:]:
                    rule_mask = rule_mask & m
            # Assign where not already assigned
            to_assign = rule_mask & (~assigned)
            self.feature_dataframe.loc[to_assign, 'manual_phenotype'] = name
            assigned = assigned | to_assign
        # If clustered_data exists, align and copy manual phenotype into it for plotting
        if self.clustered_data is not None:
            if 'manual_phenotype' not in self.clustered_data.columns:
                self.clustered_data['manual_phenotype'] = ''
            self.clustered_data.loc[:, 'manual_phenotype'] = self.feature_dataframe.loc[self.clustered_data.index, 'manual_phenotype'].values
        # Update color options and refresh plot
        self._populate_color_by_options()
        if getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()
        else:
            self._create_heatmap()

    def _save_gating_rules(self):
        """Save current gating rules to JSON."""
        import json
        if not self.gating_rules:
            QtWidgets.QMessageBox.information(self, "No Rules", "There are no gating rules to save.")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Gating Rules", "gating_rules.json", "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            with open(file_path, 'w') as f:
                json.dump(self.gating_rules, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Saved", f"Gating rules saved to: {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving gating rules: {str(e)}")

    def _load_gating_rules(self):
        """Load gating rules from JSON and apply."""
        import json
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Gating Rules", "", "JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            with open(file_path, 'r') as f:
                rules = json.load(f)
            if isinstance(rules, list):
                self.gating_rules = rules
                self._apply_gating_rules()
                QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded {len(self.gating_rules)} gating rules.")
            else:
                raise ValueError("JSON must be a list of rules")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Error loading gating rules: {str(e)}")

    def _open_annotation_dialog(self):
        """Open a dialog to annotate clusters with phenotype names. Includes save/load controls."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clusters", "Please run clustering first.")
            return
        unique_clusters = self._sorted_cluster_ids(self.clustered_data['cluster'].unique(), canonical=True)
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Annotate Phenotypes")
        v = QtWidgets.QVBoxLayout(dlg)
        
        # Create scrollable area for the form
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # Create widget to hold the form
        form_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_widget)
        editors = {}
        normalized_annotations = normalize_cluster_annotation_map(self.cluster_annotation_map)
        for cid in unique_clusters:
            le = QtWidgets.QLineEdit()
            if cid in normalized_annotations:
                le.setText(normalized_annotations[cid])
            form.addRow(format_default_cluster_label(cid), le)
            editors[cid] = le
        
        scroll_area.setWidget(form_widget)
        
        # Set max height to 90% of main window height
        try:
            # Use the parent dialog (CellClusteringDialog) height, or fall back to screen height
            if hasattr(self, 'height') and self.height() > 0:
                max_height = int(self.height() * 0.9)
            else:
                # Fallback to screen height
                screen = QtWidgets.QApplication.primaryScreen()
                if screen:
                    max_height = int(screen.availableGeometry().height() * 0.9)
                else:
                    max_height = 800
            scroll_area.setMaximumHeight(max_height)
        except Exception:
            scroll_area.setMaximumHeight(800)  # Fallback
        
        v.addWidget(scroll_area)
        
        # (Load/Save removed)
        # LLM assist row
        llm_row = QtWidgets.QHBoxLayout()
        llm_btn = QtWidgets.QPushButton("Suggest phenotypes with LLM…")
        llm_btn.setToolTip("Requires OpenAI API key. Uses per-cluster marker statistics.")
        llm_row.addWidget(llm_btn)
        llm_row.addStretch()
        v.addLayout(llm_row)
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        # (Load/Save handlers removed)
        def open_llm_dialog():
            def apply_names(display_name_map, backend_name_map):
                # Set display names in the UI
                for cid, name in display_name_map.items():
                    if cid in editors and isinstance(name, str):
                        editors[cid].setText(name)
                # Store backend names for CSV export
                self.cluster_backend_names.update(backend_name_map)
            # Ensure llm_phenotype_cache is initialized
            if not hasattr(self, 'llm_phenotype_cache') or self.llm_phenotype_cache is None:
                self.llm_phenotype_cache = {}
            d = PhenotypeSuggestionDialog(self, unique_clusters, apply_names, self.llm_phenotype_cache, self.normalization_config)
            d.exec_()
        llm_btn.clicked.connect(open_llm_dialog)

        # Make the dialog wider for better usability
        dlg.resize(500, dlg.sizeHint().height())

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Save mapping from editors
            self.cluster_annotation_map = normalize_cluster_annotation_map({
                cid: editors[cid].text().strip() for cid in unique_clusters if editors[cid].text().strip()
            })
            self.cluster_backend_names = dict(self.cluster_annotation_map)
            self._apply_cluster_annotations()
            
            # Log annotation operation
            logger = get_logger()
            logger.log_class_annotation(
                annotation_map=self.cluster_annotation_map,
                method="manual",
                acquisitions=self._get_logging_acquisitions(),
                notes=f"Annotated {len(self.cluster_annotation_map)} clusters",
                source_file=self._get_source_file_summary_for_logging()
            )
            
            QtWidgets.QMessageBox.information(self, "Annotations Applied", "Cluster annotations have been applied.")

    def _open_merge_clusters_dialog(self):
        """Open a dialog to merge two clusters."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clusters", "Please run clustering first.")
            return
        
        unique_clusters = self._sorted_cluster_ids(
            [c for c in self.clustered_data['cluster'].unique() if c != 0],
            canonical=False,
        )  # Exclude noise cluster
        if len(unique_clusters) < 2:
            QtWidgets.QMessageBox.warning(self, "Not Enough Clusters", "At least two clusters (excluding noise) are required to merge.")
            return
        
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Merge Clusters")
        dlg.setMinimumWidth(400)
        v = QtWidgets.QVBoxLayout(dlg)
        
        # Instructions
        instructions = QtWidgets.QLabel("Select two clusters to merge. The second cluster will be merged into the first.")
        instructions.setWordWrap(True)
        v.addWidget(instructions)
        
        # First cluster selector
        form = QtWidgets.QFormLayout()
        cluster_options = [self._get_cluster_display_name(cid) for cid in unique_clusters]
        merge_cluster1_combo = QtWidgets.QComboBox()
        merge_cluster1_combo.addItems(cluster_options)
        form.addRow("First cluster (target):", merge_cluster1_combo)
        
        # Second cluster selector
        merge_cluster2_combo = QtWidgets.QComboBox()
        merge_cluster2_combo.addItems(cluster_options)
        form.addRow("Second cluster (to merge):", merge_cluster2_combo)
        v.addLayout(form)
        
        # Buttons
        btns = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset All Merges")
        reset_btn.setToolTip("Reset all cluster merges to original assignments")
        reset_btn.clicked.connect(lambda: self._reset_cluster_merges_from_dialog(dlg))
        ok = QtWidgets.QPushButton("Merge")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addWidget(reset_btn)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Get selected cluster IDs
            idx1 = merge_cluster1_combo.currentIndex()
            idx2 = merge_cluster2_combo.currentIndex()
            
            if idx1 == idx2:
                QtWidgets.QMessageBox.warning(self, "Invalid Selection", "Please select two different clusters.")
                return
            
            cluster1_id = unique_clusters[idx1]
            cluster2_id = unique_clusters[idx2]
            
            # Perform merge
            self._merge_clusters(cluster1_id, cluster2_id)
            
            # Refresh plots
            self._refresh_all_plots()
            
            QtWidgets.QMessageBox.information(self, "Clusters Merged", 
                f"Cluster {cluster2_id} has been merged into cluster {cluster1_id}.")
    
    def _merge_clusters(self, target_cluster_id, source_cluster_id):
        """Merge source_cluster_id into target_cluster_id."""
        if self.clustered_data is None:
            return
        
        # Update cluster labels in clustered_data
        mask = self.clustered_data['cluster'] == source_cluster_id
        self.clustered_data.loc[mask, 'cluster'] = target_cluster_id
        
        # Update cluster labels in clustered_data_unscaled if it exists
        if self.clustered_data_unscaled is not None and 'cluster' in self.clustered_data_unscaled.columns:
            mask_unscaled = self.clustered_data_unscaled['cluster'] == source_cluster_id
            self.clustered_data_unscaled.loc[mask_unscaled, 'cluster'] = target_cluster_id
        
        # Update feature_dataframe if cluster column exists
        if 'cluster' in self.feature_dataframe.columns:
            mask_feature = self.feature_dataframe['cluster'] == source_cluster_id
            self.feature_dataframe.loc[mask_feature, 'cluster'] = target_cluster_id
        
        # Update cluster_annotation_map: merge annotations if both have them
        if source_cluster_id in self.cluster_annotation_map:
            source_annotation = self.cluster_annotation_map[source_cluster_id]
            if target_cluster_id not in self.cluster_annotation_map:
                # If target doesn't have annotation, use source's annotation
                self.cluster_annotation_map[target_cluster_id] = source_annotation
            # Remove source cluster from annotation map
            del self.cluster_annotation_map[source_cluster_id]
        
        # Update cluster_backend_names similarly
        if source_cluster_id in self.cluster_backend_names:
            source_backend_name = self.cluster_backend_names[source_cluster_id]
            if target_cluster_id not in self.cluster_backend_names:
                self.cluster_backend_names[target_cluster_id] = source_backend_name
            del self.cluster_backend_names[source_cluster_id]
        
        # Reapply cluster annotations to update cluster_phenotype column
        if self.cluster_annotation_map:
            self._apply_cluster_annotations()
    
    def _reset_cluster_merges(self):
        """Reset all cluster merges by restoring original cluster assignments."""
        if self.original_cluster_assignments is None or self.clustered_data is None:
            return
        
        # Restore original cluster assignments
        # Ensure indices match (they should since original_cluster_assignments was copied from clustered_data)
        if self.original_cluster_assignments.index.equals(self.clustered_data.index):
            self.clustered_data['cluster'] = self.original_cluster_assignments
        else:
            # If indices don't match, align them
            self.clustered_data['cluster'] = self.original_cluster_assignments.reindex(
                self.clustered_data.index, fill_value=0
            )
        
        # Update clustered_data_unscaled if it exists
        if self.clustered_data_unscaled is not None and 'cluster' in self.clustered_data_unscaled.columns:
            # clustered_data_unscaled should have the same index as clustered_data
            if self.clustered_data_unscaled.index.equals(self.clustered_data.index):
                self.clustered_data_unscaled['cluster'] = self.clustered_data['cluster']
            else:
                # If indices don't match, align them
                self.clustered_data_unscaled['cluster'] = self.clustered_data['cluster'].reindex(
                    self.clustered_data_unscaled.index, fill_value=0
                )
        
        # Update feature_dataframe if cluster column exists
        if 'cluster' in self.feature_dataframe.columns:
            # Restore original assignments for clustered cells
            matching_indices = self.clustered_data.index.intersection(self.feature_dataframe.index)
            if len(matching_indices) > 0:
                self.feature_dataframe.loc[matching_indices, 'cluster'] = self.clustered_data.loc[matching_indices, 'cluster']
        
        # Refresh plots
        self._refresh_all_plots()
    
    def _reset_cluster_merges_from_dialog(self, dialog):
        """Reset cluster merges from the merge dialog and close it."""
        self._reset_cluster_merges()
        dialog.accept()
        QtWidgets.QMessageBox.information(self, "Merges Reset", 
            "All cluster merges have been reset to original assignments.")
    
    def _refresh_all_plots(self):
        """Refresh all plots to reflect cluster changes."""
        current_view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        
        # Update stats cluster combo if it exists
        if hasattr(self, 'stats_cluster_combo'):
            self._update_stats_cluster_combo()
        
        # Refresh the current view
        self._on_view_changed(current_view)

    def _open_patient_annotation_dialog(self):
        """Open a dialog to customize patient/source file labels."""
        # Determine which column to use for patient annotation
        patient_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            patient_col = self.patient_annotation_column
        else:
            # Default priority order: standard columns first, then metadata
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in self.feature_dataframe.columns:
                    patient_col = col
                    break
            
            # If no standard column, check metadata
            if patient_col is None:
                metadata_cols = self._get_metadata_columns(self.feature_dataframe)
                if metadata_cols:
                    # Prefer columns that might be batch identifiers
                    priority_metadata = [col for col in metadata_cols 
                                       if any(keyword in col.lower() for keyword in ['pid', 'patient', 'batch', 'sample', 'subject'])]
                    if priority_metadata:
                        patient_col = priority_metadata[0]
                    else:
                        patient_col = metadata_cols[0]
        
        if not patient_col or patient_col not in self.feature_dataframe.columns:
            QtWidgets.QMessageBox.warning(self, "No Patient Annotation Data", 
                                          "No patient annotation column (source_file, batch_group, or source_well) is available in the data.")
            return
        
        # Get unique values from the selected column
        unique_values = sorted([f for f in self.feature_dataframe[patient_col].unique() if pd.notna(f)])
        if not unique_values:
            QtWidgets.QMessageBox.warning(self, "No Values", 
                                          f"No values found in {patient_col} column.")
            return
        
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Customize Patient/Source Labels")
        v = QtWidgets.QVBoxLayout(dlg)
        
        # Add instruction label
        instruction_text = f"Customize labels for each value in {patient_col}. Leave blank to use the original value."
        if patient_col == "source_file":
            instruction_text += " The source_file column stores the source data file basename; for MCD workflows this is the source .mcd file."
        instruction = QtWidgets.QLabel(instruction_text)
        instruction.setWordWrap(True)
        v.addWidget(instruction)
        
        form = QtWidgets.QFormLayout()
        editors = {}
        for value in unique_values:
            le = QtWidgets.QLineEdit()
            # Use custom label if available, otherwise use the value
            if value in self.patient_annotation_map:
                le.setText(self.patient_annotation_map[value])
            else:
                # Use basename if it's a file path, otherwise use value as-is
                if os.sep in str(value) or '/' in str(value) or '\\' in str(value):
                    default_label = os.path.basename(str(value))
                else:
                    default_label = str(value)
                le.setText(default_label)
            # Display label for the form row
            if os.sep in str(value) or '/' in str(value) or '\\' in str(value):
                display_name = os.path.basename(str(value))
            else:
                display_name = str(value)
            form.addRow(f"{patient_col}:\n{display_name}", le)
            editors[value] = le
        v.addLayout(form)
        
        # Add cohort management button
        cohort_btn = QtWidgets.QPushButton("Manage Cohorts...")
        cohort_btn.setToolTip("Group patients into cohorts. Patients in the same cohort will share the same color.")
        cohort_btn.clicked.connect(lambda: self._open_cohort_management_dialog(unique_values))
        v.addWidget(cohort_btn)
        
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        # Make the dialog wider for better usability
        dlg.resize(600, dlg.sizeHint().height())

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Save mapping from editors
            self.patient_annotation_map = {
                value: editors[value].text().strip() 
                for value in unique_values 
                if editors[value].text().strip()
            }
            # Refresh current view if patient labels are used
            view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
            if view == 'Heatmap' and hasattr(self, 'patient_annotation_checkbox') and self.patient_annotation_checkbox.isChecked():
                self._show_heatmap()
            elif view == 'UMAP' and self.umap_embedding is not None:
                # Refresh UMAP plot to update patient labels in legend
                self._create_umap_plot()
            elif view == 't-SNE' and self.tsne_embedding is not None:
                # Refresh t-SNE plot to update patient labels in legend
                self._create_tsne_plot()
            QtWidgets.QMessageBox.information(self, "Labels Applied", "Patient labels have been applied.")

    def _open_cohort_management_dialog(self, unique_values):
        """Open a dialog to manage patient cohorts (grouping patients)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Manage Patient Cohorts")
        dlg.resize(800, 600)
        
        main_layout = QtWidgets.QVBoxLayout(dlg)
        
        # Instructions
        instruction = QtWidgets.QLabel(
            "Create cohorts to group patients together. Patients in the same cohort will share the same color. "
            "Patients not assigned to any cohort will use individual colors."
        )
        instruction.setWordWrap(True)
        main_layout.addWidget(instruction)
        
        # Split into two columns: cohorts on left, patients on right
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # Left side: Cohort management
        cohort_widget = QtWidgets.QWidget()
        cohort_layout = QtWidgets.QVBoxLayout(cohort_widget)
        
        cohort_label = QtWidgets.QLabel("Cohorts:")
        cohort_layout.addWidget(cohort_label)
        
        # Cohort list with add/delete buttons
        cohort_list_widget = QtWidgets.QListWidget()
        cohort_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        
        # Populate existing cohorts
        existing_cohorts = set()
        for patient in unique_values:
            if patient in self.patient_cohort_map:
                existing_cohorts.add(self.patient_cohort_map[patient])
        
        for cohort_name in sorted(existing_cohorts):
            cohort_list_widget.addItem(cohort_name)
        
        cohort_layout.addWidget(cohort_list_widget)
        
        # Buttons for cohort management
        cohort_btn_layout = QtWidgets.QHBoxLayout()
        add_cohort_btn = QtWidgets.QPushButton("Add Cohort")
        delete_cohort_btn = QtWidgets.QPushButton("Delete Cohort")
        cohort_btn_layout.addWidget(add_cohort_btn)
        cohort_btn_layout.addWidget(delete_cohort_btn)
        cohort_layout.addLayout(cohort_btn_layout)
        
        splitter.addWidget(cohort_widget)
        
        # Right side: Patient assignment
        patient_widget = QtWidgets.QWidget()
        patient_layout = QtWidgets.QVBoxLayout(patient_widget)
        
        patient_label = QtWidgets.QLabel("Patients:")
        patient_layout.addWidget(patient_label)
        
        # Patient list (checkboxes for multi-select)
        patient_list_widget = QtWidgets.QListWidget()
        patient_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        
        # Populate patients with their current cohort assignment
        for value in unique_values:
            item = QtWidgets.QListWidgetItem(str(value))
            if value in self.patient_cohort_map:
                cohort = self.patient_cohort_map[value]
                item.setText(f"{value} → {cohort}")
            patient_list_widget.addItem(item)
        
        patient_layout.addWidget(patient_list_widget)
        
        # Buttons for patient assignment
        assign_btn_layout = QtWidgets.QHBoxLayout()
        assign_to_cohort_btn = QtWidgets.QPushButton("Assign Selected to Cohort")
        remove_from_cohort_btn = QtWidgets.QPushButton("Remove from Cohort")
        assign_btn_layout.addWidget(assign_to_cohort_btn)
        assign_btn_layout.addWidget(remove_from_cohort_btn)
        patient_layout.addLayout(assign_btn_layout)
        
        splitter.addWidget(patient_widget)
        
        # Set splitter sizes (equal)
        splitter.setSizes([400, 400])
        main_layout.addWidget(splitter)
        
        # Store reference to list widgets for callbacks
        dlg.cohort_list = cohort_list_widget
        dlg.patient_list = patient_list_widget
        
        # Add cohort callback
        def add_cohort():
            text, ok = QtWidgets.QInputDialog.getText(dlg, "New Cohort", "Cohort name:")
            if ok and text.strip():
                cohort_name = text.strip()
                if cohort_list_widget.findItems(cohort_name, QtCore.Qt.MatchExactly):
                    QtWidgets.QMessageBox.warning(dlg, "Duplicate", f"Cohort '{cohort_name}' already exists.")
                    return
                cohort_list_widget.addItem(cohort_name)
                cohort_list_widget.setCurrentItem(cohort_list_widget.item(cohort_list_widget.count() - 1))
        
        # Delete cohort callback
        def delete_cohort():
            current_item = cohort_list_widget.currentItem()
            if not current_item:
                QtWidgets.QMessageBox.warning(dlg, "No Selection", "Please select a cohort to delete.")
                return
            
            cohort_name = current_item.text()
            reply = QtWidgets.QMessageBox.question(
                dlg, "Confirm Delete",
                f"Delete cohort '{cohort_name}'? Patients in this cohort will be unassigned.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                # Remove cohort from all patients
                for i in range(patient_list_widget.count()):
                    item = patient_list_widget.item(i)
                    patient_value = unique_values[i]
                    if patient_value in self.patient_cohort_map and self.patient_cohort_map[patient_value] == cohort_name:
                        self.patient_cohort_map.pop(patient_value, None)
                        # Update display
                        item.setText(str(patient_value))
                
                # Remove from list
                row = cohort_list_widget.row(current_item)
                cohort_list_widget.takeItem(row)
        
        # Assign patients to cohort callback
        def assign_to_cohort():
            current_cohort_item = cohort_list_widget.currentItem()
            selected_patient_items = patient_list_widget.selectedItems()
            
            if not current_cohort_item:
                QtWidgets.QMessageBox.warning(dlg, "No Cohort Selected", "Please select a cohort first.")
                return
            
            if not selected_patient_items:
                QtWidgets.QMessageBox.warning(dlg, "No Patients Selected", "Please select one or more patients.")
                return
            
            cohort_name = current_cohort_item.text()
            
            for patient_item in selected_patient_items:
                row = patient_list_widget.row(patient_item)
                patient_value = unique_values[row]
                self.patient_cohort_map[patient_value] = cohort_name
                # Update display
                patient_item.setText(f"{patient_value} → {cohort_name}")
        
        # Remove patients from cohort callback
        def remove_from_cohort():
            selected_patient_items = patient_list_widget.selectedItems()
            
            if not selected_patient_items:
                QtWidgets.QMessageBox.warning(dlg, "No Patients Selected", "Please select one or more patients.")
                return
            
            for patient_item in selected_patient_items:
                row = patient_list_widget.row(patient_item)
                patient_value = unique_values[row]
                if patient_value in self.patient_cohort_map:
                    self.patient_cohort_map.pop(patient_value)
                    # Update display
                    patient_item.setText(str(patient_value))
        
        # Connect buttons
        add_cohort_btn.clicked.connect(add_cohort)
        delete_cohort_btn.clicked.connect(delete_cohort)
        assign_to_cohort_btn.clicked.connect(assign_to_cohort)
        remove_from_cohort_btn.clicked.connect(remove_from_cohort)
        
        # Dialog buttons
        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)
        
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Cohorts are already updated in self.patient_cohort_map via callbacks
            # Write cohorts to features column
            self._write_cohorts_to_features()
            # Refresh color-by options to include Cohorts
            self._populate_color_by_options()
            # Refresh group-by combo to include cohort option
            if hasattr(self, 'group_by_combo'):
                current_selection = self.group_by_combo.currentText()
                self.group_by_combo.clear()
                candidate_cols = [
                    'roi', 'ROI', 'slide', 'Slide', 'condition', 'Condition',
                    'acquisition_name', 'well', 'acquisition_id'
                ]
                available_group_cols = [c for c in candidate_cols if c in self.feature_dataframe.columns]
                
                # Build priority columns list in order (all should be options, not replacements)
                priority_cols = []
                if 'source_file' in self.feature_dataframe.columns:
                    priority_cols.append('source_file')
                if 'batch_group' in self.feature_dataframe.columns:
                    priority_cols.append('batch_group')
                if 'cohort' in self.feature_dataframe.columns:
                    priority_cols.append('cohort')
                if 'source_well' in self.feature_dataframe.columns:
                    priority_cols.append('source_well')
                
                # Add priority columns at the beginning, avoiding duplicates
                for col in reversed(priority_cols):
                    if col not in available_group_cols:
                        available_group_cols.insert(0, col)
                
                # Add source_file_acquisition_id if both source_file and acquisition_id exist
                if 'source_file' in self.feature_dataframe.columns and 'acquisition_id' in self.feature_dataframe.columns:
                    if 'source_file_acquisition_id' not in self.feature_dataframe.columns:
                        self.feature_dataframe = self.feature_dataframe.assign(
                            source_file_acquisition_id=(
                                self.feature_dataframe['source_file'].astype(str) + '_' + 
                                self.feature_dataframe['acquisition_id'].astype(str)
                            )
                        )
                    if 'source_file_acquisition_id' not in available_group_cols:
                        available_group_cols.insert(0, 'source_file_acquisition_id')
                if not available_group_cols:
                    available_group_cols = ['acquisition_name'] if 'acquisition_name' in self.feature_dataframe.columns else []
                
                # Add metadata columns for grouping
                metadata_cols = self._get_metadata_columns(self.feature_dataframe)
                if metadata_cols:
                    if available_group_cols:
                        available_group_cols.extend(metadata_cols)
                    else:
                        available_group_cols = metadata_cols
                
                for col in available_group_cols:
                    self.group_by_combo.addItem(col)
                
                # Restore selection if it still exists, otherwise keep default
                if current_selection and current_selection in available_group_cols:
                    self.group_by_combo.setCurrentText(current_selection)
            
            # Update cohort checkbox visibility
            if hasattr(self, 'use_cohort_checkbox'):
                has_cohorts = bool(self.patient_cohort_map)
                view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
                self.use_cohort_checkbox.setVisible(view in ['UMAP', 't-SNE'] and has_cohorts)
            # Refresh plots if needed
            view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
            if view == 'Heatmap' and hasattr(self, 'patient_annotation_checkbox') and self.patient_annotation_checkbox.isChecked():
                self._show_heatmap()
            elif view == 'UMAP' and self.umap_embedding is not None:
                self._create_umap_plot()
            elif view == 't-SNE' and self.tsne_embedding is not None:
                self._create_tsne_plot()

    def _open_plot_config_dialog(self):
        """Open the plot configuration dialog."""
        dlg = PlotConfigDialog(self, parent=self)
        dlg.exec_()

    def _open_feature_labels_dialog(self):
        """Open a dialog to customize feature labels (friendly names for y-axis ticks)."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Run clustering first to customize feature labels.")
            return
        
        # Respect current display feature selection (if any), matching plotted features.
        feature_cols = self._select_feature_columns(self.clustered_data)
        
        if not feature_cols:
            QtWidgets.QMessageBox.warning(self, "No Features", "No feature columns found in the data.")
            return
        
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Customize Feature Labels")
        dlg.resize(700, 600)
        v = QtWidgets.QVBoxLayout(dlg)
        
        # Add instruction label
        instruction = QtWidgets.QLabel("Set custom display names for features (e.g., 'Vimentin_mean' -> 'Mean Vimentin'). Leave blank to use original name.")
        instruction.setWordWrap(True)
        v.addWidget(instruction)
        
        # Create scroll area for many features
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(scroll_content)
        
        editors = {}
        for feature_name in sorted(feature_cols):
            le = QtWidgets.QLineEdit()
            # Use custom label if available, otherwise use original name
            if feature_name in self.feature_label_map:
                le.setText(self.feature_label_map[feature_name])
            else:
                le.setText(feature_name)
            form.addRow(feature_name, le)
            editors[feature_name] = le
        
        scroll.setWidget(scroll_content)
        v.addWidget(scroll)
        
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Update labels for features shown in this dialog only.
            # Keep labels for non-displayed features intact.
            updated_label_map = dict(self.feature_label_map)
            for feature_name in feature_cols:
                label_text = editors[feature_name].text().strip()
                if label_text and label_text != feature_name:
                    updated_label_map[feature_name] = label_text
                else:
                    updated_label_map.pop(feature_name, None)
            self.feature_label_map = updated_label_map
            # Refresh current view to apply new labels
            view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
            if view == 'Heatmap':
                self._show_heatmap()
            elif view == 'Differential Expression':
                self._show_differential_expression()
            elif view == 'Stacked Bars':
                self._show_stacked_bars()
            elif view == 'Boxplot/Violin Plot':
                self._show_boxplot_violin()
            QtWidgets.QMessageBox.information(self, "Labels Applied", "Feature labels have been applied.")

    def _get_feature_display_name(self, feature_name: str) -> str:
        """Return display label for a feature, using custom label if available."""
        if feature_name in self.feature_label_map:
            return self.feature_label_map[feature_name]
        return feature_name
    
    def _get_metadata_columns(self, df) -> List[str]:
        """Identify metadata columns (non-feature, non-standard columns) in the dataframe."""
        if df is None or df.empty:
            return []
        
        # Standard metadata columns to exclude
        exclude_cols = {
            'label', 'cell_id', 'acquisition_id', 'acquisition_name', 'acquisition_label',
            'well', 'cluster', 'source_file', 'source_well', 'source_file_acquisition_id',
            'centroid_x', 'centroid_y', 'batch_group', 'cluster_phenotype', 'cluster_id',
            'cohort'  # Cohort column (if written to features)
        }
        
        # Also filter out pandas index columns like "Unnamed: 0"
        exclude_patterns = ['Unnamed:', 'unnamed:']
        
        # Identify feature columns (intensity and morphology)
        feature_cols = set()
        for col in df.columns:
            if col in exclude_cols:
                continue
            # Check if it's a feature column (has intensity suffix or is morphology)
            if any(col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos']):
                feature_cols.add(col)
            elif col in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity',
                        'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um',
                        'aspect_ratio', 'bbox_area_um2', 'touches_border', 'touches_edge', 'holes_count']:
                feature_cols.add(col)
        
        # Metadata columns are everything else (excluding index columns)
        metadata_cols = []
        for col in df.columns:
            if col in exclude_cols or col in feature_cols:
                continue
            # Filter out pandas index columns like "Unnamed: 0"
            if any(pattern in str(col).lower() for pattern in ['unnamed:']):
                continue
            metadata_cols.append(col)
        
        return sorted(metadata_cols)

    def _write_cohorts_to_features(self):
        """Write cohort assignments to a 'cohort' column in the feature dataframe."""
        if not self.patient_cohort_map:
            # Remove cohort column if no cohorts exist
            if 'cohort' in self.feature_dataframe.columns:
                self.feature_dataframe = self.feature_dataframe.drop(columns=['cohort'])
            return
        
        # Determine which column to use for cohort mapping
        # Try patient annotation column first, then source_file, batch_group, source_well
        mapping_col = None
        if hasattr(self, 'patient_annotation_column') and self.patient_annotation_column:
            mapping_col = self.patient_annotation_column
        else:
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in self.feature_dataframe.columns:
                    mapping_col = col
                    break
        
        if mapping_col and mapping_col in self.feature_dataframe.columns:
            # Map cohorts to feature dataframe
            self.feature_dataframe['cohort'] = self.feature_dataframe[mapping_col].map(
                lambda x: self.patient_cohort_map.get(x, '')
            ).fillna('')

    def _on_cohort_coloring_changed(self, state: int):
        """Handle cohort coloring checkbox state change."""
        self.use_cohort_coloring = (state == 2)  # 2 = checked
        # Refresh current view
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'UMAP' and self.umap_embedding is not None:
            self._create_umap_plot()
        elif view == 't-SNE' and self.tsne_embedding is not None:
            self._create_tsne_plot()
        elif view == 'Heatmap' and hasattr(self, 'patient_annotation_checkbox') and self.patient_annotation_checkbox.isChecked():
            self._show_heatmap()

    def _apply_cluster_annotations(self):
        """Apply current annotation map to clustered_data and feature_dataframe as 'cluster_phenotype'."""
        if not self.clustered_data is None and self.cluster_annotation_map:
            self.cluster_annotation_map = normalize_cluster_annotation_map(self.cluster_annotation_map)
            # Persist the current user-visible names into exported phenotype columns.
            self.cluster_backend_names = dict(self.cluster_annotation_map)
            
            self.clustered_data['cluster_phenotype'] = self.clustered_data['cluster'].map(self.cluster_annotation_map).fillna('')
            if self.clustered_data_unscaled is not None and 'cluster' in self.clustered_data_unscaled.columns:
                self.clustered_data_unscaled['cluster_phenotype'] = self.clustered_data_unscaled['cluster'].map(
                    self.cluster_annotation_map
                ).fillna('')
            # Write back to feature_dataframe aligned by index
            aligned = self.feature_dataframe.reindex(self.clustered_data.index)
            if 'cluster_phenotype' not in self.feature_dataframe.columns:
                self.feature_dataframe['cluster_phenotype'] = ''
            self.feature_dataframe.loc[self.clustered_data.index, 'cluster_phenotype'] = self.clustered_data['cluster_phenotype'].values
            # Update color-by options
            self._populate_color_by_options()
            # Redraw the currently selected view
            current_view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
            if current_view == 'UMAP':
                self._create_umap_plot()
            elif current_view == 'Heatmap':
                self._create_heatmap()
            elif current_view == 'Stacked Bars':
                self._show_stacked_bars()
            elif current_view == 'Differential Expression':
                self._show_differential_expression()

    # Top-level save/load removed; handled inside annotation dialog
    
    def _explore_clusters(self):
        """Open cluster explorer window."""
        if self.clustered_data is None:
            return
        if not self._has_segmentation_masks_for_cluster_explorer():
            QtWidgets.QMessageBox.information(
                self,
                "Segmentation Masks Required",
                "Load segmentation masks to enable Cluster Explorer previews.",
            )
            self._update_cluster_action_buttons()
            return
        
        # Get cluster info
        cluster_info = []
        for cluster_id in self._sorted_cluster_ids(self.clustered_data['cluster'].unique(), canonical=False):
            cluster_cells = self.clustered_data[self.clustered_data['cluster'] == cluster_id]
            cluster_info.append({
                'cluster_id': cluster_id,
                'size': len(cluster_cells),
                'cells': cluster_cells.index.tolist()
            })
        
        # Open explorer dialog
        explorer = ClusterExplorerDialog(cluster_info, self.feature_dataframe, self.parent(), label_provider=self)
        explorer.exec_()
    
    def _save_current_plot(self):
        """Save whatever plot is currently shown in the canvas."""
        if self.figure is None:
            return
        default = "plot.png"
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'UMAP':
            default = 'umap_plot.png'
        elif view == 'Heatmap':
            default = 'cell_clustering_heatmap.png'
        elif view == 'Stacked Bars':
            default = 'stacked_bars.png'
        elif view == 'Differential Expression':
            default = 'differential_expression_heatmap.png'

        if view == 'Differential Expression':
            previous_context = getattr(self, '_de_render_context', 'gui')
            try:
                self._de_render_context = 'save'
                self._show_differential_expression()
                saved = save_figure_with_options(self.figure, default, self)
            finally:
                self._de_render_context = previous_context
                self._show_differential_expression()
        else:
            saved = save_figure_with_options(self.figure, default, self)

        if saved:
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def _save_clustering_output(self):
        """Save clustering output as CSV with all features and labels."""
        if self.clustered_data is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No clustering data available to save.")
            return
        
        # Get default filename
        default = "clustering_output.csv"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Clustering Output", default,
            "CSV Files (*.csv)"
        )
        if not file_path:
            return
        
        try:
            # Start with only the rows that were included in clustering
            # This ensures that if filters were applied (touches_edge, area, percentile, etc.),
            # only the filtered rows are included in the output
            output_df = self.feature_dataframe.loc[self.clustered_data.index].copy()
            
            # Add cluster labels (indices already match, so direct assignment)
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                output_df['cluster'] = self.clustered_data['cluster'].values
            
            # Add cluster phenotype annotations if available
            if self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
                output_df['cluster_phenotype'] = self.clustered_data['cluster_phenotype'].values
            
            # Add manual phenotype annotations if available
            if self.clustered_data is not None and 'manual_phenotype' in self.clustered_data.columns:
                output_df['manual_phenotype'] = self.clustered_data['manual_phenotype'].values
            
            # Save to CSV
            output_df.to_csv(file_path, index=True)
            
            # Show success message with summary
            total_cells = len(output_df)
            n_clusters = len(output_df['cluster'].unique()) if 'cluster' in output_df.columns else 0
            n_annotated = len(output_df[output_df['cluster_phenotype'].notna() & (output_df['cluster_phenotype'] != '')]) if 'cluster_phenotype' in output_df.columns else 0
            n_manual = len(output_df[output_df['manual_phenotype'].notna() & (output_df['manual_phenotype'] != '')]) if 'manual_phenotype' in output_df.columns else 0
            
            summary = f"Saved {total_cells} cells with {n_clusters} clusters"
            if n_annotated > 0:
                summary += f", {n_annotated} with cluster annotations"
            if n_manual > 0:
                summary += f", {n_manual} with manual annotations"
            
            QtWidgets.QMessageBox.information(self, "Success", f"Clustering output saved to: {file_path}\n\n{summary}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving clustering output: {str(e)}")

def _compute_centered_cluster_explorer_crop_bounds(
    image_shape: Tuple[int, int],
    center_y: int,
    center_x: int,
    crop_size: int,
) -> Dict[str, int]:
    """Return source/destination slices for a centered crop with black padding."""
    half_crop = crop_size // 2
    requested_y_start = int(center_y) - half_crop
    requested_x_start = int(center_x) - half_crop
    requested_y_end = requested_y_start + crop_size
    requested_x_end = requested_x_start + crop_size

    src_y_start = max(0, requested_y_start)
    src_x_start = max(0, requested_x_start)
    src_y_end = min(image_shape[0], requested_y_end)
    src_x_end = min(image_shape[1], requested_x_end)

    dst_y_start = src_y_start - requested_y_start
    dst_x_start = src_x_start - requested_x_start
    dst_y_end = dst_y_start + max(0, src_y_end - src_y_start)
    dst_x_end = dst_x_start + max(0, src_x_end - src_x_start)

    return {
        "src_y_start": src_y_start,
        "src_y_end": src_y_end,
        "src_x_start": src_x_start,
        "src_x_end": src_x_end,
        "dst_y_start": dst_y_start,
        "dst_y_end": dst_y_end,
        "dst_x_start": dst_x_start,
        "dst_x_end": dst_x_end,
    }


def _extract_centered_cluster_explorer_crop(
    array: np.ndarray,
    crop_bounds: Dict[str, int],
    crop_size: int,
) -> np.ndarray:
    """Extract a centered crop and pad outside-ROI content with zeros."""
    if array.ndim == 2:
        cropped = np.zeros((crop_size, crop_size), dtype=array.dtype)
        cropped[
            crop_bounds["dst_y_start"] : crop_bounds["dst_y_end"],
            crop_bounds["dst_x_start"] : crop_bounds["dst_x_end"],
        ] = array[
            crop_bounds["src_y_start"] : crop_bounds["src_y_end"],
            crop_bounds["src_x_start"] : crop_bounds["src_x_end"],
        ]
        return cropped

    cropped = np.zeros((crop_size, crop_size, array.shape[2]), dtype=array.dtype)
    cropped[
        crop_bounds["dst_y_start"] : crop_bounds["dst_y_end"],
        crop_bounds["dst_x_start"] : crop_bounds["dst_x_end"],
        :,
    ] = array[
        crop_bounds["src_y_start"] : crop_bounds["src_y_end"],
        crop_bounds["src_x_start"] : crop_bounds["src_x_end"],
        :,
    ]
    return cropped


def _draw_cluster_explorer_scale_bar(
    ax,
    image_shape: Tuple[int, int],
    scale_bar_length_um: float,
    pixel_size_um: float,
) -> None:
    """Draw a simple 1 px scale bar in the lower-right corner."""
    if scale_bar_length_um <= 0:
        return

    height, width = image_shape[:2]
    if height < 3 or width < 3:
        return

    pixel_size_um = max(float(pixel_size_um), 1e-6)
    target_pixels = int(round(scale_bar_length_um / pixel_size_um))
    target_pixels = max(1, min(target_pixels, max(1, width - 4)))
    x_end = width - 2
    x_start = max(1, x_end - target_pixels)
    y_pos = height - 2
    ax.plot([x_start, x_end], [y_pos, y_pos], color="white", linewidth=1.0, solid_capstyle="butt")


def _render_cluster_explorer_panel(ax_image, ax_bar, spec: Dict[str, Any], title_fontsize: int = 8) -> None:
    """Render one cluster explorer tile into the supplied axes."""
    image = spec["image"]
    if spec.get("is_rgb", False):
        image_artist = ax_image.imshow(image)
    else:
        image_artist = ax_image.imshow(
            image,
            cmap="gray",
            vmin=spec.get("vmin"),
            vmax=spec.get("vmax"),
        )

    if spec.get("show_title", False):
        ax_image.set_title(spec.get("title", ""), fontsize=title_fontsize)
    else:
        ax_image.set_title("")
    ax_image.axis("off")

    if spec.get("show_scale_bar"):
        _draw_cluster_explorer_scale_bar(
            ax_image,
            image.shape[:2],
            float(spec.get("scale_bar_length_um", 0.0)),
            float(spec.get("pixel_size_um", 1.0)),
        )

    mask_outline = spec.get("mask_outline")
    if mask_outline is not None and np.any(mask_outline):
        ax_image.contour(
            mask_outline.astype(np.float32),
            levels=[0.5],
            colors=[spec.get("mask_outline_color", "#00e5ff")],
            linewidths=float(spec.get("mask_outline_width", 1.1)),
        )

    if ax_bar is None:
        return

    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    colorbar_axis = ax_bar.inset_axes([0.56, 0.06, 0.26, 0.88])
    colorbar = ax_image.figure.colorbar(image_artist, cax=colorbar_axis)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.tick_params(
        labelsize=6,
        length=2,
        pad=1,
        labelleft=True,
        labelright=False,
    )


@dataclass
class _ClusterExplorerSettings:
    cluster_id: Any = None
    channel: str = ""
    rgb_mode: bool = False
    rgb_channels: Dict[str, str] = field(
        default_factory=lambda: {"R": "", "G": "", "B": ""}
    )
    rgb_scales: Dict[str, float] = field(
        default_factory=lambda: {"R": 1.0, "G": 1.0, "B": 1.0}
    )
    sample_count: int = 10
    column_count: int = 4
    sample_mode: str = "random"
    sample_feature: str = ""
    link_intensity_scale: bool = True
    show_tile_titles: bool = False
    balance_sources: bool = False
    lock_sampling: bool = True
    show_scale_bar: bool = False
    show_mask_outline: bool = False
    scale_bar_length_um: float = 10.0

    def clone(self) -> "_ClusterExplorerSettings":
        return _ClusterExplorerSettings(
            cluster_id=self.cluster_id,
            channel=self.channel,
            rgb_mode=self.rgb_mode,
            rgb_channels=dict(self.rgb_channels),
            rgb_scales=dict(self.rgb_scales),
            sample_count=self.sample_count,
            column_count=self.column_count,
            sample_mode=self.sample_mode,
            sample_feature=self.sample_feature,
            link_intensity_scale=self.link_intensity_scale,
            show_tile_titles=self.show_tile_titles,
            balance_sources=self.balance_sources,
            lock_sampling=self.lock_sampling,
            show_scale_bar=self.show_scale_bar,
            show_mask_outline=self.show_mask_outline,
            scale_bar_length_um=self.scale_bar_length_um,
        )


class _ClusterExplorerTileCanvas(FigureCanvas):
    """Reusable tile canvas for explorer previews."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(2.2, 2.45))
        super().__init__(self.figure)
        self.setParent(parent)
        self.image_ax = None
        self.intensity_ax = None

    def render_tile(self, spec: Dict[str, Any]) -> None:
        self.figure.clear()
        if spec.get("show_intensity_bar"):
            grid = self.figure.add_gridspec(1, 2, width_ratios=[16.6, 2.1], wspace=0.08)
            self.image_ax = self.figure.add_subplot(grid[0, 0])
            self.intensity_ax = self.figure.add_subplot(grid[0, 1])
        else:
            self.image_ax = self.figure.add_subplot(111)
            self.intensity_ax = None

        _render_cluster_explorer_panel(self.image_ax, self.intensity_ax, spec)
        top = 0.87 if spec.get("show_title", False) else 0.96
        self.figure.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=top, wspace=0.08)
        self.draw_idle()


class ClusterExplorerSettingsDialog(QtWidgets.QDialog):
    """Modal settings dialog for Cluster Explorer configuration."""

    def __init__(self, explorer: "ClusterExplorerDialog", settings: _ClusterExplorerSettings):
        super().__init__(explorer)
        self.setWindowTitle("Cluster Explorer Settings")
        self.setModal(True)
        self.resize(560, 520)

        self._explorer = explorer
        self._initial_settings = settings.clone()

        self._create_ui()
        self._populate_from_settings(self._initial_settings)
        self._sync_mode_widgets()

    def _create_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        selection_group = QtWidgets.QGroupBox("Selection")
        selection_layout = QtWidgets.QFormLayout(selection_group)

        self.cluster_combo = QtWidgets.QComboBox()
        for info in self._explorer.cluster_info:
            cluster_text = self._explorer._cluster_combo_text(info)
            self.cluster_combo.addItem(cluster_text, info["cluster_id"])
        selection_layout.addRow("Cluster", self.cluster_combo)

        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(self._explorer.available_channels)
        selection_layout.addRow("Marker", self.channel_combo)

        self.rgb_mode_chk = QtWidgets.QCheckBox("Render RGB composite")
        self.rgb_mode_chk.toggled.connect(self._sync_mode_widgets)
        selection_layout.addRow("", self.rgb_mode_chk)

        rgb_widget = QtWidgets.QWidget()
        rgb_layout = QtWidgets.QGridLayout(rgb_widget)
        rgb_layout.setContentsMargins(0, 0, 0, 0)
        rgb_layout.addWidget(QtWidgets.QLabel("Channel"), 0, 1)
        rgb_layout.addWidget(QtWidgets.QLabel("Brightness"), 0, 2)
        self.rgb_channel_combos: Dict[str, QtWidgets.QComboBox] = {}
        self.rgb_scale_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        for row, key in enumerate(("R", "G", "B"), start=1):
            rgb_layout.addWidget(QtWidgets.QLabel(key), row, 0)
            combo = QtWidgets.QComboBox()
            combo.addItems(self._explorer.available_channels)
            rgb_layout.addWidget(combo, row, 1)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, 5.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(1.0)
            rgb_layout.addWidget(spin, row, 2)
            self.rgb_channel_combos[key] = combo
            self.rgb_scale_spins[key] = spin
        selection_layout.addRow("RGB", rgb_widget)
        self.rgb_widget = rgb_widget
        layout.addWidget(selection_group)

        sampling_group = QtWidgets.QGroupBox("Sampling")
        sampling_layout = QtWidgets.QFormLayout(sampling_group)
        self.sample_count_spin = QtWidgets.QSpinBox()
        self.sample_count_spin.setRange(1, max(1, len(self._explorer.feature_dataframe)))
        sampling_layout.addRow("Images", self.sample_count_spin)

        self.column_count_spin = QtWidgets.QSpinBox()
        self.column_count_spin.setRange(1, 12)
        sampling_layout.addRow("Columns", self.column_count_spin)

        self.sample_mode_combo = QtWidgets.QComboBox()
        self.sample_mode_combo.addItem("Random sample", "random")
        self.sample_mode_combo.addItem("Top cells by feature", "top_feature")
        self.sample_mode_combo.currentIndexChanged.connect(self._sync_mode_widgets)
        sampling_layout.addRow("Sampling", self.sample_mode_combo)

        self.sample_feature_combo = QtWidgets.QComboBox()
        self.sample_feature_combo.addItems(self._explorer.rankable_features)
        sampling_layout.addRow("Feature", self.sample_feature_combo)

        self.link_intensity_scale_chk = QtWidgets.QCheckBox("Link intensity scale across crops")
        sampling_layout.addRow("", self.link_intensity_scale_chk)

        self.balance_sources_chk = QtWidgets.QCheckBox("Balance across source files")
        self.balance_sources_chk.setEnabled(self._explorer._has_multiple_source_files())
        self.balance_sources_chk.setToolTip(
            "Balance the preview across distinct source_file values "
            "(the source data file basename; for MCD workflows, the source .mcd file)."
        )
        sampling_layout.addRow("", self.balance_sources_chk)

        self.lock_sampling_chk = QtWidgets.QCheckBox("Lock sampling")
        sampling_layout.addRow("", self.lock_sampling_chk)

        self.resample_btn = QtWidgets.QPushButton("Resample")
        self.resample_btn.clicked.connect(self._resample_now)
        sampling_layout.addRow("", self.resample_btn)
        layout.addWidget(sampling_group)

        overlay_group = QtWidgets.QGroupBox("Overlays")
        overlay_layout = QtWidgets.QFormLayout(overlay_group)
        self.show_scale_bar_chk = QtWidgets.QCheckBox("Show scale bar")
        self.show_scale_bar_chk.toggled.connect(self._sync_mode_widgets)
        overlay_layout.addRow("", self.show_scale_bar_chk)

        self.show_mask_outline_chk = QtWidgets.QCheckBox("Show mask outline")
        overlay_layout.addRow("", self.show_mask_outline_chk)

        self.show_tile_titles_chk = QtWidgets.QCheckBox("Show tile titles")
        overlay_layout.addRow("", self.show_tile_titles_chk)

        self.scale_bar_length_spin = QtWidgets.QDoubleSpinBox()
        self.scale_bar_length_spin.setRange(1.0, 500.0)
        self.scale_bar_length_spin.setDecimals(1)
        self.scale_bar_length_spin.setSingleStep(1.0)
        overlay_layout.addRow("Scale bar (μm)", self.scale_bar_length_spin)
        layout.addWidget(overlay_group)

        layout.addStretch()

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Apply
        )
        self.button_box.accepted.connect(self._accept_with_apply)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self._apply_only)
        layout.addWidget(self.button_box)

    def _populate_from_settings(self, settings: _ClusterExplorerSettings) -> None:
        cluster_index = self.cluster_combo.findData(settings.cluster_id)
        if cluster_index >= 0:
            self.cluster_combo.setCurrentIndex(cluster_index)

        self.channel_combo.setCurrentText(settings.channel)
        self.rgb_mode_chk.setChecked(settings.rgb_mode)
        for key, combo in self.rgb_channel_combos.items():
            combo.setCurrentText(settings.rgb_channels.get(key, ""))
        for key, spin in self.rgb_scale_spins.items():
            spin.setValue(float(settings.rgb_scales.get(key, 1.0)))

        self.sample_count_spin.setValue(int(settings.sample_count))
        self.column_count_spin.setValue(int(settings.column_count))
        sample_mode_index = self.sample_mode_combo.findData(settings.sample_mode)
        if sample_mode_index >= 0:
            self.sample_mode_combo.setCurrentIndex(sample_mode_index)
        self.sample_feature_combo.setCurrentText(settings.sample_feature)
        self.link_intensity_scale_chk.setChecked(bool(settings.link_intensity_scale))
        self.balance_sources_chk.setChecked(bool(settings.balance_sources))
        self.lock_sampling_chk.setChecked(bool(settings.lock_sampling))
        self.show_scale_bar_chk.setChecked(bool(settings.show_scale_bar))
        self.show_mask_outline_chk.setChecked(bool(settings.show_mask_outline))
        self.show_tile_titles_chk.setChecked(bool(settings.show_tile_titles))
        self.scale_bar_length_spin.setValue(float(settings.scale_bar_length_um))

    def _sync_mode_widgets(self) -> None:
        rgb_mode = self.rgb_mode_chk.isChecked()
        self.channel_combo.setEnabled(not rgb_mode)
        self.rgb_widget.setEnabled(rgb_mode)
        self.scale_bar_length_spin.setEnabled(self.show_scale_bar_chk.isChecked())
        sample_mode = self.sample_mode_combo.currentData()
        self.sample_feature_combo.setEnabled(sample_mode == "top_feature")

    def collect_settings(self) -> _ClusterExplorerSettings:
        return _ClusterExplorerSettings(
            cluster_id=self.cluster_combo.currentData(),
            channel=self.channel_combo.currentText(),
            rgb_mode=self.rgb_mode_chk.isChecked(),
            rgb_channels={key: combo.currentText() for key, combo in self.rgb_channel_combos.items()},
            rgb_scales={key: spin.value() for key, spin in self.rgb_scale_spins.items()},
            sample_count=self.sample_count_spin.value(),
            column_count=self.column_count_spin.value(),
            sample_mode=str(self.sample_mode_combo.currentData() or "random"),
            sample_feature=self.sample_feature_combo.currentText(),
            link_intensity_scale=self.link_intensity_scale_chk.isChecked(),
            balance_sources=self.balance_sources_chk.isChecked(),
            lock_sampling=self.lock_sampling_chk.isChecked(),
            show_scale_bar=self.show_scale_bar_chk.isChecked(),
            show_mask_outline=self.show_mask_outline_chk.isChecked(),
            show_tile_titles=self.show_tile_titles_chk.isChecked(),
            scale_bar_length_um=self.scale_bar_length_spin.value(),
        )

    def _apply_only(self) -> None:
        self._explorer.apply_settings(self.collect_settings())

    def _accept_with_apply(self) -> None:
        self._apply_only()
        self.accept()

    def _resample_now(self) -> None:
        self._explorer.apply_settings(self.collect_settings(), force_resample=True)


# --------------------------
# Cluster Explorer Dialog
# --------------------------
class ClusterExplorerDialog(QtWidgets.QDialog):
    PREVIEW_CROP_SIZE = 30

    def __init__(self, cluster_info, feature_dataframe, parent=None, label_provider=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Explorer")
        self.setModal(True)

        if parent is not None:
            parent_size = parent.size()
            self.resize(int(parent_size.width() * 0.9), int(parent_size.height() * 0.9))

        self.setMinimumSize(1000, 700)
        self.cluster_info = list(cluster_info)
        self.feature_dataframe = feature_dataframe
        self._label_provider = label_provider
        self.current_cluster = None
        self.cell_images: List[Dict[str, Any]] = []
        self.available_channels = self._discover_channels()
        self.rankable_features = self._discover_rankable_features()

        self._cluster_lookup = {
            canonicalize_cluster_id(info["cluster_id"]): info for info in self.cluster_info
        }
        self._rng = np.random.default_rng(1337)
        self._stack_cache: Dict[str, np.ndarray] = {}
        self._acq_channel_cache: Dict[str, List[str]] = {}
        self._cell_preview_cache: Dict[Any, Optional[Dict[str, Any]]] = {}
        self._crop_cache: Dict[Tuple[Any, str], np.ndarray] = {}
        self._sample_orders: Dict[Tuple[Any, bool], List[Any]] = {}
        self._unlocked_sample_orders: Dict[Tuple[Any, bool], List[Any]] = {}
        self._rgb_range_cache: Dict[Tuple[Tuple[str, str, str], Tuple[str, ...]], Dict[str, Tuple[float, float]]] = {}
        self._tile_canvases: List[_ClusterExplorerTileCanvas] = []
        self._current_render_specs: List[Dict[str, Any]] = []
        self.current_preview_records: List[Dict[str, Any]] = []
        self._settings = self._build_default_settings()

        self._create_ui()
        self._sync_current_cluster()
        self._update_suggested_markers(use_suggested_channel=not bool(self._settings.channel))
        self._update_summary_label()
        self._update_mask_availability_ui()

    def _create_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        title_label = QtWidgets.QLabel("Cluster Explorer")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12pt; }")
        layout.addWidget(title_label)

        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "QLabel { padding: 6px; background: #f5f5f5; border: 1px solid #d9d9d9; }"
        )
        layout.addWidget(self.summary_label)

        self.suggested_markers_label = QtWidgets.QLabel("")
        self.suggested_markers_label.setWordWrap(True)
        self.suggested_markers_label.setStyleSheet(
            "QLabel { color: #0066cc; font-style: italic; padding: 5px; }"
        )
        layout.addWidget(self.suggested_markers_label)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(420)
        self.scroll_area.viewport().installEventFilter(self)
        layout.addWidget(self.scroll_area, 1)

        self.grid_widget = QtWidgets.QWidget()
        self.image_grid = QtWidgets.QGridLayout(self.grid_widget)
        self.image_grid.setContentsMargins(8, 8, 8, 8)
        self.image_grid.setHorizontalSpacing(8)
        self.image_grid.setVerticalSpacing(8)
        self.image_grid.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.scroll_area.setWidget(self.grid_widget)

        self.status_label = QtWidgets.QLabel("Adjust settings if needed, then click Load/Refresh to preview cells.")
        layout.addWidget(self.status_label)

        button_layout = QtWidgets.QHBoxLayout()
        self.settings_btn = QtWidgets.QPushButton("Settings...")
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        button_layout.addWidget(self.settings_btn)

        self.load_btn = QtWidgets.QPushButton("Load/Refresh")
        self.load_btn.clicked.connect(lambda: self._load_cell_images())
        button_layout.addWidget(self.load_btn)

        self.export_grid_btn = QtWidgets.QPushButton("Export grid")
        self.export_grid_btn.clicked.connect(self._export_grid)
        button_layout.addWidget(self.export_grid_btn)

        self.export_btn = QtWidgets.QPushButton("Export to HDF5")
        self.export_btn.setToolTip("Export full cluster crops, features, channels, and masks to HDF5.")
        self.export_btn.clicked.connect(self._export_to_hdf5)
        button_layout.addWidget(self.export_btn)

        button_layout.addStretch()

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)

    def _build_default_settings(self) -> _ClusterExplorerSettings:
        cluster_id = self.cluster_info[0]["cluster_id"] if self.cluster_info else None
        default_channel = self.available_channels[0] if self.available_channels else ""
        rgb_defaults = {"R": default_channel, "G": default_channel, "B": default_channel}
        if self.available_channels:
            rgb_defaults["R"] = self.available_channels[0]
            rgb_defaults["G"] = self.available_channels[min(1, len(self.available_channels) - 1)]
            rgb_defaults["B"] = self.available_channels[min(2, len(self.available_channels) - 1)]
        return _ClusterExplorerSettings(
            cluster_id=cluster_id,
            channel=default_channel,
            rgb_channels=rgb_defaults,
            sample_feature=self.rankable_features[0] if self.rankable_features else "",
        )

    def _discover_channels(self) -> List[str]:
        intensity_suffixes = [
            "_mean",
            "_std",
            "_p10",
            "_p90",
            "_integrated",
            "_frac_pos",
            "_median",
            "_mad",
        ]
        channels = set()
        for col in self.feature_dataframe.columns:
            for suffix in intensity_suffixes:
                if col.endswith(suffix):
                    channels.add(col[: -len(suffix)])
                    break
        return sorted(channels)

    def _discover_rankable_features(self) -> List[str]:
        features = []
        for col in self.feature_dataframe.columns:
            if col == "cluster":
                continue
            if pd.api.types.is_numeric_dtype(self.feature_dataframe[col]):
                features.append(col)
        return sorted(features)

    def _open_settings_dialog(self) -> None:
        self._settings_dialog = ClusterExplorerSettingsDialog(self, self._settings)
        self._settings_dialog.exec_()

    def _cluster_combo_text(self, info: Dict[str, Any]) -> str:
        label = self._get_cluster_label(info["cluster_id"])
        return f"{label} ({info['size']} cells)"

    def _has_multiple_source_files(self) -> bool:
        if "source_file" not in self.feature_dataframe.columns:
            return False
        values = {
            self._clean_optional_text(value)
            for value in self.feature_dataframe["source_file"].tolist()
            if self._clean_optional_text(value)
        }
        return len(values) > 1

    def _clean_optional_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() == "nan" else text

    def _coerce_settings(self, settings: _ClusterExplorerSettings) -> _ClusterExplorerSettings:
        coerced = settings.clone()
        if self.cluster_info:
            cluster_key = canonicalize_cluster_id(coerced.cluster_id)
            if cluster_key not in self._cluster_lookup:
                coerced.cluster_id = self.cluster_info[0]["cluster_id"]

        if self.available_channels:
            if coerced.channel not in self.available_channels:
                coerced.channel = self.available_channels[0]
            defaults = [
                self.available_channels[0],
                self.available_channels[min(1, len(self.available_channels) - 1)],
                self.available_channels[min(2, len(self.available_channels) - 1)],
            ]
            for key, fallback in zip(("R", "G", "B"), defaults):
                if coerced.rgb_channels.get(key) not in self.available_channels:
                    coerced.rgb_channels[key] = fallback
        else:
            coerced.channel = ""
            coerced.rgb_channels = {"R": "", "G": "", "B": ""}

        coerced.sample_count = max(1, int(coerced.sample_count))
        coerced.column_count = max(1, int(coerced.column_count))
        if coerced.sample_mode not in {"random", "top_feature"}:
            coerced.sample_mode = "random"
        if self.rankable_features:
            if coerced.sample_feature not in self.rankable_features:
                coerced.sample_feature = self.rankable_features[0]
        else:
            coerced.sample_feature = ""
            coerced.sample_mode = "random"
        coerced.scale_bar_length_um = max(1.0, float(coerced.scale_bar_length_um))
        return coerced

    def _sync_current_cluster(self) -> None:
        cluster_key = canonicalize_cluster_id(self._settings.cluster_id)
        self.current_cluster = self._cluster_lookup.get(cluster_key)
        if self.current_cluster is None and self.cluster_info:
            self.current_cluster = self.cluster_info[0]
            self._settings.cluster_id = self.current_cluster["cluster_id"]

    def _cluster_order_key(self) -> Tuple[Any, bool]:
        return (
            canonicalize_cluster_id(self._settings.cluster_id),
            bool(self._settings.balance_sources),
            str(self._settings.sample_mode),
            str(self._settings.sample_feature),
        )

    def _current_sample_order(self) -> List[Any]:
        key = self._cluster_order_key()
        if self._settings.lock_sampling:
            if key in self._sample_orders:
                return list(self._sample_orders[key])
            return list(self._unlocked_sample_orders.get(key, []))
        return list(self._unlocked_sample_orders.get(key, []))

    def apply_settings(self, new_settings: _ClusterExplorerSettings, force_resample: bool = False) -> None:
        previous_key = self._cluster_order_key()
        previous_order = self._current_sample_order()
        self._settings = self._coerce_settings(new_settings)
        key = self._cluster_order_key()
        if (
            self._settings.lock_sampling
            and key == previous_key
            and key not in self._sample_orders
            and previous_order
        ):
            self._sample_orders[key] = list(previous_order)
        self._sync_current_cluster()
        self._update_suggested_markers()
        self._update_summary_label()
        self._load_cell_images(force_resample=force_resample)

    def _update_summary_label(self) -> None:
        if not self.current_cluster:
            self.summary_label.setText("No cluster selected.")
            return

        cluster_label = self._get_cluster_label(self.current_cluster["cluster_id"])
        if self._settings.rgb_mode:
            marker_summary = "RGB: {R}/{G}/{B}".format(**self._settings.rgb_channels)
        else:
            marker_summary = f"Marker: {self._settings.channel}"
        if self._settings.sample_mode == "top_feature" and self._settings.sample_feature:
            sampling_summary = f"Top by {self._settings.sample_feature}"
        else:
            sampling_summary = "Random sample"

        flags = []
        flags.append("scale linked" if self._settings.link_intensity_scale else "scale per crop")
        flags.append("titles on" if self._settings.show_tile_titles else "titles off")
        flags.append("mask outline on" if self._settings.show_mask_outline else "mask outline off")
        flags.append("balanced" if self._settings.balance_sources else "unbalanced")
        flags.append("locked" if self._settings.lock_sampling else "unlocked")
        self.summary_label.setText(
            f"{cluster_label} | {marker_summary} | "
            f"{sampling_summary} | Preview {self._settings.sample_count} | Columns {self._settings.column_count} | "
            f"{', '.join(flags)}"
        )

    def _compute_suggested_markers(self, cluster_id: Any) -> Tuple[str, Optional[str]]:
        provider = self._label_provider or self.parent()
        clustered_data = getattr(provider, "clustered_data", None)
        if clustered_data is None or "cluster" not in clustered_data.columns:
            return "", None

        intensity_suffixes = [
            "_mean",
            "_median",
            "_integrated",
            "_std",
            "_p10",
            "_p90",
            "_frac_pos",
            "_mad",
        ]
        intensity_cols = [
            col
            for col in clustered_data.columns
            if any(col.endswith(suffix) for suffix in intensity_suffixes)
        ]
        if not intensity_cols:
            return "", None

        cluster_means = clustered_data.groupby("cluster")[intensity_cols].mean()
        cluster_lookup = {
            canonicalize_cluster_id(idx): idx for idx in cluster_means.index.tolist()
        }
        actual_cluster_id = cluster_lookup.get(canonicalize_cluster_id(cluster_id))
        if actual_cluster_id is None:
            return "", None

        feature_means = cluster_means.mean(axis=0)
        feature_stds = cluster_means.std(axis=0).replace(0, 1)
        cluster_z_scores = (cluster_means.loc[actual_cluster_id] - feature_means) / feature_stds
        sorted_markers = cluster_z_scores.sort_values(ascending=False)
        high_markers = sorted_markers[sorted_markers > 1.0].head(3)
        low_markers = sorted_markers[sorted_markers < -1.0].tail(3).sort_values(ascending=True)

        def _extract_channel_name(feature_name: str) -> str:
            for suffix in intensity_suffixes:
                if feature_name.endswith(suffix):
                    return feature_name[: -len(suffix)]
            return feature_name

        suggestions = []
        suggested_channel = None
        if len(high_markers) > 0:
            high_channels = [_extract_channel_name(marker) for marker in high_markers.index]
            suggestions.append(f"High: {', '.join(high_channels)}")
            suggested_channel = high_channels[0]
        if len(low_markers) > 0:
            low_channels = [_extract_channel_name(marker) for marker in low_markers.index]
            suggestions.append(f"Low: {', '.join(low_channels)}")

        if not suggestions:
            return "No strong marker patterns detected", suggested_channel
        return "Suggested markers: " + " | ".join(suggestions), suggested_channel

    def _update_suggested_markers(self, use_suggested_channel: bool = False) -> None:
        if not self.current_cluster:
            self.suggested_markers_label.setText("")
            return

        suggestion_text, suggested_channel = self._compute_suggested_markers(
            self.current_cluster["cluster_id"]
        )
        self.suggested_markers_label.setText(suggestion_text)
        if use_suggested_channel and suggested_channel and suggested_channel in self.available_channels:
            self._settings.channel = suggested_channel

    def _get_parent_window(self):
        return self.parent()

    def _get_segmentation_masks(self):
        parent_window = self._get_parent_window()
        if parent_window is None:
            return None
        return getattr(parent_window, "segmentation_masks", None)

    def _has_segmentation_masks(self) -> bool:
        segmentation_masks = self._get_segmentation_masks()
        if segmentation_masks is None:
            return False
        try:
            return len(segmentation_masks) > 0
        except Exception:
            try:
                return bool(list(segmentation_masks.keys()))
            except Exception:
                return False

    def _update_mask_availability_ui(self) -> None:
        masks_available = self._has_segmentation_masks()
        for widget in (
            self.scroll_area,
            self.settings_btn,
            self.load_btn,
            self.export_grid_btn,
            self.export_btn,
        ):
            widget.setEnabled(masks_available)
        if not masks_available:
            self.status_label.setText(
                "Segmentation masks are not loaded. Cluster Explorer previews are unavailable until masks are loaded."
            )

    def _get_loader_for_acquisition(self, acq_id: str):
        parent_window = self._get_parent_window()
        if parent_window is not None and hasattr(parent_window, "_get_loader_for_acquisition"):
            return parent_window._get_loader_for_acquisition(acq_id)
        return getattr(parent_window, "loader", None)

    def _get_original_acq_id(self, acq_id: str) -> str:
        parent_window = self._get_parent_window()
        if parent_window is not None and hasattr(parent_window, "_get_original_acq_id"):
            return parent_window._get_original_acq_id(acq_id)
        return acq_id

    def _get_pixel_size_um_for_acquisition(self, unique_acq_id: str) -> float:
        parent_window = self._get_parent_window()
        if parent_window is not None and hasattr(parent_window, "_get_pixel_size_um"):
            try:
                return float(parent_window._get_pixel_size_um(unique_acq_id))
            except Exception:
                original_id = self._get_original_acq_id(unique_acq_id)
                try:
                    return float(parent_window._get_pixel_size_um(original_id))
                except Exception:
                    return 1.0
        return 1.0

    def _get_cell_row(self, cell_idx: Any) -> Optional[pd.Series]:
        if cell_idx not in self.feature_dataframe.index:
            return None
        row = self.feature_dataframe.loc[cell_idx]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _resolve_unique_acq_id(self, acq_id: str, cell_row: pd.Series) -> Optional[str]:
        parent_window = self._get_parent_window()
        segmentation_masks = self._get_segmentation_masks()
        if segmentation_masks is None:
            return None
        if acq_id in segmentation_masks:
            return acq_id

        matching_keys = [
            key for key in segmentation_masks.keys() if str(key).startswith(f"{acq_id}__file_")
        ]
        if not matching_keys:
            return None
        if len(matching_keys) == 1:
            return matching_keys[0]

        source_file = self._clean_optional_text(cell_row.get("source_file"))
        if source_file and hasattr(parent_window, "acq_to_file"):
            for key in matching_keys:
                file_path = parent_window.acq_to_file.get(key)
                if not file_path:
                    continue
                file_name = os.path.basename(str(file_path))
                if source_file in str(file_path) or source_file == file_name:
                    return key
        return matching_keys[0]

    def _source_key_for_cell_index(self, cell_idx: Any) -> str:
        row = self._get_cell_row(cell_idx)
        if row is None:
            return str(cell_idx)
        source_file = self._clean_optional_text(row.get("source_file"))
        if source_file:
            return source_file
        acq_id = self._clean_optional_text(row.get("acquisition_id"))
        if acq_id:
            return acq_id
        return str(cell_idx)

    def _feature_value_for_cell_index(self, cell_idx: Any, feature_name: str) -> float:
        row = self._get_cell_row(cell_idx)
        if row is None or feature_name not in row.index:
            return float("-inf")
        value = row.get(feature_name)
        try:
            value_float = float(value)
        except Exception:
            return float("-inf")
        if not np.isfinite(value_float):
            return float("-inf")
        return value_float

    def _build_ranked_sample_order(self, cell_indices: List[Any], balance_sources: bool, feature_name: str) -> List[Any]:
        scored_cells = [
            (cell_idx, self._feature_value_for_cell_index(cell_idx, feature_name))
            for cell_idx in cell_indices
        ]
        if not balance_sources:
            scored_cells.sort(key=lambda item: (item[1], str(item[0])), reverse=True)
            return [cell_idx for cell_idx, _ in scored_cells]

        grouped: Dict[str, List[Tuple[Any, float]]] = {}
        for cell_idx, score in scored_cells:
            grouped.setdefault(self._source_key_for_cell_index(cell_idx), []).append((cell_idx, score))

        for values in grouped.values():
            values.sort(key=lambda item: (item[1], str(item[0])), reverse=True)

        source_keys = sorted(
            grouped.keys(),
            key=lambda key: (
                grouped[key][0][1] if grouped[key] else float("-inf"),
                key,
            ),
            reverse=True,
        )

        interleaved: List[Any] = []
        remaining = True
        while remaining:
            remaining = False
            for source_key in source_keys:
                values = grouped[source_key]
                if values:
                    interleaved.append(values.pop(0)[0])
                    remaining = True
        return interleaved

    def _build_sample_order(self, cell_indices: List[Any], balance_sources: bool) -> List[Any]:
        if self._settings.sample_mode == "top_feature" and self._settings.sample_feature:
            return self._build_ranked_sample_order(
                cell_indices,
                balance_sources,
                self._settings.sample_feature,
            )

        order = list(cell_indices)
        if not balance_sources:
            self._rng.shuffle(order)
            return order

        grouped: Dict[str, List[Any]] = {}
        for cell_idx in order:
            grouped.setdefault(self._source_key_for_cell_index(cell_idx), []).append(cell_idx)

        source_keys = list(grouped.keys())
        self._rng.shuffle(source_keys)
        for values in grouped.values():
            self._rng.shuffle(values)

        interleaved: List[Any] = []
        remaining = True
        while remaining:
            remaining = False
            for source_key in source_keys:
                values = grouped[source_key]
                if values:
                    interleaved.append(values.pop())
                    remaining = True
        return interleaved

    def _get_sample_order(self, force_resample: bool = False) -> List[Any]:
        if not self.current_cluster:
            return []
        key = self._cluster_order_key()
        cell_indices = list(self.current_cluster.get("cells", []))
        if not cell_indices:
            return []

        if self._settings.lock_sampling:
            if force_resample or key not in self._sample_orders:
                if not force_resample and key in self._unlocked_sample_orders:
                    self._sample_orders[key] = list(self._unlocked_sample_orders[key])
                else:
                    self._sample_orders[key] = self._build_sample_order(
                        cell_indices, self._settings.balance_sources
                    )
            return list(self._sample_orders[key])

        if force_resample or key not in self._unlocked_sample_orders:
            self._unlocked_sample_orders[key] = self._build_sample_order(
                cell_indices, self._settings.balance_sources
            )
        return list(self._unlocked_sample_orders[key])

    def _resolve_preview_record(self, cell_idx: Any) -> Optional[Dict[str, Any]]:
        if cell_idx in self._cell_preview_cache:
            return self._cell_preview_cache[cell_idx]

        row = self._get_cell_row(cell_idx)
        if row is None:
            self._cell_preview_cache[cell_idx] = None
            return None

        parent_window = self._get_parent_window()
        segmentation_masks = self._get_segmentation_masks()
        if segmentation_masks is None:
            self._cell_preview_cache[cell_idx] = None
            return None

        acq_id = self._clean_optional_text(row.get("acquisition_id"))
        if not acq_id:
            self._cell_preview_cache[cell_idx] = None
            return None

        unique_acq_id = self._resolve_unique_acq_id(acq_id, row)
        if unique_acq_id is None or unique_acq_id not in segmentation_masks:
            self._cell_preview_cache[cell_idx] = None
            return None

        mask = segmentation_masks[unique_acq_id]
        try:
            cell_id = int(row["cell_id"])
        except Exception:
            self._cell_preview_cache[cell_idx] = None
            return None

        cell_mask = mask == cell_id
        if not np.any(cell_mask):
            self._cell_preview_cache[cell_idx] = None
            return None

        center_x = row.get("centroid_x")
        center_y = row.get("centroid_y")
        try:
            center_x = int(round(float(center_x))) if np.isfinite(float(center_x)) else None
        except Exception:
            center_x = None
        try:
            center_y = int(round(float(center_y))) if np.isfinite(float(center_y)) else None
        except Exception:
            center_y = None

        if center_x is None or center_y is None:
            coords = np.argwhere(cell_mask)
            center_y = int(round(coords[:, 0].mean()))
            center_x = int(round(coords[:, 1].mean()))

        crop_bounds = _compute_centered_cluster_explorer_crop_bounds(
            mask.shape,
            center_y,
            center_x,
            self.PREVIEW_CROP_SIZE,
        )
        cropped_mask = _extract_centered_cluster_explorer_crop(
            cell_mask.astype(bool),
            crop_bounds,
            self.PREVIEW_CROP_SIZE,
        )

        source_file = self._clean_optional_text(row.get("source_file"))
        source_name = os.path.basename(source_file) if source_file else "Unknown"
        well_name = self._clean_optional_text(row.get("well")) or self._clean_optional_text(
            row.get("source_well")
        )
        title = f"{well_name} [{source_name}]" if well_name else source_name
        original_acq_id = self._get_original_acq_id(unique_acq_id)

        record = {
            "cell_index": cell_idx,
            "cell_id": cell_id,
            "unique_acq_id": unique_acq_id,
            "original_acq_id": original_acq_id,
            "source_key": source_file or unique_acq_id,
            "source_file": source_name,
            "well_name": well_name,
            "title": title,
            "pixel_size_um": self._get_pixel_size_um_for_acquisition(unique_acq_id),
            "crop_bounds": crop_bounds,
            "cropped_mask": cropped_mask,
        }
        self._cell_preview_cache[cell_idx] = record
        return record

    def _select_preview_records(self, sample_order: List[Any], count: int) -> List[Dict[str, Any]]:
        preview_records: List[Dict[str, Any]] = []
        for cell_idx in sample_order:
            record = self._resolve_preview_record(cell_idx)
            if record is None:
                continue
            preview_records.append(record)
            if len(preview_records) >= count:
                break
        return preview_records

    def _get_channels_for_acquisition(self, unique_acq_id: str, original_acq_id: Optional[str] = None) -> List[str]:
        if unique_acq_id in self._acq_channel_cache:
            return self._acq_channel_cache[unique_acq_id]

        original_acq_id = original_acq_id or self._get_original_acq_id(unique_acq_id)
        channels: List[str] = []
        loader = self._get_loader_for_acquisition(unique_acq_id)
        if loader is not None and hasattr(loader, "get_channels"):
            try:
                channels = list(loader.get_channels(original_acq_id))
            except Exception:
                channels = []

        if not channels:
            parent_window = self._get_parent_window()
            acquisitions = getattr(parent_window, "acquisitions", [])
            for acq in acquisitions:
                if getattr(acq, "id", None) in {unique_acq_id, original_acq_id}:
                    channels = list(getattr(acq, "channels", []) or [])
                    break

        if not channels:
            channels = list(self.available_channels)

        self._acq_channel_cache[unique_acq_id] = channels
        return channels

    def _get_cached_stack(self, unique_acq_id: str, original_acq_id: Optional[str] = None) -> np.ndarray:
        if unique_acq_id in self._stack_cache:
            return self._stack_cache[unique_acq_id]

        loader = self._get_loader_for_acquisition(unique_acq_id)
        if loader is None:
            raise RuntimeError(f"No loader available for acquisition {unique_acq_id}.")

        original_acq_id = original_acq_id or self._get_original_acq_id(unique_acq_id)
        try:
            stack = loader.get_all_channels(original_acq_id)
            if stack.ndim == 2:
                stack = stack[..., np.newaxis]
        except Exception:
            channel_names = self._get_channels_for_acquisition(unique_acq_id, original_acq_id)
            stack = np.stack(
                [loader.get_image(original_acq_id, channel) for channel in channel_names],
                axis=-1,
            )
        self._stack_cache[unique_acq_id] = stack
        return stack

    def _get_channel_crop(self, record: Dict[str, Any], channel: str) -> np.ndarray:
        cache_key = (record["cell_index"], channel)
        if cache_key in self._crop_cache:
            return self._crop_cache[cache_key]

        stack = self._get_cached_stack(record["unique_acq_id"], record["original_acq_id"])
        channel_names = self._get_channels_for_acquisition(
            record["unique_acq_id"], record["original_acq_id"]
        )
        if channel not in channel_names:
            crop = np.zeros((self.PREVIEW_CROP_SIZE, self.PREVIEW_CROP_SIZE), dtype=np.float32)
            self._crop_cache[cache_key] = crop
            return crop

        channel_index = channel_names.index(channel)
        channel_image = stack[..., channel_index].astype(np.float32, copy=False)
        cropped = _extract_centered_cluster_explorer_crop(
            channel_image,
            record["crop_bounds"],
            self.PREVIEW_CROP_SIZE,
        ).astype(np.float32)
        self._crop_cache[cache_key] = cropped
        return cropped

    def _compute_single_channel_specs(self, preview_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        channel = self._settings.channel
        crops = [self._get_channel_crop(record, channel) for record in preview_records]
        masked_values = [crop[record["cropped_mask"]] for crop, record in zip(crops, preview_records)]
        shared_limits = None
        if self._settings.link_intensity_scale:
            non_empty_values = [values for values in masked_values if values.size > 0]
            if non_empty_values:
                vmin = float(min(np.min(values) for values in non_empty_values))
                vmax = float(max(np.max(values) for values in non_empty_values))
            else:
                vmin, vmax = 0.0, 1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0
            shared_limits = (vmin, vmax)

        specs = []
        for crop, record, values in zip(crops, preview_records, masked_values):
            if shared_limits is None:
                if values.size > 0:
                    vmin = float(np.min(values))
                    vmax = float(np.max(values))
                else:
                    vmin, vmax = 0.0, 1.0
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    vmax = vmin + 1.0
            else:
                vmin, vmax = shared_limits
            specs.append(
                {
                    "image": crop,
                    "title": record["title"],
                    "is_rgb": False,
                    "vmin": vmin,
                    "vmax": vmax,
                    "show_intensity_bar": True,
                    "show_title": self._settings.show_tile_titles,
                    "show_scale_bar": self._settings.show_scale_bar,
                    "scale_bar_length_um": self._settings.scale_bar_length_um,
                    "pixel_size_um": record["pixel_size_um"],
                    "mask_outline": (
                        record["cropped_mask"] if self._settings.show_mask_outline else None
                    ),
                    "cell_id": record["cell_id"],
                }
            )
        return specs

    def _compute_rgb_ranges(self, preview_records: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        if not self._settings.link_intensity_scale:
            return {}
        channels = (
            self._settings.rgb_channels["R"],
            self._settings.rgb_channels["G"],
            self._settings.rgb_channels["B"],
        )
        acq_ids = tuple(sorted({record["unique_acq_id"] for record in preview_records}))
        cache_key = (channels, acq_ids)
        if cache_key in self._rgb_range_cache:
            return self._rgb_range_cache[cache_key]

        ranges: Dict[str, Tuple[float, float]] = {}
        for channel in channels:
            mins: List[float] = []
            maxs: List[float] = []
            for record in preview_records:
                crop = self._get_channel_crop(record, channel)
                values = crop[record["cropped_mask"]]
                if values.size == 0:
                    continue
                mins.append(float(np.min(values)))
                maxs.append(float(np.max(values)))
            if mins and maxs:
                low = min(mins)
                high = max(maxs)
                if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                    high = low + 1.0
                ranges[channel] = (low, high)
            else:
                ranges[channel] = (0.0, 1.0)

        self._rgb_range_cache[cache_key] = ranges
        return ranges

    def _compute_rgb_specs(self, preview_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranges = self._compute_rgb_ranges(preview_records)
        specs = []
        for record in preview_records:
            channels = []
            for key in ("R", "G", "B"):
                channel_name = self._settings.rgb_channels[key]
                crop = self._get_channel_crop(record, channel_name)
                if self._settings.link_intensity_scale:
                    low, high = ranges.get(channel_name, (0.0, 1.0))
                else:
                    values = crop[record["cropped_mask"]]
                    if values.size > 0:
                        low = float(np.min(values))
                        high = float(np.max(values))
                    else:
                        low, high = 0.0, 1.0
                    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                        high = low + 1.0
                normalized = (crop - low) / (high - low + 1e-8)
                normalized = np.clip(
                    normalized * float(self._settings.rgb_scales.get(key, 1.0)),
                    0.0,
                    1.0,
                )
                channels.append(normalized)
            rgb_image = np.stack(channels, axis=-1)
            specs.append(
                {
                    "image": rgb_image,
                    "title": record["title"],
                    "is_rgb": True,
                    "show_intensity_bar": False,
                    "show_title": self._settings.show_tile_titles,
                    "show_scale_bar": self._settings.show_scale_bar,
                    "scale_bar_length_um": self._settings.scale_bar_length_um,
                    "pixel_size_um": record["pixel_size_um"],
                    "mask_outline": (
                        record["cropped_mask"] if self._settings.show_mask_outline else None
                    ),
                    "cell_id": record["cell_id"],
                }
            )
        return specs

    def _build_render_specs(self, preview_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._settings.rgb_mode:
            return self._compute_rgb_specs(preview_records)
        return self._compute_single_channel_specs(preview_records)

    def _build_preview_tooltip(self, record: Dict[str, Any]) -> str:
        """Return a hover tooltip for a cluster explorer preview tile."""
        return (
            f"source_file: {record.get('source_file', 'Unknown')}\n"
            f"acquisition_id: {record.get('original_acq_id', 'Unknown')}\n"
            f"cell_id: {record.get('cell_id', 'Unknown')}"
        )

    def _ensure_tile_canvases(self, count: int) -> None:
        while len(self._tile_canvases) < count:
            canvas = _ClusterExplorerTileCanvas(self.grid_widget)
            canvas.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._tile_canvases.append(canvas)

    def _rebuild_grid_layout(self, count: int) -> None:
        while self.image_grid.count():
            item = self.image_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.grid_widget)

        visible_columns = max(1, min(self._settings.column_count, max(1, count)))
        for index, canvas in enumerate(self._tile_canvases):
            if index < count:
                canvas.show()
                self.image_grid.addWidget(canvas, index // visible_columns, index % visible_columns)
            else:
                canvas.hide()

    def _update_tile_geometry(self) -> None:
        count = len(self._current_render_specs)
        if count == 0:
            return

        visible_columns = max(1, min(self._settings.column_count, count))
        viewport_width = max(1, self.scroll_area.viewport().width())
        margins = self.image_grid.contentsMargins()
        spacing = self.image_grid.horizontalSpacing()
        if spacing < 0:
            spacing = 8
        available_width = viewport_width - margins.left() - margins.right() - spacing * (visible_columns - 1)
        tile_width = max(170, available_width // visible_columns)
        tile_height = int(tile_width * 1.15)
        for canvas in self._tile_canvases[:count]:
            canvas.setFixedSize(tile_width, tile_height)

    def _load_cell_images(self, force_resample: bool = False) -> None:
        if not self.current_cluster:
            self.status_label.setText("No cluster selected.")
            return
        if not self.available_channels:
            self.status_label.setText("No marker channels were found in the feature dataframe.")
            return
        if not self._has_segmentation_masks():
            self._update_mask_availability_ui()
            return

        try:
            sample_order = self._get_sample_order(force_resample=force_resample)
            preview_records = self._select_preview_records(sample_order, self._settings.sample_count)
            if not preview_records:
                self.current_preview_records = []
                self._current_render_specs = []
                self.cell_images = []
                self.status_label.setText("No valid cells were available for this cluster preview.")
                self._ensure_tile_canvases(0)
                self._rebuild_grid_layout(0)
                return

            render_specs = self._build_render_specs(preview_records)
            self.current_preview_records = preview_records
            self._current_render_specs = render_specs
            self.cell_images = [
                {
                    "cell_id": record["cell_id"],
                    "acquisition_id": record["unique_acq_id"],
                    "image": spec["image"],
                }
                for record, spec in zip(preview_records, render_specs)
            ]

            self._ensure_tile_canvases(len(render_specs))
            self._rebuild_grid_layout(len(render_specs))
            for canvas, spec, record in zip(
                self._tile_canvases[: len(render_specs)],
                render_specs,
                preview_records,
            ):
                canvas.render_tile(spec)
                canvas.setToolTip(self._build_preview_tooltip(record))
            self._update_tile_geometry()
            self.status_label.setText(
                f"Loaded {len(render_specs)} preview crops for {self._get_cluster_label(self.current_cluster['cluster_id'])}."
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading cell images: {exc}")

    def _current_preview_cell_ids(self) -> List[int]:
        return [int(record["cell_id"]) for record in self.current_preview_records]

    def _export_grid(self) -> None:
        if not self._current_render_specs:
            self._load_cell_images()
        if not self._current_render_specs:
            return

        count = len(self._current_render_specs)
        columns = max(1, min(self._settings.column_count, count))
        rows = int(math.ceil(count / columns))
        fig = Figure(figsize=(columns * 2.3, rows * 2.6))
        outer = fig.add_gridspec(rows, columns, wspace=0.12, hspace=0.22)

        for index, spec in enumerate(self._current_render_specs):
            row = index // columns
            col = index % columns
            if spec.get("show_intensity_bar"):
                inner = outer[row, col].subgridspec(1, 2, width_ratios=[16.6, 2.1], wspace=0.08)
                ax_image = fig.add_subplot(inner[0, 0])
                ax_bar = fig.add_subplot(inner[0, 1])
            else:
                ax_image = fig.add_subplot(outer[row, col])
                ax_bar = None
            _render_cluster_explorer_panel(ax_image, ax_bar, spec)

        fig.subplots_adjust(left=0.04, right=0.98, bottom=0.03, top=0.97, wspace=0.18, hspace=0.24)
        default = f"cluster_{self.current_cluster['cluster_id']}_grid.png"
        try:
            save_figure_with_options(fig, default, self)
        finally:
            plt.close(fig)

    def eventFilter(self, obj, event):
        if obj is self.scroll_area.viewport() and event.type() == QtCore.QEvent.Resize:
            QtCore.QTimer.singleShot(0, self._update_tile_geometry)
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._update_tile_geometry)

    def _export_to_hdf5(self):
        """Export cell images, features, channels, and masks to HDF5 file."""
        if not self.current_cluster:
            QtWidgets.QMessageBox.warning(self, "No Cluster", "Please select a cluster first.")
            return

        crop_size, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Crop Size",
            "Enter crop size (square, in pixels):",
            value=30,
            min=10,
            max=200,
            step=5,
        )
        if not ok:
            return

        default = f"cluster_{self.current_cluster['cluster_id']}_export.h5"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export to HDF5",
            default,
            "HDF5 Files (*.h5 *.hdf5)",
        )
        if not file_path:
            return

        try:
            import h5py
        except ImportError:
            QtWidgets.QMessageBox.critical(
                self,
                "Missing Dependency",
                "h5py is required for HDF5 export. Please install it: pip install h5py",
            )
            return

        progress_dlg = QtWidgets.QProgressDialog("Exporting to HDF5...", "Cancel", 0, 100, self)
        progress_dlg.setWindowModality(QtCore.Qt.WindowModal)
        progress_dlg.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            parent_window = self.parent()
            if parent_window is None:
                QtWidgets.QMessageBox.warning(self, "Error", "Cannot access parent window.")
                return
            if not self._has_segmentation_masks():
                QtWidgets.QMessageBox.warning(
                    self, "No Masks", "Segmentation masks are required for export."
                )
                return

            cluster_cells = list(self.current_cluster.get("cells", []))
            total_cells = len(cluster_cells)
            if total_cells == 0:
                QtWidgets.QMessageBox.warning(self, "No Cells", "No cells in selected cluster.")
                return

            channels = list(self.available_channels)
            if not channels:
                QtWidgets.QMessageBox.warning(
                    self, "No Channels", "No channels found in feature dataframe."
                )
                return

            cells_by_acq: Dict[str, List[Any]] = {}
            for cell_idx in cluster_cells:
                cell_row = self._get_cell_row(cell_idx)
                if cell_row is None:
                    continue

                acq_id = self._clean_optional_text(cell_row.get("acquisition_id"))
                unique_acq_id = self._resolve_unique_acq_id(acq_id, cell_row)
                if unique_acq_id is None or unique_acq_id not in parent_window.segmentation_masks:
                    continue
                cells_by_acq.setdefault(unique_acq_id, []).append(cell_idx)

            if not cells_by_acq:
                QtWidgets.QMessageBox.warning(
                    self, "No Valid Cells", "No valid cells could be found for export."
                )
                return

            acq_tasks = []
            for unique_acq_id, cell_indices in cells_by_acq.items():
                loader = parent_window._get_loader_for_acquisition(unique_acq_id)
                if loader is None:
                    continue

                original_acq_id = parent_window._get_original_acq_id(unique_acq_id)
                mask = parent_window.segmentation_masks[unique_acq_id]
                file_path_for_loader = None
                if hasattr(parent_window, "acq_to_file") and unique_acq_id in parent_window.acq_to_file:
                    file_path_for_loader = parent_window.acq_to_file[unique_acq_id]

                loader_type = "mcd"
                if (
                    hasattr(parent_window, "mcd_loaders")
                    and file_path_for_loader
                    and file_path_for_loader in parent_window.mcd_loaders
                ):
                    loader_type = "mcd"
                elif file_path_for_loader and os.path.isdir(file_path_for_loader):
                    loader_type = "ometiff"
                elif hasattr(parent_window, "loader") and parent_window.loader is not None:
                    if file_path_for_loader and file_path_for_loader.endswith((".mcd", ".mcdx")):
                        loader_type = "mcd"
                    else:
                        loader_type = "ometiff"

                cell_data_list = []
                for cell_idx in cell_indices:
                    cell_row = self._get_cell_row(cell_idx)
                    if cell_row is None:
                        continue
                    cell_data = cell_row.to_dict()
                    cell_data["_cell_idx"] = cell_idx
                    cell_data_list.append(cell_data)

                acq_tasks.append(
                    (
                        unique_acq_id,
                        original_acq_id,
                        file_path_for_loader,
                        loader_type,
                        cell_data_list,
                        channels,
                        crop_size,
                        mask,
                    )
                )

            max_workers = max(1, mp.cpu_count() - 2)
            total_acqs = len(acq_tasks)
            progress_dlg.setMaximum(total_acqs)
            progress_dlg.setLabelText(
                f"Processing {total_acqs} acquisitions with {max_workers} workers..."
            )
            QtWidgets.QApplication.processEvents()

            images_list = []
            masks_list = []
            features_list = []
            valid_cell_indices = []

            with mp.Pool(processes=max_workers) as pool:
                futures = [pool.apply_async(_process_acquisition_export, task) for task in acq_tasks]
                completed = 0
                for future in futures:
                    if progress_dlg.wasCanceled():
                        pool.terminate()
                        pool.join()
                        return

                    try:
                        result = future.get(timeout=600)
                        if result:
                            acq_images, acq_masks, acq_features, acq_indices = result
                            images_list.extend(acq_images)
                            masks_list.extend(acq_masks)
                            features_list.extend(acq_features)
                            valid_cell_indices.extend(acq_indices)
                    except Exception as exc:
                        print(f"[ClusterExplorer] Error processing acquisition: {exc}")
                        continue

                    completed += 1
                    progress_dlg.setValue(completed)
                    progress_dlg.setLabelText(
                        f"Processed {completed}/{total_acqs} acquisitions ({len(images_list)} cells)..."
                    )
                    QtWidgets.QApplication.processEvents()

            if not images_list:
                QtWidgets.QMessageBox.warning(
                    self, "No Valid Cells", "No valid cells could be processed for export."
                )
                return

            progress_dlg.setLabelText("Saving to HDF5 file...")
            QtWidgets.QApplication.processEvents()

            images_array = np.array(images_list)
            masks_array = np.array(masks_list)
            features_df = pd.DataFrame(features_list)
            features_df.index = valid_cell_indices

            with h5py.File(file_path, "w") as f:
                f.create_dataset("images", data=images_array, compression="gzip", compression_opts=4)
                f.create_dataset("masks", data=masks_array, compression="gzip", compression_opts=4)
                channels_array = np.array([ch.encode("utf-8") for ch in channels], dtype="S")
                f.create_dataset("channels", data=channels_array)
                features_rec = features_df.to_records(index=False)
                f.create_dataset("features", data=features_rec, compression="gzip", compression_opts=4)
                f["features"].attrs["columns"] = [col.encode("utf-8") for col in features_df.columns]
                f["features"].attrs["index"] = [str(idx).encode("utf-8") for idx in features_df.index]

            progress_dlg.setValue(total_acqs)
            QtWidgets.QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(images_list)} cells to:\n{file_path}\n\n"
                f"Shape: {images_array.shape}\n"
                f"Channels: {len(channels)}\n"
                f"Crop size: {crop_size}x{crop_size}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting to HDF5: {exc}",
            )
            import traceback

            traceback.print_exc()
        finally:
            progress_dlg.close()

    def _get_cluster_label(self, cluster_id):
        """Get annotated cluster name from parent dialog if available."""
        provider = self._label_provider or self.parent()
        if provider is not None and hasattr(provider, "_get_cluster_display_name"):
            try:
                return provider._get_cluster_display_name(cluster_id)
            except Exception:
                pass
        return f"Cluster {cluster_id}"


def _process_acquisition_export(unique_acq_id, original_acq_id, file_path, loader_type, cell_data_list, channels, crop_size, mask):
    """Worker function to process all cells for one acquisition.
    
    This function runs in a separate process and processes all cells for a single acquisition.
    Returns: (images_list, masks_list, features_list, cell_indices_list)
    """
    try:
        # Import here to avoid issues with multiprocessing
        from openimc.data.mcd_loader import MCDLoader
        from openimc.data.ometiff_loader import OMETIFFLoader
        from skimage.measure import regionprops
        
        # Create loader
        if loader_type == 'mcd' and file_path and os.path.isfile(file_path):
            loader = MCDLoader()
            loader.open(file_path)
        elif file_path and os.path.isdir(file_path):
            # OME-TIFF directory
            loader = OMETIFFLoader(channel_format='CHW')
            loader.open(file_path)
        else:
            # For single file or if we can't determine, we'll need to handle differently
            # This is a fallback - in practice, we should have file_path
            return None
        
        images_list = []
        masks_list = []
        features_list = []
        cell_indices_list = []
        
        # Process all cells for this acquisition
        for cell_data_dict in cell_data_list:
            cell_idx = cell_data_dict.pop('_cell_idx')
            cell_id = int(cell_data_dict['cell_id'])
            
            # Get cell mask
            cell_mask = (mask == cell_id).astype(np.uint8)
            
            if not np.any(cell_mask):
                continue
            
            # Get cell center
            props = regionprops(cell_mask)
            if not props:
                continue
            
            center_y, center_x = props[0].centroid
            center_y, center_x = int(center_y), int(center_x)
            
            crop_bounds = _compute_centered_cluster_explorer_crop_bounds(
                mask.shape,
                center_y,
                center_x,
                crop_size,
            )
            
            # Load all channels for this cell
            try:
                available_channels = loader.get_channels(original_acq_id)
                cell_image_channels = []
                
                for channel in channels:
                    if channel not in available_channels:
                        # Fill with zeros if channel not available
                        cell_image_channels.append(np.zeros((crop_size, crop_size), dtype=np.float32))
                    else:
                        try:
                            channel_img = loader.get_image(original_acq_id, channel)
                            cropped_channel = _extract_centered_cluster_explorer_crop(
                                channel_img,
                                crop_bounds,
                                crop_size,
                            )
                            cell_image_channels.append(cropped_channel.astype(np.float32))
                        except Exception:
                            # Fill with zeros if loading fails
                            cell_image_channels.append(np.zeros((crop_size, crop_size), dtype=np.float32))
                
                # Stack channels: shape will be (H, W, C)
                cell_image = np.stack(cell_image_channels, axis=-1)
                
                # Crop mask to same size and convert to binary (0-1)
                cropped_mask = _extract_centered_cluster_explorer_crop(
                    cell_mask.astype(np.float32),
                    crop_bounds,
                    crop_size,
                )
                # Ensure binary (0 or 1)
                cropped_mask = (cropped_mask > 0).astype(np.float32)
                
                images_list.append(cell_image)
                masks_list.append(cropped_mask)
                features_list.append(cell_data_dict)
                cell_indices_list.append(cell_idx)
                
            except Exception as e:
                print(f"[ClusterExplorer] Error processing cell {cell_id} in acquisition {unique_acq_id}: {e}")
                continue
        
        return (images_list, masks_list, features_list, cell_indices_list)
        
    except Exception as e:
        print(f"[ClusterExplorer] Error processing acquisition {unique_acq_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


class GatingRulesDialog(QtWidgets.QDialog):
    def __init__(self, rules, available_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Gating Rules")
        self.setModal(True)
        
        # Set size to 90% of parent window if available
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)
        
        self.setMinimumSize(700, 500)
        self._available_columns = list(sorted(set(available_columns)))
        self._rules = [r.copy() for r in (rules or [])]
        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        # Existing rules list
        self.rules_list = QtWidgets.QListWidget()
        self._refresh_rules_list()
        layout.addWidget(self.rules_list)

        # Buttons + Save/Load
        btns = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Rule")
        edit_btn = QtWidgets.QPushButton("Edit")
        del_btn = QtWidgets.QPushButton("Delete")
        load_btn = QtWidgets.QPushButton("Load…")
        save_btn = QtWidgets.QPushButton("Save…")
        btns.addWidget(add_btn)
        btns.addWidget(edit_btn)
        btns.addWidget(del_btn)
        btns.addSpacing(20)
        btns.addWidget(load_btn)
        btns.addWidget(save_btn)
        btns.addStretch()
        layout.addLayout(btns)

        # OK/Cancel
        ok_cancel = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        ok_cancel.addStretch()
        ok_cancel.addWidget(ok)
        ok_cancel.addWidget(cancel)
        layout.addLayout(ok_cancel)

        # Wire actions
        add_btn.clicked.connect(self._on_add)
        edit_btn.clicked.connect(self._on_edit)
        del_btn.clicked.connect(self._on_delete)
        def do_load():
            from PyQt5 import QtWidgets, QtCore as _QtW
            import json
            path, _ = _QtW.QFileDialog.getOpenFileName(self, "Load Gating Rules", "", "JSON Files (*.json)")
            if not path:
                return
            try:
                with open(path, 'r') as f:
                    rules = json.load(f)
                if isinstance(rules, list):
                    self._rules = rules
                    self._refresh_rules_list()
                else:
                    raise ValueError("JSON must be a list of rules")
            except Exception as e:
                _QtW.QMessageBox.critical(self, "Load Error", f"Error loading gating rules: {str(e)}")
        def do_save():
            from PyQt5 import QtWidgets, QtCore as _QtW
            import json
            path, _ = _QtW.QFileDialog.getSaveFileName(self, "Save Gating Rules", "gating_rules.json", "JSON Files (*.json)")
            if not path:
                return
            try:
                with open(path, 'w') as f:
                    json.dump(self._rules, f, indent=2)
                _QtW.QMessageBox.information(self, "Saved", f"Gating rules saved to: {path}")
            except Exception as e:
                _QtW.QMessageBox.critical(self, "Save Error", f"Error saving gating rules: {str(e)}")
        load_btn.clicked.connect(do_load)
        save_btn.clicked.connect(do_save)

    def _refresh_rules_list(self):
        self.rules_list.clear()
        for r in self._rules:
            name = r.get('name', '(unnamed)')
            logic = r.get('logic', 'AND')
            conds = r.get('conditions', [])
            desc_parts = [f"{c.get('column')} {c.get('op')} {c.get('threshold')}" for c in conds]
            item = QtWidgets.QListWidgetItem(f"{name}  [{logic}]  ::  " + " AND ".join(desc_parts))
            self.rules_list.addItem(item)

    def _on_add(self):
        rule = self._edit_rule_dialog()
        if rule:
            self._rules.append(rule)
            self._refresh_rules_list()

    def _on_edit(self):
        row = self.rules_list.currentRow()
        if row < 0 or row >= len(self._rules):
            return
        rule = self._edit_rule_dialog(self._rules[row])
        if rule:
            self._rules[row] = rule
            self._refresh_rules_list()

    def _on_delete(self):
        row = self.rules_list.currentRow()
        if row < 0 or row >= len(self._rules):
            return
        del self._rules[row]
        self._refresh_rules_list()

    def _edit_rule_dialog(self, existing=None):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Edit Rule")
        v = QtWidgets.QVBoxLayout(dlg)

        # Name
        name_edit = QtWidgets.QLineEdit()
        if existing and existing.get('name'):
            name_edit.setText(existing['name'])
        form = QtWidgets.QFormLayout()
        form.addRow("Phenotype name:", name_edit)
        v.addLayout(form)

        # Logic
        logic_combo = QtWidgets.QComboBox()
        logic_combo.addItems(["AND", "OR"])
        if existing and existing.get('logic'):
            idx = logic_combo.findText(existing['logic'].upper())
            if idx >= 0:
                logic_combo.setCurrentIndex(idx)
        v.addWidget(QtWidgets.QLabel("Combine conditions with:"))
        v.addWidget(logic_combo)

        # Conditions table
        table = QtWidgets.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Feature", "Operator", "Threshold"])
        table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(table)

        # Row buttons
        row_btns = QtWidgets.QHBoxLayout()
        add_row = QtWidgets.QPushButton("Add Condition")
        del_row = QtWidgets.QPushButton("Delete Condition")
        row_btns.addWidget(add_row)
        row_btns.addWidget(del_row)
        row_btns.addStretch()
        v.addLayout(row_btns)

        def add_condition_row(cond=None):
            r = table.rowCount()
            table.insertRow(r)
            # Feature combo
            feat = QtWidgets.QComboBox()
            feat.addItems(self._available_columns)
            if cond and cond.get('column') in self._available_columns:
                feat.setCurrentText(cond['column'])
            table.setCellWidget(r, 0, feat)
            # Operator combo
            op = QtWidgets.QComboBox()
            op.addItems(['>', '>=', '<', '<=', '==', '!='])
            if cond and cond.get('op'):
                idx = op.findText(cond['op'])
                if idx >= 0:
                    op.setCurrentIndex(idx)
            table.setCellWidget(r, 1, op)
            # Threshold edit
            thr = QtWidgets.QDoubleSpinBox()
            thr.setRange(-1e12, 1e12)
            thr.setDecimals(6)
            thr.setSingleStep(0.1)
            if cond and cond.get('threshold') is not None:
                try:
                    thr.setValue(float(cond['threshold']))
                except Exception:
                    pass
            table.setCellWidget(r, 2, thr)

        # Seed from existing
        if existing and existing.get('conditions'):
            for cond in existing['conditions']:
                add_condition_row(cond)
        else:
            add_condition_row()

        add_row.clicked.connect(lambda: add_condition_row())
        def delete_selected_rows():
            rows = sorted({i.row() for i in table.selectedIndexes()}, reverse=True)
            for r in rows:
                table.removeRow(r)
        del_row.clicked.connect(delete_selected_rows)

        # OK/Cancel
        okc = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("OK")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        okc.addStretch()
        okc.addWidget(ok)
        okc.addWidget(cancel)
        v.addLayout(okc)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Build rule
            rule = {
                'name': name_edit.text().strip(),
                'logic': logic_combo.currentText(),
                'conditions': []
            }
            for r in range(table.rowCount()):
                feat = table.cellWidget(r, 0).currentText()
                op = table.cellWidget(r, 1).currentText()
                thr = table.cellWidget(r, 2).value()
                rule['conditions'].append({'column': feat, 'op': op, 'threshold': float(thr)})
            if rule['name'] and rule['conditions']:
                return rule
            return None
        return None

    def get_rules(self):
        return [r.copy() for r in self._rules]


class PhenotypeSuggestionDialog(QtWidgets.QDialog):
    def __init__(self, parent_dialog: 'CellClusteringDialog', cluster_ids, apply_callback, cache_dict=None, normalization_config=None, parent=None):
        super().__init__(parent or parent_dialog)
        self.setWindowTitle("Suggest Phenotypes with LLM (Based on Markers Used in Clustering)")
        self.setModal(True)
        self._parent_dialog = parent_dialog
        self._cluster_ids = list(cluster_ids)
        self._apply_callback = apply_callback
        # Ensure cache_dict is properly initialized and is a reference to parent's cache
        if cache_dict is None:
            if parent_dialog and hasattr(parent_dialog, 'llm_phenotype_cache'):
                self._cache_dict = parent_dialog.llm_phenotype_cache
            else:
                self._cache_dict = {}
        else:
            self._cache_dict = cache_dict
        # Ensure it's a dict
        if not isinstance(self._cache_dict, dict):
            if parent_dialog and hasattr(parent_dialog, 'llm_phenotype_cache'):
                parent_dialog.llm_phenotype_cache = {}
                self._cache_dict = parent_dialog.llm_phenotype_cache
            else:
                self._cache_dict = {}
        self.normalization_config = normalization_config
        self._create_ui()
        # Resize dialog to 75% of the parent window size for better usability
        try:
            base_widget = parent_dialog if parent_dialog is not None else self.parent()
            if base_widget is not None:
                base_size = base_widget.size()
                w = max(600, int(base_size.width() * 0.75))
                h = max(400, int(base_size.height() * 0.75))
            else:
                # Fallback to primary screen available geometry
                screen = QtWidgets.QApplication.primaryScreen()
                geo = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1200, 800)
                w = int(geo.width() * 0.75)
                h = int(geo.height() * 0.75)
            self.resize(w, h)
        except Exception:
            pass

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.api_key_edit = QtWidgets.QLineEdit()
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-... OpenAI API Key")
        form.addRow("OpenAI API Key:", self.api_key_edit)

        api_key_note = QtWidgets.QLabel(
            "The key is used only for this dialog session and is not saved by OpenIMC."
        )
        api_key_note.setWordWrap(True)
        api_key_note.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        form.addRow("", api_key_note)

        self.context_edit = QtWidgets.QLineEdit()
        self.context_edit.setPlaceholderText("e.g., human colorectal cancer (optional)")
        form.addRow("Cohort/tissue context:", self.context_edit)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-5.2"])
        self.model_combo.setCurrentText("gpt-5.2")  # Set gpt-5.2 as default
        form.addRow("Model:", self.model_combo)

        # Reasoning level dropdown (only visible for gpt-5.2)
        self.reasoning_combo = QtWidgets.QComboBox()
        self.reasoning_combo.addItems(["none", "low", "medium", "high"])
        self.reasoning_combo.setCurrentText("none")  # Set none as default
        # Create a container widget that includes both label and combo for proper hiding
        # This ensures both the label and dropdown are hidden/shown together
        self._reasoning_container = QtWidgets.QWidget()
        reasoning_container_layout = QtWidgets.QHBoxLayout(self._reasoning_container)
        reasoning_container_layout.setContentsMargins(0, 0, 0, 0)
        reasoning_label = QtWidgets.QLabel("Reasoning level:")
        reasoning_container_layout.addWidget(reasoning_label)
        reasoning_container_layout.addWidget(self.reasoning_combo)
        reasoning_container_layout.addStretch()
        # Add as a row with empty label since we include the label in the container
        form.addRow("", self._reasoning_container)
        self._reasoning_container.hide()  # Hidden by default

        # System prompt selection (fine vs broad cell types)
        self.system_prompt_combo = QtWidgets.QComboBox()
        self.system_prompt_combo.addItems(["Fine cell types (detailed)", "Broad cell types (Myeloid, Tumor, Stroma, etc.)"])
        form.addRow("System prompt:", self.system_prompt_combo)

        # Feature mode selection for markers used in LLM prompt
        self.feature_mode_combo = QtWidgets.QComboBox()
        self.feature_mode_combo.addItems(["Markers only", "Morphometrics only", "Both"])
        form.addRow("Feature mode:", self.feature_mode_combo)

        # Per-type K controls
        self.k_int_spin = QtWidgets.QSpinBox()
        self.k_int_spin.setRange(1, 30)
        self.k_int_spin.setValue(5)
        self.k_morpho_spin = QtWidgets.QSpinBox()
        self.k_morpho_spin.setRange(1, 30)
        self.k_morpho_spin.setValue(5)

        # Container widgets for visibility toggling
        self._k_int_row = QtWidgets.QWidget()
        kint_layout = QtWidgets.QHBoxLayout(self._k_int_row)
        kint_layout.setContentsMargins(0,0,0,0)
        kint_layout.addWidget(self.k_int_spin)
        form.addRow("Top-K intensity:", self._k_int_row)

        self._k_morpho_row = QtWidgets.QWidget()
        kmorph_layout = QtWidgets.QHBoxLayout(self._k_morpho_row)
        kmorph_layout.setContentsMargins(0,0,0,0)
        kmorph_layout.addWidget(self.k_morpho_spin)
        form.addRow("Top-K morphometric:", self._k_morpho_row)

        # Default visibility
        def _update_feature_mode():
            mode = self.feature_mode_combo.currentText()
            if mode == "Markers only":
                self._k_int_row.show()
                self._k_morpho_row.hide()
            elif mode == "Morphometrics only":
                self._k_int_row.hide()
                self._k_morpho_row.show()
            else:
                self._k_int_row.show()
                self._k_morpho_row.show()
        self.feature_mode_combo.currentTextChanged.connect(lambda _t: _update_feature_mode())
        # Default to Both
        self.feature_mode_combo.setCurrentText("Both")
        _update_feature_mode()

        # Update reasoning level visibility when model changes
        def _update_reasoning_visibility():
            model = self.model_combo.currentText()
            if model == "gpt-5.2":
                self._reasoning_container.show()
            else:
                self._reasoning_container.hide()
        self.model_combo.currentTextChanged.connect(lambda _t: _update_reasoning_visibility())
        _update_reasoning_visibility()  # Set initial state

        layout.addLayout(form)

        # Create export button before adding to layout
        self.export_btn = QtWidgets.QPushButton("Export LLM Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_llm_results)
        self.export_btn.hide()  # Hidden until results are available

        btns = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run Suggestion")
        self.run_btn.clicked.connect(self._run)
        self.apply_btn = QtWidgets.QPushButton("Apply Names")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(self.run_btn)
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.export_btn)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        # Progress bar for long-running suggestions
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        # Area to render per-cluster choices after suggestions arrive (scrollable for many clusters)
        self.choices_widget = QtWidgets.QWidget()
        self.choices_layout = QtWidgets.QVBoxLayout(self.choices_widget)
        self.choices_layout.setContentsMargins(8, 8, 8, 8)
        self.choices_layout.setSpacing(8)
        self.choices_scroll = QtWidgets.QScrollArea()
        self.choices_scroll.setWidgetResizable(True)
        self.choices_scroll.setWidget(self.choices_widget)
        layout.addWidget(self.choices_scroll)

        # Holds QButtonGroup per cluster for selection
        self._cluster_choice_groups = {}
        
        # Initialize suggestions dict - will be populated from cache or new results
        self._suggestions = {}  # cluster_id -> parsed json
        
        # Check for cached results and display them immediately
        self._check_and_display_cached_results()

    def closeEvent(self, event):
        """Handle dialog closing to preserve cache and apply suggestions."""
        # Ensure the current suggestions are cached for future use
        # This is a backup in case results weren't cached during _run
        if self._suggestions:
            # Save all suggestions to cache
            self._save_results_to_cache(self._suggestions)
        
        # Automatically apply suggestions when closing if they exist
        # This ensures annotations persist even if user doesn't click "Apply Names"
        if self._suggestions:
            display_name_map = {}
            backend_name_map = {}
            # Prefer user-selected guess when available, otherwise use first suggestion
            for cid, obj in self._suggestions.items():
                # Ensure cid is an integer for consistent handling
                cid_int = int(cid)
                try:
                    selected_idx = None
                    grp = self._cluster_choice_groups.get(cid_int)
                    if grp is not None:
                        id_ = grp.checkedId()
                        if id_ != -1:
                            selected_idx = id_
                    guesses = obj.get('phenotype_guesses') or []
                    chosen = None
                    if selected_idx is not None and 0 <= selected_idx < len(guesses):
                        chosen = guesses[selected_idx]
                    elif guesses:
                        chosen = guesses[0]
                    if chosen:
                        name = str(chosen.get('name', '')).strip()
                        if name:
                            # Store human-readable name for display
                            display_name_map[cid_int] = name
                            
                            # Create normalized name for backend CSV
                            norm = name.replace(' ', '_')
                            if norm.lower() == 't_cell':
                                norm = 'T_cell'
                            if 'macrophage' in norm.lower():
                                norm = 'Myeloid_Macrophage'
                            backend_name_map[cid_int] = norm
                except Exception:
                    continue
            if display_name_map:
                # Apply suggestions silently (no message box) when closing
                self._apply_callback(display_name_map, backend_name_map)
        
        event.accept()

    def _reset_progress_bar(self):
        """Reset the progress bar to its default state."""
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Suggestion")

    def _save_results_to_cache(self, results_dict):
        """Save results to cache immediately. Ensures cache persists even without clicking Apply."""
        if self._cache_dict is None:
            # If cache_dict is None, try to get it from parent dialog
            if self._parent_dialog and hasattr(self._parent_dialog, 'llm_phenotype_cache'):
                self._cache_dict = self._parent_dialog.llm_phenotype_cache
            else:
                return
        
        # Ensure _cache_dict is a proper dict
        if not isinstance(self._cache_dict, dict):
            if self._parent_dialog and hasattr(self._parent_dialog, 'llm_phenotype_cache'):
                self._parent_dialog.llm_phenotype_cache = {}
                self._cache_dict = self._parent_dialog.llm_phenotype_cache
            else:
                return
        
        # Update cache with results, ensuring integer keys
        cache_update = {}
        for cid, result in results_dict.items():
            cid_int = int(cid) if isinstance(cid, str) else cid
            cache_update[cid_int] = result
        
        # Update the cache (this is a reference to parent's llm_phenotype_cache)
        self._cache_dict.update(cache_update)
        
        # Also ensure parent's cache is updated (in case reference was lost)
        if self._parent_dialog and hasattr(self._parent_dialog, 'llm_phenotype_cache'):
            if not isinstance(self._parent_dialog.llm_phenotype_cache, dict):
                self._parent_dialog.llm_phenotype_cache = {}
            self._parent_dialog.llm_phenotype_cache.update(cache_update)
    
    def _check_and_display_cached_results(self):
        """Check if we have cached results for the current cluster set and display them."""
        # Ensure _cache_dict is properly initialized
        if self._cache_dict is None:
            self._cache_dict = {}
            return
        
        if not isinstance(self._cache_dict, dict) or not self._cache_dict:
            return
            
        # Check if we have cached results for current clusters
        # Use integer keys consistently for both cache lookup and storage
        cached_results = {}
        for cid in self._cluster_ids:
            # Convert cluster ID to int for consistent comparison
            cid_int = int(cid)
            # Check both the original cid (string) and converted int version
            # Also check if the cache has the cluster ID in any form
            if cid_int in self._cache_dict:
                cached_results[cid_int] = self._cache_dict[cid_int]
            elif cid in self._cache_dict:
                # If found with string key, convert to int for consistency
                cached_results[cid_int] = self._cache_dict[cid]
            elif str(cid_int) in self._cache_dict:
                # Also check string version of int
                cached_results[cid_int] = self._cache_dict[str(cid_int)]
        
        # If we have cached results for any clusters, display them
        if cached_results:
            # Store the cached results in _suggestions so they can be applied
            # Use integer keys consistently
            self._suggestions.update(cached_results)
            self._render_choices(cached_results)
            self.apply_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            self.export_btn.show()
            # If we have results for all clusters, disable run button
            if len(cached_results) == len(self._cluster_ids):
                self.run_btn.setEnabled(False)
                self.run_btn.setText("Results Cached - Re-run to refresh")
            else:
                # Partial cache - allow re-running but show that some results are cached
                self.run_btn.setEnabled(True)
                cached_count = len(cached_results)
                total_count = len(self._cluster_ids)
                self.run_btn.setText(f"Run Suggestion ({cached_count}/{total_count} cached)")

    def _apply(self):
        display_name_map = {}
        backend_name_map = {}
        # Prefer user-selected guess when available
        for cid, obj in self._suggestions.items():
            # Ensure cid is an integer for consistent handling
            cid_int = int(cid)
            try:
                selected_idx = None
                grp = self._cluster_choice_groups.get(cid_int)
                if grp is not None:
                    id_ = grp.checkedId()
                    if id_ != -1:
                        selected_idx = id_
                guesses = obj.get('phenotype_guesses') or []
                chosen = None
                if selected_idx is not None and 0 <= selected_idx < len(guesses):
                    chosen = guesses[selected_idx]
                elif guesses:
                    chosen = guesses[0]
                if chosen:
                    name = str(chosen.get('name', '')).strip()
                    if name:
                        # Store human-readable name for display
                        display_name_map[cid_int] = name
                        
                        # Create normalized name for backend CSV
                        norm = name.replace(' ', '_')
                        if norm.lower() == 't_cell':
                            norm = 'T_cell'
                        if 'macrophage' in norm.lower():
                            norm = 'Myeloid_Macrophage'
                        backend_name_map[cid_int] = norm
            except Exception:
                continue
        if display_name_map:
            self._apply_callback(display_name_map, backend_name_map)
            # Ensure the current suggestions are cached for future use
            if self._suggestions:
                self._save_results_to_cache(self._suggestions)
            QtWidgets.QMessageBox.information(self, "Applied", f"Applied {len(display_name_map)} suggested names.")

    def _export_llm_results(self):
        """Export LLM results to a JSON file."""
        if not self._suggestions:
            QtWidgets.QMessageBox.warning(self, "No Results", "No LLM results available to export.")
            return
        
        # Prepare export data - convert integer keys to strings for JSON compatibility
        export_data = {}
        for cid_int, result in self._suggestions.items():
            # Convert cluster_id to string for JSON export
            export_result = result.copy()
            export_result['cluster_id'] = str(cid_int)
            export_data[str(cid_int)] = export_result
        
        # Open file dialog to select save location
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export LLM Results",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Ensure .json extension
                if not file_path.endswith('.json'):
                    file_path += '.json'
                
                # Write JSON with pretty formatting
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"LLM results exported to:\n{file_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export LLM results:\n{str(e)}"
                )

    def _process_single_cluster(self, cid, api_key, stats_per_cluster, k_int, context_str, model_name):
        """Process a single cluster with LLM API call. Used as worker function for ThreadPoolExecutor."""
        import time
        
        cid_int = int(cid)
        max_retries = 3  # Maximum number of retries for timeout/connection errors
        retry_delay = 2  # Initial delay between retries (seconds)
        
        for attempt in range(max_retries):
            try:
                payload = self._build_prompt_payload(cid, stats_per_cluster.get(cid, {}), k_int, context_str)
                payload['model'] = model_name
                # Validate payload before sending
                suggestion = self._call_openai(api_key, payload)
                # Validate JSON
                obj = self._validate_json(suggestion, cid)
                if obj is None:
                    # One retry with repair instruction
                    suggestion = self._call_openai(api_key, payload, repair=True)
                    obj = self._validate_json(suggestion, cid)
                if obj is not None:
                    # Return with consistent integer keys
                    return (cid_int, obj)
                return (cid_int, None)
            except Exception as e:
                error_msg = str(e).lower()
                is_timeout = "timeout" in error_msg or "timed out" in error_msg
                is_connection = "connection" in error_msg or "network" in error_msg
                
                # Retry for timeout/connection errors, but not for other errors (auth, rate limit, etc.)
                if (is_timeout or is_connection) and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    # Return error information for this cluster (no more retries or non-retryable error)
                    return (cid_int, {'error': str(e)})
        
        # Should not reach here, but just in case
        return (cid_int, {'error': 'Max retries exceeded'})

    def _run(self):
        # Disable the run button to prevent multiple clicks
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        
        # Show immediate feedback that processing has started
        self.progress.setRange(0, 0)  # Indeterminate progress
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Starting LLM analysis...")
        QtWidgets.QApplication.processEvents()  # Force UI update
        
        try:
            api_key = self.api_key_edit.text().strip()
            if not api_key:
                self._reset_progress_bar()
                QtWidgets.QMessageBox.warning(self, "API Key Required", "Please enter an OpenAI API key.")
                return
            if self._parent_dialog.clustered_data is None:
                self._reset_progress_bar()
                QtWidgets.QMessageBox.warning(self, "No Clusters", "Run clustering first.")
                return
            
            mode = self.feature_mode_combo.currentText()
            k_int = self.k_int_spin.value()
            k_morpho = self.k_morpho_spin.value()
            model_name = self.model_combo.currentText()
            context_str = self.context_edit.text().strip() or "IMC panel of single cells"
            
            # Use k_int as the K parameter for consistency
            self.progress.setFormat("Computing cluster statistics...")
            QtWidgets.QApplication.processEvents()
            stats_per_cluster = self._compute_stats(self._parent_dialog.clustered_data, K=k_int, mode=mode, k_int=k_int, k_morpho=k_morpho)
            
            total = max(1, len(self._cluster_ids))
            self.progress.setRange(0, total)
            self.progress.setValue(0)
            self.progress.setFormat("Processing clusters with LLM...")
            QtWidgets.QApplication.processEvents()
            
            # Use ThreadPoolExecutor for parallel API calls (up to 10 concurrent requests)
            max_workers = min(10, len(self._cluster_ids))
            results = {}
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks - maintain order by storing futures with their cluster IDs
                future_to_cid = {}
                for cid in self._cluster_ids:
                    future = executor.submit(
                        self._process_single_cluster,
                        cid, api_key, stats_per_cluster, k_int, context_str, model_name
                    )
                    future_to_cid[future] = cid
                
                # Collect results as they complete (maintain order by processing in submission order)
                # Use as_completed to process results as soon as they're ready, but track order
                results_dict = {}  # Temporary storage for results
                
                # Process completed futures
                for future in as_completed(future_to_cid):
                    try:
                        cid_int, obj = future.result()
                        results_dict[cid_int] = obj
                        completed_count += 1
                        # Update progress
                        self.progress.setValue(completed_count)
                        self.progress.setFormat(f"Processing clusters with LLM... ({completed_count}/{total})")
                        QtWidgets.QApplication.processEvents()
                    except Exception as e:
                        cid = future_to_cid[future]
                        cid_int = int(cid)
                        # Store error in results_dict
                        results_dict[cid_int] = {'error': str(e)}
                        completed_count += 1
                        self.progress.setValue(completed_count)
                        QtWidgets.QApplication.processEvents()
                
                # Now collect results in the original cluster order to maintain consistency
                for cid in self._cluster_ids:
                    cid_int = int(cid)
                    if cid_int in results_dict:
                        obj = results_dict[cid_int]
                        if obj is not None and 'error' not in obj:
                            results[cid_int] = obj
                            self._suggestions[cid_int] = obj
                        elif obj is not None and 'error' in obj:
                            # Error will be shown to user after all processing completes
                            pass
            
            # Track errors for user notification
            errors = []
            for cid in self._cluster_ids:
                cid_int = int(cid)
                if cid_int in results_dict:
                    obj = results_dict[cid_int]
                    if obj is not None and 'error' in obj:
                        errors.append((cid_int, obj['error']))
            
            if results:
                # Cache the results immediately (using integer keys consistently)
                # This ensures cache persists even if dialog is closed without clicking "Apply Names"
                self._save_results_to_cache(results)
                self._render_choices(results)
                self.apply_btn.setEnabled(True)
                self.export_btn.setEnabled(True)
                self.export_btn.show()
            
            # Show error notification if any clusters failed
            if errors:
                error_msg = f"The following clusters encountered errors:\n\n"
                for cid, error in errors:
                    error_msg += f"Cluster {cid}: {error}\n"
                error_msg += f"\nTotal: {len(errors)} cluster(s) failed out of {len(self._cluster_ids)}"
                QtWidgets.QMessageBox.warning(
                    self,
                    "LLM Processing Errors",
                    error_msg
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "LLM Error", f"Error suggesting phenotypes: {str(e)}")
        finally:
            # Reset progress bar and button state
            self._reset_progress_bar()

    def _numeric_feature_columns(self, df: pd.DataFrame):
        exclude = {'cluster', 'cluster_phenotype', 'manual_phenotype'}
        cols = []
        for c in df.columns:
            if c in exclude:
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cols.append(c)
            except Exception:
                continue
        return cols

    def _sorted_cluster_ids(self, values, *, canonical: bool = False):
        """Return cluster ids ordered numerically using the shared cluster helper."""
        annotation_map = None
        if self._parent_dialog is not None and hasattr(self._parent_dialog, 'cluster_annotation_map'):
            annotation_map = self._parent_dialog.cluster_annotation_map
        return sort_cluster_values(values, annotation_map=annotation_map, canonical=canonical)

    def _base_marker_name(self, feature_name: str) -> str:
        # First check if there's a custom feature label from the parent dialog
        if self._parent_dialog and hasattr(self._parent_dialog, 'feature_label_map'):
            if feature_name in self._parent_dialog.feature_label_map:
                custom_label = self._parent_dialog.feature_label_map[feature_name]
                custom_label = str(custom_label).strip()
                if custom_label:
                    # Preserve user-provided label as-is.
                    return custom_label
        
        # Strip trailing suffix like _mean/_median/etc.
        if '_' in feature_name:
            return feature_name.rsplit('_', 1)[0]
        return feature_name

    def _feature_label_for_prompt(self, feature_name: str) -> str:
        """Return the best display label to send in LLM context for a feature column."""
        if self._parent_dialog and hasattr(self._parent_dialog, 'feature_label_map'):
            label_map = getattr(self._parent_dialog, 'feature_label_map', {}) or {}
            if feature_name in label_map:
                label = str(label_map[feature_name]).strip()
                if label:
                    return label
        return feature_name

    def _synonymize(self, name: str) -> str:
        m = name
        repl = {
            'KRT8/18': 'CK8/18',
            'EPCAM': 'EpCAM',
            'KRT8': 'CK8',
            'KRT18': 'CK18',
        }
        k = m.upper()
        for src, dst in repl.items():
            if src == k:
                return dst
        return name

    def _compute_stats(self, clustered_df: pd.DataFrame, K: int=None, mode: str="Both", k_int: int=5, k_morpho: int=5):
        # Use K as default for k_int and k_morpho if not provided
        if K is not None:
            k_int = K
            k_morpho = K
        from math import log2
        try:
            from sklearn.metrics import roc_auc_score
        except Exception:
            roc_auc_score = None
        eps = 1e-6
        cols = self._numeric_feature_columns(clustered_df)
        # Split into intensity vs morphometric by suffix
        intensity_suffixes = ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos']
        intensity_cols = [c for c in cols if any(c.endswith(s) for s in intensity_suffixes)]
        morpho_cols = [c for c in cols if c not in intensity_cols]
        
        # Filter out DNA markers (positive in all cells) and ICSK membrane markers
        exclude_tokens = ['DNA1', 'DNA2', 'DNA_', 'IR191', 'IR193', 'ICSK']
        excluded_markers_int = [col for col in intensity_cols if any(tok in col.upper() for tok in exclude_tokens)]
        intensity_cols = [col for col in intensity_cols if col not in excluded_markers_int]
        # For morphometrics keep all (no DNA markers)

        # Choose working columns based on mode
        if mode == 'Markers only':
            work_cols = intensity_cols
        elif mode == 'Morphometrics only':
            work_cols = morpho_cols
        else:
            work_cols = intensity_cols + morpho_cols
        
        # Prepare per-cluster means
        cluster_ids = self._sorted_cluster_ids(clustered_df['cluster'].unique(), canonical=False)
        means = clustered_df.groupby('cluster')[work_cols].mean()
        stats = {}
        # Precompute across-cluster ranges per feature (based on means per cluster)
        across_range = (means.max(axis=0) - means.min(axis=0))
        for cid in cluster_ids:
            this_mean = means.loc[cid]
            rest_mean = means.drop(index=cid).mean()
            # z across clusters per feature
            col_means = means.mean(axis=0)
            col_stds = means.std(axis=0).replace(0, np.nan)
            z = (this_mean - col_means) / col_stds
            # Within-cluster distribution stats
            in_cluster = clustered_df[clustered_df['cluster'] == cid]
            in_min = in_cluster[work_cols].min(axis=0)
            in_max = in_cluster[work_cols].max(axis=0)
            in_mean = in_cluster[work_cols].mean(axis=0)
            in_median = in_cluster[work_cols].median(axis=0)
            # logFC with robust clipping to avoid log of non-positive values
            ratio = (this_mean + eps) / (rest_mean + eps)
            # Replace inf/-inf with NaN and non-positive ratios with NaN
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            ratio = ratio.where(ratio > 0, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                logfc = np.log2(ratio)
            # AUROC
            auroc = pd.Series(index=work_cols, dtype=float)
            if roc_auc_score is not None:
                labels = (clustered_df['cluster'] == cid).astype(int).values
                for f in work_cols:
                    vals = clustered_df[f].values
                    try:
                        if labels.sum() > 0 and labels.sum() < len(labels):
                            auroc[f] = roc_auc_score(labels, vals)
                        else:
                            auroc[f] = np.nan
                    except Exception:
                        auroc[f] = np.nan
            else:
                auroc[:] = np.nan
            # pct_pos at threshold tau (0 by default on normalized scale)
            tau = 0.0
            out_cluster = clustered_df[clustered_df['cluster'] != cid]
            pct_pos_in = (in_cluster[work_cols] > tau).sum(axis=0) / max(1, len(in_cluster))
            pct_pos_out = (out_cluster[work_cols] > tau).sum(axis=0) / max(1, len(out_cluster))

            # Ranking: by z-score only (descending)
            ranked = z.sort_values(ascending=False).index.tolist()
            # Select counts based on mode
            if mode == 'Both':
                # Split selection across intensity and morpho
                ranked_int = [f for f in ranked if f in intensity_cols][:k_int]
                ranked_morpho = [f for f in ranked if f in morpho_cols][:k_morpho]
                selected_up = ranked_int + ranked_morpho
            else:
                k = k_int if mode == 'Markers only' else k_morpho
                selected_up = ranked[:k]
            top_up = []
            for f in selected_up:
                base = self._synonymize(self._base_marker_name(f))
                top_up.append({
                    'feature_column': f,
                    'feature_label': self._feature_label_for_prompt(f),
                    'marker': base,
                    'auroc': None if pd.isna(auroc[f]) else float(auroc[f]),
                    'logFC': None if pd.isna(logfc[f]) else float(logfc[f]),
                    'z': None if pd.isna(z[f]) else float(z[f]),
                    'mean': None if pd.isna(this_mean[f]) else float(this_mean[f]),
                    'pct_pos': None if pd.isna(pct_pos_in[f]) else float(pct_pos_in[f]),
                    'within_min': None if pd.isna(in_min[f]) else float(in_min[f]),
                    'within_mean': None if pd.isna(in_mean[f]) else float(in_mean[f]),
                    'within_median': None if pd.isna(in_median[f]) else float(in_median[f]),
                    'within_max': None if pd.isna(in_max[f]) else float(in_max[f]),
                    'range_across_clusters': None if pd.isna(across_range[f]) else float(across_range[f])
                })
            # Down markers: lowest z-scores — take bottom K as requested
            down_ranked = z.sort_values(ascending=True).index.tolist()
            if mode == 'Both':
                down_int = [f for f in down_ranked if f in intensity_cols][:k_int]
                down_morpho = [f for f in down_ranked if f in morpho_cols][:k_morpho]
                selected_down = down_int + down_morpho
            else:
                k = k_int if mode == 'Markers only' else k_morpho
                selected_down = down_ranked[:k]
            top_down = []
            for f in selected_down:
                base = self._synonymize(self._base_marker_name(f))
                top_down.append({
                    'feature_column': f,
                    'feature_label': self._feature_label_for_prompt(f),
                    'marker': base,
                    'auroc': None if pd.isna(auroc[f]) else float(auroc[f]),
                    'logFC': None if pd.isna(logfc[f]) else float(logfc[f]),
                    'z': None if pd.isna(z[f]) else float(z[f]),
                    'within_min': None if pd.isna(in_min[f]) else float(in_min[f]),
                    'within_mean': None if pd.isna(in_mean[f]) else float(in_mean[f]),
                    'within_median': None if pd.isna(in_median[f]) else float(in_median[f]),
                    'within_max': None if pd.isna(in_max[f]) else float(in_max[f]),
                    'range_across_clusters': None if pd.isna(across_range[f]) else float(across_range[f])
                })
            stats[cid] = {
                'top_up': top_up,
                'top_down': top_down,
            }
        return stats

    def _render_choices(self, results: dict):
        # Clear previous choices
        for i in reversed(range(self.choices_layout.count())):
            item = self.choices_layout.takeAt(i)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cluster_choice_groups = {}

        # Build UI for each cluster
        for cid in sorted(
            results.keys(),
            key=lambda value: cluster_sort_key(value, annotation_map=self._parent_dialog.cluster_annotation_map if self._parent_dialog else None),
        ):
            obj = results[cid]
            guesses = obj.get('phenotype_guesses') or []

            group_box = QtWidgets.QGroupBox(f"Cluster {cid} – Select phenotype")
            v = QtWidgets.QVBoxLayout(group_box)
            btn_group = QtWidgets.QButtonGroup(group_box)
            btn_group.setExclusive(True)

            # Create a radio per guess and show rationale with confidence
            for idx, g in enumerate(guesses):
                name = str(g.get('name', '')).strip() or 'Unknown'
                confidence = g.get('confidence')
                if confidence is not None:
                    confidence_str = f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else str(confidence)
                    display_name = f"{name} ({confidence_str})"
                else:
                    display_name = name
                rb = QtWidgets.QRadioButton(display_name)
                btn_group.addButton(rb, idx)
                if idx == 0:
                    rb.setChecked(True)
                v.addWidget(rb)
                rationale = str(g.get('rationale', '')).strip()
                if rationale:
                    rationale_lbl = QtWidgets.QLabel(rationale)
                    rationale_lbl.setWordWrap(True)
                    rationale_lbl.setStyleSheet("color: #555;")
                    v.addWidget(rationale_lbl)

            # If no guesses, indicate
            if not guesses:
                v.addWidget(QtWidgets.QLabel("No plausible types returned."))

            self._cluster_choice_groups[int(cid)] = btn_group
            self.choices_layout.addWidget(group_box)

        self.choices_layout.addStretch(1)

    def _build_prompt_payload(self, cid, stat_obj, K: int, context_str: str):
        # Select system prompt based on user choice
        prompt_type = self.system_prompt_combo.currentText()
        if "Broad" in prompt_type:
            # Broad cell types system prompt
            system_prompt = (
                "You are assisting with IMC cell type suggestions. Use only the provided marker statistics. "
                "Classify cells into broad categories: Myeloid (macrophages, monocytes, dendritic cells, neutrophils, etc.), "
                "Tumor (cancer cells), Stroma (fibroblasts, endothelial cells, etc.), Lymphoid (T cells, B cells, NK cells, etc.), "
                "or other broad categories as appropriate. Give exactly 3 plausible phenotypes per cluster. "
                "When you have high confidence (typically when confidence is above 50%), you may specify more specific cell types within the broad categories. "
                "For example, within Lymphoid you can specify 'T cells', 'B cells', or 'NK cells' if the marker profile strongly supports it. "
                "Within Myeloid, you can specify 'Macrophages', 'Dendritic cells', 'Neutrophils', etc. if confident. "
                "For lower confidence predictions, use the broad category names (e.g., 'Lymphoid', 'Myeloid', 'Stroma', 'Tumor'). "
                "Rank the 3 phenotypes from most likely to least likely, and provide a confidence percentage for each phenotype. "
                "The three confidence percentages must sum to exactly 100%. "
                "Consider the range of values across and within clusters for each marker to help determine if the marker is truly unique to the cluster. "
                "Consider if the z-score and mean value is true expression of the marker or if it is due to noise. "
                "Try to give varied phenotypes, rather than the same phenotype with different names. "
                "Focus on different marker combinations to avoid giving the same phenotype with different names. "
                "If uncertain, return \"Unknown\". Output valid JSON exactly matching the given schema. Do not invent markers. "
                "When feature_label_overrides are provided, prefer those user-supplied names over raw channel names. "
                "Return valid JSON only and no prose/explanations."
            )
        else:
            # Fine cell types system prompt (default)
            system_prompt = (
                "You are assisting with IMC cell type suggestions. Use only the provided marker statistics. "
                "Prefer canonical immune/epithelial/stromal names and give exactly 3 plausible phenotypes per cluster. "
                "Rank the 3 phenotypes from most likely to least likely, and provide a confidence percentage for each phenotype. "
                "The three confidence percentages must sum to exactly 100%. "
                "Consider the range of values across and within clusters for each marker to help determine if the marker is truly unique to the cluster. "
                "Consider if the z-score and mean value is true expression of the marker or if it is due to noise. "
                "Try to give varied phenotypes, rather than the same phenotype with different names. "
                "Focus on different marker combinations to avoid giving the same phenotype with different names. "
                "If uncertain, return \"Unknown\". Output valid JSON exactly matching the given schema. Do not invent markers. "
                "When feature_label_overrides are provided, prefer those user-supplied names over raw channel names. "
                "Return valid JSON only and no prose/explanations."
            )
        schema = {
            "cluster_id": str(cid),
            "phenotype_guesses": [ { "name": "", "rationale": "", "confidence": 0 } ],
            "key_markers_positive": [],
            "key_markers_negative": [],
            "notes": ""
        }
        # Determine if arcsinh transformation was used during feature extraction
        arcsinh_used = (self.normalization_config is not None and 
                       self.normalization_config.get('method') == 'arcsinh')
        
        # Set semantics based on whether arcsinh transformation was applied
        if arcsinh_used:
            semantics = 'intensities are arcsinh-transformed; higher = more expression'
        else:
            semantics = 'intensities are raw values; higher = more expression'
        
        user_context = {
            'context': context_str,
            'semantics': semantics,
            'cluster_id': str(cid),
            'top_up': stat_obj.get('top_up', []),
            'top_down': stat_obj.get('top_down', []),
        }
        # Add an explicit raw-column -> user label mapping for features in this cluster payload.
        feature_label_overrides = {}
        for marker_obj in (stat_obj.get('top_up', []) + stat_obj.get('top_down', [])):
            col = marker_obj.get('feature_column')
            label = marker_obj.get('feature_label')
            if isinstance(col, str) and isinstance(label, str):
                col = col.strip()
                label = label.strip()
                if col and label:
                    feature_label_overrides[col] = label
        if feature_label_overrides:
            user_context['feature_label_overrides'] = feature_label_overrides
        # Build Responses API input structure
        input_msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt + " Schema: {\\n  \\\"cluster_id\\\": \\\"string\\\",\\n  \\\"phenotype_guesses\\\": [\\n    { \\\"name\\\": \\\"string\\\", \\\"rationale\\\": \\\"string\\\", \\\"confidence\\\": number }\\n  ],\\n  \\\"key_markers_positive\\\": [\\\"string\\\"],\\n  \\\"key_markers_negative\\\": [\\\"string\\\"],\\n  \\\"notes\\\": \\\"string\\\"\\n}"}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(user_context)}
                ]
            }
        ]
        return {
            'model': 'gpt-5',
            'temperature': 0.1,
            'max_tokens': 2000,
            'input': input_msgs
        }

    def _call_openai(self, api_key: str, payload: dict, repair: bool=False) -> str:
        # Use OpenAI official SDK per developer guide
        from openai import OpenAI
        
        # Increased timeout for LLM calls (120 seconds) - LLM responses can take time
        client = OpenAI(api_key=api_key, timeout=120.0)
        data = payload.copy()
        # Append repair instruction as another system content block if needed
        input_payload = data.get('input')
        if repair and isinstance(input_payload, list):
            input_payload = input_payload + [{"role": "system", "content": [{"type": "input_text", "text": "Return valid JSON only, no prose."}]}]
        try:
            pass
        except Exception:
            pass
        # Responses API call
        try:
            model_name = data.get('model', 'gpt-5')
            create_kwargs = {
                'model': model_name,
                'max_output_tokens': data.get('max_tokens', 2000),
                'input': input_payload
            }
            # Only add reasoning parameter for gpt-5.2
            if model_name == 'gpt-5.2':
                reasoning_level = self.reasoning_combo.currentText()
                if reasoning_level != 'none':
                    create_kwargs['reasoning'] = {'effort': reasoning_level}
            resp = client.responses.create(**create_kwargs)
        except Exception as e:
            error_msg = str(e)
            
            # Provide more specific error information
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                raise Exception(f"Connection error: {error_msg}. Please check your internet connection and try again.")
            elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise Exception(f"Authentication error: {error_msg}. Please check your API key.")
            elif "rate_limit" in error_msg.lower():
                raise Exception(f"Rate limit exceeded: {error_msg}. Please wait a moment and try again.")
            elif "model" in error_msg.lower():
                raise Exception(f"Model error: {error_msg}. Please try a different model.")
            else:
                raise Exception(f"OpenAI API error: {error_msg}")
        # SDK v1.42+ provides output_text for assembled content
        content = getattr(resp, 'output_text', None)
        if not content:
            try:
                # Fallback: extract text from output items
                pieces = []
                for item in getattr(resp, 'output', []) or []:
                    for block in item.get('content', []) or []:
                        if block.get('type') in ('output_text', 'summary_text'):
                            pieces.append(block.get('text', ''))
                content = "\n".join([p for p in pieces if p]) or "{}"
            except Exception:
                content = "{}"
        try:
            pass
        except Exception:
            pass
        return content

    def _validate_json(self, s: str, cid) -> dict:
        try:
            obj = json.loads(s)
            # Basic schema checks
            if str(obj.get('cluster_id', '')) != str(cid):
                obj['cluster_id'] = str(cid)
            if not isinstance(obj.get('phenotype_guesses', []), list):
                obj['phenotype_guesses'] = []
            # Ensure confidence values are present and validate they sum to 100%
            guesses = obj.get('phenotype_guesses', [])
            if guesses:
                confidences = []
                for g in guesses:
                    if 'confidence' not in g:
                        g['confidence'] = None
                    else:
                        conf = g.get('confidence')
                        if conf is not None:
                            try:
                                confidences.append(float(conf))
                            except (ValueError, TypeError):
                                confidences.append(None)
                # If all confidences are provided, normalize to sum to 100
                if confidences and all(c is not None for c in confidences):
                    total = sum(confidences)
                    if total > 0 and abs(total - 100.0) > 0.1:  # Allow small floating point differences
                        # Normalize to sum to 100
                        for i, g in enumerate(guesses):
                            if i < len(confidences) and confidences[i] is not None:
                                g['confidence'] = round((confidences[i] / total) * 100.0, 1)
            if not isinstance(obj.get('key_markers_positive', []), list):
                obj['key_markers_positive'] = []
            if not isinstance(obj.get('key_markers_negative', []), list):
                obj['key_markers_negative'] = []
            if 'notes' not in obj:
                obj['notes'] = ""
            return obj
        except Exception:
            return None
