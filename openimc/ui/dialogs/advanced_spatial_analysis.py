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
Advanced Spatial Analysis Dialog for OpenIMC

This module provides the advanced spatial analysis dialog using squidpy.
"""

import os
import sys
# CRITICAL: Configure dask BEFORE any imports that might trigger dask.dataframe import
# This must be done at the very top, before any other imports
os.environ.setdefault('DASK_DATAFRAME__QUERY_PLANNING', 'True')

# Also try direct config if dask is available (must be before squidpy import)
try:
    import dask
    # Check if dask.dataframe has already been imported (too late to configure)
    dask_dataframe_imported = 'dask.dataframe' in sys.modules
    # Set configuration before dask.dataframe is imported
    dask.config.set({'dataframe.query-planning': True})
except (ImportError, AttributeError) as e:
    pass
except Exception as e:
    import traceback
    traceback.print_exc()
    pass

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Delaunay
from scipy import stats
import seaborn as sns
from openimc.ui.cluster_utils import (
    canonicalize_cluster_id,
    extract_cluster_annotation_map_from_dataframe,
    format_default_cluster_label,
    get_cluster_display_name,
    map_cluster_series_to_display_names,
    normalize_cluster_annotation_map,
    sort_cluster_values,
)
from openimc.ui.figure_layout import dense_heatmap_style, fit_canvas_and_draw, refresh_canvas
from openimc.ui.utils import (
    benjamini_hochberg_adjust,
    benjamini_hochberg_adjust_matrix,
    combine_pvalues_fisher,
)
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.label_customization_dialogs import (
    edit_cluster_annotation_map,
    edit_feature_label_map,
)
from openimc.ui.dialogs.progress_dialog import run_blocking_task_with_progress
from openimc.ui.dialogs import spatial_analysis as spatial_analysis_module
from openimc.ui.dialogs.spatial_analysis import (
    SourceFileFilterDialog,
    _HAVE_SQUIDPY,
    _HAVE_SPARSE,
    _get_squidpy_modules,
    _get_vivid_colors,
)
from openimc.core import (
    dataframe_to_anndata,
    build_spatial_graph_anndata,
    spatial_neighborhood_enrichment,
    spatial_cooccurrence,
    spatial_autocorrelation,
    spatial_ripley,
    export_anndata
)

sq = None
ad = None
_HAVE_SQUIDPY_LOCAL = False
_SQUIDPY_IMPORT_ERROR = None


def _ensure_local_squidpy() -> bool:
    """Load squidpy on demand for the advanced dialog."""
    global sq, ad, _HAVE_SQUIDPY_LOCAL, _SQUIDPY_IMPORT_ERROR

    if _HAVE_SQUIDPY_LOCAL and sq is not None and ad is not None:
        return True

    squidpy_modules = _get_squidpy_modules()
    if squidpy_modules is None:
        _HAVE_SQUIDPY_LOCAL = False
        _SQUIDPY_IMPORT_ERROR = getattr(spatial_analysis_module, "_SQUIDPY_IMPORT_ERROR", None)
        return False

    sq, _scanpy_module, ad = squidpy_modules
    _HAVE_SQUIDPY_LOCAL = True
    _SQUIDPY_IMPORT_ERROR = None
    return True

try:
    from scipy import sparse as sp
except Exception:
    sp = None


class AdvancedSpatialAnalysisDialog(QtWidgets.QDialog):
    """Advanced Spatial Analysis Dialog using squidpy for all analyses."""
    def __init__(self, feature_dataframe: pd.DataFrame, batch_corrected_dataframe=None, clustered_cells_dataframe=None, parent=None):
        
        # Check both the imported flag and local import status
        if not (_HAVE_SQUIDPY or _ensure_local_squidpy()):
            error_msg = "squidpy is required for AdvancedSpatialAnalysisDialog. Please install with: pip install squidpy anndata"
            if '_SQUIDPY_IMPORT_ERROR' in globals() and _SQUIDPY_IMPORT_ERROR:
                error_msg += f"\n\nImport error details: {_SQUIDPY_IMPORT_ERROR}"
            raise RuntimeError(error_msg)
        
        
        super().__init__(parent)
        self.setWindowTitle("Advanced Spatial Analysis (Squidpy)")
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
        # Filtering happens dynamically when creating AnnData objects based on current clustering state
        
        # Store AnnData objects per ROI
        self.anndata_cache: Dict[str, 'ad.AnnData'] = {}
        self.spatial_graph_built = False
        
        # Track which analyses have been run per ROI
        self.analysis_status: Dict[str, Dict[str, bool]] = {}  # {roi_id: {analysis_type: bool}}
        
        
        # Cluster annotation mapping
        self.cluster_annotation_map = {}
        self.feature_label_map = {}
        
        # Source file filtering
        self.selected_source_files = set()
        self.available_source_files = set()
        
        # Track processed ROIs
        self.processed_rois: Dict[str, Dict[str, Any]] = {}  # {roi_id: {graph_built: bool, analyses: []}}
        
        # Store aggregated results for plotting
        self.aggregated_results: Dict[str, Any] = {}  # {analysis_type: aggregated_data}
        
        self._create_ui()
        self._plot_resize_in_progress = False
        self._plot_resize_timer = QtCore.QTimer(self)
        self._plot_resize_timer.setSingleShot(True)
        self._plot_resize_timer.timeout.connect(self._refresh_current_plot_after_resize)
        if hasattr(self, 'tabs'):
            self.tabs.currentChanged.connect(self._queue_plot_resize_refresh)
        
        if hasattr(self, 'source_file_status_label'):
            self._update_source_file_status_label()

    def _fit_canvas(self, canvas, *, rect=None, pad: float = 0.95):
        """Fit the supplied canvas to the current tab size and redraw it."""
        if canvas is None or canvas.width() < 10 or canvas.height() < 10:
            return
        fit_canvas_and_draw(canvas, rect=rect, pad=pad, allow_text_compaction=True)

    def _current_plot_canvas_layout(self):
        """Return the visible analysis canvas plus any reserved layout rect."""
        if not hasattr(self, 'tabs'):
            return None
        current_tab = self.tabs.currentWidget()
        if current_tab is getattr(self, 'sq_nhood_tab', None) and self.sq_nhood_canvas.figure.axes:
            return self.sq_nhood_canvas, None, 0.95
        if current_tab is getattr(self, 'sq_cooccur_tab', None) and self.sq_cooccur_canvas.figure.axes:
            rect = None
            if hasattr(self, 'sq_cooccur_plot_type_combo') and self.sq_cooccur_plot_type_combo.currentText() != "Heatmap":
                rect = [0.0, 0.0, 0.84, 1.0]
            return self.sq_cooccur_canvas, rect, 0.95
        if current_tab is getattr(self, 'sq_autocorr_tab', None) and self.sq_autocorr_canvas.figure.axes:
            return self.sq_autocorr_canvas, None, 0.95
        if current_tab is getattr(self, 'sq_ripley_tab', None) and self.sq_ripley_canvas.figure.axes:
            return self.sq_ripley_canvas, [0.0, 0.0, 0.84, 1.0], 0.95
        return None

    def _queue_plot_resize_refresh(self, *_args):
        """Debounce layout reflow for the currently visible advanced plot."""
        if self._plot_resize_in_progress or not self.isVisible():
            return
        if self._current_plot_canvas_layout() is None:
            return
        self._plot_resize_timer.start(140)

    def _refresh_current_plot_after_resize(self):
        """Refit the current advanced spatial plot to the resized canvas."""
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
        """Keep the active advanced spatial plot fitted to the dialog."""
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
    
    def _get_pixel_size_um(self, roi_id: str) -> float:
        """Get pixel size in micrometers for a specific ROI."""
        pixel_size_um = 1.0  # Default
        parent = self.parent()
        if parent is not None:
            try:
                if hasattr(parent, '_get_pixel_size_um'):
                    pixel_size_um = float(parent._get_pixel_size_um(roi_id))
            except Exception:
                pass
        return pixel_size_um
    
    def _get_or_create_anndata(self, roi_id: str) -> Optional['ad.AnnData']:
        """Get or create AnnData object for a specific ROI."""
        if roi_id in self.anndata_cache:
            return self.anndata_cache[roi_id]
        
        roi_col = self._get_roi_column()
        filtered_df = self._get_filtered_dataframe()
        
        # Get pixel size
        pixel_size_um = self._get_pixel_size_um(roi_id)
        
        # Use core function
        adata = dataframe_to_anndata(
            filtered_df,
            roi_id=roi_id,
            roi_column=roi_col,
            pixel_size_um=pixel_size_um
        )
        
        if adata is not None:
            # Ensure cluster columns are categorical (required by squidpy)
            for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
                if col in adata.obs.columns:
                    adata.obs[col] = adata.obs[col].astype('category')
            self.anndata_cache[roi_id] = adata
        
        return adata
    
    def _create_ui(self):
        """Create the UI with squidpy-specific tabs."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Feature set selector
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
        
        # Source file filter
        if 'source_file' in self.feature_dataframe.columns:
            source_files = sorted(self.feature_dataframe['source_file'].dropna().unique())
            self.available_source_files = set(source_files)
            if len(source_files) > 1:
                source_file_layout = QtWidgets.QHBoxLayout()
                source_file_layout.addWidget(QtWidgets.QLabel("Source Files:"))
                self.source_file_status_label = QtWidgets.QLabel("All files")
                source_file_layout.addWidget(self.source_file_status_label)
                self.source_file_config_btn = QtWidgets.QPushButton("Configure...")
                self.source_file_config_btn.clicked.connect(self._open_source_file_filter_dialog)
                source_file_layout.addWidget(self.source_file_config_btn)
                source_file_layout.addStretch()
                layout.addLayout(source_file_layout)
        
        # Graph creation section - Spatial Graph Construction
        params_group = QtWidgets.QGroupBox("Spatial Graph Construction")
        params_group_layout = QtWidgets.QVBoxLayout(params_group)
        params_layout = QtWidgets.QGridLayout()

        self.graph_method_combo = QtWidgets.QComboBox()
        self.graph_method_combo.addItems(["kNN", "Radius", "Delaunay"])
        self.graph_method_combo.currentTextChanged.connect(self._on_graph_method_changed)
        
        self.graph_k_spin = QtWidgets.QSpinBox()
        self.graph_k_spin.setRange(1, 64)
        self.graph_k_spin.setValue(20)
        
        self.graph_radius_spin = QtWidgets.QDoubleSpinBox()
        self.graph_radius_spin.setRange(0.1, 500.0)
        self.graph_radius_spin.setDecimals(1)
        self.graph_radius_spin.setValue(20.0)
        
        params_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        params_layout.addWidget(self.graph_method_combo, 0, 1)
        
        self.k_label = QtWidgets.QLabel("k:")
        params_layout.addWidget(self.k_label, 0, 2)
        params_layout.addWidget(self.graph_k_spin, 0, 3)
        
        self.radius_label = QtWidgets.QLabel("Radius (µm):")
        params_layout.addWidget(self.radius_label, 0, 4)
        params_layout.addWidget(self.graph_radius_spin, 0, 5)
        
        params_layout.addWidget(QtWidgets.QLabel("Random Seed:"), 0, 6)
        self.seed_spinbox = QtWidgets.QSpinBox()
        self.seed_spinbox.setRange(0, 2**31 - 1)
        self.seed_spinbox.setValue(42)
        self.seed_spinbox.setToolTip("Random seed for reproducibility (default: 42)")
        params_layout.addWidget(self.seed_spinbox, 0, 7)
        
        self.create_graph_btn = QtWidgets.QPushButton("Build Graph")
        self.create_graph_btn.setToolTip("Build the spatial graph using the selected mode and parameters")
        params_layout.addWidget(self.create_graph_btn, 0, 8)
        
        # Status label below the grid
        status_layout = QtWidgets.QHBoxLayout()
        self.graph_status_label = QtWidgets.QLabel("Graph not created")
        status_layout.addWidget(self.graph_status_label)
        status_layout.addStretch()
        
        params_group_layout.addLayout(params_layout)
        params_group_layout.addLayout(status_layout)
        
        self._on_graph_method_changed()
        layout.addWidget(params_group)
        
        # Actions
        action_row = QtWidgets.QHBoxLayout()
        self.export_anndata_btn = QtWidgets.QPushButton("Export to AnnData…")
        self.export_anndata_btn.setToolTip("Export data to AnnData format")
        action_row.addWidget(self.export_anndata_btn)
        self.cluster_labels_btn = QtWidgets.QPushButton("Customize Cluster Names…")
        self.cluster_labels_btn.setToolTip("Set custom display names for spatial-analysis cluster labels.")
        action_row.addWidget(self.cluster_labels_btn)
        self.feature_labels_btn = QtWidgets.QPushButton("Customize Feature Labels…")
        self.feature_labels_btn.setToolTip("Set custom display names for features used in spatial analysis.")
        action_row.addWidget(self.feature_labels_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        
        # Tabs for squidpy-specific analyses
        self.tabs = QtWidgets.QTabWidget()
        
        # Create squidpy-specific tabs
        self._create_squidpy_tabs()
        
        layout.addWidget(self.tabs, 1)
        
        # Wire signals
        self.export_anndata_btn.clicked.connect(self._export_to_anndata)
        self.cluster_labels_btn.clicked.connect(self._open_cluster_labels_dialog)
        self.feature_labels_btn.clicked.connect(self._open_feature_labels_dialog)
        self.create_graph_btn.clicked.connect(self._create_spatial_graph)
        
        # Initialize
        self._load_cluster_annotations()
        self._populate_roi_combo()
        self._update_button_states()
    
    def _create_squidpy_tabs(self):
        """Create squidpy-specific analysis tabs."""
        # Note: Spatial Graph tab removed - merged into Spatial Visualization tab
        
        # Add other squidpy tabs
        self._create_sq_nhood_tab()
        self._create_sq_cooccur_tab()
        self._create_sq_autocorr_tab()
        self._create_sq_ripley_tab()
    
    def _create_sq_nhood_tab(self):
        """Create neighborhood enrichment tab."""
        self.sq_nhood_tab = QtWidgets.QWidget()
        sq_nhood_layout = QtWidgets.QVBoxLayout(self.sq_nhood_tab)
        
        sq_nhood_info = QtWidgets.QLabel(
            "Test whether certain cell types are observed as neighbors more or less than expected by chance.\n"
            "Produces a cluster-cluster enrichment matrix."
        )
        sq_nhood_info.setWordWrap(True)
        sq_nhood_layout.addWidget(sq_nhood_info)
        
        sq_nhood_params = QtWidgets.QHBoxLayout()
        sq_nhood_params.addWidget(QtWidgets.QLabel("ROI (optional, 'All' for aggregation):"))
        self.sq_nhood_roi_combo = QtWidgets.QComboBox()
        self.sq_nhood_roi_combo.addItem("All ROIs", None)  # None means all ROIs
        sq_nhood_params.addWidget(self.sq_nhood_roi_combo)
        
        # Aggregation label and combo - will be shown/hidden based on ROI selection
        self.sq_nhood_agg_label = QtWidgets.QLabel("Aggregation:")
        self.sq_nhood_agg_combo = QtWidgets.QComboBox()
        self.sq_nhood_agg_combo.addItems(["Mean", "Sum"])
        sq_nhood_params.addWidget(self.sq_nhood_agg_label)
        sq_nhood_params.addWidget(self.sq_nhood_agg_combo)
        
        sq_nhood_params.addWidget(QtWidgets.QLabel("Cluster column:"))
        self.sq_nhood_cluster_combo = QtWidgets.QComboBox()
        self.sq_nhood_cluster_combo.addItems(["cluster", "cluster_phenotype", "cluster_id"])
        sq_nhood_params.addWidget(self.sq_nhood_cluster_combo)
        sq_nhood_params.addStretch()
        
        sq_nhood_btn_layout = QtWidgets.QHBoxLayout()
        self.sq_nhood_run_btn = QtWidgets.QPushButton("Run Neighborhood Enrichment")
        self.sq_nhood_save_btn = QtWidgets.QPushButton("Save Plot")
        self.sq_nhood_save_btn.setEnabled(False)
        self.sq_nhood_export_btn = QtWidgets.QPushButton("Export Results…")
        self.sq_nhood_export_btn.setEnabled(False)
        sq_nhood_btn_layout.addWidget(self.sq_nhood_run_btn)
        sq_nhood_btn_layout.addWidget(self.sq_nhood_save_btn)
        sq_nhood_btn_layout.addWidget(self.sq_nhood_export_btn)
        sq_nhood_btn_layout.addStretch()
        
        sq_nhood_layout.addLayout(sq_nhood_params)
        sq_nhood_layout.addLayout(sq_nhood_btn_layout)
        
        # Add navigation toolbar
        self.sq_nhood_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_nhood_toolbar = NavigationToolbar(self.sq_nhood_canvas, self)
        sq_nhood_layout.addWidget(self.sq_nhood_toolbar)
        sq_nhood_layout.addWidget(self.sq_nhood_canvas)
        self.tabs.addTab(self.sq_nhood_tab, "Neighborhood Enrichment")
        
        self.sq_nhood_run_btn.clicked.connect(self._run_sq_nhood_enrichment)
        self.sq_nhood_save_btn.clicked.connect(self._save_sq_nhood_plot)
        self.sq_nhood_export_btn.clicked.connect(self._export_sq_nhood_results)
        self.sq_nhood_roi_combo.currentIndexChanged.connect(self._on_sq_nhood_roi_changed)
        self.sq_nhood_agg_combo.currentIndexChanged.connect(self._on_sq_nhood_agg_changed)
        self.sq_nhood_cluster_combo.currentIndexChanged.connect(self._on_sq_nhood_cluster_changed)
    
    def _create_sq_cooccur_tab(self):
        """Create co-occurrence tab."""
        self.sq_cooccur_tab = QtWidgets.QWidget()
        sq_cooccur_layout = QtWidgets.QVBoxLayout(self.sq_cooccur_tab)
        
        sq_cooccur_info = QtWidgets.QLabel(
            "Measure how likely two cell types are to co-occur within neighborhoods of increasing radii.\n"
            "Gives co-occurrence curves across distance scales."
        )
        sq_cooccur_info.setWordWrap(True)
        sq_cooccur_layout.addWidget(sq_cooccur_info)
        
        sq_cooccur_params = QtWidgets.QHBoxLayout()
        sq_cooccur_params.addWidget(QtWidgets.QLabel("ROI (optional, 'All' for aggregation):"))
        self.sq_cooccur_roi_combo = QtWidgets.QComboBox()
        self.sq_cooccur_roi_combo.addItem("All ROIs", None)
        sq_cooccur_params.addWidget(self.sq_cooccur_roi_combo)
        
        sq_cooccur_params.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.sq_cooccur_agg_combo = QtWidgets.QComboBox()
        self.sq_cooccur_agg_combo.addItems(["Mean", "Sum"])
        sq_cooccur_params.addWidget(self.sq_cooccur_agg_combo)
        
        sq_cooccur_params.addWidget(QtWidgets.QLabel("Cluster column:"))
        self.sq_cooccur_cluster_combo = QtWidgets.QComboBox()
        self.sq_cooccur_cluster_combo.addItems(["cluster", "cluster_phenotype", "cluster_id"])
        sq_cooccur_params.addWidget(self.sq_cooccur_cluster_combo)
        
        sq_cooccur_params.addWidget(QtWidgets.QLabel("Reference cluster:"))
        self.sq_cooccur_ref_cluster_combo = QtWidgets.QComboBox()
        self.sq_cooccur_ref_cluster_combo.addItem("All clusters", None)
        sq_cooccur_params.addWidget(self.sq_cooccur_ref_cluster_combo)
        
        interval_label = QtWidgets.QLabel("Interval (comma-separated distances, e.g., '10,20,30,50'):")
        interval_label.setToolTip("Co-occurrence requires multiple distances to create curves. Provide at least 2 distances separated by commas (e.g., '10,20,30,50').")
        sq_cooccur_params.addWidget(interval_label)
        self.sq_cooccur_sizes_edit = QtWidgets.QLineEdit("10,20,30,50,100")
        self.sq_cooccur_sizes_edit.setToolTip("Enter multiple distances separated by commas. At least 2 distances are required (e.g., '10,20,30,50').")
        sq_cooccur_params.addWidget(self.sq_cooccur_sizes_edit)
        sq_cooccur_params.addStretch()
        
        sq_cooccur_btn_layout = QtWidgets.QHBoxLayout()
        self.sq_cooccur_run_btn = QtWidgets.QPushButton("Run Co-occurrence Analysis")
        self.sq_cooccur_save_btn = QtWidgets.QPushButton("Save Plot")
        self.sq_cooccur_save_btn.setEnabled(False)
        self.sq_cooccur_export_btn = QtWidgets.QPushButton("Export Results…")
        self.sq_cooccur_export_btn.setEnabled(False)
        sq_cooccur_btn_layout.addWidget(self.sq_cooccur_run_btn)
        sq_cooccur_btn_layout.addWidget(self.sq_cooccur_save_btn)
        sq_cooccur_btn_layout.addWidget(self.sq_cooccur_export_btn)
        sq_cooccur_btn_layout.addStretch()
        
        # Plot type selection
        sq_cooccur_plot_layout = QtWidgets.QHBoxLayout()
        sq_cooccur_plot_layout.addWidget(QtWidgets.QLabel("Plot type:"))
        self.sq_cooccur_plot_type_combo = QtWidgets.QComboBox()
        self.sq_cooccur_plot_type_combo.addItems(["Curves", "Heatmap"])
        self.sq_cooccur_plot_type_combo.setToolTip("Curves: Show co-occurrence across all distances. Heatmap: Show co-occurrence matrix at a selected distance.")
        sq_cooccur_plot_layout.addWidget(self.sq_cooccur_plot_type_combo)
        
        sq_cooccur_plot_layout.addWidget(QtWidgets.QLabel("Distance for heatmap (µm):"))
        self.sq_cooccur_distance_spin = QtWidgets.QDoubleSpinBox()
        self.sq_cooccur_distance_spin.setRange(0.0, 100000.0)
        self.sq_cooccur_distance_spin.setDecimals(3)
        self.sq_cooccur_distance_spin.setSingleStep(5.0)
        self.sq_cooccur_distance_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.sq_cooccur_distance_spin.setKeyboardTracking(False)
        self.sq_cooccur_distance_spin.setToolTip(
            "Type one of the computed interval distances to display in the heatmap."
        )
        self.sq_cooccur_distance_spin.setEnabled(False)  # Enabled only when heatmap is selected
        sq_cooccur_plot_layout.addWidget(self.sq_cooccur_distance_spin)
        sq_cooccur_plot_layout.addStretch()
        
        sq_cooccur_layout.addLayout(sq_cooccur_params)
        sq_cooccur_layout.addLayout(sq_cooccur_btn_layout)
        sq_cooccur_layout.addLayout(sq_cooccur_plot_layout)
        
        # Add navigation toolbar
        self.sq_cooccur_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_cooccur_toolbar = NavigationToolbar(self.sq_cooccur_canvas, self)
        sq_cooccur_layout.addWidget(self.sq_cooccur_toolbar)
        sq_cooccur_layout.addWidget(self.sq_cooccur_canvas)
        self.tabs.addTab(self.sq_cooccur_tab, "Co-occurrence")
        
        self.sq_cooccur_run_btn.clicked.connect(self._run_sq_cooccurrence)
        self.sq_cooccur_save_btn.clicked.connect(self._save_sq_cooccur_plot)
        self.sq_cooccur_export_btn.clicked.connect(self._export_sq_cooccur_results)
        self.sq_cooccur_roi_combo.currentIndexChanged.connect(self._on_sq_cooccur_roi_changed)
        self.sq_cooccur_ref_cluster_combo.currentIndexChanged.connect(self._on_sq_cooccur_ref_cluster_changed)
        self.sq_cooccur_cluster_combo.currentIndexChanged.connect(self._on_sq_cooccur_cluster_column_changed)
        self.sq_cooccur_plot_type_combo.currentTextChanged.connect(self._on_sq_cooccur_plot_type_changed)
        self.sq_cooccur_distance_spin.valueChanged.connect(self._on_sq_cooccur_distance_changed)
        
        # Store interval for distance selection
        self.sq_cooccur_interval = None
        self.sq_cooccur_heatmap_distance = None
    
    def _create_sq_autocorr_tab(self):
        """Create spatial autocorrelation tab."""
        self.sq_autocorr_tab = QtWidgets.QWidget()
        sq_autocorr_layout = QtWidgets.QVBoxLayout(self.sq_autocorr_tab)
        
        sq_autocorr_info = QtWidgets.QLabel(
            "Compute Moran's I spatial autocorrelation for marker expression. "
            "Tests whether expression of markers is spatially clustered. "
            "Positive values indicate spatial clustering, negative values indicate spatial dispersion. "
            "Red bars indicate statistically significant results (p < 0.05)."
        )
        sq_autocorr_info.setWordWrap(True)
        sq_autocorr_layout.addWidget(sq_autocorr_info)
        
        sq_autocorr_params = QtWidgets.QHBoxLayout()
        sq_autocorr_params.addWidget(QtWidgets.QLabel("ROI (optional, 'All' for aggregation):"))
        self.sq_autocorr_roi_combo = QtWidgets.QComboBox()
        self.sq_autocorr_roi_combo.addItem("All ROIs", None)
        sq_autocorr_params.addWidget(self.sq_autocorr_roi_combo)
        
        sq_autocorr_params.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.sq_autocorr_agg_combo = QtWidgets.QComboBox()
        self.sq_autocorr_agg_combo.addItems(["Mean", "Sum"])
        sq_autocorr_params.addWidget(self.sq_autocorr_agg_combo)
        
        sq_autocorr_params.addWidget(QtWidgets.QLabel("Markers (comma-separated, or 'all'):"))
        self.sq_autocorr_markers_edit = QtWidgets.QLineEdit("all")
        sq_autocorr_params.addWidget(self.sq_autocorr_markers_edit)
        
        sq_autocorr_params.addWidget(QtWidgets.QLabel("Top K:"))
        self.sq_autocorr_topk_spin = QtWidgets.QSpinBox()
        self.sq_autocorr_topk_spin.setRange(1, 100)
        self.sq_autocorr_topk_spin.setValue(20)
        sq_autocorr_params.addWidget(self.sq_autocorr_topk_spin)
        sq_autocorr_params.addStretch()
        
        sq_autocorr_btn_layout = QtWidgets.QHBoxLayout()
        self.sq_autocorr_run_btn = QtWidgets.QPushButton("Run Spatial Autocorrelation")
        self.sq_autocorr_save_btn = QtWidgets.QPushButton("Save Plot")
        self.sq_autocorr_save_btn.setEnabled(False)
        self.sq_autocorr_export_btn = QtWidgets.QPushButton("Export Results…")
        self.sq_autocorr_export_btn.setEnabled(False)
        sq_autocorr_btn_layout.addWidget(self.sq_autocorr_run_btn)
        sq_autocorr_btn_layout.addWidget(self.sq_autocorr_save_btn)
        sq_autocorr_btn_layout.addWidget(self.sq_autocorr_export_btn)
        sq_autocorr_btn_layout.addStretch()
        
        # Visualization controls
        sq_autocorr_viz_layout = QtWidgets.QHBoxLayout()
        sq_autocorr_viz_layout.addWidget(QtWidgets.QLabel("Visualization Type:"))
        self.sq_autocorr_viz_type_combo = QtWidgets.QComboBox()
        self.sq_autocorr_viz_type_combo.addItems(["Bar Plot (Top K)", "Moran Scatter Plot", "Spatial Map"])
        self.sq_autocorr_viz_type_combo.setCurrentText("Bar Plot (Top K)")
        self.sq_autocorr_viz_type_combo.setToolTip(
            "Bar Plot: Top K features by Moran's I\n"
            "Moran Scatter Plot: Canonical visualization showing variable vs spatial lag\n"
            "Spatial Map: Spatial coordinates colored by variable value"
        )
        sq_autocorr_viz_layout.addWidget(self.sq_autocorr_viz_type_combo)
        
        sq_autocorr_viz_layout.addWidget(QtWidgets.QLabel("Variable:"))
        self.sq_autocorr_var_combo = QtWidgets.QComboBox()
        self.sq_autocorr_var_combo.setToolTip("Select variable to visualize (for Moran scatter plot or spatial map)")
        self.sq_autocorr_var_combo.setEnabled(False)  # Enabled when visualization type requires it
        sq_autocorr_viz_layout.addWidget(self.sq_autocorr_var_combo)
        
        self.sq_autocorr_plot_viz_btn = QtWidgets.QPushButton("Plot Visualization")
        self.sq_autocorr_plot_viz_btn.setEnabled(False)
        self.sq_autocorr_plot_viz_btn.setToolTip("Generate selected visualization for the chosen variable")
        sq_autocorr_viz_layout.addWidget(self.sq_autocorr_plot_viz_btn)
        sq_autocorr_viz_layout.addStretch()
        
        sq_autocorr_layout.addLayout(sq_autocorr_params)
        sq_autocorr_layout.addLayout(sq_autocorr_btn_layout)
        sq_autocorr_layout.addLayout(sq_autocorr_viz_layout)
        
        # Add navigation toolbar
        self.sq_autocorr_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_autocorr_toolbar = NavigationToolbar(self.sq_autocorr_canvas, self)
        sq_autocorr_layout.addWidget(self.sq_autocorr_toolbar)
        sq_autocorr_layout.addWidget(self.sq_autocorr_canvas)
        self.tabs.addTab(self.sq_autocorr_tab, "Spatial Autocorrelation")
        
        self.sq_autocorr_run_btn.clicked.connect(self._run_sq_autocorrelation)
        self.sq_autocorr_save_btn.clicked.connect(self._save_sq_autocorr_plot)
        self.sq_autocorr_export_btn.clicked.connect(self._export_sq_autocorr_results)
        self.sq_autocorr_roi_combo.currentIndexChanged.connect(self._on_sq_autocorr_roi_changed)
        self.sq_autocorr_topk_spin.valueChanged.connect(self._on_sq_autocorr_topk_changed)
        self.sq_autocorr_viz_type_combo.currentTextChanged.connect(self._on_sq_autocorr_viz_type_changed)
        self.sq_autocorr_plot_viz_btn.clicked.connect(self._plot_sq_autocorr_visualization)
    
    def _create_sq_ripley_tab(self):
        """Create Ripley tab."""
        self.sq_ripley_tab = QtWidgets.QWidget()
        sq_ripley_layout = QtWidgets.QVBoxLayout(self.sq_ripley_tab)
        
        sq_ripley_info = QtWidgets.QLabel(
            "Compute Ripley's F, G, and L functions using squidpy for spatial clustering analysis."
        )
        sq_ripley_info.setWordWrap(True)
        sq_ripley_layout.addWidget(sq_ripley_info)
        
        sq_ripley_params = QtWidgets.QHBoxLayout()
        sq_ripley_params.addWidget(QtWidgets.QLabel("ROI (optional, 'All' for aggregation):"))
        self.sq_ripley_roi_combo = QtWidgets.QComboBox()
        self.sq_ripley_roi_combo.addItem("All ROIs", None)
        sq_ripley_params.addWidget(self.sq_ripley_roi_combo)
        
        sq_ripley_params.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.sq_ripley_agg_combo = QtWidgets.QComboBox()
        self.sq_ripley_agg_combo.addItems(["Mean", "Sum"])
        sq_ripley_params.addWidget(self.sq_ripley_agg_combo)
        
        sq_ripley_params.addWidget(QtWidgets.QLabel("Mode:"))
        self.sq_ripley_mode_combo = QtWidgets.QComboBox()
        self.sq_ripley_mode_combo.addItems(["F", "G", "L"])
        self.sq_ripley_mode_combo.setCurrentText("L")  # Set L as default
        self.sq_ripley_mode_combo.currentIndexChanged.connect(self._on_sq_ripley_mode_changed)
        sq_ripley_params.addWidget(self.sq_ripley_mode_combo)
        
        sq_ripley_params.addWidget(QtWidgets.QLabel("Max distance (µm):"))
        self.sq_ripley_r_max_spin = QtWidgets.QDoubleSpinBox()
        self.sq_ripley_r_max_spin.setRange(1.0, 200.0)
        self.sq_ripley_r_max_spin.setDecimals(1)
        self.sq_ripley_r_max_spin.setValue(50.0)
        sq_ripley_params.addWidget(self.sq_ripley_r_max_spin)
        
        sq_ripley_params.addWidget(QtWidgets.QLabel("Cluster column:"))
        self.sq_ripley_cluster_combo = QtWidgets.QComboBox()
        self.sq_ripley_cluster_combo.addItems(["cluster", "cluster_phenotype", "cluster_id"])
        sq_ripley_params.addWidget(self.sq_ripley_cluster_combo)
        sq_ripley_params.addStretch()
        
        sq_ripley_btn_layout = QtWidgets.QHBoxLayout()
        self.sq_ripley_run_btn = QtWidgets.QPushButton("Run Ripley Analysis")
        self.sq_ripley_save_btn = QtWidgets.QPushButton("Save Plot")
        self.sq_ripley_save_btn.setEnabled(False)
        self.sq_ripley_export_btn = QtWidgets.QPushButton("Export Results…")
        self.sq_ripley_export_btn.setEnabled(False)
        sq_ripley_btn_layout.addWidget(self.sq_ripley_run_btn)
        sq_ripley_btn_layout.addWidget(self.sq_ripley_save_btn)
        sq_ripley_btn_layout.addWidget(self.sq_ripley_export_btn)
        sq_ripley_btn_layout.addStretch()
        
        sq_ripley_layout.addLayout(sq_ripley_params)
        sq_ripley_layout.addLayout(sq_ripley_btn_layout)
        
        # Add navigation toolbar
        self.sq_ripley_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_ripley_toolbar = NavigationToolbar(self.sq_ripley_canvas, self)
        sq_ripley_layout.addWidget(self.sq_ripley_toolbar)
        sq_ripley_layout.addWidget(self.sq_ripley_canvas)
        self.tabs.addTab(self.sq_ripley_tab, "Ripley Functions")
        
        self.sq_ripley_run_btn.clicked.connect(self._run_sq_ripley)
        self.sq_ripley_save_btn.clicked.connect(self._save_sq_ripley_plot)
        self.sq_ripley_export_btn.clicked.connect(self._export_sq_ripley_results)
        self.sq_ripley_roi_combo.currentIndexChanged.connect(self._on_sq_ripley_roi_changed)
        self.sq_ripley_cluster_combo.currentIndexChanged.connect(self._on_sq_ripley_cluster_changed)
    
    def _on_graph_method_changed(self):
        """Handle method change for graph creation."""
        method = self.graph_method_combo.currentText()
        is_knn = method == "kNN"
        is_delaunay = method == "Delaunay"

        self.k_label.setVisible(is_knn)
        self.graph_k_spin.setVisible(is_knn)
        
        self.radius_label.setVisible(not is_knn and not is_delaunay)
        self.graph_radius_spin.setVisible(not is_knn and not is_delaunay)
    
    def _open_source_file_filter_dialog(self):
        """Open source file filter dialog."""
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
    
    def _update_source_file_status_label(self):
        """Update source file status label."""
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
        """Mirror cluster display names into dataframe and cached AnnData phenotype columns."""
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

        for adata in self.anndata_cache.values():
            if 'cluster' not in adata.obs.columns:
                continue
            phenotype_series = map_cluster_series_to_display_names(
                adata.obs['cluster'],
                annotation_map=self.cluster_annotation_map,
            )
            adata.obs['cluster_phenotype'] = phenotype_series
            adata.obs['cluster_phenotype'] = adata.obs['cluster_phenotype'].astype('category')

        for result in self.aggregated_results.values():
            if not hasattr(result, 'obs') or 'cluster' not in result.obs.columns:
                continue
            phenotype_series = map_cluster_series_to_display_names(
                result.obs['cluster'],
                annotation_map=self.cluster_annotation_map,
            )
            result.obs['cluster_phenotype'] = phenotype_series
            try:
                result.obs['cluster_phenotype'] = result.obs['cluster_phenotype'].astype('category')
            except Exception:
                pass

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
        """Return numeric feature columns that users may want to relabel for spatial plots."""
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
        return [column for column in numeric_cols if column not in exclude]

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
        self._update_autocorr_var_combo()
        self._on_sq_nhood_roi_changed()
        self._on_sq_cooccur_roi_changed()
        self._on_sq_autocorr_roi_changed()
        self._on_sq_ripley_roi_changed()

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
    
    def _populate_roi_combo(self):
        """Populate all ROI combo boxes."""
        filtered_df = self._get_filtered_dataframe()
        roi_col = self._get_roi_column()
        unique_rois = sorted(filtered_df[roi_col].unique())
        
        # List of ROI combo boxes that need "All ROIs" option (analyses)
        analysis_roi_combos = [
            'sq_nhood_roi_combo',
            'sq_cooccur_roi_combo',
            'sq_autocorr_roi_combo',
            'sq_ripley_roi_combo'
        ]
        
        # Populate analysis ROI combos with "All ROIs" option
        for combo_name in analysis_roi_combos:
            if hasattr(self, combo_name):
                combo = getattr(self, combo_name)
                # Check if "All ROIs" already exists
                has_all = False
                for i in range(combo.count()):
                    if combo.itemData(i) is None:
                        has_all = True
                        break
                if not has_all:
                    combo.insertItem(0, "All ROIs", None)
                # Update ROI list
                for i in range(combo.count() - 1, -1, -1):  # Iterate backwards to avoid index issues
                    if combo.itemData(i) is not None:
                        combo.removeItem(i)
                for roi_id in unique_rois:
                    combo.addItem(str(roi_id), roi_id)
    
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
    
    def _on_feature_set_changed(self):
        """Handle feature set change - invalidate cache and refresh."""
        self._update_active_feature_dataframe()
        
        # Reload cluster annotations since dataframe may have changed
        self._load_cluster_annotations()
        
        # Clear cache since data changed
        self.anndata_cache = {}
        self.processed_rois = {}
        self.analysis_status = {}
        self.aggregated_results = {}
        self.spatial_graph_built = False

        parent = self.parent()
        if parent is not None and hasattr(parent, '_set_analysis_feature_set_preference'):
            parent._set_analysis_feature_set_preference(self.get_active_feature_set_key(), source_dialog=self)
        
        # Refresh UI
        self._populate_roi_combo()
        self._update_button_states()
        
        # Clear all plots
        for canvas_name in ['sq_nhood_canvas', 'sq_cooccur_canvas', 'sq_autocorr_canvas', 'sq_ripley_canvas']:
            if hasattr(self, canvas_name):
                canvas = getattr(self, canvas_name)
                canvas.figure.clear()
                canvas.draw()

        for btn_name in [
            'sq_nhood_save_btn',
            'sq_nhood_export_btn',
            'sq_cooccur_save_btn',
            'sq_cooccur_export_btn',
            'sq_autocorr_save_btn',
            'sq_autocorr_export_btn',
            'sq_ripley_save_btn',
            'sq_ripley_export_btn',
        ]:
            if hasattr(self, btn_name):
                getattr(self, btn_name).setEnabled(False)
        
        # Update status
        if hasattr(self, 'graph_status_label'):
            self.graph_status_label.setText("Graph not created (feature set changed)")
            self.graph_status_label.setStyleSheet("")
    
    def _update_button_states(self):
        """Update button enabled/disabled states based on prerequisites."""
        graph_built = self.spatial_graph_built
        
        # Buttons that require graph
        graph_dependent_buttons = [
            'sq_nhood_run_btn', 'sq_cooccur_run_btn',
            'sq_autocorr_run_btn', 'sq_ripley_run_btn'
        ]
        
        for btn_name in graph_dependent_buttons:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                btn.setEnabled(graph_built)
    
    def _get_selected_roi(self, combo):
        """Get selected ROI from combo box."""
        if combo is None:
            return None
        return combo.currentData()

    def _normalize_cooccur_distances(
        self,
        raw_interval: Any,
        *,
        expected_size: Optional[int] = None,
        fallback: Any = None,
    ) -> List[float]:
        """Return co-occurrence distances aligned to the occurrence array depth."""

        def _to_float_list(values: Any) -> List[float]:
            if values is None:
                return []
            try:
                return [float(value) for value in np.asarray(values, dtype=float).reshape(-1).tolist()]
            except Exception:
                return []

        interval_values = _to_float_list(raw_interval)
        fallback_values = _to_float_list(fallback)

        if not interval_values:
            interval_values = list(fallback_values)

        if expected_size is None:
            return interval_values

        if len(interval_values) == expected_size + 1:
            distances = interval_values[1:]
        else:
            distances = interval_values[:expected_size]

        if len(distances) < expected_size and fallback_values:
            if len(fallback_values) == expected_size + 1:
                fallback_slice = fallback_values[1:]
            else:
                fallback_slice = fallback_values[:expected_size]
            for value in fallback_slice:
                if len(distances) >= expected_size:
                    break
                if not any(np.isclose(value, existing, rtol=1e-6, atol=1e-6) for existing in distances):
                    distances.append(value)

        if len(distances) < expected_size:
            distances.extend(float(idx) for idx in range(len(distances), expected_size))

        return [float(distance) for distance in distances[:expected_size]]

    def _match_cooccur_distance(self, requested_distance: Any, distances: List[float]) -> Optional[float]:
        """Return the matching available distance, accounting for float roundoff."""
        if requested_distance is None:
            return None

        try:
            requested_value = float(requested_distance)
        except (TypeError, ValueError):
            return None

        for distance in distances:
            if np.isclose(float(distance), requested_value, rtol=1e-6, atol=1e-6):
                return float(distance)
        return None

    def _set_sq_cooccur_distance_metadata(self, distances: List[float]) -> None:
        """Update the heatmap distance input metadata without changing the current value."""
        cleaned_distances = [float(distance) for distance in distances]
        self.sq_cooccur_interval = cleaned_distances or None

        if not hasattr(self, 'sq_cooccur_distance_spin'):
            return

        if len(cleaned_distances) > 1:
            diffs = sorted(
                abs(cleaned_distances[idx] - cleaned_distances[idx - 1])
                for idx in range(1, len(cleaned_distances))
                if not np.isclose(cleaned_distances[idx], cleaned_distances[idx - 1], rtol=1e-6, atol=1e-6)
            )
            if diffs:
                self.sq_cooccur_distance_spin.setSingleStep(float(diffs[0]))

        if cleaned_distances:
            available_text = ", ".join(f"{distance:g}" for distance in cleaned_distances)
            self.sq_cooccur_distance_spin.setToolTip(
                "Type one of the computed interval distances to display in the heatmap. "
                f"Available: {available_text} µm."
            )
        else:
            self.sq_cooccur_distance_spin.setToolTip(
                "Run co-occurrence analysis first to populate valid heatmap distances."
            )

    def _sync_sq_cooccur_distance_input(
        self,
        distances: List[float],
        *,
        preferred_distance: Optional[float] = None,
    ) -> None:
        """Sync the heatmap distance input to the available co-occurrence distances."""
        cleaned_distances = [float(distance) for distance in distances]
        self._set_sq_cooccur_distance_metadata(cleaned_distances)

        if not cleaned_distances or not hasattr(self, 'sq_cooccur_distance_spin'):
            return

        selected_distance = self._match_cooccur_distance(preferred_distance, cleaned_distances)
        if selected_distance is None:
            selected_distance = self._match_cooccur_distance(self.sq_cooccur_heatmap_distance, cleaned_distances)
        if selected_distance is None:
            selected_distance = self._match_cooccur_distance(self.sq_cooccur_distance_spin.value(), cleaned_distances)
        if selected_distance is None:
            selected_distance = cleaned_distances[0]

        blocker = QtCore.QSignalBlocker(self.sq_cooccur_distance_spin)
        self.sq_cooccur_distance_spin.setValue(selected_distance)
        del blocker
        self.sq_cooccur_heatmap_distance = selected_distance

    def _get_sq_cooccur_plot_adata(self) -> Optional[Any]:
        """Return the co-occurrence AnnData to plot for the current ROI selection."""
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id is not None:
            return self.anndata_cache.get(roi_id)

        for adata in self.anndata_cache.values():
            if self._find_uns_key(adata, ['co_occurrence'], contains=['co', 'occur']) is not None:
                return adata
        return None
    
    def _on_sq_nhood_roi_changed(self):
        """Handle ROI change in neighborhood enrichment tab."""
        roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
        
        # Show/hide aggregation controls based on ROI selection
        if roi_id is None:
            # "All ROIs" selected - show aggregation options
            self.sq_nhood_agg_label.setVisible(True)
            self.sq_nhood_agg_combo.setVisible(True)
            # Use aggregated result if available
            if 'nhood_enrichment' in self.aggregated_results:
                self._plot_sq_nhood_enrichment(self.aggregated_results['nhood_enrichment'])
                self.sq_nhood_save_btn.setEnabled(True)
        else:
            # Single ROI selected - hide aggregation options and re-run analysis
            self.sq_nhood_agg_label.setVisible(False)
            self.sq_nhood_agg_combo.setVisible(False)
            
            # Auto-refresh plot for this single ROI
            if roi_id in self.anndata_cache:
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' in adata.obsp:
                    # Re-run enrichment for this ROI
                    cluster_key = self.sq_nhood_cluster_combo.currentText()
                    if cluster_key in adata.obs.columns:
                        try:
                            if not hasattr(adata.obs[cluster_key], 'cat'):
                                adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
                            sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
                            if 'nhood_enrichment' in adata.uns:
                                self._plot_sq_nhood_enrichment(adata)
                                self.sq_nhood_save_btn.setEnabled(True)
                        except Exception as e:
                            pass
                elif 'nhood_enrichment' in adata.uns:
                    # Use existing results if available
                    self._plot_sq_nhood_enrichment(adata)
                    self.sq_nhood_save_btn.setEnabled(True)
    
    def _on_sq_nhood_agg_changed(self):
        """Handle aggregation change in neighborhood enrichment tab."""
        roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
        if roi_id is None and 'nhood_enrichment' in self.aggregated_results:
            # Re-aggregate and plot
            self._run_sq_nhood_enrichment()
    
    def _on_sq_cooccur_roi_changed(self):
        """Handle ROI change in co-occurrence tab."""
        adata = self._get_sq_cooccur_plot_adata()
        if adata is None:
            return

        # Update reference cluster combo (even if no co-occurrence data yet)
        self._update_cooccur_ref_cluster_combo(adata)

        has_cooccur = self._find_uns_key(adata, ['co_occurrence'], contains=['co', 'occur']) is not None
        if has_cooccur:
            self._plot_sq_cooccurrence(adata)
            self.sq_cooccur_save_btn.setEnabled(True)
    
    def _on_sq_cooccur_ref_cluster_changed(self):
        """Handle reference cluster change in co-occurrence tab."""
        if self.sq_cooccur_plot_type_combo.currentText() != "Curves":
            return

        adata = self._get_sq_cooccur_plot_adata()
        if adata is None:
            return

        has_cooccur = self._find_uns_key(adata, ['co_occurrence'], contains=['co', 'occur']) is not None
        if has_cooccur:
            self._plot_sq_cooccurrence(adata)
            self.sq_cooccur_save_btn.setEnabled(True)

    def _on_sq_cooccur_cluster_column_changed(self):
        """Handle cluster column change in co-occurrence tab."""
        adata = self._get_sq_cooccur_plot_adata()
        if adata is None:
            return

        self._update_cooccur_ref_cluster_combo(adata)
        has_cooccur = self._find_uns_key(adata, ['co_occurrence'], contains=['co', 'occur']) is not None
        if has_cooccur:
            self._plot_sq_cooccurrence(adata)
            self.sq_cooccur_save_btn.setEnabled(True)

    def _on_sq_cooccur_plot_type_changed(self):
        """Handle plot type change in co-occurrence tab."""
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        # Enable/disable distance input based on plot type
        self.sq_cooccur_distance_spin.setEnabled(plot_type == "Heatmap")
        self.sq_cooccur_ref_cluster_combo.setEnabled(plot_type == "Curves")
        
        adata = self._get_sq_cooccur_plot_adata()
        if adata is None:
            return

        has_cooccur = self._find_uns_key(adata, ['co_occurrence'], contains=['co', 'occur']) is not None
        if has_cooccur:
            self._plot_sq_cooccurrence(adata)

    def _on_sq_cooccur_distance_changed(self, *_args):
        """Handle distance selection change for heatmap in co-occurrence tab."""
        # Replot if heatmap is selected and data is available
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        if plot_type == "Heatmap":
            adata = self._get_sq_cooccur_plot_adata()
            if adata is None:
                return

            has_cooccur = self._find_uns_key(adata, ['co_occurrence'], contains=['co', 'occur']) is not None
            if has_cooccur:
                self._plot_sq_cooccurrence(adata)
    
    def _on_sq_nhood_cluster_changed(self):
        """Handle cluster column change in neighborhood enrichment tab."""
        roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
        if roi_id is None:
            # Re-run aggregation if "All ROIs" is selected
            if 'nhood_enrichment' in self.aggregated_results:
                self._run_sq_nhood_enrichment()
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            if 'spatial_connectivities' in adata.obsp:
                cluster_key = self.sq_nhood_cluster_combo.currentText()
                if cluster_key in adata.obs.columns:
                    try:
                        if not hasattr(adata.obs[cluster_key], 'cat'):
                            adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
                        sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
                        if 'nhood_enrichment' in adata.uns:
                            self._plot_sq_nhood_enrichment(adata)
                            self.sq_nhood_save_btn.setEnabled(True)
                    except Exception as e:
                        pass
    
    def _update_cooccur_ref_cluster_combo(self, adata: 'ad.AnnData', preserve_selection: bool = True):
        """Update the reference cluster combo box with available clusters."""
        cluster_key = self.sq_cooccur_cluster_combo.currentText()
        
        # Preserve current selection if requested
        current_selection = None
        if preserve_selection:
            current_selection = self.sq_cooccur_ref_cluster_combo.currentData()
        
        categories = []
        if cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories = list(adata.obs[cluster_key].cat.categories)
            else:
                categories = sort_cluster_values(
                    adata.obs[cluster_key].unique(),
                    annotation_map=self.cluster_annotation_map,
                    canonical=False,
                )

        target_selection = None
        if preserve_selection and current_selection is not None and current_selection in categories:
            target_selection = current_selection
        else:
            # Default to cluster "1" when available, otherwise the first cluster.
            for cat in categories:
                if str(cat) == "1" or cat == 1:
                    target_selection = cat
                    break
            if target_selection is None and categories:
                target_selection = categories[0]

        blocker = QtCore.QSignalBlocker(self.sq_cooccur_ref_cluster_combo)
        self.sq_cooccur_ref_cluster_combo.clear()
        self.sq_cooccur_ref_cluster_combo.addItem("All clusters", None)

        if categories:
            for cat in categories:
                self.sq_cooccur_ref_cluster_combo.addItem(
                    self._get_cluster_display_name(cat), cat
                )

        if target_selection is None:
            self.sq_cooccur_ref_cluster_combo.setCurrentIndex(0)
        else:
            for i in range(self.sq_cooccur_ref_cluster_combo.count()):
                if self.sq_cooccur_ref_cluster_combo.itemData(i) == target_selection:
                    self.sq_cooccur_ref_cluster_combo.setCurrentIndex(i)
                    break
        del blocker
    
    def _on_sq_autocorr_roi_changed(self):
        """Handle ROI change in autocorrelation tab."""
        roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
        if roi_id is None:
            # "All ROIs" selected - use aggregated result if available
            if 'autocorrelation' in self.aggregated_results:
                self._plot_sq_autocorrelation(self.aggregated_results['autocorrelation'])
                self.sq_autocorr_save_btn.setEnabled(True)
            self._update_autocorr_var_combo()
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            if 'moranI' in adata.uns:
                self._plot_sq_autocorrelation(adata)
                self.sq_autocorr_save_btn.setEnabled(True)
            self._update_autocorr_var_combo()
    
    def _on_sq_autocorr_viz_type_changed(self):
        """Handle visualization type change in autocorrelation tab."""
        viz_type = self.sq_autocorr_viz_type_combo.currentText()
        # Enable variable combo only for Moran scatter plot or spatial map
        needs_var = viz_type in ["Moran Scatter Plot", "Spatial Map"]
        self.sq_autocorr_var_combo.setEnabled(needs_var)
        self.sq_autocorr_plot_viz_btn.setEnabled(needs_var)
        
        # If switching to bar plot, show the default plot
        if viz_type == "Bar Plot (Top K)":
            self._on_sq_autocorr_roi_changed()
    
    def _update_autocorr_var_combo(self):
        """Update variable combo box with available variables, ordered by p-value."""
        self.sq_autocorr_var_combo.clear()
        
        roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
        adata = None
        
        if roi_id is None:
            # Use aggregated result if available
            if 'autocorrelation' in self.aggregated_results:
                adata = self.aggregated_results['autocorrelation']
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
        
        if adata is not None and hasattr(adata, 'uns'):
            # Check for moranI results
            moran_key = None
            if 'moranI' in adata.uns:
                moran_key = 'moranI'
            else:
                # Try alternative key names
                for key in adata.uns.keys():
                    if 'moran' in key.lower() or 'autocorr' in key.lower():
                        moran_key = key
                        break
            
            if moran_key is None:
                return
            
            moran_data = adata.uns[moran_key]
            
            # Extract variable names and p-values
            var_pval_list = []
            
            if isinstance(moran_data, pd.DataFrame):
                # DataFrame format
                pval_col, _ = self._get_preferred_moran_pvalue_columns(list(moran_data.columns))
                
                # Get variable names from index
                if 'var_names' in moran_data.columns:
                    var_names = moran_data['var_names'].values
                elif moran_data.index.name == 'var_names' or all(isinstance(x, str) for x in moran_data.index):
                    var_names = moran_data.index.values
                else:
                    var_names = [str(x) for x in moran_data.index]
                
                # Pair with p-values
                if pval_col:
                    for i, var in enumerate(var_names):
                        if var != 'touches_edge':
                            pval = moran_data.iloc[i][pval_col] if i < len(moran_data) else 1.0
                            var_pval_list.append((var, float(pval)))
                else:
                    # No p-values, just use variable names
                    var_pval_list = [(var, 1.0) for var in var_names if var != 'touches_edge']
            
            elif isinstance(moran_data, dict):
                # Dict format
                var_names = moran_data.get('var_names', [])
                p_values = None
                for key in ['pval_sim', 'pval_z_sim', 'pval_norm', 'pval']:
                    if key in moran_data:
                        p_values = moran_data.get(key)
                        break
                
                if var_names and p_values is not None:
                    if isinstance(p_values, (list, np.ndarray)):
                        for var, pval in zip(var_names, p_values):
                            if var != 'touches_edge':
                                var_pval_list.append((var, float(pval)))
                    else:
                        # Single p-value for all
                        var_pval_list = [(var, 1.0) for var in var_names if var != 'touches_edge']
                else:
                    # No p-values, just use variable names
                    var_pval_list = [(var, 1.0) for var in var_names if var != 'touches_edge']
            
            # Sort by p-value (ascending - most significant first)
            var_pval_list.sort(key=lambda x: x[1])
            
            # Add to combo with p-value annotation
            for var, pval in var_pval_list:
                display_var = self._get_feature_display_name(str(var))
                if pval < 0.001:
                    label = f"{display_var} (p < 0.001)"
                elif pval < 0.01:
                    label = f"{display_var} (p = {pval:.3f})"
                elif pval < 0.05:
                    label = f"{display_var} (p = {pval:.2f})"
                else:
                    label = f"{display_var} (p = {pval:.2f})"
                self.sq_autocorr_var_combo.addItem(label, var)  # Store actual var name as data
            
            # Enable if there are variables and visualization type requires it
            viz_type = self.sq_autocorr_viz_type_combo.currentText()
            if var_pval_list and viz_type in ["Moran Scatter Plot", "Spatial Map"]:
                self.sq_autocorr_var_combo.setEnabled(True)
                self.sq_autocorr_plot_viz_btn.setEnabled(True)
    
    def _on_sq_ripley_roi_changed(self):
        """Handle ROI change in Ripley tab."""
        roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
        if roi_id is None:
            if 'ripley' in self.aggregated_results:
                cluster_key = self.sq_ripley_cluster_combo.currentText()
                self._plot_sq_ripley(self.aggregated_results['ripley'], cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
                self.sq_ripley_export_btn.setEnabled(True)
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            # Check for any ripley key
            has_ripley = False
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    has_ripley = True
                    break
            if has_ripley:
                self._plot_sq_ripley(adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
                self.sq_ripley_export_btn.setEnabled(True)
    
    def _on_sq_ripley_mode_changed(self):
        """Handle mode change (F/G/L) in Ripley tab - auto-refresh if data exists."""
        roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            
            # Check if any ripley data exists (for any mode)
            has_ripley = False
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    has_ripley = True
                    break
            
            if has_ripley:
                self._plot_sq_ripley(adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
    
    def _on_sq_ripley_cluster_changed(self):
        """Handle cluster column change in Ripley tab."""
        roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            # Check for any ripley key
            has_ripley = False
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    has_ripley = True
                    break
            if has_ripley:
                self._plot_sq_ripley(adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
    
    def _on_sq_autocorr_topk_changed(self):
        """Handle top K change in autocorrelation tab - auto-refresh plot."""
        roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
        if roi_id is None:
            # "All ROIs" selected - use aggregated result if available
            if 'autocorrelation' in self.aggregated_results:
                self._plot_sq_autocorrelation(self.aggregated_results['autocorrelation'])
                self.sq_autocorr_save_btn.setEnabled(True)
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            if 'moranI' in adata.uns:
                self._plot_sq_autocorrelation(adata)
                self.sq_autocorr_save_btn.setEnabled(True)
    
    def _get_all_rois(self):
        """Get all ROI IDs from filtered dataframe."""
        filtered_df = self._get_filtered_dataframe()
        roi_col = self._get_roi_column()
        return sorted(filtered_df[roi_col].unique())
    
    def _create_spatial_graph(self):
        """Create spatial graph for all ROIs using core function."""
        if not self._validate_data():
            return
        
        try:
            method = self.graph_method_combo.currentText()
            k = int(self.graph_k_spin.value()) if method == "kNN" else None
            radius = float(self.graph_radius_spin.value()) if method == "Radius" else None
            seed = int(self.seed_spinbox.value())
            
            roi_col = self._get_roi_column()
            filtered_df = self._get_filtered_dataframe()
            roi_cell_counts = {}
            if roi_col in filtered_df.columns:
                roi_cell_counts = (
                    filtered_df
                    .dropna(subset=["centroid_x", "centroid_y"])
                    .groupby(roi_col)
                    .size()
                    .to_dict()
                )
            
            # Get pixel size (use first ROI as default)
            all_rois = self._get_all_rois()
            if not all_rois:
                QtWidgets.QMessageBox.warning(self, "No ROIs", "No ROIs found in the data.")
                return
            
            pixel_size_um = self._get_pixel_size_um(all_rois[0])

            def _spatial_graph_task():
                return build_spatial_graph_anndata(
                    features_df=filtered_df,
                    method=method,
                    k_neighbors=k if k else 20,
                    radius=radius,
                    pixel_size_um=pixel_size_um,
                    roi_column=roi_col,
                    roi_id=None,
                    seed=seed,
                )

            anndata_dict = run_blocking_task_with_progress(
                parent=self,
                window_title="Building Spatial Graph",
                initial_message="Creating spatial graph",
                detail_text="Building spatial neighbor graph for selected ROI set.",
                task=_spatial_graph_task,
            )
            
            if anndata_dict:
                # Update cache with new AnnData objects
                self.anndata_cache.update(anndata_dict)
                self.spatial_graph_built = True
                success_count = len(anndata_dict)
                skipped_rois = []
                if roi_cell_counts:
                    successful_roi_ids = set(anndata_dict.keys())
                    for raw_roi_id, n_cells in roi_cell_counts.items():
                        roi_id_str = str(raw_roi_id)
                        if roi_id_str not in successful_roi_ids:
                            skipped_rois.append((roi_id_str, int(n_cells)))
                adjusted_k_rois = []
                if method == "kNN" and k is not None and roi_cell_counts:
                    for raw_roi_id, n_cells in roi_cell_counts.items():
                        if n_cells >= 2 and k >= n_cells:
                            adjusted_k_rois.append((str(raw_roi_id), int(k), int(n_cells - 1)))
                
                if skipped_rois:
                    self.graph_status_label.setText(
                        f"Graph created for {success_count} ROI(s), skipped {len(skipped_rois)} ROI(s)"
                    )
                    self.graph_status_label.setStyleSheet("color: #b36b00;")
                else:
                    self.graph_status_label.setText(f"Graph created for {success_count} ROI(s)")
                    self.graph_status_label.setStyleSheet("color: green;")
                
                # Update processed ROIs tracking
                for roi_id in anndata_dict.keys():
                    if roi_id not in self.processed_rois:
                        self.processed_rois[roi_id] = {}
                    self.processed_rois[roi_id]['graph_built'] = True
                
                # Populate reference cluster combo if we have data
                if self.anndata_cache:
                    first_adata = list(self.anndata_cache.values())[0]
                    cluster_key = self.sq_cooccur_cluster_combo.currentText()
                    if cluster_key in first_adata.obs.columns:
                        self._update_cooccur_ref_cluster_combo(first_adata)
                
                self._update_button_states()

                details = [f"Spatial graph created successfully for {success_count} ROI(s)."]
                if adjusted_k_rois:
                    preview = ", ".join(
                        f"{roi} ({old_k}->{new_k})" for roi, old_k, new_k in adjusted_k_rois[:6]
                    )
                    if len(adjusted_k_rois) > 6:
                        preview += f", ... ({len(adjusted_k_rois)} total)"
                    details.append(
                        f"k was auto-adjusted for {len(adjusted_k_rois)} ROI(s) with fewer cells than requested neighbors: {preview}"
                    )
                if skipped_rois:
                    preview = ", ".join(
                        f"{roi} (n={n_cells})" for roi, n_cells in skipped_rois[:6]
                    )
                    if len(skipped_rois) > 6:
                        preview += f", ... ({len(skipped_rois)} total)"
                    if method == "kNN":
                        details.append(
                            f"Skipped {len(skipped_rois)} ROI(s) that could not build a valid graph (typically n < 2): {preview}"
                        )
                    elif method == "Delaunay":
                        details.append(
                            f"Skipped {len(skipped_rois)} ROI(s) that could not build Delaunay graph (requires at least 3 non-degenerate points): {preview}"
                        )
                    else:
                        details.append(
                            f"Skipped {len(skipped_rois)} ROI(s) due to graph construction constraints: {preview}"
                        )

                QtWidgets.QMessageBox.information(self, "Graph Created", "\n\n".join(details))
            else:
                self.graph_status_label.setText("Graph creation failed")
                self.graph_status_label.setStyleSheet("color: red;")
                QtWidgets.QMessageBox.warning(self, "Graph Creation Failed", 
                    "Failed to create spatial graph for any ROI.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating spatial graph: {str(e)}")
    
    def _run_sq_nhood_enrichment(self):
        """Run neighborhood enrichment analysis using core function."""
        import sys
        sys.stdout.flush()
        if not self.spatial_graph_built:
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            cluster_key = self.sq_nhood_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
            agg_method = self.sq_nhood_agg_combo.currentText().lower()  # "mean" or "sum"
            sys.stdout.flush()
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Get AnnData dict - filter to selected ROI if needed
            if roi_id is None:
                # Use all cached AnnData objects with graphs
                anndata_dict = {rid: adata for rid, adata in self.anndata_cache.items() 
                               if 'spatial_connectivities' in adata.obsp}
            else:
                # Use only selected ROI
                if roi_id not in self.anndata_cache:
                    QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
                    return
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' not in adata.obsp:
                    QtWidgets.QMessageBox.warning(self, "No Graph", f"No spatial graph found for ROI {roi_id}.")
                    return
                anndata_dict = {roi_id: adata}
            
            if not anndata_dict:
                QtWidgets.QMessageBox.warning(self, "No Data", "No AnnData objects with spatial graphs found.")
                return
            
            def _nhood_task():
                return spatial_neighborhood_enrichment(
                    anndata_dict=anndata_dict,
                    cluster_key=cluster_key,
                    aggregation=agg_method,
                )

            results = run_blocking_task_with_progress(
                parent=self,
                window_title="Neighborhood Enrichment",
                initial_message="Running neighborhood enrichment",
                detail_text="Computing enrichment statistics across the selected ROI set.",
                task=_nhood_task,
            )
            
            # Update cache with results
            self.anndata_cache.update(results['results'])
            
            # Update status
            for roi_id in results['results'].keys():
                if roi_id not in self.analysis_status:
                    self.analysis_status[roi_id] = {}
                self.analysis_status[roi_id]['nhood_enrichment'] = True
            
            # Create temporary adata for aggregated plotting
            if results['aggregated'] is not None:
                
                class TempAnnData:
                    def __init__(
                        self,
                        matrix,
                        cluster_key,
                        obs,
                        significant_counts=None,
                        p_values=None,
                        p_values_adjusted=None,
                    ):
                        self.uns = {'nhood_enrichment': {'zscore': matrix}}
                        if p_values is not None:
                            self.uns['nhood_enrichment']['pvalue'] = p_values
                        if p_values_adjusted is not None:
                            self.uns['nhood_enrichment']['pvalue_fdr_bh'] = p_values_adjusted
                        self.obs = obs
                        self._cluster_key = cluster_key
                        self._significant_counts = significant_counts
                
                import pandas as pd
                cluster_categories = results['cluster_categories']
                obs_df = pd.DataFrame({cluster_key: cluster_categories})
                obs_df.index = [str(c) for c in cluster_categories]
                obs_df[cluster_key] = obs_df[cluster_key].astype('category')
                
                temp_adata = TempAnnData(
                    results['aggregated'], 
                    cluster_key, 
                    obs_df,
                    significant_counts=results.get('significant_counts'),
                    p_values=results.get('aggregated_p_values'),
                    p_values_adjusted=results.get('aggregated_p_values_adjusted'),
                )
                
                self.aggregated_results['nhood_enrichment'] = temp_adata
                
                QtWidgets.QMessageBox.information(self, "Enrichment Complete", 
                    f"Neighborhood enrichment completed for {len(results['results'])} ROI(s). "
                    f"Aggregated using {agg_method}.")
                
                # Plot aggregated results
                if hasattr(temp_adata, '_significant_counts') and temp_adata._significant_counts is not None:
                    pass
                # Switch to neighborhood enrichment tab to show the plot
                nhood_tab_idx = None
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) == "Neighborhood Enrichment":
                        nhood_tab_idx = i
                        break
                if nhood_tab_idx is not None:
                    self.tabs.setCurrentIndex(nhood_tab_idx)
                
                self._plot_sq_nhood_enrichment(temp_adata)
                # Ensure canvas is visible
                self.sq_nhood_canvas.setVisible(True)
                self.sq_nhood_canvas.show()
                self.sq_nhood_canvas.update()
                self.sq_nhood_save_btn.setEnabled(True)
                self.sq_nhood_export_btn.setEnabled(True)
            else:
                # Plot first ROI result
                if results['results']:
                    first_adata = list(results['results'].values())[0]
                    if 'nhood_enrichment' in first_adata.uns:
                        pass
                    # Switch to neighborhood enrichment tab to show the plot
                    nhood_tab_idx = None
                    for i in range(self.tabs.count()):
                        if self.tabs.tabText(i) == "Neighborhood Enrichment":
                            nhood_tab_idx = i
                            break
                    if nhood_tab_idx is not None:
                        self.tabs.setCurrentIndex(nhood_tab_idx)
                    
                    self._plot_sq_nhood_enrichment(first_adata)
                    # Ensure canvas is visible
                    self.sq_nhood_canvas.setVisible(True)
                    self.sq_nhood_canvas.show()
                    self.sq_nhood_canvas.update()
                    self.sq_nhood_save_btn.setEnabled(True)
                    self.sq_nhood_export_btn.setEnabled(True)
                    QtWidgets.QMessageBox.information(self, "Enrichment Complete", 
                        f"Neighborhood enrichment completed for {len(results['results'])} ROI(s).")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running enrichment: {str(e)}")
    
    def _plot_sq_nhood_enrichment(self, adata: 'ad.AnnData'):
        """Plot neighborhood enrichment results with improved visualization."""
        if adata is None:
            self.sq_nhood_canvas.figure.clear()
            ax = self.sq_nhood_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_nhood_canvas.draw()
            return
        
        
        if 'nhood_enrichment' not in adata.uns:
            self.sq_nhood_canvas.figure.clear()
            ax = self.sq_nhood_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No enrichment data found.\nPlease run enrichment analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_nhood_canvas.draw()
            return
        
        # Clear figure and maximize the main heatmap area.
        self.sq_nhood_canvas.figure.clear()
        ax = self.sq_nhood_canvas.figure.add_subplot(111)
        
        enrichment_data = adata.uns['nhood_enrichment']
        cluster_key = self.sq_nhood_cluster_combo.currentText()
        
        if isinstance(enrichment_data, dict):
            pass
        
        # Extract matrix from enrichment data
        matrix = None
        if isinstance(enrichment_data, dict):
            if 'zscore' in enrichment_data:
                matrix = enrichment_data['zscore']
            elif 'count' in enrichment_data:
                matrix = enrichment_data['count']
            elif 'stat' in enrichment_data:
                matrix = enrichment_data['stat']
            else:
                # Try to find a matrix-like value
                for key, value in enrichment_data.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        matrix = value
                        break
                if matrix is None and len(enrichment_data) > 0:
                    first_val = list(enrichment_data.values())[0]
                    if isinstance(first_val, np.ndarray) and first_val.ndim == 2:
                        matrix = first_val
        elif isinstance(enrichment_data, np.ndarray):
            matrix = enrichment_data
        
        # Get cluster labels
        categories = None
        if hasattr(adata, 'obs') and cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories = list(adata.obs[cluster_key].cat.categories)
            else:
                categories = sort_cluster_values(
                    adata.obs[cluster_key].unique(),
                    annotation_map=self.cluster_annotation_map,
                    canonical=False,
                )
        
        if matrix is not None and isinstance(matrix, np.ndarray) and matrix.ndim == 2:
            try:
                # Handle NaN/inf values
                if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                    matrix = np.nan_to_num(matrix, nan=0.0, posinf=3.0, neginf=-3.0)
                
                # Determine color scale
                finite_vals = matrix[np.isfinite(matrix)]
                if len(finite_vals) > 0:
                    data_min, data_max = np.min(finite_vals), np.max(finite_vals)
                    # Use symmetric scale centered at 0 for better visualization
                    abs_max = max(abs(data_min), abs(data_max))
                    vmin, vmax = -max(abs_max, 1), max(abs_max, 1)
                    # Clamp to reasonable range
                    vmin = max(vmin, -5)
                    vmax = min(vmax, 5)
                else:
                    vmin, vmax = -3, 3
                
                
                # Create DataFrame for seaborn heatmap
                if categories is not None and len(categories) == matrix.shape[0] == matrix.shape[1]:
                    cluster_labels = [self._get_cluster_display_name(c) for c in categories]
                    df = pd.DataFrame(matrix, index=cluster_labels, columns=cluster_labels)
                else:
                    df = pd.DataFrame(matrix)
                style = dense_heatmap_style(
                    n_rows=df.shape[0],
                    n_cols=df.shape[1],
                    row_labels=df.index.astype(str).tolist(),
                    col_labels=df.columns.astype(str).tolist(),
                    base_tick_fontsize=10.0,
                    base_annotation_fontsize=9.0,
                    allow_annotations=True,
                )
                annot_data = np.round(df.to_numpy(), 2) if style['show_annotations'] else False

                sns.heatmap(
                    df,
                    annot=annot_data,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    vmin=vmin,
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
                    annot_kws={
                        'size': style['annotation_fontsize'],
                        'weight': 'normal',
                        'color': 'black',
                    },
                    xticklabels=True,
                    yticklabels=True,
                )

                ax.set_xticklabels(
                    ax.get_xticklabels(),
                    rotation=style['x_rotation'],
                    ha='right',
                    fontsize=style['tick_fontsize'],
                )
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=style['tick_fontsize'])

                ax.set_xlabel("Neighbor Cluster", fontsize=style['axis_fontsize'], fontweight='bold')
                ax.set_ylabel("Cell Cluster", fontsize=style['axis_fontsize'], fontweight='bold')
                ax.set_title(
                    "Neighborhood Enrichment Analysis",
                    fontsize=style['title_fontsize'],
                    fontweight='bold',
                    pad=12,
                )
                ax.tick_params(axis='both', labelsize=style['tick_fontsize'])

                colorbar = ax.collections[0].colorbar if ax.collections else None
                if colorbar is not None:
                    colorbar.ax.tick_params(labelsize=style['colorbar_fontsize'])
                    colorbar.set_label('Z-Score', fontsize=style['axis_fontsize'])

                
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
        else:
            # Debug: show what we got
            debug_info = f"Data type: {type(enrichment_data)}\n"
            if isinstance(enrichment_data, dict):
                debug_info += f"Keys: {list(enrichment_data.keys())}\n"
                for k, v in enrichment_data.items():
                    debug_info += f"  {k}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}\n"
            else:
                debug_info += f"Value: {enrichment_data}\n"
            ax.text(0.5, 0.5, f'Unable to plot enrichment data.\nData format not recognized.\n\n{debug_info}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
        
        try:
            self._fit_canvas(self.sq_nhood_canvas, pad=0.95)
            refresh_canvas(self.sq_nhood_canvas, draw=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def _save_sq_nhood_plot(self):
        """Save the neighborhood enrichment plot."""
        if save_figure_with_options(self.sq_nhood_canvas.figure, "squidpy_nhood_enrichment.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _run_sq_cooccurrence(self):
        """Run co-occurrence analysis using core function."""
        if not self.spatial_graph_built:
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            cluster_key = self.sq_cooccur_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
            plot_type = self.sq_cooccur_plot_type_combo.currentText()
            
            # Parse neighborhood sizes
            sizes_str = self.sq_cooccur_sizes_edit.text().strip()
            if sizes_str.lower() == 'all' or not sizes_str:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                    "Co-occurrence analysis requires multiple distances.\n"
                    "Please enter at least 2 distances separated by commas (e.g., '10,20,30,50').")
                return
            
            try:
                # Parse as comma-separated list
                nhood_sizes = [float(x.strip()) for x in sizes_str.split(',') if x.strip()]
                if len(nhood_sizes) < 2:
                    QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                        f"Co-occurrence analysis requires at least 2 distances.\n"
                        f"You provided {len(nhood_sizes)} distance(s): {nhood_sizes}\n"
                        f"Please enter multiple distances separated by commas (e.g., '10,20,30,50').")
                    return
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                    f"Could not parse distances. Please enter comma-separated numbers (e.g., '10,20,30,50').\n"
                    f"Error: {str(e)}")
                return
            
            
            self._sync_sq_cooccur_distance_input(
                nhood_sizes,
                preferred_distance=self.sq_cooccur_heatmap_distance,
            )
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Get AnnData dict - filter to selected ROI if needed
            if roi_id is None:
                # Use all cached AnnData objects with graphs
                anndata_dict = {rid: adata for rid, adata in self.anndata_cache.items() 
                               if 'spatial_connectivities' in adata.obsp}
            else:
                # Use only selected ROI
                if roi_id not in self.anndata_cache:
                    QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
                    return
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' not in adata.obsp:
                    QtWidgets.QMessageBox.warning(self, "No Graph", f"No spatial graph found for ROI {roi_id}.")
                    return
                anndata_dict = {roi_id: adata}
            
            if not anndata_dict:
                QtWidgets.QMessageBox.warning(self, "No Data", "No AnnData objects with spatial graphs found.")
                return
            
            def _cooccur_task():
                return spatial_cooccurrence(
                    anndata_dict=anndata_dict,
                    cluster_key=cluster_key,
                    interval=nhood_sizes,
                    reference_cluster=self.sq_cooccur_ref_cluster_combo.currentData() if plot_type == "Curves" else None,
                )

            results = run_blocking_task_with_progress(
                parent=self,
                window_title="Co-occurrence Analysis",
                initial_message="Running co-occurrence analysis",
                detail_text="Estimating neighborhood co-occurrence across distances.",
                task=_cooccur_task,
            )
            
            # Update cache with results
            self.anndata_cache.update(results)
            
            # Update status
            for roi_id in results.keys():
                if roi_id not in self.analysis_status:
                    self.analysis_status[roi_id] = {}
                self.analysis_status[roi_id]['co_occurrence'] = True
            
            if results:
                # Use first ROI for plotting and reference cluster combo
                plot_adata = list(results.values())[0]
                # Update reference cluster combo (preserve selection if combo already has items)
                preserve = self.sq_cooccur_ref_cluster_combo.count() > 0
                self._update_cooccur_ref_cluster_combo(plot_adata, preserve_selection=preserve)
                
                QtWidgets.QMessageBox.information(self, "Co-occurrence Complete", 
                    f"Co-occurrence analysis completed for {len(results)} ROI(s).")
                
                # Plot results (will use the default reference cluster "1")
                self._plot_sq_cooccurrence(plot_adata)
                self.sq_cooccur_save_btn.setEnabled(True)
                self.sq_cooccur_export_btn.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(self, "No Results", 
                    "Co-occurrence analysis completed but no results to plot.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running co-occurrence: {str(e)}")
    
    def _plot_sq_cooccurrence(self, adata: 'ad.AnnData'):
        """Plot co-occurrence results using squidpy's plotting function."""
        if adata is None:
            self.sq_cooccur_canvas.figure.clear()
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_cooccur_canvas.draw()
            return
        
        
        # Check for co_occurrence key or alternatives
        cooccur_key = None
        if 'co_occurrence' in adata.uns:
            cooccur_key = 'co_occurrence'
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    cooccur_key = key
                    break
        
        if cooccur_key is None:
            self.sq_cooccur_canvas.figure.clear()
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No co-occurrence data found.\nPlease run co-occurrence analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_cooccur_canvas.draw()
            return
        
        # Use manual plotting (squidpy's plotting functions don't work well with custom canvases)
        # sq.pl.co_occurrence() doesn't support custom axes and creates its own figure
        cooccur_data = adata.uns[cooccur_key]
        cluster_key = self.sq_cooccur_cluster_combo.currentText()
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        
        
        # Get cluster categories
        if cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories = list(adata.obs[cluster_key].cat.categories)
            else:
                categories = sort_cluster_values(
                    adata.obs[cluster_key].unique(),
                    annotation_map=self.cluster_annotation_map,
                    canonical=False,
                )
        else:
            categories = []
        
        
        occ_array = None
        if isinstance(cooccur_data, dict) and 'occ' in cooccur_data:
            try:
                occ_array = np.asarray(cooccur_data['occ'], dtype=float)
            except Exception:
                occ_array = None

        available_distances = self._normalize_cooccur_distances(
            cooccur_data.get('interval', cooccur_data.get('distances')) if isinstance(cooccur_data, dict) else None,
            expected_size=occ_array.shape[2] if occ_array is not None and occ_array.ndim == 3 else None,
            fallback=self.sq_cooccur_interval,
        )
        had_distance_metadata = self.sq_cooccur_interval is not None
        self._set_sq_cooccur_distance_metadata(available_distances)

        current_distance_value = self.sq_cooccur_distance_spin.value() if hasattr(self, 'sq_cooccur_distance_spin') else None
        current_distance_valid = self._match_cooccur_distance(current_distance_value, available_distances) is not None
        stored_distance_valid = self._match_cooccur_distance(self.sq_cooccur_heatmap_distance, available_distances) is not None
        if available_distances and (not had_distance_metadata or (not current_distance_valid and not stored_distance_valid)):
            self._sync_sq_cooccur_distance_input(available_distances)

        # Handle heatmap plotting
        if plot_type == "Heatmap":
            if not available_distances:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No distances available.\nPlease run co-occurrence analysis first.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return

            selected_distance = self._match_cooccur_distance(
                self.sq_cooccur_distance_spin.value(),
                available_distances,
            )
            if selected_distance is None:
                requested_distance = self.sq_cooccur_distance_spin.value()
                available_text = ", ".join(f"{distance:g}" for distance in available_distances)
                self.sq_cooccur_canvas.figure.clear()
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    f"Distance {requested_distance:g} µm is not available.\nAvailable distances: {available_text} µm.",
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                )
                self.sq_cooccur_canvas.draw()
                return

            if occ_array is None or occ_array.ndim != 3:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Co-occurrence data format not suitable for heatmap.\nExpected 3D array.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return

            self.sq_cooccur_heatmap_distance = selected_distance
            if not np.isclose(self.sq_cooccur_distance_spin.value(), selected_distance, rtol=1e-6, atol=1e-6):
                blocker = QtCore.QSignalBlocker(self.sq_cooccur_distance_spin)
                self.sq_cooccur_distance_spin.setValue(selected_distance)
                del blocker

            distance_idx = next(
                idx for idx, distance in enumerate(available_distances)
                if np.isclose(distance, selected_distance, rtol=1e-6, atol=1e-6)
            )

            heatmap_data = occ_array[:, :, distance_idx]
            self.sq_cooccur_canvas.figure.clear()
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            
            # Create DataFrame for better visualization
            if len(categories) == heatmap_data.shape[0] == heatmap_data.shape[1]:
                cluster_labels = [self._get_cluster_display_name(c) for c in categories]
                df = pd.DataFrame(heatmap_data, index=cluster_labels, columns=cluster_labels)
            else:
                df = pd.DataFrame(heatmap_data)
            style = dense_heatmap_style(
                n_rows=df.shape[0],
                n_cols=df.shape[1],
                row_labels=df.index.astype(str).tolist(),
                col_labels=df.columns.astype(str).tolist(),
                base_tick_fontsize=9.5,
                base_annotation_fontsize=8.5,
                allow_annotations=True,
            )
            annot_data = np.round(df.to_numpy(), 3) if style['show_annotations'] else False

            sns.heatmap(
                df,
                annot=annot_data,
                fmt='.3f',
                cmap='YlOrRd',
                square=style['square_cells'],
                linewidths=style['linewidths'],
                cbar_kws={
                    'label': 'Co-occurrence Score',
                    'shrink': style['colorbar_shrink'],
                    'fraction': style['colorbar_fraction'],
                    'pad': style['colorbar_pad'],
                },
                ax=ax,
                annot_kws={
                    'size': style['annotation_fontsize'],
                    'weight': 'normal',
                    'color': 'black',
                },
                xticklabels=True,
                yticklabels=True,
            )

            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=style['x_rotation'],
                ha='right',
                fontsize=style['tick_fontsize'],
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=style['tick_fontsize'])

            ax.set_xlabel('To Phenotype', fontsize=style['axis_fontsize'], fontweight='bold')
            ax.set_ylabel('From Phenotype', fontsize=style['axis_fontsize'], fontweight='bold')
            ax.set_title(
                f'Co-occurrence Analysis at {selected_distance:g} µm',
                fontsize=style['title_fontsize'],
                fontweight='bold',
                pad=12,
            )
            ax.tick_params(axis='both', labelsize=style['tick_fontsize'])

            colorbar = ax.collections[0].colorbar if ax.collections else None
            if colorbar is not None:
                colorbar.ax.tick_params(labelsize=style['colorbar_fontsize'])
                colorbar.set_label('Co-occurrence Score', fontsize=style['axis_fontsize'])

            self._fit_canvas(self.sq_cooccur_canvas, pad=0.95)
            return
        
        # Plot using squidpy's exact approach
        # Based on sq.pl.co_occurrence source code
        occurrence_data = cooccur_data
        out = occurrence_data["occ"]
        interval = self._normalize_cooccur_distances(
            occurrence_data.get("interval", occurrence_data.get("distances")),
            expected_size=out.shape[2] if len(out.shape) >= 3 else None,
            fallback=self.sq_cooccur_interval,
        )
        
        # IMPORTANT: The 'out' array shape is determined by the categories used when
        # co-occurrence was computed. We need to use those categories, not necessarily
        # the current ROI's categories. The array shape is (n_clusters, n_clusters, n_distances)
        # So we need to get the categories that match the array dimensions.
        out_shape = out.shape
        
        # The out array should be (n_clusters, n_clusters, n_distances)
        # Use the actual dimensions from the array
        n_clusters_in_data = out_shape[0] if len(out_shape) >= 2 else 0
        
        # Try to get categories from the data structure if available
        # Otherwise, use categories from current ROI but ensure they match array size
        if n_clusters_in_data > 0:
            # Check if categories match the array size
            if len(categories) == n_clusters_in_data:
                categories_list = categories
            else:
                # Categories don't match - try to get from adata.obs categories
                # or use indices if categories are missing
                # Use the categories from the adata that was used to compute co-occurrence
                # If the current ROI has fewer categories, we need to handle this
                if len(categories) < n_clusters_in_data:
                    # Current ROI has fewer categories - this means some clusters are missing
                    # We should use the categories from when co-occurrence was computed
                    # For now, create placeholder categories or use indices
                    # Try to get all possible categories from the cluster column's categories
                    if cluster_key in adata.obs.columns and hasattr(adata.obs[cluster_key], 'cat'):
                        all_categories = list(adata.obs[cluster_key].cat.categories)
                        if len(all_categories) >= n_clusters_in_data:
                            categories_list = all_categories[:n_clusters_in_data]
                        else:
                            # Still not enough - use indices as fallback
                            categories_list = [f"Cluster_{i}" for i in range(n_clusters_in_data)]
                    else:
                        categories_list = [f"Cluster_{i}" for i in range(n_clusters_in_data)]
                else:
                    # Current ROI has more categories - use first n_clusters_in_data
                    categories_list = categories[:n_clusters_in_data]
        else:
            categories_list = categories
        
        
        # Get clusters to plot (use reference cluster if selected, otherwise all)
        clusters_to_plot = None
        ref_cluster = self.sq_cooccur_ref_cluster_combo.currentData()
        if ref_cluster is not None:
            # Plot only the reference cluster (like squidpy's clusters parameter)
            clusters_to_plot = [ref_cluster] if ref_cluster in categories_list else categories_list
        else:
            # Plot all clusters
            clusters_to_plot = categories_list
        
        # Filter to valid clusters
        clusters_to_plot = [c for c in clusters_to_plot if c in categories_list]
        
        if not clusters_to_plot:
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No valid clusters to plot.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_cooccur_canvas.draw()
            return
        
        # Create subplots like squidpy does (1 row, len(clusters) columns)
        n_clusters = len(clusters_to_plot)
        self.sq_cooccur_canvas.figure.clear()
        
        # Get palette using tab20/tab60 for consistent cluster coloring
        # Use _get_vivid_colors which provides tab20, tab20b, tab20c for up to 60 clusters
        n_categories = len(categories_list)
        if n_categories > 0:
            # Get colors using tab20/tab60 scheme
            color_array = _get_vivid_colors(n_categories)
            # Convert to dict mapping category to color (RGBA tuple)
            palette = {cat: tuple(color_array[i]) for i, cat in enumerate(categories_list)}
        else:
            palette = {}
        
        # Create subplots
        if n_clusters == 1:
            axs = [self.sq_cooccur_canvas.figure.add_subplot(111)]
        else:
            # Create grid of subplots
            n_cols = min(n_clusters, 4)  # Limit columns for readability
            n_rows = (n_clusters + n_cols - 1) // n_cols
            axs = []
            for i in range(n_clusters):
                ax = self.sq_cooccur_canvas.figure.add_subplot(n_rows, n_cols, i + 1)
                axs.append(ax)
        
        # Plot each cluster like squidpy does
        for g, ax in zip(clusters_to_plot, axs):
            # Find index of cluster in categories_list
            try:
                idx = categories_list.index(g)
            except ValueError:
                continue
            
            # Create DataFrame like squidpy: out[idx, :, :].T with columns=categories
            # Shape: (n_distances, n_clusters)
            cluster_data = out[idx, :, :].T
            actual_n_cols = cluster_data.shape[1]
            
            # Ensure the number of columns matches categories_list length
            if actual_n_cols != len(categories_list):
                # Use categories that match the actual data shape
                if actual_n_cols <= len(categories_list):
                    # Use first actual_n_cols categories
                    df_columns = categories_list[:actual_n_cols]
                else:
                    # More columns than categories - pad with placeholder names
                    df_columns = list(categories_list) + [f"Cluster_{i}" for i in range(len(categories_list), actual_n_cols)]
                df = pd.DataFrame(cluster_data, columns=df_columns)
            else:
                df = pd.DataFrame(cluster_data, columns=categories_list)
            
            cluster_display_map = {cat: self._get_cluster_display_name(cat) for cat in df.columns}

            for column in df.columns:
                display_name = cluster_display_map.get(column, str(column))
                ax.plot(
                    interval,
                    df[column].to_numpy(),
                    linewidth=2.0,
                    color=palette.get(column, '#444444'),
                    label=display_name,
                )

            legend_kwargs = {"loc": "center left", "bbox_to_anchor": (1, 0.5), "fontsize": 8}
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                seen = set()
                unique_handles = []
                unique_labels = []
                for handle, label in zip(handles, labels):
                    if label in seen:
                        continue
                    seen.add(label)
                    unique_handles.append(handle)
                    unique_labels.append(label)
                ax.legend(unique_handles, unique_labels, **legend_kwargs)

            g_display = self._get_cluster_display_name(g)
            ax.set_title(rf"$\frac{{p(exp|{g_display})}}{{p(exp)}}$", fontsize=10)
            ax.set_xlabel("Distance (µm)", fontsize=9)
            ax.set_ylabel("Co-occurrence score", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid(True, alpha=0.25)
        
        self._fit_canvas(self.sq_cooccur_canvas, rect=[0.0, 0.0, 0.84, 1.0], pad=0.95)
    
    def _save_sq_cooccur_plot(self):
        """Save the co-occurrence plot."""
        if save_figure_with_options(self.sq_cooccur_canvas.figure, "squidpy_cooccurrence.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _run_sq_autocorrelation(self):
        """Run spatial autocorrelation analysis using core function."""
        if not self.spatial_graph_built:
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            markers_str = self.sq_autocorr_markers_edit.text().strip()
            roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
            agg_method = self.sq_autocorr_agg_combo.currentText().lower()
            
            # Parse markers
            markers = None
            if markers_str.lower() != 'all' and markers_str:
                markers = [
                    self._get_feature_name_from_display(marker.strip())
                    for marker in markers_str.split(',')
                    if marker.strip()
                ]
            
            # Get AnnData dict - filter to selected ROI if needed
            if roi_id is None:
                # Use all cached AnnData objects with graphs
                anndata_dict = {rid: adata for rid, adata in self.anndata_cache.items() 
                               if 'spatial_connectivities' in adata.obsp}
            else:
                # Use only selected ROI
                if roi_id not in self.anndata_cache:
                    QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
                    return
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' not in adata.obsp:
                    QtWidgets.QMessageBox.warning(self, "No Graph", f"No spatial graph found for ROI {roi_id}.")
                    return
                anndata_dict = {roi_id: adata}
            
            if not anndata_dict:
                QtWidgets.QMessageBox.warning(self, "No Data", "No AnnData objects with spatial graphs found.")
                return
            
            def _autocorr_task():
                return spatial_autocorrelation(
                    anndata_dict=anndata_dict,
                    markers=markers,
                    aggregation=agg_method,
                )

            results = run_blocking_task_with_progress(
                parent=self,
                window_title="Spatial Autocorrelation",
                initial_message="Running spatial autocorrelation analysis",
                detail_text="Computing Moran statistics for selected markers.",
                task=_autocorr_task,
            )
            
            # Update cache with results
            self.anndata_cache.update(results['results'])
            
            # Update status
            for roi_id in results['results'].keys():
                if roi_id not in self.analysis_status:
                    self.analysis_status[roi_id] = {}
                self.analysis_status[roi_id]['autocorrelation'] = True
            
            # Store aggregated result if available
            if results['aggregated'] is not None:
                self.aggregated_results['autocorrelation'] = results['aggregated']
                plot_adata = results['aggregated']
            elif results['results']:
                plot_adata = list(results['results'].values())[0]
            else:
                QtWidgets.QMessageBox.warning(self, "No Results", "No autocorrelation results to plot.")
                return
            
            QtWidgets.QMessageBox.information(self, "Autocorrelation Complete", 
                f"Spatial autocorrelation completed for {len(results['results'])} ROI(s). "
                f"{'Aggregated using ' + agg_method if len(results['results']) > 1 else ''}")
            
            # Update variable combo box
            self._update_autocorr_var_combo()
            
            # Plot results
            self._plot_sq_autocorrelation(plot_adata)
            self.sq_autocorr_save_btn.setEnabled(True)
            self.sq_autocorr_export_btn.setEnabled(True)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running autocorrelation: {str(e)}")

    def _extract_moran_arrays(self, moran_data):
        """Extract Moran's I values, p-values, and variable names from supported result formats."""
        I_values = None
        p_values = None
        gene_names = None

        if isinstance(moran_data, pd.DataFrame):
            if 'I' in moran_data.columns:
                I_values = moran_data['I'].to_numpy()
            elif 'moranI' in moran_data.columns:
                I_values = moran_data['moranI'].to_numpy()

            p_column, _ = self._get_preferred_moran_pvalue_columns(list(moran_data.columns))
            if p_column is not None:
                p_values = moran_data[p_column].to_numpy()

            if 'var_names' in moran_data.columns:
                gene_names = moran_data['var_names'].astype(str).to_numpy()
            elif moran_data.index.name == 'var_names' or all(isinstance(x, str) for x in moran_data.index):
                gene_names = moran_data.index.astype(str).to_numpy()
            else:
                gene_names = np.array([str(x) for x in moran_data.index], dtype=object)

        elif isinstance(moran_data, dict):
            I_values = moran_data.get('I')
            for key in ['pval_sim', 'pval_z_sim', 'pval_norm', 'pval']:
                if key in moran_data:
                    p_values = moran_data.get(key)
                    break
            gene_names = moran_data.get('var_names')

        return (
            self._coerce_numeric_vector(I_values, allow_none=True),
            self._coerce_numeric_vector(p_values, allow_none=True),
            np.asarray(gene_names, dtype=object).flatten() if gene_names is not None else np.array([], dtype=object),
        )

    def _coerce_numeric_vector(self, values, *, allow_none=False):
        """Convert vector-like values to a 1D float array, coercing invalid entries to NaN."""
        if values is None:
            return None if allow_none else np.array([], dtype=float)
        if hasattr(values, 'toarray'):
            values = values.toarray()
        array = np.asarray(values).reshape(-1)
        if array.size == 0:
            return np.array([], dtype=float)
        return pd.to_numeric(pd.Series(array), errors='coerce').to_numpy(dtype=float)
    
    def _plot_sq_autocorrelation(self, adata: 'ad.AnnData'):
        """Plot spatial autocorrelation results using squidpy's plotting function."""
        if adata is None:
            self.sq_autocorr_canvas.figure.clear()
            ax = self.sq_autocorr_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_autocorr_canvas.draw()
            return
        
        
        # Check for moranI key or alternatives
        moran_key = None
        if 'moranI' in adata.uns:
            moran_key = 'moranI'
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'moran' in key.lower() or 'autocorr' in key.lower():
                    moran_key = key
                    break
        
        if moran_key is None:
            self.sq_autocorr_canvas.figure.clear()
            ax = self.sq_autocorr_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No autocorrelation data found.\nPlease run autocorrelation analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_autocorr_canvas.draw()
            return
        
        # Squidpy doesn't have a spatial_autocorr plotting function
        # Use manual plotting which works well
        self.sq_autocorr_canvas.figure.clear()
        ax = self.sq_autocorr_canvas.figure.add_subplot(111)
        moran_data = adata.uns[moran_key]

        I_values, p_values, gene_names = self._extract_moran_arrays(moran_data)
        if (gene_names.size == 0) and hasattr(adata, 'var_names'):
            gene_names = np.asarray(adata.var_names, dtype=object).flatten()

        # Plot the data (works for both DataFrame and dict formats)
        if I_values is not None and len(I_values) > 0:
            finite_mask = np.isfinite(I_values)
            if p_values is not None and len(p_values) == len(I_values):
                p_values = np.where(np.isfinite(p_values), p_values, 1.0)
            if gene_names.size == len(I_values):
                gene_names = gene_names[finite_mask]
            I_values = I_values[finite_mask]
            if p_values is not None and len(p_values) == len(finite_mask):
                p_values = p_values[finite_mask]

            if len(I_values) == 0:
                ax.text(0.5, 0.5, 'No finite Moran\'s I values were available to plot.',
                       ha='center', va='center', transform=ax.transAxes)
                self._fit_canvas(self.sq_autocorr_canvas, pad=0.95)
                return

            # Create bar plot
            if p_values is not None and len(p_values) == len(I_values):
                sorted_idx = np.lexsort((p_values, -I_values))
            else:
                sorted_idx = np.argsort(I_values)[::-1]
            top_n = min(self.sq_autocorr_topk_spin.value(), len(sorted_idx))  # Use spinbox value
            
            top_I = I_values[sorted_idx[:top_n]]
            top_p = p_values[sorted_idx[:top_n]] if p_values is not None and len(p_values) > 0 else None
            top_genes = (
                [self._get_feature_display_name(str(gene_names[i])) for i in sorted_idx[:top_n]]
                if len(gene_names) > 0
                else [f"Feature_{i}" for i in sorted_idx[:top_n]]
            )
            
            colors = ['red' if (top_p is not None and p < 0.05) else 'gray' for p in (top_p if top_p is not None else [1.0] * top_n)]
            ax.barh(range(top_n), top_I, color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_genes, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Moran's I")
            ax.set_title(f"Spatial Autocorrelation (Top {top_n})")
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Unable to extract Moran\'s I values.\nData format not recognized.\nCheck debug output for details.', 
                   ha='center', va='center', transform=ax.transAxes)
        
        self._fit_canvas(self.sq_autocorr_canvas, pad=0.95)
    
    def _save_sq_autocorr_plot(self):
        """Save the spatial autocorrelation plot."""
        if save_figure_with_options(self.sq_autocorr_canvas.figure, "squidpy_autocorrelation.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _plot_sq_autocorr_visualization(self):
        """Plot Moran scatter plot or spatial map for selected variable."""
        viz_type = self.sq_autocorr_viz_type_combo.currentText()
        
        # Get the actual variable name from combo box data (not the display text)
        current_idx = self.sq_autocorr_var_combo.currentIndex()
        if current_idx < 0:
            QtWidgets.QMessageBox.warning(self, "No Variable", "Please select a variable to visualize.")
            return
        
        var_name = self.sq_autocorr_var_combo.itemData(current_idx)
        if not var_name:
            # Fallback to text if no data stored (shouldn't happen)
            var_name = self.sq_autocorr_var_combo.currentText()
            # Remove p-value annotation if present
            if ' (p' in var_name:
                var_name = var_name.split(' (p')[0]
        
        if not var_name:
            QtWidgets.QMessageBox.warning(self, "No Variable", "Please select a variable to visualize.")
            return
        
        roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
        adata = None
        
        if roi_id is None:
            # Use aggregated result if available, but for visualization we need individual ROI
            # Try to get first available ROI
            if self.anndata_cache:
                roi_id = list(self.anndata_cache.keys())[0]
                adata = self.anndata_cache[roi_id]
            else:
                QtWidgets.QMessageBox.warning(self, "No Data", 
                    "Spatial visualizations require a specific ROI. Please select an ROI.")
                return
        elif roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
        else:
            QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
            return
        
        if adata is None:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data available for visualization.")
            return
        
        # Check if spatial graph exists
        if 'spatial_connectivities' not in adata.obsp:
            QtWidgets.QMessageBox.warning(self, "No Graph", 
                "Spatial graph not found. Please build the spatial graph first.")
            return
        
        # Get variable values
        var_values = None
        if var_name in adata.var_names:
            # Variable is in var (feature matrix)
            var_idx = list(adata.var_names).index(var_name)
            var_values = self._coerce_numeric_vector(adata.X[:, var_idx], allow_none=True)
        elif hasattr(adata, 'obs') and var_name in adata.obs.columns:
            # Variable is in obs (metadata)
            if not pd.api.types.is_numeric_dtype(adata.obs[var_name]):
                QtWidgets.QMessageBox.warning(self, "Invalid Variable", 
                    f"Variable '{var_name}' is not numeric. Please select a numeric variable.")
                return
            var_values = self._coerce_numeric_vector(adata.obs[var_name].values, allow_none=True)
        else:
            QtWidgets.QMessageBox.warning(self, "Variable Not Found", 
                f"Variable '{var_name}' not found in data.")
            return
        
        # Validate variable values
        if var_values is None or len(var_values) == 0:
            QtWidgets.QMessageBox.warning(self, "No Data", 
                f"No data available for variable '{var_name}'.")
            return

        n_obs = len(adata.obs) if hasattr(adata, 'obs') else None
        if n_obs is not None and len(var_values) != n_obs:
            QtWidgets.QMessageBox.warning(
                self,
                "Dimension Mismatch",
                f"Variable '{var_name}' has {len(var_values)} values but the ROI contains {n_obs} cells.",
            )
            return
        
        # Handle NaN and inf values
        finite_mask = np.isfinite(var_values)
        if np.any(~finite_mask):
            n_invalid = int(np.sum(~finite_mask))
            QtWidgets.QMessageBox.warning(self, "Invalid Values", 
                f"Variable '{var_name}' contains {n_invalid} NaN or infinite values. "
                "These will be replaced with 0 for visualization.")
            var_values = np.nan_to_num(var_values, nan=0.0, posinf=0.0, neginf=0.0)

        if np.allclose(var_values, var_values[0]):
            QtWidgets.QMessageBox.warning(
                self,
                "Low Variability",
                f"Variable '{var_name}' is constant within the selected ROI. "
                "Moran visualizations require at least some variability.",
            )
            return
        
        if viz_type == "Moran Scatter Plot":
            self._plot_moran_scatter(adata, var_name, var_values)
        elif viz_type == "Spatial Map":
            self._plot_spatial_map(adata, var_name, var_values)
        
        self.sq_autocorr_save_btn.setEnabled(True)
    
    def _plot_moran_scatter(self, adata: 'ad.AnnData', var_name: str, var_values: np.ndarray):
        """Plot Moran scatter plot: variable vs spatial lag."""
        self.sq_autocorr_canvas.figure.clear()
        ax = self.sq_autocorr_canvas.figure.add_subplot(111)
        display_var_name = self._get_feature_display_name(str(var_name))
        
        var_values = self._coerce_numeric_vector(var_values)
        if len(var_values) == 0:
            QtWidgets.QMessageBox.warning(self, "No Data", f"No numeric values were available for '{var_name}'.")
            return
        
        # Get spatial weights matrix
        W = adata.obsp['spatial_connectivities']
        if hasattr(W, 'toarray'):
            W = W.toarray()
        else:
            W = np.array(W)
        W = np.asarray(W, dtype=float)

        if W.shape[0] != len(var_values) or W.shape[1] != len(var_values):
            QtWidgets.QMessageBox.warning(
                self,
                "Dimension Mismatch",
                f"Spatial weights for '{var_name}' do not match the number of cells in this ROI.",
            )
            return
        
        # Standardize variable (mean-center and scale)
        var_centered = var_values - np.mean(var_values)
        var_std = np.std(var_centered)
        if var_std > 0:
            var_standardized = var_centered / var_std
        else:
            var_standardized = var_centered
        
        # Ensure var_standardized is 1D
        var_standardized = np.asarray(var_standardized).flatten()
        
        # Compute spatial lag: W * x
        spatial_lag = W @ var_standardized
        
        # Ensure spatial_lag is 1D
        spatial_lag = np.asarray(spatial_lag).flatten()
        if len(spatial_lag) != len(var_standardized):
            QtWidgets.QMessageBox.warning(
                self,
                "Dimension Mismatch",
                f"Spatial lag for '{var_name}' could not be computed consistently.",
            )
            return
        
        # Compute Moran's I from the slope
        # Moran's I = (n/W0) * (x' W x) / (x' x)
        # where W0 is sum of weights
        n = len(var_standardized)
        W0 = np.sum(W)
        denom = var_standardized @ var_standardized
        if W0 > 0 and denom > 0:
            moran_I = (n / W0) * (var_standardized @ W @ var_standardized) / (var_standardized @ var_standardized)
        else:
            moran_I = 0.0
        
        # Create scatter plot
        ax.scatter(var_standardized, spatial_lag, alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        
        # Fit regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(var_standardized, spatial_lag)
        x_line = np.linspace(var_standardized.min(), var_standardized.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f"Slope = {slope:.3f} (Moran's I ≈ {moran_I:.3f})")
        
        # Add quadrant lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add quadrant labels
        x_range = var_standardized.max() - var_standardized.min()
        y_range = spatial_lag.max() - spatial_lag.min()
        ax.text(0.95 * var_standardized.max(), 0.95 * spatial_lag.max(), 'HH', 
                ha='right', va='top', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.text(0.95 * var_standardized.min(), 0.95 * spatial_lag.min(), 'LL', 
                ha='left', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.95 * var_standardized.max(), 0.95 * spatial_lag.min(), 'HL', 
                ha='right', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.text(0.95 * var_standardized.min(), 0.95 * spatial_lag.max(), 'LH', 
                ha='left', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Labels and title
        ax.set_xlabel(f'{display_var_name} (standardized)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Spatial Lag (Wx)', fontsize=11, fontweight='bold')
        ax.set_title(f"Moran Scatter Plot: {display_var_name}\nSlope = {slope:.3f} (Moran's I ≈ {moran_I:.3f})", 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        self._fit_canvas(self.sq_autocorr_canvas, pad=0.95)
    
    def _plot_spatial_map(self, adata: 'ad.AnnData', var_name: str, var_values: np.ndarray):
        """Plot spatial map colored by variable."""
        self.sq_autocorr_canvas.figure.clear()
        ax = self.sq_autocorr_canvas.figure.add_subplot(111)
        display_var_name = self._get_feature_display_name(str(var_name))
        
        var_values = self._coerce_numeric_vector(var_values)
        if len(var_values) == 0:
            QtWidgets.QMessageBox.warning(self, "No Data", f"No numeric values were available for '{var_name}'.")
            return
        
        # Get spatial coordinates
        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial']
        elif 'centroid_x' in adata.obs.columns and 'centroid_y' in adata.obs.columns:
            coords = np.column_stack([adata.obs['centroid_x'].values, adata.obs['centroid_y'].values])
        else:
            QtWidgets.QMessageBox.warning(self, "No Coordinates", 
                "Spatial coordinates not found. Cannot create spatial map.")
            return
        coords = np.asarray(coords, dtype=float)
        
        # Validate dimensions match
        if len(var_values) != len(coords):
            QtWidgets.QMessageBox.warning(self, "Dimension Mismatch", 
                f"Variable has {len(var_values)} values but there are {len(coords)} cells. "
                "Cannot create spatial map.")
            return
        
        # Create scatter plot colored by variable
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=var_values, 
                           cmap='viridis', s=10, alpha=0.7, edgecolors='black', linewidths=0.1)
        
        # Add colorbar
        cbar = self.sq_autocorr_canvas.figure.colorbar(scatter, ax=ax)
        cbar.set_label(display_var_name, fontsize=11, fontweight='bold')
        
        # Labels and title
        ax.set_xlabel('X coordinate (µm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y coordinate (µm)', fontsize=11, fontweight='bold')
        ax.set_title(f"Spatial Map: {display_var_name}\n(This map visualizes the variable used to compute Moran's I)", 
                     fontsize=12, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        self._fit_canvas(self.sq_autocorr_canvas, pad=0.95)
    
    def _run_sq_ripley(self):
        """Run Ripley analysis using core function."""
        if not self.spatial_graph_built:
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            mode = self.sq_ripley_mode_combo.currentText()  # F, G, or L
            max_dist = float(self.sq_ripley_r_max_spin.value())
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Get AnnData dict - filter to selected ROI if needed
            if roi_id is None:
                # Use all cached AnnData objects with graphs
                anndata_dict = {rid: adata for rid, adata in self.anndata_cache.items() 
                               if 'spatial_connectivities' in adata.obsp}
            else:
                # Use only selected ROI
                if roi_id not in self.anndata_cache:
                    QtWidgets.QMessageBox.warning(self, "No Data", f"No data found for ROI {roi_id}.")
                    return
                adata = self.anndata_cache[roi_id]
                if 'spatial_connectivities' not in adata.obsp:
                    QtWidgets.QMessageBox.warning(self, "No Graph", f"No spatial graph found for ROI {roi_id}.")
                    return
                anndata_dict = {roi_id: adata}
            
            if not anndata_dict:
                QtWidgets.QMessageBox.warning(self, "No Data", "No AnnData objects with spatial graphs found.")
                return
            
            def _ripley_task():
                return spatial_ripley(
                    anndata_dict=anndata_dict,
                    cluster_key=cluster_key,
                    mode=mode,
                    max_dist=max_dist,
                )

            results = run_blocking_task_with_progress(
                parent=self,
                window_title="Ripley Analysis",
                initial_message="Running Ripley analysis",
                detail_text="Estimating spatial dispersion/clustering statistics.",
                task=_ripley_task,
            )
            
            # Update cache with results
            self.anndata_cache.update(results)
            
            # Update status
            for roi_id in results.keys():
                if roi_id not in self.analysis_status:
                    self.analysis_status[roi_id] = {}
                self.analysis_status[roi_id]['ripley'] = True
            
            if results:
                # Check if aggregation is needed (multiple ROIs and aggregation selected)
                agg_method = self.sq_ripley_agg_combo.currentText().lower()
                should_aggregate = (roi_id is None and len(results) > 1 and agg_method in ['mean', 'sum'])
                
                if should_aggregate:
                    # Aggregate results across ROIs
                    plot_adata = self._aggregate_ripley_results(results, cluster_key, mode, agg_method)
                else:
                    # Use first ROI for plotting
                    plot_adata = list(results.values())[0]

                if roi_id is None:
                    self.aggregated_results['ripley'] = plot_adata
                else:
                    self.aggregated_results.pop('ripley', None)
                
                QtWidgets.QMessageBox.information(self, "Ripley Complete", 
                    f"Ripley analysis completed for {len(results)} ROI(s).")
                
                # Plot results
                self._plot_sq_ripley(plot_adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
                self.sq_ripley_export_btn.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(self, "No Results", 
                    "Ripley analysis completed but no results to plot. "
                    "This can happen when clusters are too small.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running Ripley: {str(e)}")
    
    def _aggregate_ripley_results(self, results: Dict[str, 'ad.AnnData'], cluster_key: str, mode: str, agg_method: str) -> 'ad.AnnData':
        """Aggregate Ripley results across multiple ROIs."""
        from squidpy._constants._constants import RipleyStat
        
        mode_enum = RipleyStat(mode)
        stat_key = f"{mode_enum.s}_stat"
        sims_stat_key = "sims_stat"
        
        # Collect all stat dataframes
        stat_dfs = []
        sims_stat_dfs = []
        pvalue_dfs = []
        all_bins = set()
        all_clusters = set()
        
        for roi_id, adata in results.items():
            if 'ripley' not in adata.uns:
                continue
            
            ripley_data = adata.uns['ripley']
            if stat_key not in ripley_data:
                continue
            
            stat_df = ripley_data[stat_key]
            stat_dfs.append(stat_df)
            
            # Collect unique clusters and bins
            if cluster_key in stat_df.columns:
                all_clusters.update(stat_df[cluster_key].unique())
            if 'bins' in stat_df.columns:
                all_bins.update(pd.to_numeric(stat_df['bins'], errors='coerce').dropna().tolist())
            
            # Collect simulation stats if available
            if sims_stat_key in ripley_data:
                sims_stat_dfs.append(ripley_data[sims_stat_key])

            raw_pvalues = ripley_data.get('pvalues')
            if raw_pvalues is not None:
                pvalue_matrix = np.asarray(raw_pvalues, dtype=float)
                if pvalue_matrix.ndim == 2:
                    if hasattr(adata, 'obs') and cluster_key in adata.obs.columns and hasattr(adata.obs[cluster_key], 'cat'):
                        cluster_categories = list(adata.obs[cluster_key].cat.categories)
                    elif cluster_key in stat_df.columns:
                        cluster_categories = list(pd.Series(stat_df[cluster_key]).dropna().unique())
                    else:
                        cluster_categories = [f'Cluster_{idx}' for idx in range(pvalue_matrix.shape[0])]

                    bins = ripley_data.get('bins')
                    if bins is None:
                        bins = np.sort(pd.to_numeric(stat_df['bins'], errors='coerce').dropna().unique())
                    else:
                        bins = np.asarray(bins, dtype=float).reshape(-1)

                    pvalue_rows = []
                    for cluster_idx, cluster_id in enumerate(cluster_categories[:pvalue_matrix.shape[0]]):
                        for bin_idx, distance_um in enumerate(bins[:pvalue_matrix.shape[1]]):
                            pvalue_rows.append(
                                {
                                    cluster_key: cluster_id,
                                    'bins': float(distance_um),
                                    'p_value': float(pvalue_matrix[cluster_idx, bin_idx]) if np.isfinite(pvalue_matrix[cluster_idx, bin_idx]) else np.nan,
                                }
                            )
                    if pvalue_rows:
                        pvalue_dfs.append(pd.DataFrame(pvalue_rows))
        
        if not stat_dfs:
            # Fallback: return first result
            return list(results.values())[0]
        
        # Aggregate stat dataframes
        # Combine all dataframes
        combined_stat_df = pd.concat(stat_dfs, ignore_index=True)
        
        # Group by bins and cluster_key, then aggregate stats
        grouped = combined_stat_df.groupby(['bins', cluster_key], as_index=False)
        if agg_method == 'mean':
            aggregated_stat_df = grouped.agg({'stats': 'mean'}).reset_index(drop=True)
        else:  # sum
            aggregated_stat_df = grouped.agg({'stats': 'sum'}).reset_index(drop=True)
        
        # Aggregate simulation stats if available
        aggregated_sims_stat = None
        if sims_stat_dfs:
            combined_sims_df = pd.concat(sims_stat_dfs, ignore_index=True)
            grouped_sims = combined_sims_df.groupby('bins', as_index=False)
            if agg_method == 'mean':
                aggregated_sims_stat = grouped_sims.agg({'stats': 'mean'}).reset_index(drop=True)
            else:  # sum
                aggregated_sims_stat = grouped_sims.agg({'stats': 'sum'}).reset_index(drop=True)

        aggregated_pvalues = None
        aggregated_pvalues_adjusted = None
        ordered_clusters = sort_cluster_values(
            all_clusters,
            annotation_map=self.cluster_annotation_map,
            canonical=False,
        ) if all_clusters else []
        ordered_bins = np.asarray(sorted(all_bins), dtype=float) if all_bins else np.array([], dtype=float)

        if pvalue_dfs and ordered_clusters and ordered_bins.size > 0:
            combined_pvalue_df = pd.concat(pvalue_dfs, ignore_index=True)
            grouped_pvalues = combined_pvalue_df.groupby([cluster_key, 'bins'])['p_value']
            combined_rows = []
            for (cluster_id, bin_value), series in grouped_pvalues:
                combined_rows.append(
                    {
                        cluster_key: cluster_id,
                        'bins': float(bin_value),
                        'p_value': combine_pvalues_fisher(series.to_numpy(dtype=float, copy=False)),
                    }
                )

            if combined_rows:
                combined_pvalue_df = pd.DataFrame(combined_rows)
                cluster_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(ordered_clusters)}
                bin_to_idx = {float(bin_value): idx for idx, bin_value in enumerate(ordered_bins)}
                aggregated_pvalues = np.full((len(ordered_clusters), len(ordered_bins)), np.nan, dtype=float)
                for _, row in combined_pvalue_df.iterrows():
                    cluster_idx = cluster_to_idx.get(row[cluster_key])
                    bin_idx = bin_to_idx.get(float(row['bins']))
                    if cluster_idx is None or bin_idx is None:
                        continue
                    aggregated_pvalues[cluster_idx, bin_idx] = float(row['p_value'])
                aggregated_pvalues_adjusted = benjamini_hochberg_adjust_matrix(aggregated_pvalues)
        
        # Create aggregated result structure
        aggregated_ripley_data = {
            stat_key: aggregated_stat_df
        }
        if aggregated_sims_stat is not None:
            aggregated_ripley_data[sims_stat_key] = aggregated_sims_stat
        if ordered_bins.size > 0:
            aggregated_ripley_data['bins'] = ordered_bins
        if aggregated_pvalues is not None:
            aggregated_ripley_data['pvalues'] = aggregated_pvalues
            aggregated_ripley_data['pvalues_fdr_bh'] = aggregated_pvalues_adjusted
        
        # Create a new AnnData object with aggregated results
        # Use the first ROI's structure as template
        first_adata = list(results.values())[0]
        aggregated_adata = first_adata.copy()
        aggregated_adata.uns['ripley'] = aggregated_ripley_data
        if ordered_clusters and hasattr(aggregated_adata, 'obs') and cluster_key in aggregated_adata.obs.columns:
            aggregated_adata.obs[cluster_key] = pd.Categorical(
                aggregated_adata.obs[cluster_key],
                categories=ordered_clusters,
                ordered=True,
            )
        
        
        return aggregated_adata
    
    def _plot_sq_ripley(self, adata: 'ad.AnnData', cluster_key: str):
        """Plot Ripley results using squidpy's plotting function."""
        if adata is None:
            self.sq_ripley_canvas.figure.clear()
            ax = self.sq_ripley_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_ripley_canvas.draw()
            return
        
        
        # Check for ripley key or alternatives
        ripley_key = None
        if 'ripley' in adata.uns:
            ripley_key = 'ripley'
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    ripley_key = key
                    break
        
        if ripley_key is None:
            self.sq_ripley_canvas.figure.clear()
            ax = self.sq_ripley_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No Ripley data found.\nPlease run Ripley analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_ripley_canvas.draw()
            return
        
        # Use manual plotting (squidpy's plotting functions don't work well with custom canvases)
        # sq.pl.ripley() doesn't support custom axes properly
        mode = self.sq_ripley_mode_combo.currentText()
        self.sq_ripley_canvas.figure.clear()
        ax = self.sq_ripley_canvas.figure.add_subplot(111)
        ripley_data = adata.uns[ripley_key]
        
        
        # Plot using squidpy's exact approach
        # Based on sq.pl.ripley source code
        from squidpy._constants._constants import RipleyStat
        
        res = ripley_data
        mode_enum = RipleyStat(mode)
        
        # Get categories and palette like squidpy does
        if cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories_list = list(adata.obs[cluster_key].cat.categories)
            else:
                categories_list = sort_cluster_values(
                    adata.obs[cluster_key].unique(),
                    annotation_map=self.cluster_annotation_map,
                    canonical=False,
                )
        else:
            categories_list = []
        
        # Get palette using tab20/tab60 for consistent cluster coloring
        # Use _get_vivid_colors which provides tab20, tab20b, tab20c for up to 60 clusters
        n_categories = len(categories_list)
        if n_categories > 0:
            # Get colors using tab20/tab60 scheme
            color_array = _get_vivid_colors(n_categories)
            # Convert to dict mapping category to color (RGBA tuple)
            palette = {cat: tuple(color_array[i]) for i, cat in enumerate(categories_list)}
        else:
            palette = {}
        
        # Get the statistic DataFrame
        stat_key = f"{mode_enum.s}_stat"
        if stat_key not in res:
            ax.text(0.5, 0.5, f'No {stat_key} found in Ripley data.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_ripley_canvas.draw()
            return
        
        stat_df = res[stat_key]
        
        # Create mapping from cluster IDs to display names
        cluster_display_map = {cat: self._get_cluster_display_name(cat) for cat in categories_list}
        
        # Draw line-only curves so the GUI view stays clean and export-ready.
        if cluster_key in stat_df.columns:
            for cluster_id in categories_list:
                cluster_df = stat_df[stat_df[cluster_key] == cluster_id].copy()
                if cluster_df.empty:
                    continue
                cluster_df = cluster_df.sort_values('bins')
                ax.plot(
                    cluster_df['bins'].to_numpy(dtype=float),
                    cluster_df['stats'].to_numpy(dtype=float),
                    linewidth=2.0,
                    color=palette.get(cluster_id, '#444444'),
                    label=cluster_display_map.get(cluster_id, str(cluster_id)),
                )

        # Keep simulations as a dashed reference line instead of an opaque interval band.
        sims_key = "sims_stat"
        if sims_key not in res:
            sims_key = f"sims_{stat_key}"
        if sims_key in res:
            sims_df = res[sims_key]
            if isinstance(sims_df, pd.DataFrame) and {'bins', 'stats'}.issubset(sims_df.columns):
                sims_summary = (
                    sims_df.groupby('bins', as_index=False)['stats']
                    .mean()
                    .sort_values('bins')
                )
                if not sims_summary.empty:
                    ax.plot(
                        sims_summary['bins'].to_numpy(dtype=float),
                        sims_summary['stats'].to_numpy(dtype=float),
                        linestyle='--',
                        linewidth=1.4,
                        color='gray',
                        alpha=0.9,
                        label='Simulation mean',
                    )

        # Set legend like squidpy (labels already have display names from stat_df_display)
        legend_kwargs = {"loc": "center left", "bbox_to_anchor": (1, 0.5)}
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            seen = set()
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, labels):
                if label in seen:
                    continue
                seen.add(label)
                unique_handles.append(handle)
                unique_labels.append(label)
            ax.legend(unique_handles, unique_labels, **legend_kwargs)

        # Set labels like squidpy
        ax.set_ylabel("value")
        ax.set_title(f"Ripley's {mode_enum.s}")
        
        self._fit_canvas(self.sq_ripley_canvas, rect=[0.0, 0.0, 0.84, 1.0], pad=0.95)
    
    def _save_sq_ripley_plot(self):
        """Save the Ripley plot."""
        if save_figure_with_options(self.sq_ripley_canvas.figure, "squidpy_ripley.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def _sanitize_export_component(self, value: Any) -> str:
        """Return a filesystem-friendly name fragment for exported CSVs."""
        text = str(value).strip()
        if not text:
            return "results"
        sanitized = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
        sanitized = sanitized.strip("_")
        return sanitized or "results"

    def _export_table_to_csv(
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

    def _find_uns_key(self, adata: Any, preferred_keys: List[str], *, contains: Optional[List[str]] = None) -> Optional[str]:
        """Return the first matching `uns` key from an AnnData-like object."""
        if adata is None or not hasattr(adata, 'uns'):
            return None

        uns_keys = list(getattr(adata, 'uns', {}).keys())
        for key in preferred_keys:
            if key in uns_keys:
                return key

        contains = contains or []
        for key in uns_keys:
            lowered = key.lower()
            if all(token in lowered for token in contains):
                return key
        return None

    def _get_obs_categories(self, adata: Any, cluster_key: str, expected_size: Optional[int] = None) -> List[Any]:
        """Return cluster categories aligned to the expected matrix width when possible."""
        categories: List[Any] = []
        if adata is not None and hasattr(adata, 'obs') and cluster_key in adata.obs.columns:
            series = adata.obs[cluster_key]
            if hasattr(series, 'cat'):
                categories = list(series.cat.categories)
            else:
                categories = list(
                    sort_cluster_values(
                        pd.Series(series).dropna().unique(),
                        annotation_map=self.cluster_annotation_map,
                        canonical=False,
                    )
                )

        if expected_size is None:
            return categories
        if len(categories) >= expected_size:
            return list(categories[:expected_size])

        padded = list(categories)
        padded.extend(f"Cluster_{idx}" for idx in range(len(padded), expected_size))
        return padded

    def _get_matrix_from_mapping(self, mapping: Dict[str, Any], *keys: str) -> Optional[np.ndarray]:
        """Return the first 2D matrix stored under any of the requested keys."""
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, np.ndarray) and value.ndim == 2:
                return np.asarray(value, dtype=float)
        return None

    def _get_preferred_moran_pvalue_columns(self, columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Pick the best available Moran p-value and adjusted-p-value columns."""
        base_priority = ['pval_sim', 'pval_z_sim', 'pval_norm', 'pval']
        adjusted_priority = [
            'pval_sim_fdr_bh',
            'pval_z_sim_fdr_bh',
            'pval_norm_fdr_bh',
            'pval_fdr_bh',
        ]

        base_column = next((column for column in base_priority if column in columns), None)
        if base_column is not None:
            preferred_adjusted = f'{base_column}_fdr_bh'
            if preferred_adjusted in columns:
                return base_column, preferred_adjusted

        adjusted_column = next((column for column in adjusted_priority if column in columns), None)
        return base_column, adjusted_column

    def _build_sq_nhood_export_df(
        self,
        adata: Any,
        *,
        roi_label: str,
        cluster_key: str,
        aggregation_method: Optional[str] = None,
    ) -> pd.DataFrame:
        """Flatten neighborhood enrichment matrices into a CSV-friendly table."""
        uns_key = self._find_uns_key(
            adata,
            ['nhood_enrichment', f'{cluster_key}_nhood_enrichment', 'nhood_enrichment_zscore'],
            contains=['nhood'],
        )
        if uns_key is None:
            return pd.DataFrame()

        enrichment_data = adata.uns[uns_key]
        matrices: Dict[str, np.ndarray] = {}
        if isinstance(enrichment_data, dict):
            for key, value in enrichment_data.items():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    matrices[key] = np.asarray(value, dtype=float)
        elif isinstance(enrichment_data, np.ndarray) and enrichment_data.ndim == 2:
            matrices['zscore'] = np.asarray(enrichment_data, dtype=float)

        if not matrices:
            return pd.DataFrame()

        z_matrix = matrices.get('zscore')
        if z_matrix is None:
            z_matrix = matrices.get('stat')
        if z_matrix is None:
            z_matrix = next(iter(matrices.values()))
        n_rows, n_cols = z_matrix.shape
        categories = self._get_obs_categories(adata, cluster_key, expected_size=max(n_rows, n_cols))
        row_categories = categories[:n_rows]
        col_categories = categories[:n_cols]
        count_matrix = matrices.get('count')
        stat_matrix = matrices.get('stat')
        pvalue_matrix = self._get_matrix_from_mapping(
            matrices,
            'pvalue',
            'p_value',
            'pval',
        )
        adjusted_pvalue_matrix = self._get_matrix_from_mapping(
            matrices,
            'pvalue_fdr_bh',
            'pvalue_adjusted',
            'p_value_adjusted',
            'pval_fdr_bh',
        )
        significant_counts = getattr(adata, '_significant_counts', None)

        rows = []
        for row_idx, cluster_a in enumerate(row_categories):
            for col_idx, cluster_b in enumerate(col_categories):
                row = {
                    'roi_id': roi_label,
                    'cluster_A': cluster_a,
                    'cluster_A_label': self._get_cluster_display_name(cluster_a),
                    'cluster_B': cluster_b,
                    'cluster_B_label': self._get_cluster_display_name(cluster_b),
                    'z_score': float(z_matrix[row_idx, col_idx]) if np.isfinite(z_matrix[row_idx, col_idx]) else np.nan,
                }
                if count_matrix is not None and count_matrix.shape == z_matrix.shape:
                    row['count'] = float(count_matrix[row_idx, col_idx]) if np.isfinite(count_matrix[row_idx, col_idx]) else np.nan
                if stat_matrix is not None and stat_matrix.shape == z_matrix.shape:
                    row['stat'] = float(stat_matrix[row_idx, col_idx]) if np.isfinite(stat_matrix[row_idx, col_idx]) else np.nan
                if pvalue_matrix is not None and pvalue_matrix.shape == z_matrix.shape:
                    row['p_value'] = float(pvalue_matrix[row_idx, col_idx]) if np.isfinite(pvalue_matrix[row_idx, col_idx]) else np.nan
                if adjusted_pvalue_matrix is not None and adjusted_pvalue_matrix.shape == z_matrix.shape:
                    row['p_value_adjusted'] = float(adjusted_pvalue_matrix[row_idx, col_idx]) if np.isfinite(adjusted_pvalue_matrix[row_idx, col_idx]) else np.nan
                if isinstance(significant_counts, np.ndarray) and significant_counts.shape == z_matrix.shape:
                    row['significant_roi_count'] = int(significant_counts[row_idx, col_idx])
                rows.append(row)

        export_df = pd.DataFrame(rows)
        if export_df.empty:
            return export_df

        if 'p_value' not in export_df.columns:
            export_df['p_value'] = np.nan
            finite_mask = np.isfinite(export_df['z_score'].to_numpy(dtype=float, copy=False))
            if np.any(finite_mask):
                export_df.loc[finite_mask, 'p_value'] = 2.0 * stats.norm.sf(
                    np.abs(export_df.loc[finite_mask, 'z_score'].to_numpy(dtype=float, copy=False))
                )
            export_df['p_value_source'] = 'z_score_normal_approximation'
        else:
            export_df['p_value_source'] = 'permutation'

        if 'p_value_adjusted' not in export_df.columns:
            export_df['p_value_adjusted'] = benjamini_hochberg_adjust(
                export_df['p_value'].to_numpy(dtype=float, copy=False)
            )
        if aggregation_method:
            export_df['aggregation_method'] = aggregation_method
        return export_df

    def _build_sq_cooccur_export_df(
        self,
        adata: Any,
        *,
        roi_label: str,
        cluster_key: str,
    ) -> pd.DataFrame:
        """Flatten co-occurrence arrays into a long-form table."""
        cooccur_key = self._find_uns_key(adata, ['co_occurrence'], contains=['co', 'occur'])
        if cooccur_key is None:
            return pd.DataFrame()

        cooccur_data = adata.uns[cooccur_key]
        if not isinstance(cooccur_data, dict) or 'occ' not in cooccur_data:
            return pd.DataFrame()

        occ = np.asarray(cooccur_data['occ'], dtype=float)
        if occ.ndim != 3:
            return pd.DataFrame()

        distances = self._normalize_cooccur_distances(
            cooccur_data.get('interval', cooccur_data.get('distances')),
            expected_size=occ.shape[2],
            fallback=self.sq_cooccur_interval,
        )

        categories = self._get_obs_categories(adata, cluster_key, expected_size=occ.shape[0])

        rows = []
        for source_idx, source_cluster in enumerate(categories):
            for target_idx, target_cluster in enumerate(categories):
                for dist_idx, distance_um in enumerate(distances[:occ.shape[2]]):
                    rows.append(
                        {
                            'roi_id': roi_label,
                            'source_cluster': source_cluster,
                            'source_cluster_label': self._get_cluster_display_name(source_cluster),
                            'target_cluster': target_cluster,
                            'target_cluster_label': self._get_cluster_display_name(target_cluster),
                            'distance_um': float(distance_um),
                            'co_occurrence_score': float(occ[source_idx, target_idx, dist_idx]),
                        }
                    )

        return pd.DataFrame(rows)

    def _build_sq_autocorr_export_df(
        self,
        adata: Any,
        *,
        roi_label: str,
        aggregation_method: Optional[str] = None,
    ) -> pd.DataFrame:
        """Flatten Moran autocorrelation results into a CSV-friendly table."""
        moran_key = self._find_uns_key(adata, ['moranI'], contains=['moran'])
        if moran_key is None:
            return pd.DataFrame()

        moran_data = adata.uns[moran_key]
        if isinstance(moran_data, pd.DataFrame):
            export_df = moran_data.copy()
            if 'var_names' not in export_df.columns:
                export_df.insert(0, 'var_names', export_df.index.astype(str))
            export_df = export_df.reset_index(drop=True)
        elif isinstance(moran_data, dict):
            I_values, p_values, gene_names = self._extract_moran_arrays(moran_data)
            i_len = len(I_values) if I_values is not None else 0
            p_len = len(p_values) if p_values is not None else 0
            n_rows = int(max(i_len, p_len, len(gene_names)))
            if n_rows == 0:
                return pd.DataFrame()

            if gene_names.size == 0:
                gene_names = np.asarray([f"feature_{idx}" for idx in range(n_rows)], dtype=object)

            data: Dict[str, Any] = {'var_names': gene_names}
            for key, value in moran_data.items():
                if key == 'var_names':
                    continue
                if np.isscalar(value):
                    data[key] = np.repeat(value, n_rows)
                    continue
                if hasattr(value, 'toarray'):
                    value = value.toarray()
                array = np.asarray(value).reshape(-1)
                if array.size == n_rows:
                    data[key] = array

            if 'I' not in data and I_values is not None and len(I_values) == n_rows:
                data['I'] = I_values
            if 'pval_norm' not in data and p_values is not None and len(p_values) == n_rows:
                data['pval_norm'] = p_values
            export_df = pd.DataFrame(data)
        else:
            return pd.DataFrame()

        if export_df.empty:
            return export_df

        export_df['roi_id'] = roi_label
        export_df['feature'] = export_df['var_names'].astype(str)
        export_df['feature_label'] = export_df['feature'].map(self._get_feature_display_name)

        moran_column = 'I' if 'I' in export_df.columns else 'moranI' if 'moranI' in export_df.columns else None
        if moran_column is not None:
            export_df['moran_i'] = pd.to_numeric(export_df[moran_column], errors='coerce')

        p_column, p_adjusted_column = self._get_preferred_moran_pvalue_columns(list(export_df.columns))
        if p_column is not None:
            export_df['p_value'] = pd.to_numeric(export_df[p_column], errors='coerce')
            export_df['p_value_source'] = p_column
            if p_adjusted_column is not None:
                export_df['p_value_adjusted'] = pd.to_numeric(export_df[p_adjusted_column], errors='coerce')
            else:
                export_df['p_value_adjusted'] = benjamini_hochberg_adjust(
                    export_df['p_value'].to_numpy(dtype=float, copy=False)
                )
        else:
            export_df['p_value'] = np.nan
            export_df['p_value_adjusted'] = np.nan

        if aggregation_method:
            export_df['aggregation_method'] = aggregation_method

        sort_columns = [column for column in ['p_value', 'moran_i'] if column in export_df.columns]
        if sort_columns:
            ascending = [True if column == 'p_value' else False for column in sort_columns]
            export_df = export_df.sort_values(sort_columns, ascending=ascending, na_position='last').reset_index(drop=True)
        return export_df

    def _build_sq_ripley_export_df(
        self,
        adata: Any,
        *,
        roi_label: str,
        cluster_key: str,
        aggregation_method: Optional[str] = None,
    ) -> pd.DataFrame:
        """Flatten Ripley results and simulation summaries into one table."""
        ripley_key = self._find_uns_key(adata, ['ripley'], contains=['ripley'])
        if ripley_key is None:
            return pd.DataFrame()

        ripley_data = adata.uns[ripley_key]
        if not isinstance(ripley_data, dict):
            return pd.DataFrame()

        mode = self.sq_ripley_mode_combo.currentText()
        stat_key = f"{mode}_stat"
        if stat_key not in ripley_data or not isinstance(ripley_data[stat_key], pd.DataFrame):
            return pd.DataFrame()

        export_df = ripley_data[stat_key].copy().reset_index(drop=True)
        if export_df.empty:
            return export_df

        export_df['roi_id'] = roi_label
        export_df['mode'] = mode
        export_df['distance_um'] = pd.to_numeric(export_df.get('bins'), errors='coerce')
        export_df['stat_value'] = pd.to_numeric(export_df.get('stats'), errors='coerce')

        cluster_column = cluster_key if cluster_key in export_df.columns else 'cluster' if 'cluster' in export_df.columns else None
        if cluster_column is not None:
            export_df['cluster'] = export_df[cluster_column]
            export_df['cluster_label'] = export_df['cluster'].map(self._get_cluster_display_name)

        sims_key = 'sims_stat' if 'sims_stat' in ripley_data else f'sims_{stat_key}'
        if sims_key in ripley_data and isinstance(ripley_data[sims_key], pd.DataFrame):
            sims_summary = (
                ripley_data[sims_key]
                .groupby('bins', as_index=False)
                .agg(
                    simulation_mean=('stats', 'mean'),
                    simulation_std=('stats', 'std'),
                    simulation_count=('stats', 'count'),
                )
                .rename(columns={'bins': 'distance_um'})
            )
            export_df = export_df.merge(sims_summary, on='distance_um', how='left')

        pvalue_matrix = None
        if 'pvalues' in ripley_data:
            raw_pvalues = np.asarray(ripley_data['pvalues'], dtype=float)
            if raw_pvalues.ndim == 2:
                pvalue_matrix = raw_pvalues
        adjusted_pvalue_matrix = None
        if 'pvalues_fdr_bh' in ripley_data:
            raw_adjusted = np.asarray(ripley_data['pvalues_fdr_bh'], dtype=float)
            if raw_adjusted.ndim == 2:
                adjusted_pvalue_matrix = raw_adjusted
        elif pvalue_matrix is not None:
            adjusted_pvalue_matrix = benjamini_hochberg_adjust_matrix(pvalue_matrix)

        if pvalue_matrix is not None:
            category_labels = self._get_obs_categories(adata, cluster_key, expected_size=pvalue_matrix.shape[0])
            bin_values = ripley_data.get('bins')
            if bin_values is None:
                bin_values = np.sort(export_df['distance_um'].dropna().unique())
            else:
                bin_values = np.asarray(bin_values, dtype=float).reshape(-1)

            pvalue_rows = []
            for cluster_idx, cluster_id in enumerate(category_labels[:pvalue_matrix.shape[0]]):
                for bin_idx, distance_um in enumerate(bin_values[:pvalue_matrix.shape[1]]):
                    pvalue_row = {
                        'cluster': cluster_id,
                        'distance_um': float(distance_um),
                        'p_value': float(pvalue_matrix[cluster_idx, bin_idx]) if np.isfinite(pvalue_matrix[cluster_idx, bin_idx]) else np.nan,
                    }
                    if adjusted_pvalue_matrix is not None and adjusted_pvalue_matrix.shape == pvalue_matrix.shape:
                        pvalue_row['p_value_adjusted'] = (
                            float(adjusted_pvalue_matrix[cluster_idx, bin_idx])
                            if np.isfinite(adjusted_pvalue_matrix[cluster_idx, bin_idx])
                            else np.nan
                        )
                    pvalue_rows.append(pvalue_row)

            if pvalue_rows:
                pvalue_df = pd.DataFrame(pvalue_rows)
                export_df = export_df.merge(pvalue_df, on=['cluster', 'distance_um'], how='left')
                export_df['p_value_source'] = 'simulation'

        if aggregation_method:
            export_df['aggregation_method'] = aggregation_method

        preferred_columns = [
            'roi_id',
            'mode',
            'distance_um',
            'cluster',
            'cluster_label',
            'stat_value',
            'p_value',
            'p_value_adjusted',
            'simulation_mean',
            'simulation_std',
            'simulation_count',
        ]
        remaining_columns = [column for column in export_df.columns if column not in preferred_columns]
        ordered_columns = [column for column in preferred_columns if column in export_df.columns] + remaining_columns
        return export_df.loc[:, ordered_columns]

    def _export_sq_nhood_results(self):
        """Export neighborhood enrichment results to CSV."""
        roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
        cluster_key = self.sq_nhood_cluster_combo.currentText()
        aggregation_method = self.sq_nhood_agg_combo.currentText().lower() if roi_id is None else None

        if roi_id is None:
            adata = self.aggregated_results.get('nhood_enrichment')
            roi_label = "All ROIs"
        else:
            adata = self.anndata_cache.get(roi_id)
            roi_label = str(roi_id)

        export_df = self._build_sq_nhood_export_df(
            adata,
            roi_label=roi_label,
            cluster_key=cluster_key,
            aggregation_method=aggregation_method,
        )
        if export_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Results", "No neighborhood enrichment results are available to export.")
            return

        filename = f"squidpy_nhood_enrichment_{self._sanitize_export_component(roi_label)}.csv"
        self._export_table_to_csv(
            export_df,
            title="Export Neighborhood Enrichment Results",
            default_filename=filename,
            success_message="Neighborhood enrichment results exported to:",
        )

    def _export_sq_cooccur_results(self):
        """Export co-occurrence results to CSV."""
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        cluster_key = self.sq_cooccur_cluster_combo.currentText()

        if roi_id is None:
            export_frames = []
            for current_roi_id, adata in self.anndata_cache.items():
                export_df = self._build_sq_cooccur_export_df(
                    adata,
                    roi_label=str(current_roi_id),
                    cluster_key=cluster_key,
                )
                if not export_df.empty:
                    export_frames.append(export_df)
            export_df = pd.concat(export_frames, ignore_index=True, sort=False) if export_frames else pd.DataFrame()
            roi_label = "all_rois"
        else:
            export_df = self._build_sq_cooccur_export_df(
                self.anndata_cache.get(roi_id),
                roi_label=str(roi_id),
                cluster_key=cluster_key,
            )
            roi_label = str(roi_id)

        if export_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Results", "No co-occurrence results are available to export.")
            return

        filename = f"squidpy_cooccurrence_{self._sanitize_export_component(roi_label)}.csv"
        self._export_table_to_csv(
            export_df,
            title="Export Co-occurrence Results",
            default_filename=filename,
            success_message="Co-occurrence results exported to:",
        )

    def _export_sq_autocorr_results(self):
        """Export spatial autocorrelation results to CSV."""
        roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
        aggregation_method = self.sq_autocorr_agg_combo.currentText().lower() if roi_id is None else None

        if roi_id is None:
            adata = self.aggregated_results.get('autocorrelation')
            roi_label = "All ROIs"
        else:
            adata = self.anndata_cache.get(roi_id)
            roi_label = str(roi_id)

        export_df = self._build_sq_autocorr_export_df(
            adata,
            roi_label=roi_label,
            aggregation_method=aggregation_method,
        )
        if export_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Results", "No spatial autocorrelation results are available to export.")
            return

        filename = f"squidpy_autocorrelation_{self._sanitize_export_component(roi_label)}.csv"
        self._export_table_to_csv(
            export_df,
            title="Export Spatial Autocorrelation Results",
            default_filename=filename,
            success_message="Spatial autocorrelation results exported to:",
        )

    def _export_sq_ripley_results(self):
        """Export Ripley analysis results to CSV."""
        roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
        cluster_key = self.sq_ripley_cluster_combo.currentText()
        aggregation_method = self.sq_ripley_agg_combo.currentText().lower() if roi_id is None else None

        if roi_id is None:
            adata = self.aggregated_results.get('ripley')
            roi_label = "All ROIs"
        else:
            adata = self.anndata_cache.get(roi_id)
            roi_label = str(roi_id)

        export_df = self._build_sq_ripley_export_df(
            adata,
            roi_label=roi_label,
            cluster_key=cluster_key,
            aggregation_method=aggregation_method,
        )
        if export_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Results", "No Ripley analysis results are available to export.")
            return

        filename = f"squidpy_ripley_{self._sanitize_export_component(roi_label)}_{self.sq_ripley_mode_combo.currentText()}.csv"
        self._export_table_to_csv(
            export_df,
            title="Export Ripley Results",
            default_filename=filename,
            success_message="Ripley analysis results exported to:",
        )
    
    def _export_to_anndata(self):
        """Export data to AnnData format using core function."""
        from PyQt5.QtWidgets import QFileDialog
        
        if not self.anndata_cache:
            QtWidgets.QMessageBox.warning(self, "No Data", 
                "No AnnData objects available. Please build spatial graphs first.")
            return
        
        # Ask user if they want combined or separate files
        reply = QtWidgets.QMessageBox.question(
            self,
            "Export Format",
            "Export as:\n\nYes = Combined file (all ROIs)\nNo = Separate files (one per ROI)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes
        )
        
        if reply == QtWidgets.QMessageBox.Cancel:
            return
        
        try:
            if reply == QtWidgets.QMessageBox.Yes:
                # Combined export
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export Combined AnnData", "combined_anndata.h5ad", 
                    "AnnData Files (*.h5ad);;All Files (*)"
                )
                if not file_path:
                    return
                
                # Use core function
                export_anndata(self.anndata_cache, file_path, combined=True)
                QtWidgets.QMessageBox.information(self, "Export Complete", 
                    f"Combined AnnData exported to:\n{file_path}")
            else:
                # Separate files
                export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
                if not export_dir:
                    return
                
                # Use core function
                export_anndata(self.anndata_cache, export_dir, combined=False)
                QtWidgets.QMessageBox.information(self, "Export Complete", 
                    f"Exported {len(self.anndata_cache)} AnnData file(s) to:\n{export_dir}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
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
    
    def reset_analysis_state(self):
        """Reset all analysis state - clear results and allow restart."""
        # Clear all caches
        self.anndata_cache = {}
        self.processed_rois = {}
        self.analysis_status = {}
        self.aggregated_results = {}
        
        # Reset analysis flags
        self.spatial_graph_built = False
        
        # Clear all canvas figures
        canvas_names = [
            'sq_nhood_canvas', 'sq_cooccur_canvas', 'sq_autocorr_canvas', 'sq_ripley_canvas'
        ]
        for canvas_name in canvas_names:
            if hasattr(self, canvas_name):
                canvas = getattr(self, canvas_name)
                canvas.figure.clear()
                canvas.draw()
        
        # Disable save buttons
        save_button_names = [
            'sq_nhood_save_btn',
            'sq_nhood_export_btn',
            'sq_cooccur_save_btn',
            'sq_cooccur_export_btn',
            'sq_autocorr_save_btn',
            'sq_autocorr_export_btn',
            'sq_ripley_save_btn',
            'sq_ripley_export_btn',
        ]
        for btn_name in save_button_names:
            if hasattr(self, btn_name):
                btn = getattr(self, btn_name)
                btn.setEnabled(False)
        
        # Update status labels
        if hasattr(self, 'graph_status_label'):
            self.graph_status_label.setText("Graph not created")
            self.graph_status_label.setStyleSheet("")
        
        # Update button states
        self._update_button_states()
    
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
            self.cluster_annotation_map = normalize_cluster_annotation_map(label_source.cluster_annotation_map or {})
        if label_source is not None and hasattr(label_source, 'feature_label_map'):
            self.feature_label_map = dict(label_source.feature_label_map or {})
        self._apply_cluster_annotations_to_dataframes()

        # Clear AnnData cache since dataframe changed
        self.anndata_cache = {}
        
        # Refresh ROI combo boxes and other UI elements that depend on dataframe
        self._populate_roi_combo()
        self._update_autocorr_var_combo()

    def apply_feature_set_preference(self, feature_set_key: Optional[str]):
        """Apply a shared feature-set preference without forcing unnecessary resets."""
        target_text = self._feature_set_text_for_key(feature_set_key)
        if hasattr(self, 'feature_set_combo'):
            if self.feature_set_combo.currentText() != target_text:
                self.feature_set_combo.setCurrentText(target_text)
                return

        self._update_active_feature_dataframe()
    
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
        """Handle dialog closing."""
        event.accept()
