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
os.environ.setdefault('DASK_DATAFRAME__QUERY_PLANNING', 'False')

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.spatial_analysis import (
    SourceFileFilterDialog,
    dataframe_to_anndata,
    _HAVE_SQUIDPY,
    _HAVE_SPARSE,
    _get_vivid_colors
)

try:
    # Suppress FutureWarning about anndata.read_text deprecation and squidpy __version__ deprecation
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*read_text.*')
        warnings.filterwarnings('ignore', category=FutureWarning, message='.*__version__.*')
        import squidpy as sq
        import anndata as ad
except ImportError:
    sq = None
    ad = None

try:
    from scipy import sparse as sp
except Exception:
    sp = None


class AdvancedSpatialAnalysisDialog(QtWidgets.QDialog):
    """Advanced Spatial Analysis Dialog using squidpy for all analyses."""
    def __init__(self, feature_dataframe: pd.DataFrame, batch_corrected_dataframe=None, parent=None):
        if not _HAVE_SQUIDPY:
            raise RuntimeError("squidpy is required for AdvancedSpatialAnalysisDialog")
        
        super().__init__(parent)
        self.setWindowTitle("Advanced Spatial Analysis (Squidpy)")
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
        
        # Store AnnData objects per ROI
        self.anndata_cache: Dict[str, 'ad.AnnData'] = {}
        self.spatial_graph_built = False
        
        # Track which analyses have been run per ROI
        self.analysis_status: Dict[str, Dict[str, bool]] = {}  # {roi_id: {analysis_type: bool}}
        
        
        # Cluster annotation mapping
        self.cluster_annotation_map = {}
        
        # Source file filtering
        self.selected_source_files = set()
        self.available_source_files = set()
        
        # Track processed ROIs
        self.processed_rois: Dict[str, Dict[str, Any]] = {}  # {roi_id: {graph_built: bool, analyses: []}}
        
        # Store aggregated results for plotting
        self.aggregated_results: Dict[str, Any] = {}  # {analysis_type: aggregated_data}
        
        self._create_ui()
        
        if hasattr(self, 'source_file_status_label'):
            self._update_source_file_status_label()
    
    def _get_roi_column(self):
        """Get the appropriate ROI column name."""
        if self.feature_dataframe is not None and 'source_well' in self.feature_dataframe.columns:
            return 'source_well'
        return 'acquisition_id'
    
    def _get_filtered_dataframe(self):
        """Get the filtered dataframe based on selected source files."""
        df = self.feature_dataframe.copy()
        if ('source_file' in df.columns and 
            hasattr(self, 'selected_source_files') and 
            self.selected_source_files and 
            len(self.selected_source_files) > 0):
            df = df[df['source_file'].isin(self.selected_source_files)]
        return df
    
    def _get_or_create_anndata(self, roi_id: str) -> Optional['ad.AnnData']:
        """Get or create AnnData object for a specific ROI."""
        print(f"[DEBUG] _get_or_create_anndata: roi_id={roi_id}")
        if roi_id in self.anndata_cache:
            print(f"[DEBUG] Using cached AnnData for ROI {roi_id}")
            return self.anndata_cache[roi_id]
        
        roi_col = self._get_roi_column()
        print(f"[DEBUG] ROI column: {roi_col}")
        filtered_df = self._get_filtered_dataframe()
        print(f"[DEBUG] Filtered dataframe shape: {filtered_df.shape}")
        print(f"[DEBUG] Filtered dataframe columns: {list(filtered_df.columns)[:10]}...")
        
        adata = dataframe_to_anndata(
            filtered_df,
            roi_id=roi_id,
            roi_column=roi_col,
            parent=self.parent()
        )
        
        if adata is not None:
            print(f"[DEBUG] Created AnnData: shape={adata.shape}, var_names={list(adata.var_names)[:5]}...")
            # Ensure cluster columns are categorical (required by squidpy)
            for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
                if col in adata.obs.columns:
                    adata.obs[col] = adata.obs[col].astype('category')
                    print(f"[DEBUG] Set {col} as categorical, categories: {list(adata.obs[col].cat.categories)}")
            self.anndata_cache[roi_id] = adata
        else:
            print(f"[DEBUG] Failed to create AnnData for ROI {roi_id}")
        
        return adata
    
    def _create_ui(self):
        """Create the UI with squidpy-specific tabs."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Feature set selector
        if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
            feature_set_layout = QtWidgets.QHBoxLayout()
            feature_set_layout.addWidget(QtWidgets.QLabel("Feature Set:"))
            self.feature_set_combo = QtWidgets.QComboBox()
            self.feature_set_combo.addItem("Original Features")
            self.feature_set_combo.addItem("Batch-Corrected Features")
            self.feature_set_combo.setCurrentText("Batch-Corrected Features")
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
        action_row.addStretch(1)
        layout.addLayout(action_row)
        
        # Tabs for squidpy-specific analyses
        self.tabs = QtWidgets.QTabWidget()
        
        # Create squidpy-specific tabs
        self._create_squidpy_tabs()
        
        layout.addWidget(self.tabs, 1)
        
        # Wire signals
        self.export_anndata_btn.clicked.connect(self._export_to_anndata)
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
        sq_nhood_btn_layout.addWidget(self.sq_nhood_run_btn)
        sq_nhood_btn_layout.addWidget(self.sq_nhood_save_btn)
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
        sq_cooccur_btn_layout.addWidget(self.sq_cooccur_run_btn)
        sq_cooccur_btn_layout.addWidget(self.sq_cooccur_save_btn)
        sq_cooccur_btn_layout.addStretch()
        
        # Plot type selection
        sq_cooccur_plot_layout = QtWidgets.QHBoxLayout()
        sq_cooccur_plot_layout.addWidget(QtWidgets.QLabel("Plot type:"))
        self.sq_cooccur_plot_type_combo = QtWidgets.QComboBox()
        self.sq_cooccur_plot_type_combo.addItems(["Curves", "Heatmap"])
        self.sq_cooccur_plot_type_combo.setToolTip("Curves: Show co-occurrence across all distances. Heatmap: Show co-occurrence matrix at a selected distance.")
        sq_cooccur_plot_layout.addWidget(self.sq_cooccur_plot_type_combo)
        
        sq_cooccur_plot_layout.addWidget(QtWidgets.QLabel("Distance for heatmap:"))
        self.sq_cooccur_distance_combo = QtWidgets.QComboBox()
        self.sq_cooccur_distance_combo.setToolTip("Select a distance from the interval to display in the heatmap.")
        self.sq_cooccur_distance_combo.setEnabled(False)  # Enabled only when heatmap is selected
        sq_cooccur_plot_layout.addWidget(self.sq_cooccur_distance_combo)
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
        self.sq_cooccur_roi_combo.currentIndexChanged.connect(self._on_sq_cooccur_roi_changed)
        self.sq_cooccur_ref_cluster_combo.currentIndexChanged.connect(self._on_sq_cooccur_ref_cluster_changed)
        self.sq_cooccur_cluster_combo.currentIndexChanged.connect(self._on_sq_cooccur_cluster_column_changed)
        self.sq_cooccur_plot_type_combo.currentTextChanged.connect(self._on_sq_cooccur_plot_type_changed)
        self.sq_cooccur_distance_combo.currentIndexChanged.connect(self._on_sq_cooccur_distance_changed)
        
        # Store interval for distance selection
        self.sq_cooccur_interval = None
    
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
        sq_autocorr_btn_layout.addWidget(self.sq_autocorr_run_btn)
        sq_autocorr_btn_layout.addWidget(self.sq_autocorr_save_btn)
        sq_autocorr_btn_layout.addStretch()
        
        sq_autocorr_layout.addLayout(sq_autocorr_params)
        sq_autocorr_layout.addLayout(sq_autocorr_btn_layout)
        
        # Add navigation toolbar
        self.sq_autocorr_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        self.sq_autocorr_toolbar = NavigationToolbar(self.sq_autocorr_canvas, self)
        sq_autocorr_layout.addWidget(self.sq_autocorr_toolbar)
        sq_autocorr_layout.addWidget(self.sq_autocorr_canvas)
        self.tabs.addTab(self.sq_autocorr_tab, "Spatial Autocorrelation")
        
        self.sq_autocorr_run_btn.clicked.connect(self._run_sq_autocorrelation)
        self.sq_autocorr_save_btn.clicked.connect(self._save_sq_autocorr_plot)
        self.sq_autocorr_roi_combo.currentIndexChanged.connect(self._on_sq_autocorr_roi_changed)
        self.sq_autocorr_topk_spin.valueChanged.connect(self._on_sq_autocorr_topk_changed)
    
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
        sq_ripley_btn_layout.addWidget(self.sq_ripley_run_btn)
        sq_ripley_btn_layout.addWidget(self.sq_ripley_save_btn)
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
        """Load cluster annotations from parent."""
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, 'cluster_annotation_map'):
                self.cluster_annotation_map = parent.cluster_annotation_map.copy()
        except Exception:
            pass
    
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
        """Get display name for cluster with annotation if available."""
        if cluster_id in self.cluster_annotation_map:
            annotation = self.cluster_annotation_map[cluster_id]
            return f"{cluster_id} ({annotation})"
        return str(cluster_id)
    
    def _on_feature_set_changed(self):
        """Handle feature set change - invalidate cache and refresh."""
        # Update feature dataframe
        if self.feature_set_combo.currentText() == "Batch-Corrected Features":
            if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
                self.feature_dataframe = self.batch_corrected_dataframe.copy()
            else:
                QtWidgets.QMessageBox.warning(self, "No Batch-Corrected Data", 
                    "Batch-corrected features are not available. Using original features.")
                self.feature_set_combo.setCurrentText("Original Features")
                return
        else:
            self.feature_dataframe = self.original_feature_dataframe.copy()
        
        # Clear cache since data changed
        self.anndata_cache = {}
        self.processed_rois = {}
        self.analysis_status = {}
        self.aggregated_results = {}
        self.spatial_graph_built = False
        
        # Refresh UI
        self._populate_roi_combo()
        self._update_button_states()
        
        # Clear all plots
        for canvas_name in ['sq_nhood_canvas', 'sq_cooccur_canvas', 'sq_autocorr_canvas', 'sq_ripley_canvas']:
            if hasattr(self, canvas_name):
                canvas = getattr(self, canvas_name)
                canvas.figure.clear()
                canvas.draw()
        
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
                            print(f"[DEBUG] Error re-running enrichment for ROI {roi_id}: {e}")
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
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            # Check for co-occurrence data with alternative key names
            has_cooccur = False
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    has_cooccur = True
                    break
            if has_cooccur:
                # Update reference cluster combo
                self._update_cooccur_ref_cluster_combo(adata)
                self._plot_sq_cooccurrence(adata)
                self.sq_cooccur_save_btn.setEnabled(True)
    
    def _on_sq_cooccur_ref_cluster_changed(self):
        """Handle reference cluster change in co-occurrence tab."""
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            # Check for co-occurrence data with alternative key names
            has_cooccur = False
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    has_cooccur = True
                    break
            if has_cooccur:
                self._plot_sq_cooccurrence(adata)
                self.sq_cooccur_save_btn.setEnabled(True)
    
    def _on_sq_cooccur_cluster_column_changed(self):
        """Handle cluster column change in co-occurrence tab."""
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            # Update reference cluster combo
            self._update_cooccur_ref_cluster_combo(adata)
            # Check for co-occurrence data with alternative key names
            has_cooccur = False
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    has_cooccur = True
                    break
            if has_cooccur:
                self._plot_sq_cooccurrence(adata)
                self.sq_cooccur_save_btn.setEnabled(True)
    
    def _on_sq_cooccur_plot_type_changed(self):
        """Handle plot type change in co-occurrence tab."""
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        # Enable/disable distance combo based on plot type
        self.sq_cooccur_distance_combo.setEnabled(plot_type == "Heatmap")
        
        # Replot if data is available
        roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
        if roi_id and roi_id in self.anndata_cache:
            adata = self.anndata_cache[roi_id]
            # Check for co-occurrence data
            has_cooccur = False
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    has_cooccur = True
                    break
            if has_cooccur:
                self._plot_sq_cooccurrence(adata)
    
    def _on_sq_cooccur_distance_changed(self):
        """Handle distance selection change for heatmap in co-occurrence tab."""
        # Replot if heatmap is selected and data is available
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        if plot_type == "Heatmap":
            roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
            if roi_id and roi_id in self.anndata_cache:
                adata = self.anndata_cache[roi_id]
                # Check for co-occurrence data
                has_cooccur = False
                for key in adata.uns.keys():
                    if 'co' in key.lower() and 'occur' in key.lower():
                        has_cooccur = True
                        break
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
                        print(f"[DEBUG] Error re-running enrichment for ROI {roi_id}: {e}")
    
    def _update_cooccur_ref_cluster_combo(self, adata: 'ad.AnnData'):
        """Update the reference cluster combo box with available clusters."""
        cluster_key = self.sq_cooccur_cluster_combo.currentText()
        self.sq_cooccur_ref_cluster_combo.clear()
        self.sq_cooccur_ref_cluster_combo.addItem("All clusters", None)
        
        if cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories = list(adata.obs[cluster_key].cat.categories)
            else:
                categories = sorted(adata.obs[cluster_key].unique())
            
            for cat in categories:
                self.sq_cooccur_ref_cluster_combo.addItem(
                    self._get_cluster_display_name(cat), cat
                )
    
    def _on_sq_autocorr_roi_changed(self):
        """Handle ROI change in autocorrelation tab."""
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
    
    def _on_sq_ripley_roi_changed(self):
        """Handle ROI change in Ripley tab."""
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
        """Create spatial graph for all ROIs - unified graph creation method."""
        print(f"[DEBUG] _create_spatial_graph: Starting")
        if not self._validate_data():
            print(f"[DEBUG] Data validation failed")
            return
        
        try:
            method = self.graph_method_combo.currentText()
            k = int(self.graph_k_spin.value()) if method == "kNN" else None
            radius = float(self.graph_radius_spin.value()) if method == "Radius" else None
            print(f"[DEBUG] Graph method: {method}, k={k}, radius={radius}")
            
            roi_col = self._get_roi_column()
            all_rois = self._get_all_rois()
            print(f"[DEBUG] Found {len(all_rois)} ROIs: {all_rois}")
            
            if not all_rois:
                print(f"[DEBUG] No ROIs found")
                QtWidgets.QMessageBox.warning(self, "No ROIs", "No ROIs found in the data.")
                return
            
            # Build graph for all ROIs
            success_count = 0
            for roi_id in all_rois:
                print(f"[DEBUG] Building graph for ROI: {roi_id}")
                adata = self._get_or_create_anndata(roi_id)
                if adata is None:
                    print(f"[DEBUG] Failed to get/create AnnData for ROI {roi_id}")
                    continue
                print(f"[DEBUG] AnnData shape: {adata.shape}, spatial coords shape: {adata.obsm['spatial'].shape}")
                coords = adata.obsm['spatial']
                print(f"[DEBUG] Spatial coords (µm) range X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
                      f"range Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
                
                try:
                    if method == "kNN":
                        # Use squidpy for kNN
                        # Coordinates are in micrometers, coord_type="generic" works with any units
                        print(f"[DEBUG] Building kNN graph with k={k}")
                        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=k, n_rings=1)
                        # Verify graph was created
                        if 'spatial_connectivities' in adata.obsp:
                            n_edges = adata.obsp['spatial_connectivities'].nnz // 2  # Divide by 2 for undirected
                            print(f"[DEBUG] kNN graph built successfully: {n_edges} edges")
                            success_count += 1
                        else:
                            print(f"[DEBUG] ERROR: kNN graph not found in obsp after creation")
                            continue
                    elif method == "Radius":
                        # Radius is in micrometers, coordinates are in micrometers
                        # So we can use radius directly without conversion
                        print(f"[DEBUG] Building Radius graph with radius={radius} µm")
                        sq.gr.spatial_neighbors(adata, coord_type="generic", radius=radius, n_rings=1)
                        # Verify graph was created
                        if 'spatial_connectivities' in adata.obsp:
                            n_edges = adata.obsp['spatial_connectivities'].nnz // 2  # Divide by 2 for undirected
                            print(f"[DEBUG] Radius graph built successfully: {n_edges} edges")
                            success_count += 1
                        else:
                            print(f"[DEBUG] ERROR: Radius graph not found in obsp after creation")
                            continue
                    elif method == "Delaunay":
                        # Delaunay triangulation - manual implementation
                        if sp is None:
                            QtWidgets.QMessageBox.warning(self, "Missing Dependency", 
                                "scipy.sparse is required for Delaunay triangulation.")
                            continue
                        
                        print(f"[DEBUG] Building Delaunay graph")
                        coords = adata.obsm['spatial']
                        tri = Delaunay(coords)
                        print(f"[DEBUG] Delaunay triangulation: {len(tri.simplices)} simplices")
                        
                        # Build connectivity matrix from Delaunay triangulation
                        n_cells = len(coords)
                        rows, cols = [], []
                        for simplex in tri.simplices:
                            # Each simplex has 3 vertices, create edges between all pairs
                            for i in range(3):
                                for j in range(i + 1, 3):
                                    rows.extend([simplex[i], simplex[j]])
                                    cols.extend([simplex[j], simplex[i]])
                        
                        # Create sparse matrix
                        data = np.ones(len(rows))
                        conn = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
                        
                        # Store in AnnData format
                        adata.obsp['spatial_connectivities'] = conn
                        
                        # Calculate distances
                        coords_array = coords
                        distances = []
                        for i, j in zip(rows, cols):
                            dist = np.linalg.norm(coords_array[i] - coords_array[j])
                            distances.append(dist)
                        dist_matrix = sp.csr_matrix((distances, (rows, cols)), shape=(n_cells, n_cells))
                        adata.obsp['spatial_distances'] = dist_matrix
                        
                        # Verify graph was created
                        if 'spatial_connectivities' in adata.obsp:
                            n_edges = adata.obsp['spatial_connectivities'].nnz // 2  # Divide by 2 for undirected
                            print(f"[DEBUG] Delaunay graph built successfully: {n_edges} edges")
                            success_count += 1
                        else:
                            print(f"[DEBUG] ERROR: Delaunay graph not found in obsp after creation")
                            continue
                    else:
                        QtWidgets.QMessageBox.warning(self, "Invalid Method", f"Unknown method: {method}")
                        return
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    QtWidgets.QMessageBox.warning(self, "Graph Build Error", 
                        f"Failed to build graph for ROI {roi_id}: {str(e)}")
                    continue
            
            if success_count > 0:
                self.spatial_graph_built = True
                self.graph_status_label.setText(f"Graph created for {success_count} ROI(s)")
                self.graph_status_label.setStyleSheet("color: green;")
                
                # Update processed ROIs tracking
                for roi_id in all_rois:
                    if roi_id in self.anndata_cache and 'spatial_connectivities' in self.anndata_cache[roi_id].obsp:
                        if roi_id not in self.processed_rois:
                            self.processed_rois[roi_id] = {}
                        self.processed_rois[roi_id]['graph_built'] = True
                
                self._update_button_states()
                QtWidgets.QMessageBox.information(self, "Graph Created", 
                    f"Spatial graph created successfully for {success_count} ROI(s).")
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
        """Run neighborhood enrichment analysis using squidpy for all ROIs or selected ROI."""
        print(f"[DEBUG] _run_sq_nhood_enrichment: Starting")
        if not self.spatial_graph_built:
            print(f"[DEBUG] Spatial graph not built")
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            cluster_key = self.sq_nhood_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_nhood_roi_combo)
            agg_method = self.sq_nhood_agg_combo.currentText().lower()  # "mean" or "sum"
            print(f"[DEBUG] Cluster key: {cluster_key}, ROI: {roi_id}, Aggregation: {agg_method}")
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                print(f"[DEBUG] Cluster column '{cluster_key}' not found in dataframe")
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Determine which ROIs to process
            if roi_id is None:
                # Process all ROIs
                all_rois = self._get_all_rois()
                print(f"[DEBUG] Processing all ROIs: {all_rois}")
            else:
                # Process single ROI
                all_rois = [roi_id]
                print(f"[DEBUG] Processing single ROI: {roi_id}")
            
            if not all_rois:
                print(f"[DEBUG] No ROIs found")
                QtWidgets.QMessageBox.warning(self, "No ROIs", "No ROIs found to process.")
                return
            
            # Run enrichment for each ROI
            enrichment_matrices = []  # List of (roi_id, matrix) tuples
            success_count = 0
            
            for current_roi_id in all_rois:
                print(f"[DEBUG] Processing ROI: {current_roi_id}")
                adata = self.anndata_cache.get(current_roi_id)
                if adata is None:
                    print(f"[DEBUG] No AnnData cached for ROI {current_roi_id}, trying to create...")
                    adata = self._get_or_create_anndata(current_roi_id)
                
                if adata is None:
                    print(f"[DEBUG] Failed to get/create AnnData for ROI {current_roi_id}")
                    continue
                
                if 'spatial_connectivities' not in adata.obsp:
                    print(f"[DEBUG] No spatial_connectivities in obsp for ROI {current_roi_id}")
                    continue
                
                if cluster_key not in adata.obs.columns:
                    print(f"[DEBUG] Cluster key '{cluster_key}' not in obs.columns for ROI {current_roi_id}")
                    continue
                
                if not hasattr(adata.obs[cluster_key], 'cat'):
                    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
                
                print(f"[DEBUG] Running nhood_enrichment for ROI {current_roi_id}, clusters: {list(adata.obs[cluster_key].cat.categories)}")
                try:
                    # Run neighborhood enrichment
                    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
                    
                    # Debug: Check what's in adata.uns
                    print(f"[DEBUG] All keys in adata.uns: {list(adata.uns.keys())}")
                    
                    # Extract matrix - try different possible keys
                    enrichment_data = adata.uns.get('nhood_enrichment', {})
                    if not enrichment_data:
                        # Try alternative key names
                        for key in adata.uns.keys():
                            if 'nhood' in key.lower() or 'enrichment' in key.lower():
                                print(f"[DEBUG] Found potential key: {key}, type: {type(adata.uns[key])}")
                                enrichment_data = adata.uns[key]
                                break
                    
                    print(f"[DEBUG] Enrichment data type: {type(enrichment_data)}, keys: {list(enrichment_data.keys()) if isinstance(enrichment_data, dict) else 'N/A'}")
                    if isinstance(enrichment_data, dict) and len(enrichment_data) > 0:
                        print(f"[DEBUG] Enrichment data contents: {list(enrichment_data.items())[:3]}...")
                    matrix = None
                    if isinstance(enrichment_data, dict):
                        if 'zscore' in enrichment_data:
                            matrix = enrichment_data['zscore']
                            print(f"[DEBUG] Found zscore matrix, shape: {matrix.shape}")
                        elif 'count' in enrichment_data:
                            matrix = enrichment_data['count']
                            print(f"[DEBUG] Found count matrix, shape: {matrix.shape}")
                        elif 'stat' in enrichment_data:
                            matrix = enrichment_data['stat']
                            print(f"[DEBUG] Found stat matrix, shape: {matrix.shape}")
                        else:
                            for value in enrichment_data.values():
                                if isinstance(value, np.ndarray) and value.ndim == 2:
                                    matrix = value
                                    print(f"[DEBUG] Found matrix in dict values, shape: {matrix.shape}")
                                    break
                    elif isinstance(enrichment_data, np.ndarray):
                        matrix = enrichment_data
                        print(f"[DEBUG] Enrichment data is array, shape: {matrix.shape}")
                    
                    if matrix is not None and isinstance(matrix, np.ndarray) and matrix.ndim == 2:
                        enrichment_matrices.append((current_roi_id, matrix))
                        success_count += 1
                        print(f"[DEBUG] Successfully extracted matrix for ROI {current_roi_id}, shape: {matrix.shape}")
                        
                        # Update status
                        if current_roi_id not in self.analysis_status:
                            self.analysis_status[current_roi_id] = {}
                        self.analysis_status[current_roi_id]['nhood_enrichment'] = True
                    else:
                        print(f"[DEBUG] Failed to extract valid matrix from enrichment_data")
                except Exception as e:
                    print(f"[DEBUG] Exception in nhood_enrichment for ROI {current_roi_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if success_count == 0:
                QtWidgets.QMessageBox.warning(self, "Enrichment Failed", 
                    "Failed to run neighborhood enrichment for any ROI.")
                return
            
            # Aggregate results
            if len(enrichment_matrices) > 1:
                # Collect all cluster categories from all ROIs to create union
                roi_cluster_map = {}  # Maps roi_id to its cluster categories
                
                for roi_id, matrix in enrichment_matrices:
                    if roi_id in self.anndata_cache:
                        adata = self.anndata_cache[roi_id]
                        if cluster_key in adata.obs.columns:
                            if hasattr(adata.obs[cluster_key], 'cat'):
                                clusters = list(adata.obs[cluster_key].cat.categories)
                            else:
                                clusters = sorted(adata.obs[cluster_key].unique())
                            roi_cluster_map[roi_id] = clusters
                
                # Get union of all clusters
                all_cluster_sets = [set(clusters) for clusters in roi_cluster_map.values()]
                all_clusters_union = sorted(set().union(*all_cluster_sets)) if all_cluster_sets else []
                
                if not all_clusters_union:
                    aggregated_matrix = enrichment_matrices[0][1]  # Extract matrix from tuple
                else:
                    # Align all matrices to the union of clusters
                    aligned_matrices = []
                    n_clusters = len(all_clusters_union)
                    
                    for roi_id, matrix in enrichment_matrices:
                        roi_clusters = roi_cluster_map.get(roi_id)
                        
                        if roi_clusters is not None:
                            # Create aligned matrix
                            aligned_matrix = np.full((n_clusters, n_clusters), np.nan)
                            
                            # Map old indices to new indices
                            cluster_to_new_idx = {clust: idx for idx, clust in enumerate(all_clusters_union)}
                            
                            # Fill in values where clusters overlap
                            for i, old_clust_i in enumerate(roi_clusters):
                                if old_clust_i in cluster_to_new_idx:
                                    new_i = cluster_to_new_idx[old_clust_i]
                                    for j, old_clust_j in enumerate(roi_clusters):
                                        if old_clust_j in cluster_to_new_idx:
                                            new_j = cluster_to_new_idx[old_clust_j]
                                            aligned_matrix[new_i, new_j] = matrix[i, j]
                            
                            aligned_matrices.append(aligned_matrix)
                        else:
                            # Fallback: if we can't align, use original matrix (will warn below)
                            aligned_matrices.append(matrix)
                    
                    # Check if all matrices are now the same size
                    shapes = [m.shape for m in aligned_matrices]
                    if len(set(shapes)) == 1:
                        # All same shape - aggregate
                        stacked = np.stack(aligned_matrices, axis=0)
                        if agg_method == 'mean':
                            aggregated_matrix = np.nanmean(stacked, axis=0)
                        else:  # sum
                            aggregated_matrix = np.nansum(stacked, axis=0)
                        print(f"[DEBUG] Aggregated {len(aligned_matrices)} matrices aligned to {n_clusters} clusters")
                    else:
                        # Still different shapes - use first one and warn
                        aggregated_matrix = aligned_matrices[0]
                        QtWidgets.QMessageBox.warning(self, "Shape Mismatch", 
                            "ROIs have different cluster counts. Could not fully align matrices. Using first ROI result.")
            else:
                aggregated_matrix = enrichment_matrices[0][1]  # Extract matrix from tuple
            
            # Create a temporary adata-like object for plotting
            class TempAnnData:
                def __init__(self, matrix, cluster_key, obs):
                    self.uns = {'nhood_enrichment': {'zscore': matrix}}
                    self.obs = obs
                    # Store cluster_key for plotting
                    self._cluster_key = cluster_key
            
            # Get cluster categories - use union if available, otherwise from first ROI
            if len(enrichment_matrices) > 1 and 'all_clusters_union' in locals() and all_clusters_union:
                # Use the union of clusters we computed
                cluster_categories = all_clusters_union
            else:
                # Get from first successful ROI
                first_adata = None
                for current_roi_id in all_rois:
                    if current_roi_id in self.anndata_cache:
                        first_adata = self.anndata_cache[current_roi_id]
                        if cluster_key in first_adata.obs.columns:
                            break
                
                if first_adata is None:
                    QtWidgets.QMessageBox.warning(self, "No Data", "Could not find data for plotting.")
                    return
                
                if hasattr(first_adata.obs[cluster_key], 'cat'):
                    cluster_categories = list(first_adata.obs[cluster_key].cat.categories)
                else:
                    cluster_categories = sorted(first_adata.obs[cluster_key].unique())
            
            # Create a minimal obs DataFrame with cluster categories for plotting
            import pandas as pd
            obs_df = pd.DataFrame({cluster_key: cluster_categories})
            obs_df.index = [str(c) for c in cluster_categories]
            # Make it categorical for plotting
            obs_df[cluster_key] = obs_df[cluster_key].astype('category')
            
            temp_adata = TempAnnData(aggregated_matrix, cluster_key, obs_df)
            
            # Store aggregated result
            self.aggregated_results['nhood_enrichment'] = temp_adata
            
            QtWidgets.QMessageBox.information(self, "Enrichment Complete", 
                f"Neighborhood enrichment completed for {success_count} ROI(s). "
                f"Aggregated using {agg_method}.")
            
            # Plot aggregated results
            self._plot_sq_nhood_enrichment(temp_adata)
            self.sq_nhood_save_btn.setEnabled(True)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running enrichment: {str(e)}")
    
    def _plot_sq_nhood_enrichment(self, adata: 'ad.AnnData'):
        """Plot neighborhood enrichment results."""
        if adata is None or 'nhood_enrichment' not in adata.uns:
            self.sq_nhood_canvas.figure.clear()
            ax = self.sq_nhood_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No enrichment data found.\nPlease run enrichment analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_nhood_canvas.draw()
            return
        
        # Manual plotting (more reliable than squidpy's plotting with custom canvas)
        self.sq_nhood_canvas.figure.clear()
        ax = self.sq_nhood_canvas.figure.add_subplot(111)
        
        enrichment_data = adata.uns['nhood_enrichment']
        cluster_key = self.sq_nhood_cluster_combo.currentText()
        
        # Convert to numpy array if needed
        matrix = None
        if isinstance(enrichment_data, dict):
            # Squidpy stores the result in a nested structure
            # Try common keys
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
                # If still not found, try first value
                if matrix is None and len(enrichment_data) > 0:
                    first_val = list(enrichment_data.values())[0]
                    if isinstance(first_val, np.ndarray) and first_val.ndim == 2:
                        matrix = first_val
        elif isinstance(enrichment_data, np.ndarray):
            matrix = enrichment_data
        
        if matrix is not None and isinstance(matrix, np.ndarray) and matrix.ndim == 2:
            # Handle NaN/inf values
            if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                # Replace with 0 for display
                matrix = np.nan_to_num(matrix, nan=0.0, posinf=3.0, neginf=-3.0)
            
            # Determine vmin/vmax from data if reasonable
            vmin, vmax = -3, 3
            if matrix.size > 0:
                finite_vals = matrix[np.isfinite(matrix)]
                if len(finite_vals) > 0:
                    data_min, data_max = np.min(finite_vals), np.max(finite_vals)
                    if abs(data_min) < 10 and abs(data_max) < 10:
                        vmin, vmax = data_min, data_max
                    else:
                        # Use percentiles for extreme values
                        vmin = np.percentile(finite_vals, 5)
                        vmax = np.percentile(finite_vals, 95)
            
            im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            self.sq_nhood_canvas.figure.colorbar(im, ax=ax, label='Z-Score')
            
            # Add labels if available
            if cluster_key in adata.obs.columns:
                if hasattr(adata.obs[cluster_key], 'cat'):
                    categories = adata.obs[cluster_key].cat.categories
                else:
                    categories = adata.obs[cluster_key].unique()
                
                n_cats = len(categories)
                if matrix.shape[0] == n_cats and matrix.shape[1] == n_cats:
                    ax.set_xticks(np.arange(n_cats))
                    ax.set_yticks(np.arange(n_cats))
                    ax.set_xticklabels([self._get_cluster_display_name(c) for c in categories], rotation=45, ha='right')
                    ax.set_yticklabels([self._get_cluster_display_name(c) for c in categories])
            
            ax.set_title("Neighborhood Enrichment")
            ax.set_xlabel("Neighbor Cluster")
            ax.set_ylabel("Cell Cluster")
        else:
            # Debug: show what we got
            debug_info = f"Data type: {type(enrichment_data)}\n"
            if isinstance(enrichment_data, dict):
                debug_info += f"Keys: {list(enrichment_data.keys())}\n"
                for k, v in enrichment_data.items():
                    debug_info += f"  {k}: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}\n"
            ax.text(0.5, 0.5, f'Unable to plot enrichment data.\nData format not recognized.\n\n{debug_info}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
        
        self.sq_nhood_canvas.figure.tight_layout()
        self.sq_nhood_canvas.draw()
    
    def _save_sq_nhood_plot(self):
        """Save the neighborhood enrichment plot."""
        if save_figure_with_options(self.sq_nhood_canvas.figure, "squidpy_nhood_enrichment.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _run_sq_cooccurrence(self):
        """Run co-occurrence analysis using squidpy for all ROIs or selected ROI."""
        print(f"[DEBUG] _run_sq_cooccurrence: Starting")
        if not self.spatial_graph_built:
            print(f"[DEBUG] Spatial graph not built")
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            cluster_key = self.sq_cooccur_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_cooccur_roi_combo)
            agg_method = self.sq_cooccur_agg_combo.currentText().lower()
            print(f"[DEBUG] Cluster key: {cluster_key}, ROI: {roi_id}, Aggregation: {agg_method}")
            
            # Parse neighborhood sizes
            sizes_str = self.sq_cooccur_sizes_edit.text().strip()
            if sizes_str.lower() == 'all' or not sizes_str:
                QtWidgets.QMessageBox.warning(self, "Invalid Input", 
                    "Co-occurrence analysis requires multiple distances.\n"
                    "Please enter at least 2 distances separated by commas (e.g., '10,20,30,50').")
                return
            
            try:
                # Parse as comma-separated list
                nhood_sizes = [int(x.strip()) for x in sizes_str.split(',') if x.strip()]
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
            
            print(f"[DEBUG] Neighborhood sizes: {nhood_sizes} (count: {len(nhood_sizes)})")
            
            # Store interval for distance selection
            self.sq_cooccur_interval = nhood_sizes
            
            # Update distance combo
            self.sq_cooccur_distance_combo.clear()
            for dist in nhood_sizes:
                self.sq_cooccur_distance_combo.addItem(f"{dist} µm", dist)
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                print(f"[DEBUG] Cluster column '{cluster_key}' not found in dataframe")
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Determine which ROIs to process
            if roi_id is None:
                all_rois = self._get_all_rois()
                print(f"[DEBUG] Processing all ROIs: {all_rois}")
            else:
                all_rois = [roi_id]
                print(f"[DEBUG] Processing single ROI: {roi_id}")
            
            if not all_rois:
                print(f"[DEBUG] No ROIs found")
                QtWidgets.QMessageBox.warning(self, "No ROIs", "No ROIs found to process.")
                return
            
            # Run co-occurrence for each ROI
            cooccur_results = []
            success_count = 0
            
            for current_roi_id in all_rois:
                print(f"[DEBUG] Processing ROI: {current_roi_id}")
                adata = self.anndata_cache.get(current_roi_id)
                if adata is None:
                    print(f"[DEBUG] No AnnData cached for ROI {current_roi_id}, trying to create...")
                    adata = self._get_or_create_anndata(current_roi_id)
                
                if adata is None:
                    print(f"[DEBUG] Failed to get/create AnnData for ROI {current_roi_id}")
                    continue
                
                if 'spatial_connectivities' not in adata.obsp:
                    print(f"[DEBUG] No spatial_connectivities in obsp for ROI {current_roi_id}")
                    continue
                
                if cluster_key not in adata.obs.columns:
                    print(f"[DEBUG] Cluster key '{cluster_key}' not in obs.columns for ROI {current_roi_id}")
                    continue
                
                if not hasattr(adata.obs[cluster_key], 'cat'):
                    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
                
                print(f"[DEBUG] Running co_occurrence for ROI {current_roi_id}, clusters: {list(adata.obs[cluster_key].cat.categories)}")
                try:
                    # Run co-occurrence analysis (always use interval since we validated it)
                    sq.gr.co_occurrence(adata, cluster_key=cluster_key, interval=nhood_sizes)
                    
                    # Debug: Check what was stored
                    print(f"[DEBUG] All keys in adata.uns after co_occurrence: {list(adata.uns.keys())}")
                    if 'co_occurrence' in adata.uns:
                        cooccur_stored = adata.uns['co_occurrence']
                        print(f"[DEBUG] Co-occurrence data type: {type(cooccur_stored)}")
                        if isinstance(cooccur_stored, dict):
                            print(f"[DEBUG] Co-occurrence keys: {list(cooccur_stored.keys())[:5]}...")
                            # Check structure of first value
                            if len(cooccur_stored) > 0:
                                first_key = list(cooccur_stored.keys())[0]
                                first_val = cooccur_stored[first_key]
                                print(f"[DEBUG] First key: {first_key}, value type: {type(first_val)}")
                                if isinstance(first_val, dict):
                                    print(f"[DEBUG] First value keys: {list(first_val.keys())}")
                    else:
                        # Check for alternative key names
                        for key in adata.uns.keys():
                            if 'co' in key.lower() or 'occur' in key.lower():
                                print(f"[DEBUG] Found potential co-occurrence key: {key}, type: {type(adata.uns[key])}")
                    
                    print(f"[DEBUG] Co-occurrence completed for ROI {current_roi_id}")
                    cooccur_results.append(adata)
                    success_count += 1
                    
                    if current_roi_id not in self.analysis_status:
                        self.analysis_status[current_roi_id] = {}
                    self.analysis_status[current_roi_id]['co_occurrence'] = True
                except Exception as e:
                    print(f"[DEBUG] Exception in co_occurrence for ROI {current_roi_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if success_count == 0:
                QtWidgets.QMessageBox.warning(self, "Co-occurrence Failed", 
                    "Failed to run co-occurrence for any ROI.")
                return
            
            # For co-occurrence, we'll plot the first ROI or aggregate if multiple
            # Co-occurrence data structure is complex, so we'll use first result for now
            # TODO: Implement proper aggregation for co-occurrence curves
            if len(cooccur_results) > 0:
                plot_adata = cooccur_results[0]
                print(f"[DEBUG] Using first ROI result for plotting: {plot_adata}")
                print(f"[DEBUG] Plot adata uns keys: {list(plot_adata.uns.keys())}")
                
                # Update reference cluster combo
                self._update_cooccur_ref_cluster_combo(plot_adata)
                
                QtWidgets.QMessageBox.information(self, "Co-occurrence Complete", 
                    f"Co-occurrence analysis completed for {success_count} ROI(s).")
                
                # Plot results
                self._plot_sq_cooccurrence(plot_adata)
                self.sq_cooccur_save_btn.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(self, "No Results", 
                    "Co-occurrence analysis completed but no results to plot.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running co-occurrence: {str(e)}")
    
    def _plot_sq_cooccurrence(self, adata: 'ad.AnnData'):
        """Plot co-occurrence results."""
        print(f"[DEBUG] _plot_sq_cooccurrence: Starting")
        print(f"[DEBUG] adata: {adata}")
        if adata is None:
            print(f"[DEBUG] adata is None")
            self.sq_cooccur_canvas.figure.clear()
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_cooccur_canvas.draw()
            return
        
        print(f"[DEBUG] adata.uns keys: {list(adata.uns.keys())}")
        
        # Check for co_occurrence key or alternatives
        cooccur_data = None
        if 'co_occurrence' in adata.uns:
            cooccur_data = adata.uns['co_occurrence']
            print(f"[DEBUG] Found 'co_occurrence' key")
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'co' in key.lower() and 'occur' in key.lower():
                    cooccur_data = adata.uns[key]
                    print(f"[DEBUG] Found alternative key: {key}")
                    break
        
        if cooccur_data is None:
            print(f"[DEBUG] No co-occurrence data found in adata.uns")
            self.sq_cooccur_canvas.figure.clear()
            ax = self.sq_cooccur_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No co-occurrence data found.\nPlease run co-occurrence analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_cooccur_canvas.draw()
            return
        
        # Manual plotting (more reliable than squidpy's plotting with custom canvas)
        self.sq_cooccur_canvas.figure.clear()
        
        cluster_key = self.sq_cooccur_cluster_combo.currentText()
        plot_type = self.sq_cooccur_plot_type_combo.currentText()
        
        print(f"[DEBUG] Co-occurrence data type: {type(cooccur_data)}")
        print(f"[DEBUG] Plot type: {plot_type}")
        
        # Get cluster categories
        if cluster_key in adata.obs.columns:
            if hasattr(adata.obs[cluster_key], 'cat'):
                categories = list(adata.obs[cluster_key].cat.categories)
            else:
                categories = sorted(adata.obs[cluster_key].unique())
        else:
            categories = []
        
        print(f"[DEBUG] Categories: {categories}")
        
        # Handle heatmap plotting
        if plot_type == "Heatmap":
            # Try to populate distance combo if empty but interval data is available
            if self.sq_cooccur_distance_combo.count() == 0:
                interval_to_populate = self.sq_cooccur_interval
                if interval_to_populate is None and isinstance(cooccur_data, dict):
                    if 'interval' in cooccur_data:
                        interval_to_populate = cooccur_data['interval']
                    elif 'distances' in cooccur_data:
                        interval_to_populate = cooccur_data['distances']
                
                if interval_to_populate is not None:
                    # Convert to list if needed
                    if isinstance(interval_to_populate, np.ndarray):
                        interval_to_populate = interval_to_populate.tolist()
                    # Populate combo
                    self.sq_cooccur_distance_combo.clear()
                    for dist in interval_to_populate:
                        self.sq_cooccur_distance_combo.addItem(f"{dist} µm", dist)
                    # Store for future use
                    if self.sq_cooccur_interval is None:
                        self.sq_cooccur_interval = interval_to_populate
                else:
                    ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                    ax.text(0.5, 0.5, 'No distances available.\nPlease run co-occurrence analysis first.', 
                           ha='center', va='center', transform=ax.transAxes)
                    self.sq_cooccur_canvas.draw()
                    return
            
            selected_distance = self.sq_cooccur_distance_combo.currentData()
            if selected_distance is None:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Please select a distance for the heatmap.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return
            
            # Find the index of the selected distance
            # Try to get interval from stored value or extract from data
            interval_to_use = self.sq_cooccur_interval
            if interval_to_use is None and isinstance(cooccur_data, dict):
                if 'interval' in cooccur_data:
                    interval_to_use = cooccur_data['interval']
                elif 'distances' in cooccur_data:
                    interval_to_use = cooccur_data['distances']
            
            if interval_to_use is None:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Interval data not available.\nPlease run co-occurrence analysis first.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return
            
            # Convert to list if it's a numpy array
            if isinstance(interval_to_use, np.ndarray):
                interval_to_use = interval_to_use.tolist()
            
            try:
                distance_idx = interval_to_use.index(selected_distance)
            except ValueError:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Distance {selected_distance} not found in interval.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return
            
            # Extract heatmap data
            if isinstance(cooccur_data, dict) and 'occ' in cooccur_data:
                occ_array = cooccur_data['occ']
                if occ_array.ndim == 3:
                    # Extract 2D slice at the selected distance
                    heatmap_data = occ_array[:, :, distance_idx]
                    ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                    
                    # Create heatmap
                    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
                    
                    # Set labels
                    if len(categories) == heatmap_data.shape[0] and len(categories) == heatmap_data.shape[1]:
                        ax.set_xticks(np.arange(len(categories)))
                        ax.set_yticks(np.arange(len(categories)))
                        ax.set_xticklabels([self._get_cluster_display_name(c) for c in categories], rotation=45, ha='right')
                        ax.set_yticklabels([self._get_cluster_display_name(c) for c in categories])
                    
                    ax.set_xlabel('To Phenotype')
                    ax.set_ylabel('From Phenotype')
                    ax.set_title(f'Co-occurrence Heatmap at {selected_distance} µm')
                    
                    # Add colorbar
                    cbar = self.sq_cooccur_canvas.figure.colorbar(im, ax=ax)
                    cbar.set_label('Co-occurrence Score')
                    
                    self.sq_cooccur_canvas.figure.tight_layout()
                    self.sq_cooccur_canvas.draw()
                    return
                else:
                    ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                    ax.text(0.5, 0.5, 'Co-occurrence data format not suitable for heatmap.\nExpected 3D array.', 
                           ha='center', va='center', transform=ax.transAxes)
                    self.sq_cooccur_canvas.draw()
                    return
            else:
                ax = self.sq_cooccur_canvas.figure.add_subplot(111)
                ax.text(0.5, 0.5, 'Co-occurrence data format not recognized for heatmap.', 
                       ha='center', va='center', transform=ax.transAxes)
                self.sq_cooccur_canvas.draw()
                return
        
        # Continue with curve plotting (original logic)
        ax = self.sq_cooccur_canvas.figure.add_subplot(111)
        
        # Extract data for plotting
        plotted = False
        
        if isinstance(cooccur_data, dict):
            print(f"[DEBUG] Co-occurrence dict with {len(cooccur_data)} keys: {list(cooccur_data.keys())[:10]}...")
            
            # Try to find interval/distances from the data structure
            interval = None
            if 'interval' in cooccur_data:
                interval = cooccur_data['interval']
                print(f"[DEBUG] Found top-level interval: {interval}")
            elif 'distances' in cooccur_data:
                interval = cooccur_data['distances']
                print(f"[DEBUG] Found top-level distances: {interval}")
            
            # Check for 'occ' key - this is the co-occurrence matrix/array
            if 'occ' in cooccur_data:
                occ_array = cooccur_data['occ']
                print(f"[DEBUG] Found 'occ' array, shape: {occ_array.shape}, ndim: {occ_array.ndim}")
                
                if interval is not None:
                    distances = np.array(interval)
                    print(f"[DEBUG] Using distances: {distances}")
                    
                    if occ_array.ndim == 3:
                        # 3D array: (cluster1, cluster2, distance_index)
                        # Shape: (n_clusters, n_clusters, n_distances)
                        print(f"[DEBUG] 3D occ array: shape {occ_array.shape}")
                        n_clusters = occ_array.shape[0]
                        n_distances = occ_array.shape[2]
                        
                        # Handle dimension mismatch: use first n_distances from interval
                        if len(distances) >= n_distances:
                            plot_distances = distances[:n_distances]
                            print(f"[DEBUG] Using first {n_distances} distances from interval: {plot_distances}")
                        elif len(distances) == n_distances:
                            plot_distances = distances
                        else:
                            # Create distance indices if interval doesn't match
                            plot_distances = np.arange(n_distances)
                            print(f"[DEBUG] Interval length mismatch, using indices: {plot_distances}")
                        
                        # Get selected reference cluster
                        ref_cluster = self.sq_cooccur_ref_cluster_combo.currentData()
                        print(f"[DEBUG] Reference cluster selected: {ref_cluster}")
                        
                        if ref_cluster is None:
                            # Plot all clusters: self-co-occurrence for each
                            for i in range(min(n_clusters, len(categories))):
                                cluster = categories[i]
                                self_occ = occ_array[i, i, :]  # Self-co-occurrence
                                if len(self_occ) == len(plot_distances):
                                    ax.plot(plot_distances, self_occ, 
                                           label=f"{self._get_cluster_display_name(cluster)} (self)",
                                           marker='o', markersize=4, linewidth=1.5)
                                    plotted = True
                                    print(f"[DEBUG] Plotted self-co-occurrence for cluster {cluster}: {len(self_occ)} points")
                        else:
                            # Plot co-occurrence with respect to selected reference cluster
                            try:
                                ref_idx = categories.index(ref_cluster)
                                if ref_idx < n_clusters:
                                    # Plot co-occurrence of all clusters with the reference cluster
                                    for i in range(min(n_clusters, len(categories))):
                                        cluster = categories[i]
                                        if i == ref_idx:
                                            # Self-co-occurrence
                                            co_occ = occ_array[i, i, :]
                                            label = f"{self._get_cluster_display_name(cluster)} (self)"
                                        else:
                                            # Co-occurrence with reference
                                            co_occ = occ_array[i, ref_idx, :]
                                            label = f"{self._get_cluster_display_name(cluster)} with {self._get_cluster_display_name(ref_cluster)}"
                                        
                                        if len(co_occ) == len(plot_distances):
                                            ax.plot(plot_distances, co_occ, 
                                                   label=label,
                                                   marker='o', markersize=4, linewidth=1.5)
                                            plotted = True
                                            print(f"[DEBUG] Plotted co-occurrence for {label}: {len(co_occ)} points")
                            except ValueError:
                                print(f"[DEBUG] Reference cluster {ref_cluster} not found in categories")
                    
                    elif occ_array.ndim == 2:
                        # 2D array: could be (cluster_pairs, distances) or (clusters, distances)
                        print(f"[DEBUG] 2D occ array: shape {occ_array.shape}")
                        if occ_array.shape[1] == len(distances):
                            # Each row is a cluster pair, columns are distances
                            for i in range(min(occ_array.shape[0], len(categories))):
                                if i < len(categories):
                                    cluster = categories[i]
                                    ax.plot(distances, occ_array[i, :], 
                                           label=self._get_cluster_display_name(cluster),
                                           marker='o', markersize=4, linewidth=1.5)
                                    plotted = True
                                    print(f"[DEBUG] Plotted curve for cluster {cluster}")
                        elif occ_array.shape[0] == len(distances):
                            # Each column is a cluster, rows are distances
                            for i in range(min(occ_array.shape[1], len(categories))):
                                if i < len(categories):
                                    cluster = categories[i]
                                    ax.plot(distances, occ_array[:, i], 
                                           label=self._get_cluster_display_name(cluster),
                                           marker='o', markersize=4, linewidth=1.5)
                                    plotted = True
                                    print(f"[DEBUG] Plotted curve for cluster {cluster}")
                    
                    elif occ_array.ndim == 1:
                        # 1D array: single curve
                        if len(occ_array) == len(distances):
                            ax.plot(distances, occ_array, 
                                   label="Co-occurrence",
                                   marker='o', markersize=4, linewidth=1.5)
                            plotted = True
                            print(f"[DEBUG] Plotted single co-occurrence curve")
            
            # Fallback: Try old format (cluster pairs as keys)
            if not plotted:
                print(f"[DEBUG] Trying fallback format - cluster pairs as keys")
                for key, value in cooccur_data.items():
                    if isinstance(value, dict):
                        # Extract interval/distances and co-occurrence values
                        if 'interval' in value:
                            distances = np.array(value['interval'])
                        elif 'distances' in value:
                            distances = np.array(value['distances'])
                        else:
                            distances = interval if interval is not None else None
                        
                        # Extract co-occurrence values
                        if 'occ' in value:
                            cooccur_values = np.array(value['occ'])
                        elif 'cooccurrence' in value:
                            cooccur_values = np.array(value['cooccurrence'])
                        else:
                            continue
                        
                        # Get cluster pair label
                        if isinstance(key, tuple) and len(key) == 2:
                            cat1, cat2 = key
                            label = f"{self._get_cluster_display_name(cat1)}-{self._get_cluster_display_name(cat2)}"
                        elif isinstance(key, str) and '_' in key:
                            parts = key.split('_', 1)
                            label = f"{self._get_cluster_display_name(parts[0])}-{self._get_cluster_display_name(parts[1])}"
                        else:
                            label = str(key)
                        
                        if distances is not None and len(distances) > 0 and len(cooccur_values) > 0:
                            # Ensure they have the same length
                            min_len = min(len(distances), len(cooccur_values))
                            if min_len > 0:
                                ax.plot(distances[:min_len], cooccur_values[:min_len], 
                                       label=label, marker='o', markersize=4, linewidth=1.5)
                                plotted = True
                                print(f"[DEBUG] Plotted curve for {label}: {min_len} points")
        
        if plotted:
            ax.set_xlabel('Distance (µm)')
            ax.set_ylabel('Co-occurrence')
            ax.set_title("Co-occurrence Analysis")
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            print(f"[DEBUG] Plot completed successfully")
        else:
            print(f"[DEBUG] Failed to plot - data format not recognized")
            ax.text(0.5, 0.5, 'Unable to plot co-occurrence data.\nData format not recognized.\nCheck debug output for details.', 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.sq_cooccur_canvas.figure.tight_layout()
        self.sq_cooccur_canvas.draw()
    
    def _save_sq_cooccur_plot(self):
        """Save the co-occurrence plot."""
        if save_figure_with_options(self.sq_cooccur_canvas.figure, "squidpy_cooccurrence.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _run_sq_autocorrelation(self):
        """Run spatial autocorrelation analysis using squidpy for all ROIs or selected ROI."""
        print(f"[DEBUG] _run_sq_autocorrelation: Starting")
        if not self.spatial_graph_built:
            print(f"[DEBUG] Spatial graph not built")
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            markers_str = self.sq_autocorr_markers_edit.text().strip()
            roi_id = self._get_selected_roi(self.sq_autocorr_roi_combo)
            agg_method = self.sq_autocorr_agg_combo.currentText().lower()
            print(f"[DEBUG] Markers: {markers_str}, ROI: {roi_id}, Aggregation: {agg_method}")
            
            # Parse markers
            if markers_str.lower() == 'all' or not markers_str:
                genes = None  # Use all features
            else:
                genes = [m.strip() for m in markers_str.split(',')]
            print(f"[DEBUG] Genes to analyze: {genes if genes else 'all'}")
            
            # Determine which ROIs to process
            if roi_id is None:
                all_rois = self._get_all_rois()
                print(f"[DEBUG] Processing all ROIs: {all_rois}")
            else:
                all_rois = [roi_id]
                print(f"[DEBUG] Processing single ROI: {roi_id}")
            
            if not all_rois:
                print(f"[DEBUG] No ROIs found")
                QtWidgets.QMessageBox.warning(self, "No ROIs", "No ROIs found to process.")
                return
            
            # Run autocorrelation for each ROI
            moran_results = []  # List of dicts with 'I' and 'var_names'
            success_count = 0
            all_genes = set()
            
            for current_roi_id in all_rois:
                print(f"[DEBUG] Processing ROI: {current_roi_id}")
                adata = self.anndata_cache.get(current_roi_id)
                if adata is None:
                    print(f"[DEBUG] No AnnData cached for ROI {current_roi_id}, trying to create...")
                    adata = self._get_or_create_anndata(current_roi_id)
                
                if adata is None:
                    print(f"[DEBUG] Failed to get/create AnnData for ROI {current_roi_id}")
                    continue
                
                if 'spatial_connectivities' not in adata.obsp:
                    print(f"[DEBUG] No spatial_connectivities in obsp for ROI {current_roi_id}")
                    continue
                
                print(f"[DEBUG] Available var_names: {list(adata.var_names)[:5]}...")
                try:
                    # Run spatial autocorrelation (Moran's I)
                    if genes is not None:
                        available_genes = [g for g in genes if g in adata.var_names]
                        print(f"[DEBUG] Available genes from requested list: {available_genes}")
                        if not available_genes:
                            print(f"[DEBUG] No requested genes found in var_names")
                            continue
                        sq.gr.spatial_autocorr(adata, mode="moran", genes=available_genes)
                        all_genes.update(available_genes)
                    else:
                        print(f"[DEBUG] Running autocorrelation for all {len(adata.var_names)} features")
                        sq.gr.spatial_autocorr(adata, mode="moran")
                        # Convert var_names to list to avoid DataFrame/Index issues
                        var_names_list = list(adata.var_names) if hasattr(adata.var_names, '__iter__') else []
                        all_genes.update(var_names_list)
                    
                    # Extract results
                    print(f"[DEBUG] Checking adata.uns keys: {list(adata.uns.keys())}")
                    moran_data = adata.uns.get('moranI', {})
                    print(f"[DEBUG] moran_data type: {type(moran_data)}")
                    
                    # Check if moran_data is not empty (handle both dict and DataFrame)
                    has_data = False
                    if isinstance(moran_data, pd.DataFrame):
                        has_data = not moran_data.empty
                        if has_data:
                            print(f"[DEBUG] moran_data DataFrame shape: {moran_data.shape}, columns: {list(moran_data.columns)}")
                    elif isinstance(moran_data, dict):
                        has_data = len(moran_data) > 0
                        if has_data:
                            print(f"[DEBUG] moran_data dict keys: {list(moran_data.keys())}")
                    elif hasattr(moran_data, '__len__'):
                        try:
                            has_data = len(moran_data) > 0
                        except (TypeError, ValueError):
                            has_data = False
                    else:
                        # For other types, use explicit check
                        has_data = moran_data is not None and moran_data != {}
                    
                    if has_data:
                        print(f"[DEBUG] Extracted Moran's I data for ROI {current_roi_id}")
                        moran_results.append({
                            'adata': adata,
                            'moranI': moran_data
                        })
                        success_count += 1
                        
                        if current_roi_id not in self.analysis_status:
                            self.analysis_status[current_roi_id] = {}
                        self.analysis_status[current_roi_id]['autocorrelation'] = True
                    else:
                        print(f"[DEBUG] No Moran's I data found in adata.uns for ROI {current_roi_id}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue
            
            if success_count == 0:
                QtWidgets.QMessageBox.warning(self, "Autocorrelation Failed", 
                    "Failed to run autocorrelation for any ROI.")
                return
            
            # Aggregate results across ROIs
            if len(moran_results) > 1:
                # Get common genes across all ROIs
                common_genes = sorted(all_genes)
                I_values_agg = []
                p_values_agg = []
                
                for gene in common_genes:
                    I_vals = []
                    p_vals = []
                    for result in moran_results:
                        moranI = result['moranI']
                        
                        # Handle DataFrame format
                        if isinstance(moranI, pd.DataFrame):
                            if gene in moranI.index:
                                # DataFrame with genes as index
                                I_val = moranI.loc[gene, 'I'] if 'I' in moranI.columns else None
                                p_val = moranI.loc[gene, 'pval_norm'] if 'pval_norm' in moranI.columns else None
                                if I_val is not None:
                                    I_vals.append(float(I_val))
                                if p_val is not None:
                                    p_vals.append(float(p_val))
                            elif 'var_names' in moranI.columns or 'var_names' in moranI.index:
                                # DataFrame might have var_names column
                                if 'var_names' in moranI.columns:
                                    gene_rows = moranI[moranI['var_names'] == gene]
                                else:
                                    gene_rows = moranI[moranI.index == gene]
                                if not gene_rows.empty:
                                    I_val = gene_rows['I'].iloc[0] if 'I' in gene_rows.columns else None
                                    p_val = gene_rows['pval_norm'].iloc[0] if 'pval_norm' in gene_rows.columns else None
                                    if I_val is not None:
                                        I_vals.append(float(I_val))
                                    if p_val is not None:
                                        p_vals.append(float(p_val))
                        # Handle dict format
                        elif isinstance(moranI, dict):
                            if 'I' in moranI and 'var_names' in moranI:
                                var_names = moranI.get('var_names', [])
                                if gene in var_names:
                                    idx = list(var_names).index(gene)
                                    I_vals.append(moranI['I'][idx] if isinstance(moranI['I'], (list, np.ndarray)) else moranI['I'])
                                    if 'pval_norm' in moranI:
                                        p_vals.append(moranI['pval_norm'][idx] if isinstance(moranI['pval_norm'], (list, np.ndarray)) else moranI['pval_norm'])
                    
                    if I_vals:
                        if agg_method == 'mean':
                            I_values_agg.append(np.nanmean(I_vals))
                        else:  # sum
                            I_values_agg.append(np.nansum(I_vals))
                        if p_vals:
                            p_values_agg.append(np.nanmean(p_vals))
                        else:
                            p_values_agg.append(1.0)
                
                # Create aggregated result
                class TempAnnData:
                    def __init__(self, I_vals, p_vals, genes):
                        self.uns = {
                            'moranI': {
                                'I': np.array(I_vals),
                                'pval_norm': np.array(p_vals) if p_vals else None,
                                'var_names': genes
                            }
                        }
                
                aggregated_adata = TempAnnData(I_values_agg, p_values_agg, common_genes)
                plot_adata = aggregated_adata
                # Store aggregated result
                self.aggregated_results['autocorrelation'] = aggregated_adata
            else:
                plot_adata = moran_results[0]['adata']
            
            QtWidgets.QMessageBox.information(self, "Autocorrelation Complete", 
                f"Spatial autocorrelation completed for {success_count} ROI(s). "
                f"{'Aggregated using ' + agg_method if len(moran_results) > 1 else ''}")
            
            # Plot results
            self._plot_sq_autocorrelation(plot_adata)
            self.sq_autocorr_save_btn.setEnabled(True)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running autocorrelation: {str(e)}")
    
    def _plot_sq_autocorrelation(self, adata: 'ad.AnnData'):
        """Plot spatial autocorrelation results."""
        print(f"[DEBUG] _plot_sq_autocorrelation: Starting")
        print(f"[DEBUG] adata: {adata}")
        if adata is None:
            print(f"[DEBUG] adata is None")
            self.sq_autocorr_canvas.figure.clear()
            ax = self.sq_autocorr_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_autocorr_canvas.draw()
            return
        
        print(f"[DEBUG] adata.uns keys: {list(adata.uns.keys())}")
        
        # Check for moranI key or alternatives
        moran_data = None
        if 'moranI' in adata.uns:
            moran_data = adata.uns['moranI']
            print(f"[DEBUG] Found 'moranI' key")
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'moran' in key.lower() or 'autocorr' in key.lower():
                    moran_data = adata.uns[key]
                    print(f"[DEBUG] Found alternative key: {key}")
                    break
        
        if moran_data is None:
            print(f"[DEBUG] No Moran's I data found in adata.uns")
            self.sq_autocorr_canvas.figure.clear()
            ax = self.sq_autocorr_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No autocorrelation data found.\nPlease run autocorrelation analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_autocorr_canvas.draw()
            return
        
        # Manual plotting (more reliable than squidpy's plotting with custom canvas)
        self.sq_autocorr_canvas.figure.clear()
        ax = self.sq_autocorr_canvas.figure.add_subplot(111)
        
        print(f"[DEBUG] moran_data type: {type(moran_data)}")
        
        # Extract I values and p-values
        I_values = None
        p_values = None
        gene_names = None
        
        if isinstance(moran_data, pd.DataFrame):
            print(f"[DEBUG] moran_data is DataFrame, shape: {moran_data.shape}, columns: {list(moran_data.columns)}")
            # DataFrame format - extract columns
            if 'I' in moran_data.columns:
                I_values = moran_data['I'].values
                print(f"[DEBUG] Extracted I values from DataFrame, shape: {I_values.shape}")
            elif 'moranI' in moran_data.columns:
                I_values = moran_data['moranI'].values
                print(f"[DEBUG] Extracted moranI values from DataFrame, shape: {I_values.shape}")
            
            if 'pval_norm' in moran_data.columns:
                p_values = moran_data['pval_norm'].values
            elif 'pval' in moran_data.columns:
                p_values = moran_data['pval'].values
            
            # Get gene names from index or var_names column
            if 'var_names' in moran_data.columns:
                gene_names = moran_data['var_names'].values
            elif moran_data.index.name == 'var_names' or all(isinstance(x, str) for x in moran_data.index):
                gene_names = moran_data.index.values
            else:
                # Try to infer from index
                gene_names = [str(x) for x in moran_data.index]
        
        elif isinstance(moran_data, dict):
            I_values = None
            p_values = None
            gene_names = None
            
            # Try different possible keys
            if 'I' in moran_data:
                I_values = moran_data['I']
                if isinstance(I_values, np.ndarray):
                    pass  # Good
                elif isinstance(I_values, (list, tuple)):
                    I_values = np.array(I_values)
                else:
                    I_values = None
            
            if 'pval_norm' in moran_data:
                p_values = moran_data['pval_norm']
                if isinstance(p_values, np.ndarray):
                    pass
                elif isinstance(p_values, (list, tuple)):
                    p_values = np.array(p_values)
                else:
                    p_values = None
            elif 'pval' in moran_data:
                p_values = moran_data['pval']
                if isinstance(p_values, np.ndarray):
                    pass
                elif isinstance(p_values, (list, tuple)):
                    p_values = np.array(p_values)
                else:
                    p_values = None
            
            # Get gene names
            if 'var_names' in moran_data:
                gene_names = moran_data['var_names']
            elif hasattr(adata, 'var_names'):
                gene_names = adata.var_names.values
            else:
                gene_names = [f"Feature_{i}" for i in range(len(I_values))] if I_values is not None else []
        
        # Plot the data (works for both DataFrame and dict formats)
        if I_values is not None and len(I_values) > 0:
            print(f"[DEBUG] Plotting {len(I_values)} Moran's I values")
            # Create bar plot
            sorted_idx = np.argsort(I_values)[::-1]  # Sort by I value descending
            top_n = min(self.sq_autocorr_topk_spin.value(), len(sorted_idx))  # Use spinbox value
            
            top_I = I_values[sorted_idx[:top_n]]
            top_p = p_values[sorted_idx[:top_n]] if p_values is not None and len(p_values) > 0 else None
            top_genes = [gene_names[i] for i in sorted_idx[:top_n]] if len(gene_names) > 0 else [f"Feature_{i}" for i in sorted_idx[:top_n]]
            
            colors = ['red' if (top_p is not None and p < 0.05) else 'gray' for p in (top_p if top_p is not None else [1.0] * top_n)]
            ax.barh(range(top_n), top_I, color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_genes, fontsize=8)
            ax.set_xlabel("Moran's I")
            ax.set_title(f"Spatial Autocorrelation (Top {top_n})")
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            print(f"[DEBUG] Plot completed successfully")
        else:
            print(f"[DEBUG] Failed to plot - I_values is None or empty")
            ax.text(0.5, 0.5, 'Unable to extract Moran\'s I values.\nData format not recognized.\nCheck debug output for details.', 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.sq_autocorr_canvas.figure.tight_layout()
        self.sq_autocorr_canvas.draw()
    
    def _save_sq_autocorr_plot(self):
        """Save the spatial autocorrelation plot."""
        if save_figure_with_options(self.sq_autocorr_canvas.figure, "squidpy_autocorrelation.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _run_sq_ripley(self):
        """Run Ripley analysis using squidpy for all ROIs or selected ROI."""
        print(f"[DEBUG] _run_sq_ripley: Starting")
        if not self.spatial_graph_built:
            print(f"[DEBUG] Spatial graph not built")
            QtWidgets.QMessageBox.warning(self, "Graph Required", 
                "Please create the spatial graph first (Step 1 at the top).")
            return
        
        try:
            mode = self.sq_ripley_mode_combo.currentText()  # F, G, or L
            max_dist = float(self.sq_ripley_r_max_spin.value())
            cluster_key = self.sq_ripley_cluster_combo.currentText()
            roi_id = self._get_selected_roi(self.sq_ripley_roi_combo)
            agg_method = self.sq_ripley_agg_combo.currentText().lower()
            print(f"[DEBUG] Mode: {mode}, max_dist: {max_dist}, cluster_key: {cluster_key}, ROI: {roi_id}, Aggregation: {agg_method}")
            
            # Check if cluster column exists
            filtered_df = self._get_filtered_dataframe()
            if cluster_key not in filtered_df.columns:
                print(f"[DEBUG] Cluster column '{cluster_key}' not found in dataframe")
                QtWidgets.QMessageBox.warning(self, "Missing Column", 
                    f"Cluster column '{cluster_key}' not found in data.")
                return
            
            # Determine which ROIs to process
            if roi_id is None:
                all_rois = self._get_all_rois()
                print(f"[DEBUG] Processing all ROIs: {all_rois}")
            else:
                all_rois = [roi_id]
                print(f"[DEBUG] Processing single ROI: {roi_id}")
            
            if not all_rois:
                print(f"[DEBUG] No ROIs found")
                QtWidgets.QMessageBox.warning(self, "No ROIs", "No ROIs found to process.")
                return
            
            # Run Ripley for each ROI
            ripley_results = []
            success_count = 0
            
            for current_roi_id in all_rois:
                print(f"[DEBUG] Processing ROI: {current_roi_id}")
                adata = self.anndata_cache.get(current_roi_id)
                if adata is None:
                    print(f"[DEBUG] No AnnData cached for ROI {current_roi_id}, trying to create...")
                    adata = self._get_or_create_anndata(current_roi_id)
                
                if adata is None:
                    print(f"[DEBUG] Failed to get/create AnnData for ROI {current_roi_id}")
                    continue
                
                if 'spatial_connectivities' not in adata.obsp:
                    print(f"[DEBUG] No spatial_connectivities in obsp for ROI {current_roi_id}")
                    continue
                
                if cluster_key not in adata.obs.columns:
                    print(f"[DEBUG] Cluster key '{cluster_key}' not in obs.columns for ROI {current_roi_id}")
                    continue
                
                if not hasattr(adata.obs[cluster_key], 'cat'):
                    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
                
                # Check cluster sizes and filter out clusters with < 2 cells
                cluster_counts = adata.obs[cluster_key].value_counts()
                min_cluster_size = cluster_counts.min()
                print(f"[DEBUG] Cluster counts: {dict(cluster_counts)}, min_size: {min_cluster_size}")
                
                # Filter out clusters with < 2 cells before running Ripley
                valid_clusters = cluster_counts[cluster_counts >= 2].index.tolist()
                if len(valid_clusters) == 0:
                    print(f"[DEBUG] Skipping ROI {current_roi_id} - no clusters with >= 2 cells")
                    continue  # Skip this ROI
                
                if len(valid_clusters) < len(cluster_counts):
                    print(f"[DEBUG] Filtering out {len(cluster_counts) - len(valid_clusters)} clusters with < 2 cells. "
                          f"Using {len(valid_clusters)} valid clusters: {valid_clusters}")
                    # Filter adata to only include valid clusters
                    adata_filtered = adata[adata.obs[cluster_key].isin(valid_clusters)].copy()
                    if adata_filtered.n_obs == 0:
                        print(f"[DEBUG] Skipping ROI {current_roi_id} - no cells after filtering")
                        continue
                    adata = adata_filtered
                
                print(f"[DEBUG] Running ripley for ROI {current_roi_id}, clusters: {list(adata.obs[cluster_key].cat.categories)}")
                try:
                    sq.gr.ripley(adata, cluster_key=cluster_key, mode=mode, max_dist=max_dist)
                    
                    # Debug: Check what was stored
                    print(f"[DEBUG] All keys in adata.uns after ripley: {list(adata.uns.keys())}")
                    if 'ripley' in adata.uns:
                        ripley_stored = adata.uns['ripley']
                        print(f"[DEBUG] Ripley data type: {type(ripley_stored)}")
                        if isinstance(ripley_stored, dict):
                            print(f"[DEBUG] Ripley keys: {list(ripley_stored.keys())[:10]}...")
                            # Check structure of first value
                            if len(ripley_stored) > 0:
                                first_key = list(ripley_stored.keys())[0]
                                first_val = ripley_stored[first_key]
                                print(f"[DEBUG] First key: {first_key}, value type: {type(first_val)}")
                                if isinstance(first_val, dict):
                                    print(f"[DEBUG] First value keys: {list(first_val.keys())}")
                        elif isinstance(ripley_stored, pd.DataFrame):
                            print(f"[DEBUG] Ripley DataFrame shape: {ripley_stored.shape}, columns: {list(ripley_stored.columns)}")
                    else:
                        # Check for alternative key names
                        for key in adata.uns.keys():
                            if 'ripley' in key.lower():
                                print(f"[DEBUG] Found potential ripley key: {key}, type: {type(adata.uns[key])}")
                    
                    print(f"[DEBUG] Ripley completed for ROI {current_roi_id}")
                    ripley_results.append(adata)
                    success_count += 1
                    
                    if current_roi_id not in self.analysis_status:
                        self.analysis_status[current_roi_id] = {}
                    self.analysis_status[current_roi_id]['ripley'] = True
                except ValueError as e:
                    if "n_neighbors" in str(e) or "n_samples_fit" in str(e):
                        print(f"[DEBUG] Skipping ROI {current_roi_id} - insufficient samples: {e}")
                        continue  # Skip this ROI
                    else:
                        raise
                except Exception as e:
                    print(f"[DEBUG] Exception in ripley for ROI {current_roi_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if success_count == 0:
                QtWidgets.QMessageBox.warning(self, "Ripley Failed", 
                    "Failed to run Ripley analysis for any ROI. "
                    "This can happen when clusters are too small.")
                return
            
            # For Ripley, we'll plot the first ROI for now
            # TODO: Implement proper aggregation for Ripley curves
            if len(ripley_results) > 0:
                plot_adata = ripley_results[0]
                print(f"[DEBUG] Using first ROI result for plotting: {plot_adata}")
                print(f"[DEBUG] Plot adata uns keys: {list(plot_adata.uns.keys())}")
                
                QtWidgets.QMessageBox.information(self, "Ripley Complete", 
                    f"Ripley analysis completed for {success_count} ROI(s).")
                
                # Plot results
                self._plot_sq_ripley(plot_adata, cluster_key)
                self.sq_ripley_save_btn.setEnabled(True)
            else:
                QtWidgets.QMessageBox.warning(self, "No Results", 
                    "Ripley analysis completed but no results to plot.")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error running Ripley: {str(e)}")
    
    def _plot_sq_ripley(self, adata: 'ad.AnnData', cluster_key: str):
        """Plot Ripley results."""
        print(f"[DEBUG] _plot_sq_ripley: Starting")
        print(f"[DEBUG] adata: {adata}, cluster_key: {cluster_key}")
        if adata is None:
            print(f"[DEBUG] adata is None")
            self.sq_ripley_canvas.figure.clear()
            ax = self.sq_ripley_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data provided for plotting.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_ripley_canvas.draw()
            return
        
        print(f"[DEBUG] adata.uns keys: {list(adata.uns.keys())}")
        
        # Check for ripley key or alternatives
        ripley_data = None
        if 'ripley' in adata.uns:
            ripley_data = adata.uns['ripley']
            print(f"[DEBUG] Found 'ripley' key")
        else:
            # Try alternative key names
            for key in adata.uns.keys():
                if 'ripley' in key.lower():
                    ripley_data = adata.uns[key]
                    print(f"[DEBUG] Found alternative key: {key}")
                    break
        
        if ripley_data is None:
            print(f"[DEBUG] No Ripley data found in adata.uns")
            self.sq_ripley_canvas.figure.clear()
            ax = self.sq_ripley_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No Ripley data found.\nPlease run Ripley analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
            self.sq_ripley_canvas.draw()
            return
        
        # Manual plotting (more reliable than squidpy's plotting with custom canvas)
        mode = self.sq_ripley_mode_combo.currentText()
        self.sq_ripley_canvas.figure.clear()
        ax = self.sq_ripley_canvas.figure.add_subplot(111)
        
        print(f"[DEBUG] ripley_data type: {type(ripley_data)}")
        
        # Extract data for plotting
        if isinstance(ripley_data, dict) and len(ripley_data) > 0:
            print(f"[DEBUG] ripley_data is dict with {len(ripley_data)} keys: {list(ripley_data.keys())[:10]}...")
            
            plotted = False
            
            # Check for the actual squidpy structure: bins, F_stat/G_stat/L_stat, etc.
            if 'bins' in ripley_data:
                bins = ripley_data['bins']
                print(f"[DEBUG] Found 'bins' key, shape: {bins.shape if hasattr(bins, 'shape') else 'N/A'}")
                distances = np.array(bins) if not isinstance(bins, np.ndarray) else bins
                
                # Look for the statistic DataFrame (F_stat, G_stat, or L_stat)
                stat_key = f'{mode}_stat'
                if stat_key in ripley_data:
                    stat_df = ripley_data[stat_key]
                    print(f"[DEBUG] Found '{stat_key}' DataFrame, shape: {stat_df.shape}, columns: {list(stat_df.columns)}")
                    
                    if isinstance(stat_df, pd.DataFrame):
                        # Check if DataFrame is in long format (bins, cluster, stats) or wide format (clusters as columns)
                        if 'cluster' in stat_df.columns and 'stats' in stat_df.columns:
                            # Long format: pivot to wide format
                            print(f"[DEBUG] DataFrame is in long format, pivoting...")
                            
                            # Get unique bins and clusters
                            unique_bins = sorted(stat_df['bins'].unique()) if 'bins' in stat_df.columns else None
                            unique_clusters = sorted(stat_df['cluster'].unique())
                            print(f"[DEBUG] Unique bins: {len(unique_bins) if unique_bins is not None else 'N/A'}, "
                                  f"Unique clusters: {len(unique_clusters)}")
                            
                            if 'bins' in stat_df.columns:
                                # Pivot using bins as index
                                try:
                                    pivot_df = stat_df.pivot(index='bins', columns='cluster', values='stats')
                                    print(f"[DEBUG] Pivoted DataFrame shape: {pivot_df.shape}, columns: {list(pivot_df.columns)}")
                                    
                                    # Align distances with pivot_df index
                                    # pivot_df.index contains bin indices (0, 1, 2, ...) or actual distance values
                                    if pivot_df.index.dtype in [np.int64, np.int32]:
                                        # Index is bin indices, use distances array
                                        if len(distances) >= len(pivot_df.index):
                                            # Map bin indices to distances
                                            plot_distances = distances[pivot_df.index.values]
                                        else:
                                            # Use index as distances if distances array is shorter
                                            plot_distances = pivot_df.index.values
                                    else:
                                        # Index might be actual distance values
                                        plot_distances = pivot_df.index.values
                                    
                                    # Store plot_distances for CSR line calculation
                                    stored_plot_distances = plot_distances.copy()
                                    
                                    # Plot each cluster column
                                    for col in pivot_df.columns:
                                        values = pivot_df[col].values
                                        # Remove NaN values
                                        valid_mask = ~np.isnan(values)
                                        if np.any(valid_mask):
                                            valid_distances = plot_distances[valid_mask]
                                            valid_values = values[valid_mask]
                                            if len(valid_distances) > 0 and len(valid_values) > 0:
                                                ax.plot(valid_distances, valid_values, 
                                                       label=self._get_cluster_display_name(col),
                                                       marker='o', markersize=4, linewidth=1.5)
                                                plotted = True
                                                print(f"[DEBUG] Plotted curve for cluster {col}: {len(valid_distances)} points")
                                    
                                    # Add expected CSR line after plotting all clusters
                                    if mode == 'L' and len(stored_plot_distances) > 0:
                                        # For L function, expected under CSR is L(r) = r
                                        ax.plot(stored_plot_distances, stored_plot_distances, 
                                               color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                                               label='Expected (CSR)', zorder=0)
                                except Exception as e:
                                    print(f"[DEBUG] Pivot failed: {e}, trying alternative approach")
                                    # Alternative: group by cluster and plot directly
                                    for cluster in unique_clusters:
                                        cluster_data = stat_df[stat_df['cluster'] == cluster]
                                        if 'bins' in cluster_data.columns:
                                            # Use bins as indices into distances array
                                            bin_indices = cluster_data['bins'].values.astype(int)
                                            cluster_values = cluster_data['stats'].values
                                            if len(bin_indices) > 0 and len(bin_indices) <= len(distances):
                                                valid_indices = bin_indices[bin_indices < len(distances)]
                                                if len(valid_indices) > 0:
                                                    cluster_distances = distances[valid_indices]
                                                    cluster_vals = cluster_values[bin_indices < len(distances)]
                                                    valid_mask = ~np.isnan(cluster_vals)
                                                    if np.any(valid_mask):
                                                        ax.plot(cluster_distances[valid_mask], cluster_vals[valid_mask],
                                                               label=self._get_cluster_display_name(cluster),
                                                               marker='o', markersize=4, linewidth=1.5)
                                                        plotted = True
                                                        print(f"[DEBUG] Plotted curve for cluster {cluster}: {len(cluster_distances[valid_mask])} points")
                                    
                                    # Add expected CSR line for L function in alternative approach
                                    if mode == 'L' and len(distances) > 0:
                                        ax.plot(distances, distances, 
                                               color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                                               label='Expected (CSR)', zorder=0)
                            else:
                                # No bins column, use index
                                pivot_df = stat_df.pivot(index=stat_df.index, columns='cluster', values='stats')
                                print(f"[DEBUG] Pivoted DataFrame shape: {pivot_df.shape}, columns: {list(pivot_df.columns)}")
                                
                                # Use distances array if it matches, otherwise use index
                                if len(distances) == len(pivot_df.index):
                                    plot_distances = distances
                                else:
                                    plot_distances = np.arange(len(pivot_df.index))
                                
                                # Plot each cluster column
                                for col in pivot_df.columns:
                                    values = pivot_df[col].values
                                    valid_mask = ~np.isnan(values)
                                    if np.any(valid_mask):
                                        valid_distances = plot_distances[valid_mask]
                                        valid_values = values[valid_mask]
                                        if len(valid_distances) > 0 and len(valid_values) > 0:
                                            ax.plot(valid_distances, valid_values, 
                                                   label=self._get_cluster_display_name(col),
                                                   marker='o', markersize=4, linewidth=1.5)
                                            plotted = True
                                            print(f"[DEBUG] Plotted curve for cluster {col}: {len(valid_distances)} points")
                        else:
                            # Wide format: each column is a cluster, rows are distance bins
                            print(f"[DEBUG] DataFrame is in wide format")
                            for col in stat_df.columns:
                                if col == 'bins':  # Skip bins column if it exists
                                    continue
                                values = stat_df[col].values
                                # Remove NaN values
                                valid_mask = ~np.isnan(values)
                                if np.any(valid_mask):
                                    valid_distances = distances[valid_mask]
                                    valid_values = values[valid_mask]
                                    if len(valid_distances) > 0 and len(valid_values) > 0:
                                        ax.plot(valid_distances, valid_values, 
                                               label=self._get_cluster_display_name(col),
                                               marker='o', markersize=4, linewidth=1.5)
                                        plotted = True
                                        print(f"[DEBUG] Plotted curve for cluster {col}: {len(valid_distances)} points")
                else:
                    print(f"[DEBUG] No '{stat_key}' found in ripley_data")
            
            # Fallback: Try old format (cluster-specific keys)
            if not plotted:
                print(f"[DEBUG] Trying fallback format - cluster-specific keys")
                # Get cluster categories
                if cluster_key in adata.obs.columns:
                    if hasattr(adata.obs[cluster_key], 'cat'):
                        categories = adata.obs[cluster_key].cat.categories
                    else:
                        categories = adata.obs[cluster_key].unique()
                else:
                    categories = []
                
                for cluster in categories:
                    for key_format in [f"{cluster_key}_{cluster}", f"{cluster}", str(cluster)]:
                        if key_format in ripley_data:
                            data = ripley_data[key_format]
                            print(f"[DEBUG] Found data for key: {key_format}, type: {type(data)}")
                            if isinstance(data, dict):
                                distances = data.get('distances', data.get('interval', []))
                                values = data.get(mode, data.get(mode.lower(), []))
                                print(f"[DEBUG] Extracted distances: {len(distances) if hasattr(distances, '__len__') else 'N/A'}, values: {len(values) if hasattr(values, '__len__') else 'N/A'}")
                                if len(distances) > 0 and len(values) > 0:
                                    distances = np.array(distances) if not isinstance(distances, np.ndarray) else distances
                                    values = np.array(values) if not isinstance(values, np.ndarray) else values
                                    min_len = min(len(distances), len(values))
                                    if min_len > 0:
                                        ax.plot(distances[:min_len], values[:min_len], 
                                               label=self._get_cluster_display_name(cluster), 
                                               marker='o', markersize=4, linewidth=1.5)
                                        plotted = True
                                        print(f"[DEBUG] Plotted curve for cluster {cluster}: {min_len} points")
                                        break
            
            # Last resort: try to find any plottable data
            if not plotted:
                print(f"[DEBUG] No cluster-specific data found, trying all keys")
                for key, value in ripley_data.items():
                    print(f"[DEBUG] Checking key: {key}, value type: {type(value)}")
                    if isinstance(value, pd.DataFrame):
                        # DataFrame - try to plot columns
                        print(f"[DEBUG] DataFrame with columns: {list(value.columns)}")
                        # Try to infer distances from index or use row numbers
                        if 'bins' in ripley_data:
                            distances = np.array(ripley_data['bins'])
                            for col in value.columns:
                                values = value[col].values
                                valid_mask = ~np.isnan(values)
                                if np.any(valid_mask) and len(distances) == len(values):
                                    valid_distances = distances[valid_mask]
                                    valid_values = values[valid_mask]
                                    if len(valid_distances) > 0:
                                        ax.plot(valid_distances, valid_values, 
                                               label=str(col), marker='o', markersize=4, linewidth=1.5)
                                        plotted = True
                                        print(f"[DEBUG] Plotted curve for column {col}")
                    elif isinstance(value, dict):
                        distances = value.get('distances', value.get('interval', []))
                        values = value.get(mode, value.get(mode.lower(), []))
                        print(f"[DEBUG] Key {key}: distances={len(distances) if hasattr(distances, '__len__') else 'N/A'}, values={len(values) if hasattr(values, '__len__') else 'N/A'}")
                        if len(distances) > 0 and len(values) > 0:
                            distances = np.array(distances) if not isinstance(distances, np.ndarray) else distances
                            values = np.array(values) if not isinstance(values, np.ndarray) else values
                            min_len = min(len(distances), len(values))
                            if min_len > 0:
                                ax.plot(distances[:min_len], values[:min_len], 
                                       label=str(key), marker='o', markersize=4, linewidth=1.5)
                                plotted = True
                                print(f"[DEBUG] Plotted curve for key {key}: {min_len} points")
            
            if plotted:
                ax.set_xlabel('Distance (µm)')
                ax.set_ylabel(f"Ripley's {mode}")
                ax.set_title(f"Ripley's {mode} Function")
                
                # Add expected random pattern line for uncentered Ripley functions
                # Squidpy returns uncentered versions:
                # - F(r): cumulative distribution function, under CSR: F(r) = 1 - exp(-λπr²) where λ is density
                # - G(r): nearest neighbor distance CDF, under CSR: G(r) = 1 - exp(-λπr²)
                # - L(r): L(r) = sqrt(K(r)/π), under CSR where K(r) = πr²: L(r) = r
                # For L function, we can plot L(r) = r as the expected line
                # For F and G, we need density estimation which is complex, so we skip for now
                if 'bins' in ripley_data and mode == 'L':
                    bins = ripley_data['bins']
                    if isinstance(bins, np.ndarray) and len(bins) > 0:
                        # For L function, expected value under CSR is L(r) = r
                        expected_values = bins
                        ax.plot(bins, expected_values, color='gray', 
                               linestyle='--', linewidth=1.5, alpha=0.7,
                               label='Expected (CSR)', zorder=0)
                
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 0:
                    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                print(f"[DEBUG] Plot completed successfully")
            else:
                print(f"[DEBUG] Failed to plot - data format not recognized")
                ax.text(0.5, 0.5, f'Unable to plot Ripley\'s {mode} data.\nData format not recognized.\nCheck debug output for details.', 
                       ha='center', va='center', transform=ax.transAxes)
        elif isinstance(ripley_data, pd.DataFrame):
            print(f"[DEBUG] ripley_data is DataFrame, shape: {ripley_data.shape}, columns: {list(ripley_data.columns)}")
            # Handle DataFrame format if needed
            ax.text(0.5, 0.5, f'Ripley data is in DataFrame format.\nPlotting not yet implemented for DataFrame.\nCheck debug output for structure.', 
                   ha='center', va='center', transform=ax.transAxes)
        else:
            print(f"[DEBUG] ripley_data is unexpected type: {type(ripley_data)}")
            ax.text(0.5, 0.5, 'No Ripley data found.\nPlease run Ripley analysis first.', 
                   ha='center', va='center', transform=ax.transAxes)
        
        self.sq_ripley_canvas.figure.tight_layout()
        self.sq_ripley_canvas.draw()
    
    def _save_sq_ripley_plot(self):
        """Save the Ripley plot."""
        if save_figure_with_options(self.sq_ripley_canvas.figure, "squidpy_ripley.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")
    
    def _export_to_anndata(self):
        """Export data to AnnData format (combined or per-ROI)."""
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
                
                # Concatenate all AnnData objects
                import anndata as ad
                adata_list = list(self.anndata_cache.values())
                if len(adata_list) == 1:
                    combined_adata = adata_list[0]
                else:
                    combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
                
                combined_adata.write(file_path)
                QtWidgets.QMessageBox.information(self, "Export Complete", 
                    f"Combined AnnData exported to:\n{file_path}")
            else:
                # Separate files
                export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
                if not export_dir:
                    return
                
                for roi_id, adata in self.anndata_cache.items():
                    file_path = os.path.join(export_dir, f"anndata_roi_{roi_id}.h5ad")
                    adata.write(file_path)
                
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
            'sq_nhood_save_btn', 'sq_cooccur_save_btn', 'sq_autocorr_save_btn', 'sq_ripley_save_btn'
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
        
        # Clear AnnData cache since dataframe changed
        self.anndata_cache = {}
        
        # Refresh ROI combo boxes and other UI elements that depend on dataframe
        self._populate_roi_combo()
    
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

