"""
Spatial Analysis Dialog for OpenIMC

This module provides comprehensive spatial analysis capabilities including:
- Graph construction (kNN and radius-based)
- Pairwise enrichment analysis with permutation testing
- Distance distribution analysis
- Ripley K/L functions for spatial clustering analysis
- Neighborhood composition analysis

IMPORTANT: Centroid coordinates (centroid_x, centroid_y) are in PIXELS.
All distance calculations are converted to micrometers using pixel_size_um.
The pixel_size_um is retrieved from the parent window for each ROI.

Required columns:
- acquisition_id: ROI identifier
- cell_id: Unique cell identifier
- centroid_x, centroid_y: Cell centroid coordinates in pixels
- cluster: Cell cluster labels (created by clustering analysis)
- cluster_phenotype: Cell phenotype labels (optional, created by annotation)

The analysis automatically detects cluster columns in order: 'cluster', 'cluster_phenotype', 'cluster_id'.
Run clustering analysis first to create the 'cluster' column.
"""

from typing import Optional, Dict, Any, Tuple, List

import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from scipy.spatial import cKDTree, Delaunay
import networkx as nx
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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


class SpatialAnalysisDialog(QtWidgets.QDialog):
    def __init__(self, feature_dataframe: pd.DataFrame, batch_corrected_dataframe=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial Analysis")
        self.setMinimumSize(900, 650)

        self.original_feature_dataframe = feature_dataframe  # Store original
        self.batch_corrected_dataframe = batch_corrected_dataframe  # Store batch-corrected
        self.feature_dataframe = feature_dataframe  # Active dataframe (can be switched)
        self.edge_df: Optional[pd.DataFrame] = None
        self.adj_matrices: Dict[str, sp.csr_matrix] = {}  # Per-ROI adjacency matrices
        self.cell_id_to_gid: Dict[Tuple[str, int], int] = {}  # (roi_id, cell_id) -> global_id
        self.gid_to_cell_id: Dict[int, Tuple[str, int]] = {}  # global_id -> (roi_id, cell_id)
        self.metadata: Dict[str, Any] = {}
        self.neighborhood_df: Optional[pd.DataFrame] = None
        self.cluster_summary_df: Optional[pd.DataFrame] = None
        self.enrichment_df: Optional[pd.DataFrame] = None
        self.distance_df: Optional[pd.DataFrame] = None
        self.ripley_df: Optional[pd.DataFrame] = None
        self.rng_seed: int = 42  # Default seed for reproducibility
        
        # Cache for spatial visualization to avoid recomputing edges
        self.spatial_viz_cache: Dict[str, Any] = {}  # roi_id -> {edges, cell_coords, roi_df}
        
        # Track which analyses have been run
        self.neighborhood_analysis_run = False
        self.enrichment_analysis_run = False
        self.distance_analysis_run = False
        self.ripley_analysis_run = False
        self.spatial_viz_run = False
        self.community_analysis_run = False
        
        # Cluster annotation mapping (will be populated from parent if available)
        self.cluster_annotation_map = {}

        self._create_ui()
    
    def _on_feature_set_changed(self):
        """Handle feature set selection change."""
        if not hasattr(self, 'feature_set_combo'):
            return
        
        selected = self.feature_set_combo.currentText()
        if selected == "Batch-Corrected Features" and self.batch_corrected_dataframe is not None:
            self.feature_dataframe = self.batch_corrected_dataframe.copy()
        else:
            self.feature_dataframe = self.original_feature_dataframe.copy()
        
        # Clear cached analyses when switching feature sets
        self.edge_df = None
        self.adj_matrices = {}
        self.cell_id_to_gid = {}
        self.gid_to_cell_id = {}
        self.neighborhood_df = None
        self.cluster_summary_df = None
        self.enrichment_df = None
        self.distance_df = None
        self.ripley_df = None
        self.spatial_viz_cache = {}
        
        # Reset analysis flags
        self.neighborhood_analysis_run = False
        self.enrichment_analysis_run = False
        self.distance_analysis_run = False
        self.ripley_analysis_run = False
        self.spatial_viz_run = False
        self.community_analysis_run = False
    
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Feature set selector (if batch-corrected data is available)
        if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
            feature_set_layout = QtWidgets.QHBoxLayout()
            feature_set_layout.addWidget(QtWidgets.QLabel("Feature Set:"))
            self.feature_set_combo = QtWidgets.QComboBox()
            self.feature_set_combo.addItem("Original Features")
            self.feature_set_combo.addItem("Batch-Corrected Features")
            self.feature_set_combo.setCurrentText("Original Features")
            self.feature_set_combo.setToolTip("Choose between original or batch-corrected feature sets")
            self.feature_set_combo.currentTextChanged.connect(self._on_feature_set_changed)
            feature_set_layout.addWidget(self.feature_set_combo)
            feature_set_layout.addStretch()
            layout.addLayout(feature_set_layout)

        # Parameters group
        params_group = QtWidgets.QGroupBox("Spatial Graph Construction")
        params_layout = QtWidgets.QGridLayout(params_group)

        self.graph_mode_combo = QtWidgets.QComboBox()
        self.graph_mode_combo.addItems(["kNN", "Radius", "Delaunay"])
        self.graph_mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 64)
        self.k_spin.setValue(6)
        
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 500.0)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setValue(20.0)
        
        # Permutation parameters for enrichment analysis (will be added to enrichment tab)
        self.n_perm_spin = QtWidgets.QSpinBox()
        self.n_perm_spin.setRange(10, 1000)
        self.n_perm_spin.setValue(100)
        
        # Ripley parameters (will be added to Ripley tab)
        self.ripley_r_max_spin = QtWidgets.QDoubleSpinBox()
        self.ripley_r_max_spin.setRange(1.0, 200.0)
        self.ripley_r_max_spin.setDecimals(1)
        self.ripley_r_max_spin.setValue(50.0)
        
        self.ripley_n_steps_spin = QtWidgets.QSpinBox()
        self.ripley_n_steps_spin.setRange(5, 50)
        self.ripley_n_steps_spin.setValue(20)

        params_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        params_layout.addWidget(self.graph_mode_combo, 0, 1)
        
        self.k_label = QtWidgets.QLabel("k:")
        params_layout.addWidget(self.k_label, 0, 2)
        params_layout.addWidget(self.k_spin, 0, 3)
        
        self.radius_label = QtWidgets.QLabel("Radius (µm):")
        params_layout.addWidget(self.radius_label, 0, 4)
        params_layout.addWidget(self.radius_spin, 0, 5)
        
        # Random seed
        params_layout.addWidget(QtWidgets.QLabel("Random Seed:"), 0, 6)
        self.seed_spinbox = QtWidgets.QSpinBox()
        self.seed_spinbox.setRange(0, 2**31 - 1)
        self.seed_spinbox.setValue(42)
        self.seed_spinbox.setToolTip("Random seed for reproducibility (default: 42)")
        params_layout.addWidget(self.seed_spinbox, 0, 7)
        
        # Initially show kNN controls, hide radius
        self._on_mode_changed()

        layout.addWidget(params_group)

        # Actions
        action_row = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export Results…")
        self.export_btn.setEnabled(False)
        self.export_graph_btn = QtWidgets.QPushButton("Export Graph…")
        self.export_graph_btn.setEnabled(False)
        action_row.addWidget(self.export_btn)
        action_row.addWidget(self.export_graph_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        # Results tabs with matplotlib canvases
        self.tabs = QtWidgets.QTabWidget()
        
        # Neighborhood Composition tab
        self.neighborhood_tab = QtWidgets.QWidget()
        neighborhood_layout = QtWidgets.QVBoxLayout(self.neighborhood_tab)
        
        # Neighborhood run button and save plot button
        neighborhood_btn_layout = QtWidgets.QHBoxLayout()
        self.neighborhood_run_btn = QtWidgets.QPushButton("Run Neighborhood Analysis")
        self.neighborhood_save_btn = QtWidgets.QPushButton("Save Plot")
        self.neighborhood_save_btn.setEnabled(False)
        neighborhood_btn_layout.addWidget(self.neighborhood_run_btn)
        neighborhood_btn_layout.addWidget(self.neighborhood_save_btn)
        neighborhood_btn_layout.addStretch()
        neighborhood_layout.addLayout(neighborhood_btn_layout)
        
        self.neighborhood_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        neighborhood_layout.addWidget(self.neighborhood_canvas)
        self.tabs.addTab(self.neighborhood_tab, "Neighborhood Composition")
        
        # Pairwise Enrichment tab
        self.enrichment_tab = QtWidgets.QWidget()
        enrichment_layout = QtWidgets.QVBoxLayout(self.enrichment_tab)
        
        # Enrichment parameters and run button
        enrichment_params = QtWidgets.QHBoxLayout()
        enrichment_params.addWidget(QtWidgets.QLabel("Permutations:"))
        enrichment_params.addWidget(self.n_perm_spin)
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
        
        # Distance run button and save plot button
        distance_btn_layout = QtWidgets.QHBoxLayout()
        self.distance_run_btn = QtWidgets.QPushButton("Run Distance Analysis")
        self.distance_save_btn = QtWidgets.QPushButton("Save Plot")
        self.distance_save_btn.setEnabled(False)
        distance_btn_layout.addWidget(self.distance_run_btn)
        distance_btn_layout.addWidget(self.distance_save_btn)
        distance_btn_layout.addStretch()
        distance_layout.addLayout(distance_btn_layout)
        
        self.distance_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        distance_layout.addWidget(self.distance_canvas)
        self.tabs.addTab(self.distance_tab, "Distance Distributions")
        
        # Ripley K/L tab
        self.ripley_tab = QtWidgets.QWidget()
        ripley_layout = QtWidgets.QVBoxLayout(self.ripley_tab)
        
        # Ripley parameters and run button
        ripley_params = QtWidgets.QHBoxLayout()
        ripley_params.addWidget(QtWidgets.QLabel("Max radius (µm):"))
        ripley_params.addWidget(self.ripley_r_max_spin)
        ripley_params.addWidget(QtWidgets.QLabel("Steps:"))
        ripley_params.addWidget(self.ripley_n_steps_spin)
        self.ripley_run_btn = QtWidgets.QPushButton("Run Ripley Analysis")
        self.ripley_save_btn = QtWidgets.QPushButton("Save Plot")
        self.ripley_save_btn.setEnabled(False)
        ripley_params.addWidget(self.ripley_run_btn)
        ripley_params.addWidget(self.ripley_save_btn)
        ripley_params.addStretch()
        ripley_layout.addLayout(ripley_params)
        
        self.ripley_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        ripley_layout.addWidget(self.ripley_canvas)
        self.tabs.addTab(self.ripley_tab, "Ripley K/L")
        
        # Spatial Visualization tab
        self.spatial_viz_tab = QtWidgets.QWidget()
        spatial_viz_layout = QtWidgets.QVBoxLayout(self.spatial_viz_tab)
        
        # Spatial visualization controls
        spatial_viz_controls = QtWidgets.QHBoxLayout()
        self.roi_label = QtWidgets.QLabel("ROI:")
        spatial_viz_controls.addWidget(self.roi_label)
        self.roi_combo = QtWidgets.QComboBox()
        spatial_viz_controls.addWidget(self.roi_combo)
        
        # Add multi-select option
        self.faceted_plot_check = QtWidgets.QCheckBox("Faceted plot (all/selected ROIs)")
        self.faceted_plot_check.setToolTip("Create a faceted plot showing multiple ROIs side by side")
        spatial_viz_controls.addWidget(self.faceted_plot_check)
        
        # ROI list widget for multi-selection (hidden by default)
        self.roi_list_widget = QtWidgets.QListWidget()
        self.roi_list_widget.setMaximumHeight(100)
        self.roi_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.roi_list_widget.setVisible(False)
        
        spatial_viz_controls.addWidget(QtWidgets.QLabel("Color by:"))
        self.spatial_color_combo = QtWidgets.QComboBox()
        self.spatial_color_combo.setEditable(True)
        self.spatial_color_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.spatial_color_combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.spatial_color_combo.setToolTip("Search and select feature for color encoding")
        spatial_viz_controls.addWidget(self.spatial_color_combo)
        
        spatial_viz_controls.addWidget(QtWidgets.QLabel("Size by:"))
        self.spatial_size_combo = QtWidgets.QComboBox()
        self.spatial_size_combo.setEditable(True)
        self.spatial_size_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.spatial_size_combo.completer().setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.spatial_size_combo.addItem("None")
        self.spatial_size_combo.setToolTip("Search and select feature for size encoding (optional)")
        spatial_viz_controls.addWidget(self.spatial_size_combo)
        
        self.spatial_viz_run_btn = QtWidgets.QPushButton("Generate Spatial Plot")
        self.spatial_viz_save_btn = QtWidgets.QPushButton("Save Plot")
        self.spatial_viz_save_btn.setEnabled(False)
        spatial_viz_controls.addWidget(self.spatial_viz_run_btn)
        spatial_viz_controls.addWidget(self.spatial_viz_save_btn)
        spatial_viz_controls.addStretch()
        spatial_viz_layout.addLayout(spatial_viz_controls)
        
        # Add ROI list widget to layout
        spatial_viz_layout.addWidget(self.roi_list_widget)
        
        self.spatial_viz_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        spatial_viz_layout.addWidget(self.spatial_viz_canvas)
        self.tabs.addTab(self.spatial_viz_tab, "Spatial Visualization")
        
        # Spatial Community Analysis tab
        self.community_tab = QtWidgets.QWidget()
        community_layout = QtWidgets.QVBoxLayout(self.community_tab)
        
        # Community analysis controls
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
        
        # Cell type exclusion controls
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
        self.neighborhood_run_btn.clicked.connect(self._run_neighborhood_analysis)
        self.enrichment_run_btn.clicked.connect(self._run_enrichment_analysis)
        self.distance_run_btn.clicked.connect(self._run_distance_analysis)
        self.ripley_run_btn.clicked.connect(self._run_ripley_analysis)
        self.neighborhood_save_btn.clicked.connect(self._save_neighborhood_plot)
        self.enrichment_save_btn.clicked.connect(self._save_enrichment_plot)
        self.distance_save_btn.clicked.connect(self._save_distance_plot)
        self.ripley_save_btn.clicked.connect(self._save_ripley_plot)
        self.spatial_viz_run_btn.clicked.connect(self._run_spatial_visualization)
        self.spatial_viz_save_btn.clicked.connect(self._save_spatial_viz_plot)
        self.faceted_plot_check.toggled.connect(self._on_faceted_plot_toggled)
        # Connect color/size changes to update plot without regenerating
        self.spatial_color_combo.currentTextChanged.connect(self._on_spatial_viz_option_changed)
        self.spatial_size_combo.currentTextChanged.connect(self._on_spatial_viz_option_changed)
        self.community_run_btn.clicked.connect(self._run_community_analysis)
        self.community_save_btn.clicked.connect(self._save_community_plot)
        self.export_btn.clicked.connect(self._export_results)
        self.export_graph_btn.clicked.connect(self._export_graph)
        
        # Initialize tab states
        self._update_tab_states()
        
        # Try to get cluster annotations from parent dialog
        self._load_cluster_annotations()
        
        # Populate ROI combo box and color options
        self._populate_roi_combo()
        self._populate_spatial_color_options()
        self._populate_spatial_size_options()
        self._populate_community_roi_combo()
        self._populate_exclude_clusters_list()
        
        # Set up a timer to periodically check for annotation updates
        self.annotation_timer = QtCore.QTimer()
        self.annotation_timer.timeout.connect(self._check_annotation_updates)
        self.annotation_timer.start(2000)  # Check every 2 seconds

    def _on_mode_changed(self):
        """Handle mode change to show/hide relevant controls."""
        mode = self.graph_mode_combo.currentText()
        is_knn = mode == "kNN"
        is_delaunay = mode == "Delaunay"

        # Show/hide k controls
        self.k_label.setVisible(is_knn)
        self.k_spin.setVisible(is_knn)
        
        # Show/hide radius controls  
        self.radius_label.setVisible(not is_knn and not is_delaunay)
        self.radius_spin.setVisible(not is_knn and not is_delaunay)

    def _update_tab_states(self):
        """Update tab enabled/disabled states based on analysis progress."""
        # Neighborhood tab is always enabled (it's the first step)
        self.tabs.setTabEnabled(0, True)  # Neighborhood Composition
        
        # Other tabs are only enabled after neighborhood analysis is run
        self.tabs.setTabEnabled(1, self.neighborhood_analysis_run)  # Pairwise Enrichment
        self.tabs.setTabEnabled(2, self.neighborhood_analysis_run)  # Distance Distributions
        self.tabs.setTabEnabled(3, self.neighborhood_analysis_run)  # Ripley K/L
        self.tabs.setTabEnabled(4, self.neighborhood_analysis_run)  # Spatial Visualization
        self.tabs.setTabEnabled(5, self.neighborhood_analysis_run)  # Spatial Communities
        
        # Update button states
        self.enrichment_run_btn.setEnabled(self.neighborhood_analysis_run)
        self.distance_run_btn.setEnabled(self.neighborhood_analysis_run)
        self.ripley_run_btn.setEnabled(self.neighborhood_analysis_run)
        
        # Update save button states
        self.neighborhood_save_btn.setEnabled(self.neighborhood_analysis_run)
        self.enrichment_save_btn.setEnabled(self.enrichment_analysis_run)
        self.distance_save_btn.setEnabled(self.distance_analysis_run)
        self.ripley_save_btn.setEnabled(self.ripley_analysis_run)
        self.spatial_viz_save_btn.setEnabled(self.spatial_viz_run)
        self.community_save_btn.setEnabled(self.community_analysis_run)
        
        # Enable export graph button if graph is built
        self.export_graph_btn.setEnabled(
            self.edge_df is not None and not self.edge_df.empty
        )

    def _load_cluster_annotations(self):
        """Load cluster annotations from parent dialog if available."""
        try:
            # Try to get annotations from parent dialog
            parent = self.parent()
            if parent is not None and hasattr(parent, 'cluster_annotation_map'):
                self.cluster_annotation_map = parent.cluster_annotation_map.copy()
                print(f"[DEBUG] Loaded {len(self.cluster_annotation_map)} cluster annotations from parent")
            elif parent is not None and hasattr(parent, '_get_cluster_display_name'):
                # If parent has the method, we can use it directly
                print("[DEBUG] Parent has cluster display name method available")
            
            # Also check if cluster_phenotype column exists in the dataframe
            if self.feature_dataframe is not None and 'cluster_phenotype' in self.feature_dataframe.columns:
                # Build annotation map from cluster_phenotype column
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
                    print(f"[DEBUG] Loaded {len(phenotype_map)} cluster annotations from cluster_phenotype column")
                    
        except Exception as e:
            print(f"[DEBUG] Could not load cluster annotations: {e}")

    def _get_cluster_display_name(self, cluster_id):
        """Return display label for a cluster id, using annotation if available."""
        # Try to get from parent dialog first (has display names)
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, '_get_cluster_display_name'):
                return parent._get_cluster_display_name(cluster_id)
        except Exception:
            pass
        
        # Fall back to annotation map, converting underscores to spaces for display
        if isinstance(self.cluster_annotation_map, dict) and cluster_id in self.cluster_annotation_map and self.cluster_annotation_map[cluster_id]:
            name = self.cluster_annotation_map[cluster_id]
            # Convert underscores to spaces for display (in case we have backend names)
            return name.replace('_', ' ')
            
        return f"Cluster {int(cluster_id)}"

    def _check_annotation_updates(self):
        """Check if cluster annotations have been updated and refresh if needed."""
        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, 'cluster_annotation_map'):
                # Check if parent's annotation map has changed
                parent_map = parent.cluster_annotation_map
                if parent_map != self.cluster_annotation_map:
                    self.cluster_annotation_map = parent_map.copy()
                    print(f"[DEBUG] Updated cluster annotations: {len(self.cluster_annotation_map)} annotations")
                    
                    # Refresh plots if they exist
                    if self.neighborhood_analysis_run:
                        self._update_neighborhood_plot()
                    if self.enrichment_analysis_run:
                        self._update_enrichment_plot()
                    if self.distance_analysis_run:
                        self._update_distance_plot()
                    if self.ripley_analysis_run:
                        self._update_ripley_plot()
                    if self.spatial_viz_run:
                        # Refresh spatial visualization with current ROI
                        current_roi = self.roi_combo.currentText()
                        if current_roi:
                            self._create_spatial_visualization(current_roi)
        except Exception as e:
            # Silently ignore errors to avoid spam
            pass

    def _validate_data(self):
        """Validate that required data is available for analysis."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            print("[DEBUG] ERROR: Feature dataframe is None or empty")
            QtWidgets.QMessageBox.warning(self, "No Data", "Feature dataframe is empty.")
            return False
            
        print(f"[DEBUG] Feature dataframe shape: {self.feature_dataframe.shape}")
        print(f"[DEBUG] Feature dataframe columns: {list(self.feature_dataframe.columns)}")
        
        required_cols = {"acquisition_id", "cell_id", "centroid_x", "centroid_y"}
        missing = [c for c in required_cols if c not in self.feature_dataframe.columns]
        if missing:
            print(f"[DEBUG] ERROR: Missing required columns: {missing}")
            QtWidgets.QMessageBox.critical(self, "Missing columns", f"Missing required columns: {', '.join(missing)}")
            return False

        # Check for cluster column - look for cluster, cluster_phenotype, or cluster_id
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
                
        if cluster_col is None:
            QtWidgets.QMessageBox.critical(self, "Missing cluster labels", 
                "No cluster column found. Please run clustering analysis first.\n"
                "Expected columns: 'cluster', 'cluster_phenotype', or 'cluster_id'")
            return False
            
        return True

    def _build_spatial_graph(self):
        """Build the spatial graph (edges and adjacency matrices)."""
        if hasattr(self, 'edge_df') and self.edge_df is not None and not self.edge_df.empty:
            print("[DEBUG] Spatial graph already built, skipping...")
            return True
            
        print("[DEBUG] Building spatial graph...")
        
        mode = self.graph_mode_combo.currentText()
        k = int(self.k_spin.value())
        radius_um = float(self.radius_spin.value())
        
        # Get seed from UI and set for reproducibility
        self.rng_seed = self.seed_spinbox.value()
        random.seed(self.rng_seed)
        np.random.seed(self.rng_seed)
        
        print(f"[DEBUG] Analysis parameters - Mode: {mode}, k: {k}, radius_um: {radius_um}")
        print(f"[DEBUG] Random seed: {self.rng_seed}")

        try:
            # Initialize global cell ID mapping
            self.cell_id_to_gid = {}
            self.gid_to_cell_id = {}
            self.adj_matrices = {}
            
            # Step 1: Build global cell ID mapping and per-ROI adjacency matrices
            edge_records = []
            global_id_counter = 0
            
            parent = self.parent() if hasattr(self, 'parent') else None
            print(f"[DEBUG] Parent available: {parent is not None}")

            # Process per ROI/acquisition to respect ROI boundaries
            roi_groups = list(self.feature_dataframe.groupby('acquisition_id'))
            print(f"[DEBUG] Found {len(roi_groups)} ROI groups")
            
            for roi_idx, (roi_id, roi_df) in enumerate(roi_groups):
                print(f"[DEBUG] Processing ROI {roi_idx + 1}/{len(roi_groups)}: {roi_id}")
                print(f"[DEBUG] ROI dataframe shape: {roi_df.shape}")
                
                roi_df = roi_df.dropna(subset=["centroid_x", "centroid_y"])  # ensure valid coordinates
                print(f"[DEBUG] After dropping NaN coordinates: {roi_df.shape}")
                
                if roi_df.empty:
                    print(f"[DEBUG] ROI {roi_id} is empty after dropping NaN, skipping")
                    continue
                    
                coords_px = roi_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
                cell_ids = roi_df["cell_id"].astype(int).to_numpy()
                
                # Build global ID mapping for this ROI
                roi_id_str = str(roi_id)
                roi_cell_to_gid = {}
                for i, cell_id in enumerate(cell_ids):
                    gid = global_id_counter + i
                    self.cell_id_to_gid[(roi_id_str, int(cell_id))] = gid
                    self.gid_to_cell_id[gid] = (roi_id_str, int(cell_id))
                    roi_cell_to_gid[int(cell_id)] = gid
                
                global_id_counter += len(cell_ids)
                
                print(f"[DEBUG] Coords shape: {coords_px.shape}, Cell IDs shape: {cell_ids.shape}")

                # Get pixel size in µm for this ROI
                pixel_size_um = 1.0
                try:
                    if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                        pixel_size_um = float(parent._get_pixel_size_um(roi_id))  # type: ignore[attr-defined]
                        print(f"[DEBUG] Got pixel size from parent: {pixel_size_um}")
                except Exception as e:
                    print(f"[DEBUG] Could not get pixel size from parent: {e}")
                    pixel_size_um = 1.0

                print(f"[DEBUG] Creating cKDTree with {len(coords_px)} points")
                tree = cKDTree(coords_px)
                print(f"[DEBUG] cKDTree created successfully")

                # Use set to deduplicate edges during construction
                roi_edges_set = set()
                roi_edges_list = []

                if mode == "kNN":
                    print(f"[DEBUG] Running kNN analysis with k={k}")
                    # Query k+1 (including self), exclude self idx 0
                    query_k = min(k + 1, max(2, len(coords_px)))
                    print(f"[DEBUG] Query k: {query_k}")
                    
                    try:
                        dists, idxs = tree.query(coords_px, k=query_k)
                        print(f"[DEBUG] Query completed - dists type: {type(dists)}, shape: {getattr(dists, 'shape', 'scalar')}")
                        print(f"[DEBUG] Query completed - idxs type: {type(idxs)}, shape: {getattr(idxs, 'shape', 'scalar')}")
                        
                        # Handle scalar case (when only 1 point or k=1)
                        if np.isscalar(dists):
                            print(f"[DEBUG] Handling scalar case")
                            dists = np.array([[dists]])
                            idxs = np.array([[idxs]])
                        # Ensure 2D for array case
                        elif dists.ndim == 1:
                            print(f"[DEBUG] Handling 1D array case")
                            dists = dists[:, None]
                            idxs = idxs[:, None]
                        
                        print(f"[DEBUG] Final dists shape: {dists.shape}, idxs shape: {idxs.shape}")
                        
                        for i in range(len(coords_px)):
                            src_cell_id = int(cell_ids[i])
                            for j in range(1, min(dists.shape[1], k + 1)):
                                nbr_idx = int(idxs[i, j])
                                if nbr_idx < 0 or nbr_idx >= len(coords_px):
                                    continue
                                dst_cell_id = int(cell_ids[nbr_idx])
                                
                                # Create canonical edge (smaller cell_id first)
                                edge_key = (min(src_cell_id, dst_cell_id), max(src_cell_id, dst_cell_id))
                                if edge_key not in roi_edges_set:
                                    roi_edges_set.add(edge_key)
                                    dist_um = float(dists[i, j]) * pixel_size_um
                                    roi_edges_list.append((src_cell_id, dst_cell_id, dist_um))
                        
                        print(f"[DEBUG] kNN analysis completed for ROI {roi_id}, found {len(roi_edges_list)} unique edges")
                        
                    except Exception as e:
                        print(f"[DEBUG] ERROR in kNN analysis for ROI {roi_id}: {e}")
                        print(f"[DEBUG] Exception type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                        raise
                        
                elif mode == "Radius":
                    print(f"[DEBUG] Running radius analysis with radius={radius_um}µm")
                    # Radius graph: convert radius µm to pixels
                    radius_px = radius_um / max(pixel_size_um, 1e-12)
                    print(f"[DEBUG] Radius in pixels: {radius_px}")
                    
                    try:
                        pairs = tree.query_pairs(r=radius_px)
                        print(f"[DEBUG] Found {len(pairs)} pairs within radius")
                        
                        for i, j in pairs:
                            a_id = int(cell_ids[int(i)])
                            b_id = int(cell_ids[int(j)])
                            
                            # Create canonical edge (smaller cell_id first)
                            edge_key = (min(a_id, b_id), max(a_id, b_id))
                            if edge_key not in roi_edges_set:
                                roi_edges_set.add(edge_key)
                                dist_um = float(np.linalg.norm(coords_px[int(i)] - coords_px[int(j)])) * pixel_size_um
                                roi_edges_list.append((a_id, b_id, dist_um))
                        
                        print(f"[DEBUG] Radius analysis completed for ROI {roi_id}, found {len(roi_edges_list)} unique edges")
                        
                    except Exception as e:
                        print(f"[DEBUG] ERROR in radius analysis for ROI {roi_id}: {e}")
                        print(f"[DEBUG] Exception type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                        raise
                        
                elif mode == "Delaunay":
                    print(f"[DEBUG] Running Delaunay triangulation for ROI {roi_id}")
                    
                    try:
                        # Delaunay triangulation
                        tri = Delaunay(coords_px)
                        print(f"[DEBUG] Delaunay triangulation created with {len(tri.simplices)} triangles")
                        
                        # Extract edges from simplices (triangles)
                        edges_set = set()
                        for simplex in tri.simplices:
                            # Each simplex has 3 vertices, create edges between all pairs
                            for i in range(3):
                                for j in range(i+1, 3):
                                    v1, v2 = simplex[i], simplex[j]
                                    # Create canonical edge (smaller index first)
                                    edge_key = (min(v1, v2), max(v1, v2))
                                    if edge_key not in edges_set:
                                        edges_set.add(edge_key)
                                        a_id = int(cell_ids[v1])
                                        b_id = int(cell_ids[v2])
                                        dist_um = float(np.linalg.norm(coords_px[v1] - coords_px[v2])) * pixel_size_um
                                        roi_edges_list.append((a_id, b_id, dist_um))
                        
                        print(f"[DEBUG] Delaunay analysis completed for ROI {roi_id}, found {len(roi_edges_list)} unique edges")
                        
                    except Exception as e:
                        print(f"[DEBUG] ERROR in Delaunay analysis for ROI {roi_id}: {e}")
                        print(f"[DEBUG] Exception type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                        raise
                else:
                    print(f"[DEBUG] ERROR: Unknown mode '{mode}' for ROI {roi_id}")
                    raise ValueError(f"Unknown graph construction mode: {mode}")

                # Build per-ROI adjacency matrix
                if _HAVE_SPARSE and len(roi_edges_list) > 0:
                    print(f"[DEBUG] Building adjacency matrix for ROI {roi_id}")
                    n_cells = len(cell_ids)
                    rows, cols, data = [], [], []
                    
                    for src_cell_id, dst_cell_id, _ in roi_edges_list:
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
                        print(f"[DEBUG] Created adjacency matrix for ROI {roi_id}: {adj_matrix.shape}")
                
                # Add edges to global edge records
                for src_cell_id, dst_cell_id, dist_um in roi_edges_list:
                    edge_records.append((roi_id_str, src_cell_id, dst_cell_id, dist_um))

            print(f"[DEBUG] Total edge records: {len(edge_records)}")

            # Create edge dataframe (no need for drop_duplicates since we deduplicated during construction)
            if edge_records:
                print(f"[DEBUG] Creating edge dataframe from {len(edge_records)} records")
                self.edge_df = pd.DataFrame(edge_records, columns=["roi_id", "cell_id_A", "cell_id_B", "distance_um"])
                print(f"[DEBUG] Edge dataframe shape: {self.edge_df.shape}")
            else:
                print(f"[DEBUG] No edge records, creating empty dataframe")
                self.edge_df = pd.DataFrame(columns=["roi_id", "cell_id_A", "cell_id_B", "distance_um"])

            # Update metadata
            self.metadata.update({
                "mode": mode,
                "k": k,
                "radius_um": radius_um,
                "rng_seed": self.rng_seed,
                "num_edges": int(len(self.edge_df)),
                "num_rois": len(roi_groups),
                "pixel_size_um": pixel_size_um,
            })

            print(f"[DEBUG] Spatial graph built successfully with {len(self.edge_df)} edges")
            
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
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
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
            print(f"[DEBUG] CRITICAL ERROR in spatial graph building: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Spatial Graph Error", f"Error: {str(e)}\n\nCheck console for detailed debug information.")
            return False

    def _run_neighborhood_analysis(self):
        """Run neighborhood composition analysis."""
        print("[DEBUG] Starting neighborhood analysis...")
        
        if not self._validate_data():
            return
            
        if not self._build_spatial_graph():
            return
            
        try:
            print(f"[DEBUG] Starting neighborhood composition analysis...")
            self._compute_neighborhood_composition()
            
            self.neighborhood_analysis_run = True
            self._update_tab_states()
            
            print(f"[DEBUG] Neighborhood analysis completed successfully")
            
            # Log neighborhood analysis
            logger = get_logger()
            acquisitions = list(self.feature_dataframe['acquisition_id'].unique()) if 'acquisition_id' in self.feature_dataframe.columns else []
            params = {
                "graph_mode": self.graph_mode_combo.currentText(),
                "k": int(self.k_spin.value()),
                "radius_um": float(self.radius_spin.value())
            }
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
            logger.log_spatial_analysis(
                analysis_type="neighborhood_composition",
                parameters=params,
                acquisitions=acquisitions,
                notes="Neighborhood composition analysis completed",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Neighborhood Analysis", "Neighborhood analysis completed successfully.")
            self.export_btn.setEnabled(True)
            
            # Update visualization
            print(f"[DEBUG] Updating neighborhood visualization...")
            self._update_neighborhood_plot()
            
        except Exception as e:
            print(f"[DEBUG] ERROR in neighborhood analysis: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Neighborhood Analysis Error", f"Error: {str(e)}")

    def _run_enrichment_analysis(self):
        """Run pairwise enrichment analysis."""
        print("[DEBUG] Starting enrichment analysis...")
        
        if not self.neighborhood_analysis_run:
            QtWidgets.QMessageBox.warning(self, "Prerequisite Required", "Please run neighborhood analysis first.")
            return
            
        try:
            n_perm = int(self.n_perm_spin.value())
            print(f"[DEBUG] Starting pairwise enrichment analysis with {n_perm} permutations...")
            self._compute_pairwise_enrichment(n_perm=n_perm)
            
            self.enrichment_analysis_run = True
            
            print(f"[DEBUG] Enrichment analysis completed successfully")
            
            # Log enrichment analysis
            logger = get_logger()
            acquisitions = list(self.feature_dataframe['acquisition_id'].unique()) if 'acquisition_id' in self.feature_dataframe.columns else []
            params = {
                "n_permutations": n_perm,
                "seed": self.seed_spinbox.value()
            }
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
            logger.log_spatial_analysis(
                analysis_type="pairwise_enrichment",
                parameters=params,
                acquisitions=acquisitions,
                notes=f"Pairwise enrichment analysis with {n_perm} permutations",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Enrichment Analysis", "Enrichment analysis completed successfully.")
            
            # Update visualization
            print(f"[DEBUG] Updating enrichment visualization...")
            self._update_enrichment_plot()
            
        except Exception as e:
            print(f"[DEBUG] ERROR in enrichment analysis: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Enrichment Analysis Error", f"Error: {str(e)}")

    def _run_distance_analysis(self):
        """Run distance distribution analysis."""
        print("[DEBUG] Starting distance analysis...")
        
        if not self.neighborhood_analysis_run:
            QtWidgets.QMessageBox.warning(self, "Prerequisite Required", "Please run neighborhood analysis first.")
            return
            
        try:
            print(f"[DEBUG] Starting distance distribution analysis...")
            self._compute_distance_distributions()
            
            self.distance_analysis_run = True
            
            print(f"[DEBUG] Distance analysis completed successfully")
            
            # Log distance analysis
            logger = get_logger()
            acquisitions = list(self.feature_dataframe['acquisition_id'].unique()) if 'acquisition_id' in self.feature_dataframe.columns else []
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
            logger.log_spatial_analysis(
                analysis_type="distance_distribution",
                parameters={},
                acquisitions=acquisitions,
                notes="Distance distribution analysis completed",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Distance Analysis", "Distance analysis completed successfully.")
            
            # Update visualization
            print(f"[DEBUG] Updating distance visualization...")
            self._update_distance_plot()
            
        except Exception as e:
            print(f"[DEBUG] ERROR in distance analysis: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Distance Analysis Error", f"Error: {str(e)}")

    def _run_ripley_analysis(self):
        """Run Ripley K/L functions analysis."""
        print("[DEBUG] Starting Ripley analysis...")
        
        if not self.neighborhood_analysis_run:
            QtWidgets.QMessageBox.warning(self, "Prerequisite Required", "Please run neighborhood analysis first.")
            return
            
        try:
            ripley_r_max = float(self.ripley_r_max_spin.value())
            ripley_n_steps = int(self.ripley_n_steps_spin.value())
            print(f"[DEBUG] Starting Ripley functions analysis with r_max={ripley_r_max}, steps={ripley_n_steps}...")
            self._compute_ripley_functions(r_max=ripley_r_max, n_steps=ripley_n_steps)
            
            self.ripley_analysis_run = True
            
            print(f"[DEBUG] Ripley analysis completed successfully")
            
            # Log Ripley analysis
            logger = get_logger()
            acquisitions = list(self.feature_dataframe['acquisition_id'].unique()) if 'acquisition_id' in self.feature_dataframe.columns else []
            params = {
                "r_max": ripley_r_max,
                "n_steps": ripley_n_steps
            }
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
            logger.log_spatial_analysis(
                analysis_type="ripley_k",
                parameters=params,
                acquisitions=acquisitions,
                notes=f"Ripley K/L functions analysis with r_max={ripley_r_max}, {ripley_n_steps} steps",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Ripley Analysis", "Ripley analysis completed successfully.")
            
            # Update visualization
            print(f"[DEBUG] Updating Ripley visualization...")
            self._update_ripley_plot()
            
        except Exception as e:
            print(f"[DEBUG] ERROR in Ripley analysis: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Ripley Analysis Error", f"Error: {str(e)}")

    def _populate_roi_combo(self):
        """Populate the ROI combo box and list widget with available ROIs."""
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            unique_rois = sorted(self.feature_dataframe['acquisition_id'].unique())
            self.roi_combo.clear()
            self.roi_combo.addItems([str(roi) for roi in unique_rois])
            
            # Also populate the list widget for multi-selection
            self.roi_list_widget.clear()
            for roi in unique_rois:
                item = QtWidgets.QListWidgetItem(str(roi))
                item.setCheckState(QtCore.Qt.Unchecked)
                self.roi_list_widget.addItem(item)
            
            print(f"[DEBUG] Populated ROI combo with {len(unique_rois)} ROIs")

    def _populate_spatial_color_options(self):
        """Populate the spatial color combo box with available features (searchable)."""
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            self.spatial_color_combo.clear()
            
            # Add cluster options first
            cluster_options = []
            for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
                if col in self.feature_dataframe.columns:
                    cluster_options.append(col)
            
            if cluster_options:
                # Use the first available cluster column as default
                default_cluster = cluster_options[0]
                self.spatial_color_combo.addItem(f"Cluster ({default_cluster})")
                
                # Add other cluster options if available
                for col in cluster_options[1:]:
                    self.spatial_color_combo.addItem(f"Cluster ({col})")
            
            # Add morphometric features
            morphometric_features = []
            for col in self.feature_dataframe.columns:
                if any(keyword in col.lower() for keyword in ['area', 'perimeter', 'diameter', 'eccentricity', 'solidity', 'extent', 'aspect_ratio', 'circularity']):
                    morphometric_features.append(col)
            
            for feature in sorted(morphometric_features):
                self.spatial_color_combo.addItem(f"Morphology: {feature}")
            
            # Add ALL marker expression features (not just hardcoded ones)
            marker_features = []
            for col in self.feature_dataframe.columns:
                if '_mean' in col.lower():
                    marker_features.append(col)
            
            for feature in sorted(marker_features):
                marker_name = feature.replace('_mean', '').replace('_', ' ')
                self.spatial_color_combo.addItem(f"Marker: {marker_name}")
            
            print(f"[DEBUG] Populated spatial color options with {self.spatial_color_combo.count()} options")

    def _populate_spatial_size_options(self):
        """Populate the spatial size combo box with available features (searchable)."""
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            self.spatial_size_combo.clear()
            self.spatial_size_combo.addItem("None")
            
            # Add morphometric features (good for size encoding)
            morphometric_features = []
            for col in self.feature_dataframe.columns:
                if any(keyword in col.lower() for keyword in ['area', 'perimeter', 'diameter', 'eccentricity', 'solidity', 'extent', 'aspect_ratio', 'circularity']):
                    morphometric_features.append(col)
            
            for feature in sorted(morphometric_features):
                self.spatial_size_combo.addItem(f"Morphology: {feature}")
            
            # Add marker expression features (can also be used for size)
            marker_features = []
            for col in self.feature_dataframe.columns:
                if '_mean' in col.lower():
                    marker_features.append(col)
            
            for feature in sorted(marker_features):
                marker_name = feature.replace('_mean', '').replace('_', ' ')
                self.spatial_size_combo.addItem(f"Marker: {marker_name}")
            
            print(f"[DEBUG] Populated spatial size options with {self.spatial_size_combo.count()} options")

    def _populate_community_roi_combo(self):
        """Populate the community ROI combo box with available ROIs."""
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            unique_rois = sorted(self.feature_dataframe['acquisition_id'].unique())
            self.community_roi_combo.clear()
            self.community_roi_combo.addItems([str(roi) for roi in unique_rois])
            print(f"[DEBUG] Populated community ROI combo with {len(unique_rois)} ROIs")

    def _populate_exclude_clusters_list(self):
        """Populate the exclude clusters list with available cluster types."""
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            self.exclude_clusters_list.clear()
            
            # Get cluster column
            cluster_col = None
            for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
                if col in self.feature_dataframe.columns:
                    cluster_col = col
                    break
            
            if cluster_col:
                unique_clusters = sorted(self.feature_dataframe[cluster_col].unique())
                for cluster in unique_clusters:
                    if pd.notna(cluster):
                        display_name = self._get_cluster_display_name(cluster)
                        item = QtWidgets.QListWidgetItem(display_name)
                        item.setData(QtCore.Qt.UserRole, cluster)
                        self.exclude_clusters_list.addItem(item)
                
                print(f"[DEBUG] Populated exclude clusters list with {self.exclude_clusters_list.count()} cluster types")

    def _on_faceted_plot_toggled(self, checked):
        """Handle faceted plot checkbox toggle."""
        self.roi_list_widget.setVisible(checked)
        self.roi_combo.setVisible(not checked)
        self.roi_label.setVisible(not checked)
        if checked:
            # Select all ROIs by default
            for i in range(self.roi_list_widget.count()):
                item = self.roi_list_widget.item(i)
                item.setCheckState(QtCore.Qt.Checked)

    def _run_spatial_visualization(self):
        """Generate spatial visualization for the selected ROI(s)."""
        print("[DEBUG] Starting spatial visualization...")
        
        if not self.neighborhood_analysis_run:
            QtWidgets.QMessageBox.warning(self, "Prerequisite Required", "Please run neighborhood analysis first.")
            return
        
        try:
            if self.faceted_plot_check.isChecked():
                # Get selected ROIs from list widget
                selected_rois = []
                for i in range(self.roi_list_widget.count()):
                    item = self.roi_list_widget.item(i)
                    if item.checkState() == QtCore.Qt.Checked:
                        selected_rois.append(item.text())
                
                if not selected_rois:
                    QtWidgets.QMessageBox.warning(self, "No ROIs Selected", "Please select at least one ROI for faceted plot.")
                    return
                
                print(f"[DEBUG] Generating faceted spatial visualization for {len(selected_rois)} ROIs")
                self._create_faceted_spatial_visualization(selected_rois)
            else:
                # Single ROI mode
                if self.roi_combo.currentText() == "":
                    QtWidgets.QMessageBox.warning(self, "No ROI Selected", "Please select an ROI to visualize.")
                    return
                
                selected_roi = self.roi_combo.currentText()
                print(f"[DEBUG] Generating spatial visualization for ROI: {selected_roi}")
                self._create_spatial_visualization(selected_roi)
            
            self.spatial_viz_run = True
            self._update_tab_states()
            
            print(f"[DEBUG] Spatial visualization completed successfully")
            QtWidgets.QMessageBox.information(self, "Spatial Visualization", "Spatial visualization completed successfully.")
            
        except Exception as e:
            print(f"[DEBUG] ERROR in spatial visualization: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Spatial Visualization Error", f"Error: {str(e)}")

    def _on_spatial_viz_option_changed(self):
        """Handle color/size option changes - update plot without regenerating edges."""
        if not self.spatial_viz_run:
            return  # Don't update if plot hasn't been generated yet
        
        if self.faceted_plot_check.isChecked():
            return  # Don't auto-update faceted plots
        
        current_roi = self.roi_combo.currentText()
        if current_roi and current_roi in self.spatial_viz_cache:
            # Update only the scatter plot, not edges
            self._update_spatial_scatter_plot(current_roi)

    def _create_spatial_visualization(self, roi_id, force_regenerate=False):
        """Create spatial visualization showing cells and edges for a specific ROI."""
        if self.feature_dataframe is None or self.edge_df is None:
            return
        
        # Determine if we should use cache BEFORE populating it
        use_cache = not force_regenerate and roi_id in self.spatial_viz_cache
            
        # Check cache first (unless forcing regeneration)
        if use_cache:
            cache = self.spatial_viz_cache[roi_id]
            roi_df = cache['roi_df']
            roi_edges = cache['roi_edges']
            cell_coords = cache['cell_coords']
        else:
            # Get data for the selected ROI
            roi_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == roi_id].copy()
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)].copy()
            
            if roi_df.empty:
                print(f"[DEBUG] No data found for ROI {roi_id}")
                return
            
            # Create a mapping from cell_id to coordinates
            cell_coords = {}
            for _, row in roi_df.iterrows():
                cell_coords[int(row['cell_id'])] = (row['centroid_x'], row['centroid_y'])
            
            # Cache the data
            self.spatial_viz_cache[roi_id] = {
                'roi_df': roi_df,
                'roi_edges': roi_edges,
                'cell_coords': cell_coords
            }
        
        # Get color and size options
        color_option = self.spatial_color_combo.currentText()
        size_option = self.spatial_size_combo.currentText()
        print(f"[DEBUG] Color option: {color_option}, Size option: {size_option}")
        
        # Clear the canvas only if forcing regeneration or not using cache
        if not use_cache:
            self.spatial_viz_canvas.figure.clear()
            ax = self.spatial_viz_canvas.figure.add_subplot(111)
            
            # Plot edges first (so they appear behind the nodes) - only if regenerating
            if not roi_edges.empty:
                print(f"[DEBUG] Plotting {len(roi_edges)} edges for ROI {roi_id}")
                
                # Plot edges
                for _, edge in roi_edges.iterrows():
                    cell_a = int(edge['cell_id_A'])
                    cell_b = int(edge['cell_id_B'])
                    
                    if cell_a in cell_coords and cell_b in cell_coords:
                        x_coords = [cell_coords[cell_a][0], cell_coords[cell_b][0]]
                        y_coords = [cell_coords[cell_a][1], cell_coords[cell_b][1]]
                        ax.plot(x_coords, y_coords, 'grey', alpha=0.3, linewidth=0.5, zorder=1)
        else:
            # Get existing axes
            ax = self.spatial_viz_canvas.figure.axes[0] if self.spatial_viz_canvas.figure.axes else None
            if ax is None:
                return
        
        # Update scatter plot (this will be called for both new and cached plots)
        self._update_spatial_scatter_plot(roi_id, ax=ax)
        
        # Customize the plot (only if not using cache)
        if not use_cache:
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            ax.set_title(f'Spatial Visualization - ROI {roi_id}')
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            self.spatial_viz_canvas.figure.tight_layout()
        
        self.spatial_viz_canvas.draw()
        print(f"[DEBUG] Spatial visualization created for ROI {roi_id} with {len(roi_df)} cells and {len(roi_edges)} edges")

    def _update_spatial_scatter_plot(self, roi_id, ax=None):
        """Update only the scatter plot (cells) without regenerating edges."""
        if roi_id not in self.spatial_viz_cache:
            return
        
        cache = self.spatial_viz_cache[roi_id]
        roi_df = cache['roi_df']
        cell_coords = cache['cell_coords']
        
        if ax is None:
            ax = self.spatial_viz_canvas.figure.axes[0] if self.spatial_viz_canvas.figure.axes else None
            if ax is None:
                return
        
        # Clear existing scatter plots and colorbars (but keep edges)
        for artist in ax.collections[:]:
            if isinstance(artist, plt.PathCollection):  # scatter plots
                artist.remove()
        for cbar in self.spatial_viz_canvas.figure.axes[1:]:  # colorbars
            cbar.remove()
        # Clear legend
        ax.legend_ = None
        
        # Get color and size options
        color_option = self.spatial_color_combo.currentText()
        size_option = self.spatial_size_combo.currentText()
        
        # Get size values if specified
        size_values = None
        size_feature_name = None
        if size_option and size_option != "None":
            if size_option.startswith("Morphology:"):
                size_feature_name = size_option.replace("Morphology: ", "")
            elif size_option.startswith("Marker:"):
                marker_name = size_option.replace("Marker: ", "").replace(" ", "_")
                size_feature_name = f"{marker_name}_mean"
            
            if size_feature_name and size_feature_name in roi_df.columns:
                size_values = roi_df[size_feature_name].values
                # Normalize sizes to reasonable range (20-200)
                valid_sizes = size_values[~np.isnan(size_values)]
                if len(valid_sizes) > 0:
                    min_size, max_size = valid_sizes.min(), valid_sizes.max()
                    if max_size > min_size:
                        size_values = 20 + 180 * (size_values - min_size) / (max_size - min_size)
                    else:
                        size_values = np.full_like(size_values, 50)
                else:
                    size_values = None
        else:
            size_values = 20  # Default size
        
        # Get all coordinates
        x_coords = roi_df['centroid_x'].values
        y_coords = roi_df['centroid_y'].values
        
        # Determine coloring method
        if color_option.startswith("Cluster"):
            # Extract cluster column name
            cluster_col = None
            if "(cluster)" in color_option:
                cluster_col = "cluster"
            elif "(cluster_phenotype)" in color_option:
                cluster_col = "cluster_phenotype"
            elif "(cluster_id)" in color_option:
                cluster_col = "cluster_id"
            else:
                # Default to first available cluster column
                for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
                    if col in roi_df.columns:
                        cluster_col = col
                        break
            
            if cluster_col is None:
                print(f"[DEBUG] No cluster column found for ROI {roi_id}")
                return
                
            # Get unique clusters and create color map
            unique_clusters = sorted(roi_df[cluster_col].unique())
            n_clusters = len(unique_clusters)
            
            # Use a vivid color palette that can handle many clusters
            colors = _get_vivid_colors(max(n_clusters, 3))
            cluster_color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
            
            # Plot cells (nodes) colored by cluster with optional size encoding
            for cluster in unique_clusters:
                cluster_mask = roi_df[cluster_col] == cluster
                cluster_cells = roi_df[cluster_mask]
                if not cluster_cells.empty:
                    cluster_x = cluster_cells['centroid_x'].values
                    cluster_y = cluster_cells['centroid_y'].values
                    
                    # Get sizes for this cluster
                    if isinstance(size_values, np.ndarray):
                        cluster_sizes = size_values[cluster_mask]
                    else:
                        cluster_sizes = size_values
                    
                    # Plot cells as points
                    ax.scatter(cluster_x, cluster_y, 
                              c=[cluster_color_map[cluster]], 
                              s=cluster_sizes, alpha=0.8, edgecolors='black', linewidth=0.5,
                              label=self._get_cluster_display_name(cluster),
                              zorder=2)
            
            # Add legend for clusters
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
        else:
            # Color by continuous feature (morphology or marker expression)
            feature_name = None
            if color_option.startswith("Morphology:"):
                feature_name = color_option.replace("Morphology: ", "")
            elif color_option.startswith("Marker:"):
                # Convert back to column name
                marker_name = color_option.replace("Marker: ", "").replace(" ", "_")
                feature_name = f"{marker_name}_mean"
            
            if feature_name and feature_name in roi_df.columns:
                # Get feature values
                feature_values = roi_df[feature_name].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(feature_values)
                if np.any(valid_mask):
                    valid_x = roi_df.loc[valid_mask, 'centroid_x'].values
                    valid_y = roi_df.loc[valid_mask, 'centroid_y'].values
                    valid_values = feature_values[valid_mask]
                    
                    # Get sizes for valid points
                    if isinstance(size_values, np.ndarray):
                        valid_sizes = size_values[valid_mask]
                    else:
                        valid_sizes = size_values
                    
                    # Create scatter plot with colorbar
                    scatter = ax.scatter(valid_x, valid_y, 
                                       c=valid_values, 
                                       s=valid_sizes, alpha=0.8, edgecolors='black', linewidth=0.5,
                                       cmap='viridis', zorder=2)
                    
                    # Add colorbar
                    cbar = self.spatial_viz_canvas.figure.colorbar(scatter, ax=ax)
                    cbar.set_label(feature_name, rotation=270, labelpad=15)
                else:
                    print(f"[DEBUG] No valid values found for feature {feature_name}")
                    return
            else:
                print(f"[DEBUG] Feature {feature_name} not found in data")
                return
        
        # Customize the plot
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'Spatial Visualization - ROI {roi_id}')
        ax.set_aspect('equal')
        
        # Invert y-axis to match typical image coordinates (origin at top-left)
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent legend cutoff
        self.spatial_viz_canvas.figure.tight_layout()
        self.spatial_viz_canvas.draw()
        
        print(f"[DEBUG] Spatial visualization created for ROI {roi_id} with {len(roi_df)} cells and {len(roi_edges)} edges")

    def _create_faceted_spatial_visualization(self, roi_ids):
        """Create faceted spatial visualization showing multiple ROIs side by side."""
        if self.feature_dataframe is None or self.edge_df is None:
            return
        
        if not roi_ids:
            return
        
        # Get color option
        color_option = self.spatial_color_combo.currentText()
        print(f"[DEBUG] Color option selected: {color_option}")
        
        # Clear the canvas
        self.spatial_viz_canvas.figure.clear()
        
        # Calculate grid dimensions
        n_rois = len(roi_ids)
        n_cols = min(4, n_rois)  # Max 4 columns
        n_rows = int(np.ceil(n_rois / n_cols))
        
        # Determine coloring method and get color map if needed
        cluster_col = None
        cluster_color_map = None
        if color_option.startswith("Cluster"):
            # Extract cluster column name
            if "(cluster)" in color_option:
                cluster_col = "cluster"
            elif "(cluster_phenotype)" in color_option:
                cluster_col = "cluster_phenotype"
            elif "(cluster_id)" in color_option:
                cluster_col = "cluster_id"
            else:
                for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
                    if col in self.feature_dataframe.columns:
                        cluster_col = col
                        break
            
            if cluster_col:
                # Get all unique clusters across all ROIs
                all_clusters = sorted(self.feature_dataframe[cluster_col].dropna().unique())
                n_clusters = len(all_clusters)
                colors = _get_vivid_colors(max(n_clusters, 3))
                cluster_color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(all_clusters)}
        
        # Create subplots
        axes = []
        for idx, roi_id in enumerate(roi_ids):
            row = idx // n_cols
            col = idx % n_cols
            ax = self.spatial_viz_canvas.figure.add_subplot(n_rows, n_cols, idx + 1)
            axes.append(ax)
            
            # Get data for this ROI
            roi_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == roi_id].copy()
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)].copy()
            
            if roi_df.empty:
                ax.text(0.5, 0.5, f'No data for ROI {roi_id}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'ROI {roi_id}')
                continue
            
            # Plot edges first
            if not roi_edges.empty:
                cell_coords = {}
                for _, row_data in roi_df.iterrows():
                    cell_coords[int(row_data['cell_id'])] = (row_data['centroid_x'], row_data['centroid_y'])
                
                for _, edge in roi_edges.iterrows():
                    cell_a = int(edge['cell_id_A'])
                    cell_b = int(edge['cell_id_B'])
                    
                    if cell_a in cell_coords and cell_b in cell_coords:
                        x_coords = [cell_coords[cell_a][0], cell_coords[cell_b][0]]
                        y_coords = [cell_coords[cell_a][1], cell_coords[cell_b][1]]
                        ax.plot(x_coords, y_coords, 'grey', alpha=0.3, linewidth=0.3, zorder=1)
            
            # Plot cells
            if cluster_col and cluster_col in roi_df.columns and cluster_color_map:
                # Color by cluster
                unique_clusters = sorted(roi_df[cluster_col].dropna().unique())
                for cluster in unique_clusters:
                    cluster_cells = roi_df[roi_df[cluster_col] == cluster]
                    if not cluster_cells.empty:
                        x_coords = cluster_cells['centroid_x'].values
                        y_coords = cluster_cells['centroid_y'].values
                        ax.scatter(x_coords, y_coords, 
                                  c=[cluster_color_map[cluster]], 
                                  s=10, alpha=0.8, edgecolors='black', linewidth=0.3,
                                  zorder=2)
            else:
                # Color by continuous feature
                feature_name = None
                if color_option.startswith("Morphology:"):
                    feature_name = color_option.replace("Morphology: ", "")
                elif color_option.startswith("Marker:"):
                    marker_name = color_option.replace("Marker: ", "").replace(" ", "_")
                    feature_name = f"{marker_name}_mean"
                
                if feature_name and feature_name in roi_df.columns:
                    feature_values = roi_df[feature_name].values
                    valid_mask = ~np.isnan(feature_values)
                    if np.any(valid_mask):
                        x_coords = roi_df.loc[valid_mask, 'centroid_x'].values
                        y_coords = roi_df.loc[valid_mask, 'centroid_y'].values
                        values = feature_values[valid_mask]
                        scatter = ax.scatter(x_coords, y_coords, 
                                           c=values, 
                                           s=10, alpha=0.8, edgecolors='black', linewidth=0.3,
                                           cmap='viridis', zorder=2)
            
            # Customize subplot
            ax.set_title(f'ROI {roi_id}', fontsize=10)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            
            # Only show axis labels on edge subplots
            if row == n_rows - 1:
                ax.set_xlabel('X (pixels)', fontsize=8)
            if col == 0:
                ax.set_ylabel('Y (pixels)', fontsize=8)
        
        # Add shared legend if using clusters
        if cluster_col and cluster_color_map:
            # Get all unique clusters across all ROIs for legend
            all_clusters = sorted(self.feature_dataframe[cluster_col].dropna().unique())
            handles = []
            labels = []
            for cluster in all_clusters:
                handles.append(plt.Rectangle((0,0),1,1, facecolor=cluster_color_map[cluster], 
                                           edgecolor='black', linewidth=0.5))
                labels.append(self._get_cluster_display_name(cluster))
            
            # Add legend to the figure (not individual subplots)
            self.spatial_viz_canvas.figure.legend(handles, labels, 
                                                 loc='upper right', bbox_to_anchor=(0.98, 0.98),
                                                 fontsize=8, ncol=min(3, len(labels)))
        
        # Adjust layout
        self.spatial_viz_canvas.figure.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space for legend
        self.spatial_viz_canvas.draw()
        
        print(f"[DEBUG] Faceted spatial visualization created for {len(roi_ids)} ROIs")

    def _run_community_analysis(self):
        """Run spatial community detection analysis."""
        print("[DEBUG] Starting spatial community analysis...")
        
        if not self.neighborhood_analysis_run:
            QtWidgets.QMessageBox.warning(self, "Prerequisite Required", "Please run neighborhood analysis first.")
            return
            
        if self.community_roi_combo.currentText() == "":
            QtWidgets.QMessageBox.warning(self, "No ROI Selected", "Please select an ROI for community analysis.")
            return
            
        try:
            selected_roi = self.community_roi_combo.currentText()
            min_cells = int(self.min_cells_spin.value())
            print(f"[DEBUG] Running community analysis for ROI: {selected_roi}, min_cells: {min_cells}")
            
            self._detect_spatial_communities(selected_roi, min_cells)
            
            self.community_analysis_run = True
            self._update_tab_states()
            
            print(f"[DEBUG] Community analysis completed successfully")
            
            # Log community analysis
            logger = get_logger()
            params = {
                "roi_id": selected_roi,
                "min_cells": min_cells,
                "seed": self.seed_spinbox.value()
            }
            # Get source file name from parent if available
            source_file = None
            if self.parent() is not None and hasattr(self.parent(), 'current_path'):
                source_file = os.path.basename(self.parent().current_path) if self.parent().current_path else None
            logger.log_spatial_analysis(
                analysis_type="community_detection",
                parameters=params,
                acquisitions=[selected_roi],
                notes=f"Community detection analysis for ROI {selected_roi} with min_cells={min_cells}",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(self, "Community Analysis", "Spatial community analysis completed successfully.")
            
        except Exception as e:
            print(f"[DEBUG] ERROR in community analysis: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Community Analysis Error", f"Error: {str(e)}")

    def _detect_spatial_communities(self, roi_id, min_cells):
        """Detect spatial communities using Louvain modularity optimization."""
        if self.feature_dataframe is None or self.edge_df is None:
            return
            
        # Get data for the selected ROI
        roi_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == roi_id].copy()
        roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)].copy()
        
        if roi_df.empty:
            print(f"[DEBUG] No data found for ROI {roi_id}")
            return
            
        # Get cluster column for exclusion
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in roi_df.columns:
                cluster_col = col
                break
        
        # Apply cell type exclusion if enabled
        if self.exclude_clusters_check.isChecked() and cluster_col:
            excluded_clusters = []
            for i in range(self.exclude_clusters_list.count()):
                item = self.exclude_clusters_list.item(i)
                if item.isSelected():
                    excluded_clusters.append(item.data(QtCore.Qt.UserRole))
            
            if excluded_clusters:
                print(f"[DEBUG] Excluding clusters: {excluded_clusters}")
                roi_df = roi_df[~roi_df[cluster_col].isin(excluded_clusters)]
                # Also filter edges to only include remaining cells
                remaining_cells = set(roi_df['cell_id'].astype(int))
                roi_edges = roi_edges[
                    (roi_edges['cell_id_A'].isin(remaining_cells)) & 
                    (roi_edges['cell_id_B'].isin(remaining_cells))
                ]
                print(f"[DEBUG] After exclusion: {len(roi_df)} cells, {len(roi_edges)} edges")
        
        if roi_df.empty or roi_edges.empty:
            print(f"[DEBUG] No data remaining after exclusion for ROI {roi_id}")
            return
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (cells)
        for _, row in roi_df.iterrows():
            G.add_node(int(row['cell_id']))
        
        # Add edges
        for _, edge in roi_edges.iterrows():
            G.add_edge(int(edge['cell_id_A']), int(edge['cell_id_B']))
        
        print(f"[DEBUG] Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Run Louvain community detection with fixed seed for reproducibility
        seed = self.seed_spinbox.value()
        np.random.seed(seed)
        random.seed(seed)
        
        try:
            communities = nx.community.louvain_communities(G, seed=seed)
            print(f"[DEBUG] Found {len(communities)} communities")
        except Exception as e:
            print(f"[DEBUG] Error in community detection: {e}")
            # Fallback to simple connected components
            communities = list(nx.connected_components(G))
            print(f"[DEBUG] Using connected components: {len(communities)} communities")
        
        # Filter communities by minimum size
        filtered_communities = [comm for comm in communities if len(comm) >= min_cells]
        print(f"[DEBUG] After filtering (min_cells={min_cells}): {len(filtered_communities)} communities")
        
        # Assign community IDs to cells
        community_assignments = {}
        for i, community in enumerate(filtered_communities):
            for cell_id in community:
                community_assignments[cell_id] = i
        
        # Add community assignments to the dataframe
        roi_df['spatial_community'] = roi_df['cell_id'].astype(int).map(community_assignments)
        
        # Store results for visualization
        self.community_results = {
            'roi_id': roi_id,
            'roi_df': roi_df,
            'roi_edges': roi_edges,
            'communities': filtered_communities,
            'community_assignments': community_assignments,
            'min_cells': min_cells
        }
        
        # Create visualization
        self._create_community_visualization()

    def _create_community_visualization(self):
        """Create spatial visualization showing communities."""
        if not hasattr(self, 'community_results'):
            return
            
        results = self.community_results
        roi_df = results['roi_df']
        roi_edges = results['roi_edges']
        communities = results['communities']
        
        # Clear the canvas
        self.community_canvas.figure.clear()
        ax = self.community_canvas.figure.add_subplot(111)
        
        # Plot edges first (so they appear behind the nodes)
        if not roi_edges.empty:
            print(f"[DEBUG] Plotting {len(roi_edges)} edges for community visualization")
            
            # Create a mapping from cell_id to coordinates
            cell_coords = {}
            for _, row in roi_df.iterrows():
                cell_coords[int(row['cell_id'])] = (row['centroid_x'], row['centroid_y'])
            
            # Plot edges
            for _, edge in roi_edges.iterrows():
                cell_a = int(edge['cell_id_A'])
                cell_b = int(edge['cell_id_B'])
                
                if cell_a in cell_coords and cell_b in cell_coords:
                    x_coords = [cell_coords[cell_a][0], cell_coords[cell_b][0]]
                    y_coords = [cell_coords[cell_a][1], cell_coords[cell_b][1]]
                    ax.plot(x_coords, y_coords, 'grey', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Plot cells colored by community
        if len(communities) > 0:
            # Create color map for communities using vivid colors
            colors = _get_vivid_colors(max(len(communities), 3))
            
            for i, community in enumerate(communities):
                community_cells = roi_df[roi_df['spatial_community'] == i]
                if not community_cells.empty:
                    x_coords = community_cells['centroid_x'].values
                    y_coords = community_cells['centroid_y'].values
                    
                    # Plot cells as points
                    ax.scatter(x_coords, y_coords, 
                              c=[colors[i % len(colors)]], 
                              s=20, alpha=0.8, edgecolors='black', linewidth=0.5,
                              label=f'Community {i} (n={len(community)})',
                              zorder=2)
        
        # Customize the plot
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'Spatial Communities - ROI {results["roi_id"]} (min_cells={results["min_cells"]})')
        ax.set_aspect('equal')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Invert y-axis to match typical image coordinates (origin at top-left)
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent legend cutoff
        self.community_canvas.figure.tight_layout()
        self.community_canvas.draw()
        
        print(f"[DEBUG] Community visualization created with {len(communities)} communities")

    def _export_results(self):
        if self.edge_df is None:
            QtWidgets.QMessageBox.warning(self, "No Results", "Run analysis before exporting.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if not out_dir:
            return
        try:
            # Export edge list
            edges_csv = os.path.join(out_dir, "edges.csv")
            self.edge_df.to_csv(edges_csv, index=False)
            edges_parquet = os.path.join(out_dir, "edges.parquet")
            try:
                self.edge_df.to_parquet(edges_parquet, index=False)
            except Exception:
                # Parquet optional, ignore if engine missing
                pass
            
            # Export per-ROI adjacency matrices
            if _HAVE_SPARSE and self.adj_matrices:
                from scipy.sparse import save_npz
                for roi_id, adj_matrix in self.adj_matrices.items():
                    adj_file = os.path.join(out_dir, f"adj_{roi_id}.npz")
                    save_npz(adj_file, adj_matrix)
                
                # Export global cell ID mapping
                mapping_file = os.path.join(out_dir, "cell_id_mapping.json")
                mapping_data = {
                    "cell_id_to_gid": {f"{roi_id}_{cell_id}": gid for (roi_id, cell_id), gid in self.cell_id_to_gid.items()},
                    "gid_to_cell_id": {str(gid): {"roi_id": roi_id, "cell_id": cell_id} for gid, (roi_id, cell_id) in self.gid_to_cell_id.items()}
                }
                with open(mapping_file, "w") as f:
                    json.dump(mapping_data, f, indent=2)
            
            # Export neighborhood composition
            if self.neighborhood_df is not None and not self.neighborhood_df.empty:
                neighborhood_csv = os.path.join(out_dir, "neighborhood_composition.csv")
                self.neighborhood_df.to_csv(neighborhood_csv, index=False)
                
                if self.cluster_summary_df is not None and not self.cluster_summary_df.empty:
                    summary_csv = os.path.join(out_dir, "cluster_summary.csv")
                    self.cluster_summary_df.to_csv(summary_csv, index=False)
            
            # Export pairwise enrichment
            if self.enrichment_df is not None and not self.enrichment_df.empty:
                enrichment_csv = os.path.join(out_dir, "pairwise_enrichment.csv")
                self.enrichment_df.to_csv(enrichment_csv, index=False)
            
            # Export distance distributions
            if self.distance_df is not None and not self.distance_df.empty:
                distance_csv = os.path.join(out_dir, "distance_distributions.csv")
                self.distance_df.to_csv(distance_csv, index=False)
            
            # Export Ripley functions
            if self.ripley_df is not None and not self.ripley_df.empty:
                ripley_csv = os.path.join(out_dir, "ripley_functions.csv")
                self.ripley_df.to_csv(ripley_csv, index=False)
            
            # Export metadata
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f, indent=2)
            
            QtWidgets.QMessageBox.information(self, "Export", f"Saved results to:\n{out_dir}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def _build_graph_with_features(self, roi_id: Optional[str] = None) -> nx.Graph:
        """
        Build a NetworkX graph with all features associated as node attributes.
        
        Args:
            roi_id: If specified, build graph for a single ROI. Otherwise, build a combined graph.
            
        Returns:
            NetworkX graph with nodes representing cells and edges representing spatial connections.
            Node attributes include all features from feature_dataframe.
            Edge attributes include distance_um and roi_id.
        """
        if self.edge_df is None or self.edge_df.empty:
            raise ValueError("Graph not built. Please run graph construction first.")
        
        G = nx.Graph()
        
        # Filter edges by ROI if specified
        if roi_id is not None:
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)].copy()
            roi_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == roi_id].copy()
        else:
            roi_edges = self.edge_df.copy()
            roi_df = self.feature_dataframe.copy()
        
        if roi_df.empty or roi_edges.empty:
            raise ValueError(f"No data available for ROI: {roi_id}")
        
        # Create a mapping from (roi_id, cell_id) to row index for quick lookup
        # For multi-ROI graphs, we need to use (roi_id, cell_id) as node identifier
        # For single ROI, we can use just cell_id
        use_roi_prefix = roi_id is None
        
        # Add nodes with all features as attributes
        for _, row in roi_df.iterrows():
            cell_id = int(row['cell_id'])
            # Ensure acquisition_id is converted to string to match edge_df roi_id format
            acquisition_id_str = str(row['acquisition_id'])
            node_id = f"{acquisition_id_str}_{cell_id}" if use_roi_prefix else cell_id
            
            # Add all features as node attributes
            node_attrs = {}
            for col in roi_df.columns:
                value = row[col]
                # Convert numpy types to Python native types for JSON serialization
                if pd.isna(value):
                    node_attrs[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    node_attrs[col] = value.item()
                elif isinstance(value, np.ndarray):
                    node_attrs[col] = value.tolist()
                else:
                    node_attrs[col] = value
            
            G.add_node(node_id, **node_attrs)
        
        # Add edges with attributes
        for _, edge in roi_edges.iterrows():
            src_roi = str(edge['roi_id'])
            src_cell_id = int(edge['cell_id_A'])
            dst_cell_id = int(edge['cell_id_B'])
            distance_um = float(edge['distance_um'])
            
            src_node = f"{src_roi}_{src_cell_id}" if use_roi_prefix else src_cell_id
            dst_node = f"{src_roi}_{dst_cell_id}" if use_roi_prefix else dst_cell_id
            
            # Only add edge if both nodes exist
            if src_node in G and dst_node in G:
                G.add_edge(src_node, dst_node, distance_um=distance_um, roi_id=src_roi)
        
        return G
    
    def _convert_to_igraph(self, nx_graph: nx.Graph) -> 'ig.Graph':
        """
        Convert a NetworkX graph to an igraph Graph.
        
        Args:
            nx_graph: NetworkX graph to convert
            
        Returns:
            igraph Graph with the same nodes, edges, and attributes
        """
        if not _HAVE_IGRAPH:
            raise ImportError("igraph is not available. Install python-igraph to use this feature.")
        
        # Get node and edge attributes
        node_attrs = {}
        edge_attrs = {}
        
        # Collect all node attribute names
        for node in nx_graph.nodes():
            for key, value in nx_graph.nodes[node].items():
                if key not in node_attrs:
                    node_attrs[key] = []
        
        # Collect all edge attribute names
        for u, v in nx_graph.edges():
            for key, value in nx_graph.edges[u, v].items():
                if key not in edge_attrs:
                    edge_attrs[key] = []
        
        # Build lists of node attributes
        node_list = list(nx_graph.nodes())
        for attr_name in node_attrs.keys():
            node_attrs[attr_name] = [nx_graph.nodes[node].get(attr_name, None) for node in node_list]
        
        # Build edge list and edge attributes
        edge_list = []
        for attr_name in edge_attrs.keys():
            edge_attrs[attr_name] = []
        
        for u, v in nx_graph.edges():
            edge_list.append((node_list.index(u), node_list.index(v)))
            for attr_name in edge_attrs.keys():
                edge_attrs[attr_name].append(nx_graph.edges[u, v].get(attr_name, None))
        
        # Create igraph graph
        ig_graph = ig.Graph(edge_list, directed=False)
        
        # Add node attributes
        for attr_name, attr_values in node_attrs.items():
            ig_graph.vs[attr_name] = attr_values
        
        # Add edge attributes
        for attr_name, attr_values in edge_attrs.items():
            ig_graph.es[attr_name] = attr_values
        
        return ig_graph
    
    def _export_graph(self):
        """Export the spatial graph in NetworkX and/or igraph format."""
        if self.edge_df is None or self.edge_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Graph", "Graph not built. Please run graph construction first.")
            return
        
        # Ask user for format and ROI selection
        format_dialog = QtWidgets.QDialog(self)
        format_dialog.setWindowTitle("Export Graph")
        format_dialog.setMinimumWidth(400)
        layout = QtWidgets.QVBoxLayout(format_dialog)
        
        # ROI selection
        roi_layout = QtWidgets.QHBoxLayout()
        roi_layout.addWidget(QtWidgets.QLabel("Export graph for:"))
        roi_combo = QtWidgets.QComboBox()
        roi_combo.addItem("All ROIs (combined)", None)
        unique_rois = sorted(self.feature_dataframe['acquisition_id'].unique())
        for roi_id in unique_rois:
            roi_combo.addItem(str(roi_id), roi_id)
        roi_layout.addWidget(roi_combo)
        roi_layout.addStretch()
        layout.addLayout(roi_layout)
        
        # Format selection
        format_layout = QtWidgets.QVBoxLayout()
        format_layout.addWidget(QtWidgets.QLabel("Export formats:"))
        networkx_check = QtWidgets.QCheckBox("NetworkX (GraphML or pickle)")
        networkx_check.setChecked(True)
        format_layout.addWidget(networkx_check)
        
        igraph_check = QtWidgets.QCheckBox("igraph (GraphML or pickle)")
        igraph_check.setEnabled(_HAVE_IGRAPH)
        if not _HAVE_IGRAPH:
            igraph_check.setToolTip("igraph not available. Install python-igraph to use this option.")
        format_layout.addWidget(igraph_check)
        layout.addLayout(format_layout)
        
        # File format selection for NetworkX
        nx_format_layout = QtWidgets.QHBoxLayout()
        nx_format_layout.addWidget(QtWidgets.QLabel("NetworkX format:"))
        nx_format_combo = QtWidgets.QComboBox()
        nx_format_combo.addItems(["GraphML (.graphml)", "Pickle (.gpickle)", "Both"])
        nx_format_layout.addWidget(nx_format_combo)
        nx_format_layout.addStretch()
        layout.addLayout(nx_format_layout)
        
        # File format selection for igraph
        ig_format_layout = QtWidgets.QHBoxLayout()
        ig_format_layout.addWidget(QtWidgets.QLabel("igraph format:"))
        ig_format_combo = QtWidgets.QComboBox()
        ig_format_combo.addItems(["GraphML (.graphml)", "Pickle (.pickle)", "Both"])
        ig_format_layout.addWidget(ig_format_combo)
        ig_format_layout.addStretch()
        layout.addLayout(ig_format_layout)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        ok_btn.clicked.connect(format_dialog.accept)
        cancel_btn.clicked.connect(format_dialog.reject)
        
        if format_dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        # Get selected options
        roi_id = roi_combo.currentData()
        export_networkx = networkx_check.isChecked()
        export_igraph = igraph_check.isChecked() and _HAVE_IGRAPH
        nx_format = nx_format_combo.currentText()
        ig_format = ig_format_combo.currentText()
        
        if not export_networkx and not export_igraph:
            QtWidgets.QMessageBox.warning(self, "No Format", "Please select at least one export format.")
            return
        
        # Get output directory
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if not out_dir:
            return
        
        try:
            # Build graph
            roi_label = str(roi_id) if roi_id is not None else "all_rois"
            print(f"[DEBUG] Building graph for ROI: {roi_label}")
            G = self._build_graph_with_features(roi_id=roi_id)
            print(f"[DEBUG] Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Export NetworkX formats
            if export_networkx:
                if "GraphML" in nx_format or nx_format == "Both":
                    graphml_file = os.path.join(out_dir, f"spatial_graph_{roi_label}.graphml")
                    nx.write_graphml(G, graphml_file)
                    print(f"[DEBUG] Exported NetworkX GraphML to: {graphml_file}")
                
                if "Pickle" in nx_format or nx_format == "Both":
                    pickle_file = os.path.join(out_dir, f"spatial_graph_{roi_label}.gpickle")
                    nx.write_gpickle(G, pickle_file)
                    print(f"[DEBUG] Exported NetworkX pickle to: {pickle_file}")
            
            # Export igraph formats
            if export_igraph:
                ig_G = self._convert_to_igraph(G)
                print(f"[DEBUG] Converted to igraph with {len(ig_G.vs)} vertices and {len(ig_G.es)} edges")
                
                if "GraphML" in ig_format or ig_format == "Both":
                    graphml_file = os.path.join(out_dir, f"spatial_graph_{roi_label}_igraph.graphml")
                    ig_G.write_graphml(graphml_file)
                    print(f"[DEBUG] Exported igraph GraphML to: {graphml_file}")
                
                if "Pickle" in ig_format or ig_format == "Both":
                    pickle_file = os.path.join(out_dir, f"spatial_graph_{roi_label}_igraph.pickle")
                    # igraph doesn't have a direct pickle method, so we'll use pickle module
                    import pickle
                    with open(pickle_file, 'wb') as f:
                        pickle.dump(ig_G, f)
                    print(f"[DEBUG] Exported igraph pickle to: {pickle_file}")
            
            # Export metadata about the graph
            metadata = {
                "roi_id": roi_label,
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "graph_mode": self.metadata.get("mode", "unknown"),
                "k": self.metadata.get("k", None),
                "radius_um": self.metadata.get("radius_um", None),
                "node_attributes": list(next(iter(G.nodes(data=True)))[1].keys()) if G.number_of_nodes() > 0 else [],
                "edge_attributes": list(next(iter(G.edges(data=True)))[1].keys()) if G.number_of_edges() > 0 else [],
            }
            
            metadata_file = os.path.join(out_dir, f"graph_metadata_{roi_label}.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            QtWidgets.QMessageBox.information(
                self, 
                "Export Complete", 
                f"Graph exported successfully!\n\n"
                f"ROI: {roi_label}\n"
                f"Nodes: {G.number_of_nodes()}\n"
                f"Edges: {G.number_of_edges()}\n\n"
                f"Saved to: {out_dir}"
            )
            
        except Exception as e:
            print(f"[DEBUG] Error exporting graph: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Error exporting graph:\n{str(e)}")

    def _compute_neighborhood_composition(self):
        """Compute neighborhood composition for each cell with vectorized operations."""
        if self.edge_df is None or self.edge_df.empty:
            return
            
        # Use detected cluster column (already validated in _run_analysis)
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        # Get unique clusters across all ROIs for stable indexing
        unique_clusters = sorted(self.feature_dataframe[cluster_col].unique())
        n_clusters = len(unique_clusters)
        
        # Create stable cluster index mapping (use actual cluster IDs as indices)
        cluster_to_idx = {cluster: cluster for cluster in unique_clusters}
        
        # Initialize neighborhood composition dataframe
        neighborhood_data = []
        
        # Process each ROI separately
        for roi_id, roi_df in self.feature_dataframe.groupby('acquisition_id'):
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)]
            
            if roi_edges.empty:
                continue
                
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
        
        self.neighborhood_df = pd.DataFrame(neighborhood_data)
        
        # Compute per-cluster summary with vectorized operations
        if not self.neighborhood_df.empty:
            cluster_summary_data = []
            for cluster in unique_clusters:
                cluster_cells = self.neighborhood_df[self.neighborhood_df['cluster_id'] == cluster]
                if not cluster_cells.empty:
                    summary_row = {'cluster_id': cluster}
                    for cluster_id in unique_clusters:
                        col_name = f'frac_cluster_{cluster_id}'
                        if col_name in cluster_cells.columns:
                            summary_row[f'avg_frac_cluster_{cluster_id}'] = cluster_cells[col_name].mean()
                    cluster_summary_data.append(summary_row)
            
            self.cluster_summary_df = pd.DataFrame(cluster_summary_data)

    def _update_neighborhood_plot(self):
        """Update the neighborhood composition visualization."""
        if self.neighborhood_df is None or self.neighborhood_df.empty:
            self.neighborhood_canvas.figure.clear()
            ax = self.neighborhood_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No neighborhood data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Neighborhood Composition')
            self.neighborhood_canvas.draw()
            return
            
        self.neighborhood_canvas.figure.clear()
        
        # Create subplots
        fig = self.neighborhood_canvas.figure
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Get cluster information
        unique_clusters = sorted(self.neighborhood_df['cluster_id'].unique())
        n_clusters = len(unique_clusters)
        
        # Plot 1: Average neighborhood composition per cluster
        if self.cluster_summary_df is not None and not self.cluster_summary_df.empty:
            cluster_labels = [self._get_cluster_display_name(c) for c in unique_clusters]
            neighbor_labels = [self._get_cluster_display_name(c) for c in unique_clusters]
            
            # Create composition matrix
            composition_matrix = np.zeros((len(unique_clusters), n_clusters))
            for i, cluster in enumerate(unique_clusters):
                cluster_row = self.cluster_summary_df[self.cluster_summary_df['cluster_id'] == cluster]
                if not cluster_row.empty:
                    for j, neighbor_cluster in enumerate(unique_clusters):
                        col_name = f'avg_frac_cluster_{neighbor_cluster}'
                        if col_name in cluster_row.columns:
                            composition_matrix[i, j] = cluster_row.iloc[0][col_name]
            
            # Heatmap
            im = ax1.imshow(composition_matrix, cmap='viridis', aspect='auto')
            ax1.set_xticks(range(n_clusters))
            # Calculate font size based on number of labels and label length
            max_label_len = max([len(str(l)) for l in neighbor_labels] + [1])
            fontsize = max(6, min(10, 12 - max_label_len // 5))
            ax1.set_xticklabels(neighbor_labels, rotation=45, ha='right', fontsize=fontsize)
            ax1.set_yticks(range(len(unique_clusters)))
            ax1.set_yticklabels(cluster_labels, fontsize=fontsize)
            ax1.set_xlabel('Neighbor Cluster')
            ax1.set_ylabel('Cell Cluster')
            ax1.set_title('Average Neighborhood Composition')
            
            # Add colorbar
            fig.colorbar(im, ax=ax1, label='Fraction')
        
        # Plot 2: Distribution of neighbor fractions for each cluster
        frac_cols = [col for col in self.neighborhood_df.columns if col.startswith('frac_cluster_')]
        if frac_cols:
            # Create box plot data
            plot_data = []
            plot_labels = []
            for col in frac_cols:
                plot_data.append(self.neighborhood_df[col].values)
                # Extract cluster ID from column name (e.g., 'frac_cluster_1' -> '1')
                cluster_id = int(col.split('_')[-1])
                plot_labels.append(self._get_cluster_display_name(cluster_id))
            
            ax2.boxplot(plot_data, labels=plot_labels)
            ax2.set_xlabel('Neighbor Cluster')
            ax2.set_ylabel('Fraction')
            ax2.set_title('Distribution of Neighbor Fractions')
            # Calculate font size based on number of labels and label length
            max_label_len = max([len(str(l)) for l in plot_labels] + [1])
            fontsize = max(6, min(10, 12 - max_label_len // 5))
            ax2.tick_params(axis='x', rotation=45, labelsize=fontsize)
        
        fig.tight_layout()
        self.neighborhood_canvas.draw()

    def _compute_pairwise_enrichment(self, n_perm=100):
        """Compute pairwise interaction enrichment analysis using permutation null."""
        if self.edge_df is None or self.edge_df.empty:
            return
            
        # Use detected cluster column (already validated in _run_analysis)
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        enrichment_data = []
        
        # Process each ROI separately
        for roi_id, roi_df in self.feature_dataframe.groupby('acquisition_id'):
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)]
            
            if roi_edges.empty:
                continue
                
            # Get unique clusters in this ROI
            unique_clusters = sorted(roi_df[cluster_col].unique())
            n_clusters = len(unique_clusters)
            
            if n_clusters < 2:
                continue  # Need at least 2 clusters for pairwise analysis
            
            # Create cell_id to cluster mapping for efficient lookup
            cell_to_cluster = dict(zip(roi_df['cell_id'], roi_df[cluster_col]))
            
            # Count observed edges between cluster pairs
            observed_edges = {}
            for _, edge in roi_edges.iterrows():
                cell_a, cell_b = int(edge['cell_id_A']), int(edge['cell_id_B'])
                
                cluster_a = cell_to_cluster.get(cell_a)
                cluster_b = cell_to_cluster.get(cell_b)
                
                if cluster_a is not None and cluster_b is not None:
                    # Create canonical pair (smaller cluster first)
                    pair = tuple(sorted([cluster_a, cluster_b]))
                    observed_edges[pair] = observed_edges.get(pair, 0) + 1
            
            # Perform permutation test for each cluster pair
            for i, cluster_a in enumerate(unique_clusters):
                for j, cluster_b in enumerate(unique_clusters):
                    if j < i:  # Avoid duplicates
                        continue
                        
                    pair = (cluster_a, cluster_b)
                    observed = observed_edges.get(pair, 0)
                    
                    # Get cells in each cluster
                    cells_a = roi_df[roi_df[cluster_col] == cluster_a]['cell_id'].tolist()
                    cells_b = roi_df[roi_df[cluster_col] == cluster_b]['cell_id'].tolist()
                    
                    # Perform permutations with seed for reproducibility
                    seed = self.seed_spinbox.value()
                    np.random.seed(seed)
                    random.seed(seed)
                    
                    permuted_counts = []
                    for perm_idx in range(n_perm):
                        # Use a different seed for each permutation to ensure reproducibility
                        # but still get different permutations
                        np.random.seed(seed + perm_idx)
                        random.seed(seed + perm_idx)
                        # Shuffle cluster labels while preserving degrees
                        shuffled_clusters = roi_df[cluster_col].values.copy()
                        np.random.shuffle(shuffled_clusters)
                        
                        # Create temporary mapping
                        temp_cell_to_cluster = dict(zip(roi_df['cell_id'], shuffled_clusters))
                        
                        # Count edges for this permutation
                        perm_count = 0
                        for _, edge in roi_edges.iterrows():
                            cell_a, cell_b = int(edge['cell_id_A']), int(edge['cell_id_B'])
                            
                            perm_cluster_a = temp_cell_to_cluster.get(cell_a)
                            perm_cluster_b = temp_cell_to_cluster.get(cell_b)
                            
                            if perm_cluster_a is not None and perm_cluster_b is not None:
                                perm_pair = tuple(sorted([perm_cluster_a, perm_cluster_b]))
                                if perm_pair == pair:
                                    perm_count += 1
                        
                        permuted_counts.append(perm_count)
                    
                    # Calculate statistics
                    expected_mean = np.mean(permuted_counts)
                    expected_std = np.std(permuted_counts)
                    
                    if expected_std > 0:
                        z_score = (observed - expected_mean) / expected_std
                        # Two-tailed p-value from permutation distribution
                        p_value = np.mean(np.abs(permuted_counts - expected_mean) >= abs(observed - expected_mean))
                    else:
                        z_score = 0.0
                        p_value = 1.0
                    
                    enrichment_data.append({
                        'roi_id': roi_id,
                        'cluster_A': cluster_a,
                        'cluster_B': cluster_b,
                        'observed_edges': observed,
                        'expected_mean': expected_mean,
                        'expected_std': expected_std,
                        'z_score': z_score,
                        'p_value': p_value,
                        'n_permutations': n_perm
                    })
        
        self.enrichment_df = pd.DataFrame(enrichment_data)

    def _normal_cdf(self, x):
        """Approximate normal CDF using error function."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

    def _update_enrichment_plot(self):
        """Update the pairwise enrichment visualization."""
        if self.enrichment_df is None or self.enrichment_df.empty:
            self.enrichment_canvas.figure.clear()
            ax = self.enrichment_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No enrichment data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Pairwise Enrichment')
            self.enrichment_canvas.draw()
            return
            
        self.enrichment_canvas.figure.clear()
        
        # Create subplots
        fig = self.enrichment_canvas.figure
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Get unique clusters across all ROIs
        all_clusters = sorted(set(self.enrichment_df['cluster_A'].unique()) | 
                             set(self.enrichment_df['cluster_B'].unique()))
        n_clusters = len(all_clusters)
        
        if n_clusters == 0:
            return
        
        # Create enrichment matrix (Z-scores)
        enrichment_matrix = np.zeros((n_clusters, n_clusters))
        pvalue_matrix = np.ones((n_clusters, n_clusters))
        
        for _, row in self.enrichment_df.iterrows():
            i = all_clusters.index(row['cluster_A'])
            j = all_clusters.index(row['cluster_B'])
            enrichment_matrix[i, j] = row['z_score']
            enrichment_matrix[j, i] = row['z_score']  # Symmetric
            pvalue_matrix[i, j] = row['p_value']
            pvalue_matrix[j, i] = row['p_value']  # Symmetric
        
        # Plot 1: Z-score heatmap with significance markers
        im1 = ax1.imshow(enrichment_matrix, cmap='RdBu_r', aspect='auto', 
                        vmin=-3, vmax=3)
        cluster_labels = [self._get_cluster_display_name(c) for c in all_clusters]
        # Calculate font size based on number of labels and label length
        max_label_len = max([len(str(l)) for l in cluster_labels] + [1])
        fontsize = max(6, min(10, 12 - max_label_len // 5))
        ax1.set_xticks(range(n_clusters))
        ax1.set_xticklabels(cluster_labels, rotation=45, ha='right', fontsize=fontsize)
        ax1.set_yticks(range(n_clusters))
        ax1.set_yticklabels(cluster_labels, fontsize=fontsize)
        ax1.set_xlabel('Cluster B')
        ax1.set_ylabel('Cluster A')
        ax1.set_title('Pairwise Interaction Enrichment (Z-scores)')
        
        # Add significance markers (dots for |z| >= 2 or p < 0.05)
        for i in range(n_clusters):
            for j in range(n_clusters):
                z_val = enrichment_matrix[i, j]
                p_val = pvalue_matrix[i, j]
                if abs(z_val) >= 2 or p_val < 0.05:
                    # Add significance marker
                    ax1.scatter(j, i, s=50, c='white', marker='o', edgecolors='black', linewidth=1)
        
        # Add colorbar
        fig.colorbar(im1, ax=ax1, label='Z-score')
        
        # Plot 2: Observed vs Expected scatter
        observed = self.enrichment_df['observed_edges'].values
        expected = self.enrichment_df['expected_mean'].values
        
        ax2.scatter(expected, observed, alpha=0.6, s=50)
        
        # Add diagonal line (observed = expected)
        max_val = max(np.max(observed), np.max(expected))
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Expected')
        
        ax2.set_xlabel('Expected Edges')
        ax2.set_ylabel('Observed Edges')
        ax2.set_title('Observed vs Expected Edges')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.enrichment_canvas.draw()

    def _compute_distance_distributions(self):
        """Compute distance distribution analysis using geometric nearest neighbor search."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return
            
        # Use detected cluster column (already validated in _run_analysis)
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        distance_data = []
        
        # Get pixel size for distance conversion
        parent = self.parent() if hasattr(self, 'parent') else None
        
        # Process each ROI separately
        for roi_id, roi_df in self.feature_dataframe.groupby('acquisition_id'):
            roi_df = roi_df.dropna(subset=["centroid_x", "centroid_y"])
            if roi_df.empty:
                continue
                
            # Get pixel size for this ROI
            pixel_size_um = 1.0
            try:
                if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                    pixel_size_um = float(parent._get_pixel_size_um(roi_id))  # type: ignore[attr-defined]
            except Exception:
                pixel_size_um = 1.0
            
            # Convert coordinates to micrometers
            coords_um = roi_df[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
            cell_ids = roi_df["cell_id"].astype(int).to_numpy()
            
            # Get unique clusters in this ROI
            unique_clusters = sorted(roi_df[cluster_col].unique())
            
            # Create KDTree for efficient nearest neighbor search
            tree = cKDTree(coords_um)
            
            # For each cell, find nearest neighbor of each cluster type
            # Use enumerate to get positional index since coords_um is zero-indexed
            for pos_idx, (df_idx, cell_row) in enumerate(roi_df.iterrows()):
                cell_id = int(cell_row['cell_id'])
                cell_cluster = cell_row[cluster_col]
                cell_coord = coords_um[pos_idx]
                
                # Find nearest neighbor for each cluster type
                for target_cluster in unique_clusters:
                    if target_cluster == cell_cluster:
                        # For same cluster, find nearest neighbor excluding self
                        target_cells = roi_df[roi_df[cluster_col] == target_cluster]
                        if len(target_cells) < 2:
                            continue
                        
                        target_coords = target_cells[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
                        target_cell_ids = target_cells["cell_id"].astype(int).to_numpy()
                        
                        # Find nearest neighbor excluding self
                        min_distance = float('inf')
                        nearest_cell_id = None
                        
                        for j, target_coord in enumerate(target_coords):
                            if target_cell_ids[j] != cell_id:  # Exclude self
                                distance = np.linalg.norm(cell_coord - target_coord)
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_cell_id = target_cell_ids[j]
                    else:
                        # For different clusters, find nearest neighbor
                        target_cells = roi_df[roi_df[cluster_col] == target_cluster]
                        if target_cells.empty:
                            continue
                        
                        target_coords = target_cells[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
                        target_cell_ids = target_cells["cell_id"].astype(int).to_numpy()
                        
                        # Find nearest neighbor
                        min_distance = float('inf')
                        nearest_cell_id = None
                        
                        for j, target_coord in enumerate(target_coords):
                            distance = np.linalg.norm(cell_coord - target_coord)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_cell_id = target_cell_ids[j]
                    
                    # Record the nearest neighbor distance
                    if min_distance != float('inf'):
                        distance_data.append({
                            'roi_id': roi_id,
                            'cell_A_id': cell_id,
                            'cell_A_cluster': cell_cluster,
                            'nearest_B_cluster': target_cluster,
                            'nearest_B_dist_um': min_distance,
                            'nearest_B_cell_id': nearest_cell_id
                        })
        
        self.distance_df = pd.DataFrame(distance_data)

    def _update_distance_plot(self):
        """Update the distance distribution visualization."""
        if self.distance_df is None or self.distance_df.empty:
            self.distance_canvas.figure.clear()
            ax = self.distance_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No distance data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Distance Distributions')
            self.distance_canvas.draw()
            return
            
        self.distance_canvas.figure.clear()
        
        # Create subplots
        fig = self.distance_canvas.figure
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        # Get unique clusters
        all_clusters = sorted(set(self.distance_df['cell_A_cluster'].unique()) | 
                             set(self.distance_df['nearest_B_cluster'].unique()))
        
        if len(all_clusters) == 0:
            return
        
        # Plot 1: Histogram of all distances
        all_distances = self.distance_df['nearest_B_dist_um'].values
        ax1.hist(all_distances, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Distance (µm)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of All Nearest Neighbor Distances')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Violin plot by cluster pair (all clusters, not limited)
        if len(all_clusters) >= 2:
            # Create violin plot data for all cluster pairs
            violin_data = []
            violin_labels = []
            
            for cell_cluster in all_clusters:
                for target_cluster in all_clusters:
                    pair_data = self.distance_df[
                        (self.distance_df['cell_A_cluster'] == cell_cluster) &
                        (self.distance_df['nearest_B_cluster'] == target_cluster)
                    ]['nearest_B_dist_um'].values
                    
                    if len(pair_data) > 0:
                        violin_data.append(pair_data)
                        violin_labels.append(f'{int(cell_cluster)}→{int(target_cluster)}')
            
            if violin_data:
                # Limit display to avoid overcrowding, but allow more than 4
                max_pairs = min(len(violin_data), 16)  # Show up to 16 pairs
                ax2.violinplot(violin_data[:max_pairs], positions=range(max_pairs))
                ax2.set_xticks(range(max_pairs))
                # Calculate font size based on number of labels and label length
                max_label_len = max([len(str(l)) for l in violin_labels[:max_pairs]] + [1])
                fontsize = max(6, min(10, 12 - max_label_len // 5))
                ax2.set_xticklabels(violin_labels[:max_pairs], rotation=45, ha='right', fontsize=fontsize)
                ax2.set_ylabel('Distance (µm)')
                ax2.set_title(f'Distance Distributions by Cluster Pair (showing {max_pairs}/{len(violin_data)})')
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: CDF curves for each cluster
        for cluster in all_clusters[:5]:  # Limit to first 5 clusters
            cluster_distances = self.distance_df[
                self.distance_df['cell_A_cluster'] == cluster
            ]['nearest_B_dist_um'].values
            
            if len(cluster_distances) > 0:
                sorted_distances = np.sort(cluster_distances)
                cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
                ax3.plot(sorted_distances, cumulative, label=self._get_cluster_display_name(cluster), linewidth=2)
        
        ax3.set_xlabel('Distance (µm)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Functions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        summary_data = []
        for roi_id, roi_df in self.distance_df.groupby('roi_id'):
            roi_distances = roi_df['nearest_B_dist_um'].values
            summary_data.append({
                'ROI': roi_id,
                'Mean': np.mean(roi_distances),
                'Median': np.median(roi_distances),
                'Std': np.std(roi_distances)
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            x_pos = range(len(summary_df))
            ax4.bar(x_pos, summary_df['Mean'], alpha=0.7, label='Mean')
            ax4.errorbar(x_pos, summary_df['Mean'], yerr=summary_df['Std'], 
                        fmt='none', color='black', capsize=5)
            ax4.set_xticks(x_pos)
            # Calculate font size based on number of labels and label length
            roi_labels = summary_df['ROI'].tolist()
            max_label_len = max([len(str(l)) for l in roi_labels] + [1])
            fontsize = max(6, min(10, 12 - max_label_len // 5))
            ax4.set_xticklabels(roi_labels, rotation=45, ha='right', fontsize=fontsize)
            ax4.set_ylabel('Distance (µm)')
            ax4.set_title('Mean Distances by ROI')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.distance_canvas.draw()

    def _compute_ripley_functions(self, r_max=50.0, n_steps=20):
        """Compute Ripley K and L functions with correct formulas and edge correction."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return
            
        # Use detected cluster column (already validated in _run_analysis)
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        ripley_data = []
        
        # Get pixel size for distance conversion
        parent = self.parent() if hasattr(self, 'parent') else None
        
        # Process each ROI separately
        for roi_id, roi_df in self.feature_dataframe.groupby('acquisition_id'):
            roi_df = roi_df.dropna(subset=["centroid_x", "centroid_y"])
            if roi_df.empty:
                continue
                
            # Get pixel size for this ROI
            pixel_size_um = 1.0
            try:
                if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                    pixel_size_um = float(parent._get_pixel_size_um(roi_id))  # type: ignore[attr-defined]
            except Exception:
                pixel_size_um = 1.0
            
            # Convert coordinates to micrometers
            coords_um = roi_df[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
            
            # Get ROI dimensions for edge correction
            roi_width = np.max(coords_um[:, 0]) - np.min(coords_um[:, 0])
            roi_height = np.max(coords_um[:, 1]) - np.min(coords_um[:, 1])
            roi_area = roi_width * roi_height
            
            # Set radius range with edge correction limit
            max_radius = min(r_max, 0.25 * min(roi_width, roi_height))
            radius_steps = np.linspace(1.0, max_radius, n_steps)
            
            # Get unique clusters in this ROI
            unique_clusters = sorted(roi_df[cluster_col].unique())
            
            # Compute Ripley functions for each cluster
            for cluster in unique_clusters:
                cluster_cells = roi_df[roi_df[cluster_col] == cluster]
                if len(cluster_cells) < 2:
                    continue
                    
                cluster_coords = cluster_cells[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
                n_points = len(cluster_coords)
                
                # Point density
                lambda_density = n_points / roi_area
                
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
        
        self.ripley_df = pd.DataFrame(ripley_data)

    def _update_ripley_plot(self):
        """Update the Ripley K/L functions visualization."""
        if self.ripley_df is None or self.ripley_df.empty:
            self.ripley_canvas.figure.clear()
            ax = self.ripley_canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No Ripley data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Ripley K/L Functions')
            self.ripley_canvas.draw()
            return
            
        self.ripley_canvas.figure.clear()
        
        # Create subplots
        fig = self.ripley_canvas.figure
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Get unique clusters
        unique_clusters = sorted(self.ripley_df['cell_type'].unique())
        
        if len(unique_clusters) == 0:
            return
        
        # Plot 1: K function curves
        for cluster in unique_clusters:
            cluster_data = self.ripley_df[self.ripley_df['cell_type'] == cluster]
            if not cluster_data.empty:
                cluster_name = self._get_cluster_display_name(cluster)
                ax1.plot(cluster_data['r_um'], cluster_data['K_obs'], 
                        label=f'{cluster_name} (Observed)', linewidth=2)
                ax1.plot(cluster_data['r_um'], cluster_data['K_exp'], 
                        '--', alpha=0.7, label=f'{cluster_name} (Expected)')
        
        ax1.set_xlabel('Radius (µm)')
        ax1.set_ylabel('K(r)')
        ax1.set_title('Ripley K Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: L function curves
        for cluster in unique_clusters:
            cluster_data = self.ripley_df[self.ripley_df['cell_type'] == cluster]
            if not cluster_data.empty:
                cluster_name = self._get_cluster_display_name(cluster)
                ax2.plot(cluster_data['r_um'], cluster_data['L_obs'], 
                        label=f'{cluster_name} (Observed)', linewidth=2)
                # L expected is always 0 under CSR
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Expected (CSR)')
        
        ax2.set_xlabel('Radius (µm)')
        ax2.set_ylabel('L(r)')
        ax2.set_title('Ripley L Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add interpretation text
        fig.text(0.5, 0.02, 
                'L(r) > 0: Clustering at scale r; L(r) < 0: Dispersion at scale r', 
                ha='center', fontsize=10, style='italic')
        
        fig.tight_layout()
        self.ripley_canvas.draw()

    def _save_neighborhood_plot(self):
        """Save the neighborhood composition plot."""
        if self.neighborhood_df is None or self.neighborhood_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No neighborhood data to save.")
            return
        
        if save_figure_with_options(self.neighborhood_canvas.figure, "neighborhood_composition.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def _save_enrichment_plot(self):
        """Save the enrichment plot."""
        if self.enrichment_df is None or self.enrichment_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No enrichment data to save.")
            return
        
        if save_figure_with_options(self.enrichment_canvas.figure, "pairwise_enrichment.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def _save_distance_plot(self):
        """Save the distance distribution plot."""
        if self.distance_df is None or self.distance_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No distance data to save.")
            return
        
        if save_figure_with_options(self.distance_canvas.figure, "distance_distributions.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def _save_ripley_plot(self):
        """Save the Ripley K/L plot."""
        if self.ripley_df is None or self.ripley_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "No Ripley data to save.")
            return
        
        if save_figure_with_options(self.ripley_canvas.figure, "ripley_functions.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def _save_spatial_viz_plot(self):
        """Save the spatial visualization plot."""
        if not self.spatial_viz_run:
            QtWidgets.QMessageBox.warning(self, "No Data", "No spatial visualization to save.")
            return
        
        if save_figure_with_options(self.spatial_viz_canvas.figure, "spatial_visualization.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def _save_community_plot(self):
        """Save the spatial community plot."""
        if not self.community_analysis_run:
            QtWidgets.QMessageBox.warning(self, "No Data", "No community analysis to save.")
            return
        
        if save_figure_with_options(self.community_canvas.figure, "spatial_communities.png", self):
            QtWidgets.QMessageBox.information(self, "Success", "Plot saved successfully")

    def closeEvent(self, event):
        """Handle dialog closing to clean up resources."""
        if hasattr(self, 'annotation_timer'):
            self.annotation_timer.stop()
        event.accept()


