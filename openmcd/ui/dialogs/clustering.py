from typing import List
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from skimage.measure import regionprops
import json
import math

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


# --------------------------
# Cell Clustering Dialog
# --------------------------
class CellClusteringDialog(QtWidgets.QDialog):
    def __init__(self, feature_dataframe, normalization_config=None, parent=None):
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
        self.feature_dataframe = feature_dataframe
        self.normalization_config = normalization_config
        self.cluster_labels = None
        self.clustered_data = None
        self.umap_embedding = None
        self.cluster_annotation_map = {}
        self.cluster_backend_names = {}  # Store normalized names for CSV export
        self.gating_rules = []  # list of dict: {name, logic, conditions: [{column, op, threshold}]}
        self.llm_phenotype_cache = {}  # Cache for LLM phenotype suggestions
        
        self._create_ui()
        self._setup_plot()
        self._on_clustering_type_changed()  # Initialize UI state
        self._on_leiden_mode_changed()  # Initialize Leiden mode state
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Cell Clustering Analysis")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 16px; }")
        layout.addWidget(title_label)
        
        # Options panel
        options_group = QtWidgets.QGroupBox("Clustering Options")
        options_layout = QtWidgets.QHBoxLayout(options_group)
        
        # (Aggregation and morphometric inclusion moved to Feature Selector dialog)
        
        # Clustering method type (first)
        options_layout.addWidget(QtWidgets.QLabel("Clustering Method:"))
        self.clustering_type = QtWidgets.QComboBox()
        clustering_types = ["Hierarchical"]
        if _HAVE_LEIDEN:
            clustering_types.append("Leiden")
        self.clustering_type.addItems(clustering_types)
        self.clustering_type.setCurrentText("Hierarchical")
        self.clustering_type.currentTextChanged.connect(self._on_clustering_type_changed)
        options_layout.addWidget(self.clustering_type)
        
        # Number of clusters (only for hierarchical)
        self.n_clusters_label = QtWidgets.QLabel("Number of clusters:")
        options_layout.addWidget(self.n_clusters_label)
        self.n_clusters = QtWidgets.QSpinBox()
        self.n_clusters.setRange(2, 20)
        self.n_clusters.setValue(5)
        options_layout.addWidget(self.n_clusters)
        
        # Hierarchical method selection (initially visible)
        self.hierarchical_label = QtWidgets.QLabel("Linkage Method:")
        self.hierarchical_method = QtWidgets.QComboBox()
        self.hierarchical_method.addItems(["ward", "complete", "average", "single"])
        self.hierarchical_method.setCurrentText("ward")
        options_layout.addWidget(self.hierarchical_label)
        options_layout.addWidget(self.hierarchical_method)
        
        # Leiden clustering options (initially hidden)
        self.leiden_options_group = QtWidgets.QGroupBox("Leiden Clustering Options")
        leiden_options_layout = QtWidgets.QVBoxLayout(self.leiden_options_group)
        
        # Resolution vs Modularity choice
        self.leiden_mode_group = QtWidgets.QButtonGroup()
        self.resolution_radio = QtWidgets.QRadioButton("Use resolution parameter")
        self.modularity_radio = QtWidgets.QRadioButton("Use modularity optimization")
        self.resolution_radio.setChecked(True)
        self.leiden_mode_group.addButton(self.resolution_radio)
        self.leiden_mode_group.addButton(self.modularity_radio)
        leiden_options_layout.addWidget(self.resolution_radio)
        leiden_options_layout.addWidget(self.modularity_radio)
        
        # Resolution parameter
        resolution_layout = QtWidgets.QHBoxLayout()
        self.resolution_label = QtWidgets.QLabel("Resolution:")
        self.resolution_spinbox = QtWidgets.QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.1, 5.0)
        self.resolution_spinbox.setSingleStep(0.1)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setDecimals(1)
        resolution_layout.addWidget(self.resolution_label)
        resolution_layout.addWidget(self.resolution_spinbox)
        leiden_options_layout.addLayout(resolution_layout)
        
        # Connect radio button changes
        self.resolution_radio.toggled.connect(self._on_leiden_mode_changed)
        self.modularity_radio.toggled.connect(self._on_leiden_mode_changed)
        
        self.leiden_options_group.setVisible(False)
        options_layout.addWidget(self.leiden_options_group)

        # Dendrogram mode (only for hierarchical methods)
        self.dendro_label = QtWidgets.QLabel("Dendrogram:")
        self.dendro_mode = QtWidgets.QComboBox()
        self.dendro_mode.addItems(["Rows only", "Rows and columns"]) 
        self.dendro_mode.setCurrentText("Rows only")
        options_layout.addWidget(self.dendro_label)
        options_layout.addWidget(self.dendro_mode)
        
        # Run clustering button
        self.run_btn = QtWidgets.QPushButton("Run Clustering")
        self.run_btn.clicked.connect(self._run_clustering)
        options_layout.addWidget(self.run_btn)
        
        # Save clustering output button
        self.save_output_btn = QtWidgets.QPushButton("Save Clustering Output")
        self.save_output_btn.clicked.connect(self._save_clustering_output)
        self.save_output_btn.setEnabled(False)
        self.save_output_btn.setToolTip("Save CSV with all features, cluster labels, and manual annotations")
        options_layout.addWidget(self.save_output_btn)
        
        options_layout.addStretch()
        layout.addWidget(options_group)
        
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
        self.view_combo.addItems(["Heatmap", "UMAP", "Stacked Bars", "Differential Expression"])
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        viz_layout.addWidget(self.view_combo)

        # Color-by control (UMAP only)
        self.color_by_label = QtWidgets.QLabel("Color by:")
        viz_layout.addWidget(self.color_by_label)
        self.color_by_combo = QtWidgets.QComboBox()
        self.color_by_combo.addItem("Cluster")
        self.color_by_combo.currentTextChanged.connect(self._on_color_by_changed)
        viz_layout.addWidget(self.color_by_combo)

        # Group-by for stacked bars (Stacked Bars only)
        self.group_by_label = QtWidgets.QLabel("Group by:")
        viz_layout.addWidget(self.group_by_label)
        self.group_by_combo = QtWidgets.QComboBox()
        candidate_cols = [
            'roi', 'ROI', 'slide', 'Slide', 'condition', 'Condition',
            'acquisition_name', 'well', 'acquisition_id'
        ]
        available_group_cols = [c for c in candidate_cols if c in self.feature_dataframe.columns]
        if not available_group_cols:
            available_group_cols = ['acquisition_name'] if 'acquisition_name' in self.feature_dataframe.columns else []
        for col in available_group_cols:
            self.group_by_combo.addItem(col)
        viz_layout.addWidget(self.group_by_combo)

        # Colormap selector (for heatmaps and differential expression)
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

        # Heatmap source selector (Clusters vs Manual Gates)
        self.heatmap_source_label = QtWidgets.QLabel("Heatmap of:")
        viz_layout.addWidget(self.heatmap_source_label)
        self.heatmap_source_combo = QtWidgets.QComboBox()
        self.heatmap_source_combo.addItems(["Heatmap of:", "Clusters", "Manual Gates"])  # temporary to ensure widget exists
        self.heatmap_source_combo.clear()
        self.heatmap_source_combo.addItems(["Clusters", "Manual Gates"])
        self.heatmap_source_combo.currentTextChanged.connect(self._on_heatmap_source_changed)
        viz_layout.addWidget(self.heatmap_source_combo)

        # Heatmap filter button
        self.heatmap_filter_btn = QtWidgets.QPushButton("Filter…")
        self.heatmap_filter_btn.setToolTip("Filter which clusters/phenotypes appear in the heatmap")
        self.heatmap_filter_btn.clicked.connect(self._open_heatmap_filter_dialog)
        viz_layout.addWidget(self.heatmap_filter_btn)

        # Top N markers selector (for differential expression only)
        self.top_n_label = QtWidgets.QLabel("Top N:")
        viz_layout.addWidget(self.top_n_label)
        self.top_n_spinbox = QtWidgets.QSpinBox()
        self.top_n_spinbox.setMinimum(1)
        self.top_n_spinbox.setMaximum(20)
        self.top_n_spinbox.setValue(5)
        self.top_n_spinbox.valueChanged.connect(self._on_top_n_changed)
        viz_layout.addWidget(self.top_n_spinbox)

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
        
    def _setup_plot(self):
        """Setup the matplotlib plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Click 'Run Clustering' to generate heatmap", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        self.canvas.draw()
        self._update_viz_controls_visibility()
        
    def _on_clustering_type_changed(self):
        """Handle clustering type change to show/hide relevant controls."""
        clustering_type = self.clustering_type.currentText()
        is_leiden = clustering_type == "Leiden"
        is_hierarchical = clustering_type == "Hierarchical"
        
        # Show/hide Leiden options group
        self.leiden_options_group.setVisible(is_leiden)
        
        # Show/hide hierarchical method selection
        self.hierarchical_label.setVisible(is_hierarchical)
        self.hierarchical_method.setVisible(is_hierarchical)
        # Show/hide number of clusters only for hierarchical
        if hasattr(self, 'n_clusters_label'):
            self.n_clusters_label.setVisible(is_hierarchical)
        if hasattr(self, 'n_clusters'):
            self.n_clusters.setVisible(is_hierarchical)
        
        # Show/hide dendrogram controls for hierarchical methods
        self.dendro_label.setVisible(is_hierarchical)
        self.dendro_mode.setVisible(is_hierarchical)
        self._update_viz_controls_visibility()
    
    def _on_leiden_mode_changed(self):
        """Handle Leiden clustering mode change (resolution vs modularity)."""
        use_resolution = self.resolution_radio.isChecked()
        self.resolution_label.setVisible(use_resolution)
        self.resolution_spinbox.setVisible(use_resolution)
        
    def _run_clustering(self):
        """Run the clustering analysis."""
        try:
            # Reset custom cluster names when re-clustering
            self.cluster_annotation_map = {}
            self.cluster_backend_names = {}
            # Clear LLM phenotype cache when re-clustering
            print(f"[DEBUG] Clearing LLM cache before re-clustering. Previous cache had {len(self.llm_phenotype_cache)} entries")
            self.llm_phenotype_cache = {}
            print("[DEBUG] LLM cache cleared")
            
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
            else:  # Hierarchical
                cluster_method = self.hierarchical_method.currentText()
            
            # Prepare data
            # Allow user to select features interactively
            available_cols = self._list_available_feature_columns(include_morpho)
            from openmcd.ui.dialogs.feature_selector_dialog import FeatureSelectorDialog
            selector = FeatureSelectorDialog(available_cols, self)
            if selector.exec_() != QtWidgets.QDialog.Accepted:
                return
            selected_columns = selector.get_selected_columns()

            data = self._prepare_clustering_data(agg_method, include_morpho, selected_columns)
            
            if data is None or data.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for clustering.")
                return
            
            # Clear canvas before clustering
            self.figure.clear()
            self.canvas.draw()
            
            # Perform clustering
            self.clustered_data, self.cluster_labels = self._perform_clustering(data, n_clusters, cluster_method)
            
            # Default to heatmap view after clustering
            self._create_heatmap()
            
            # Force canvas refresh
            self.canvas.draw()
            
            # Enable buttons
            self.explore_btn.setEnabled(True)
            self.annotate_btn.setEnabled(True)
            self.save_plot_btn.setEnabled(True)
            self.save_output_btn.setEnabled(True)
            # If UMAP was previously run, keep that available
            # Otherwise, selecting UMAP will prompt to run

            # Auto-apply annotations if already loaded for these cluster ids
            if self.cluster_annotation_map:
                self._apply_cluster_annotations()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Clustering Error", f"Error during clustering: {str(e)}")
    
    def _list_available_feature_columns(self, include_morpho):
        marker_cols = [col for col in self.feature_dataframe.columns 
                      if any(col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        morpho_cols = []
        if include_morpho:
            morpho_cols = [col for col in self.feature_dataframe.columns 
                          if col in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity', 
                                   'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um', 
                                   'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count']]
        # Note: centroid_x and centroid_y are excluded from clustering as they are spatial coordinates
        return sorted(set(marker_cols + morpho_cols))

    def _prepare_clustering_data(self, agg_method, include_morpho, selected_columns):
        """Prepare data for clustering."""
        feature_cols = list(selected_columns or [])
        
        if not feature_cols:
            return None
        
        # Check if all selected columns exist in the dataframe
        missing_cols = [col for col in feature_cols if col not in self.feature_dataframe.columns]
        if missing_cols:
            return None
        
        # Extract data
        data = self.feature_dataframe[feature_cols].copy()
        # Handle missing/infinite values safely
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median(numeric_only=True))
        
        # Normalize data (z-score) and drop any residual non-finite rows/cols
        # Handle columns with zero variance (all values are the same)
        data_means = data.mean()
        data_stds = data.std(ddof=0)

        # For columns with zero variance, set them to 0 (centered but not scaled)
        zero_var_cols = data_stds == 0
        if zero_var_cols.any():
            # Set zero variance columns to 0 (centered)
            data.loc[:, zero_var_cols] = 0
            # Only normalize non-zero variance columns
            non_zero_var_cols = ~zero_var_cols
            if non_zero_var_cols.any():
                # Ensure dtype compatibility by converting to float64 for calculation, then back to original dtype
                normalized_data = (data.loc[:, non_zero_var_cols] - data_means[non_zero_var_cols]) / data_stds[non_zero_var_cols]
                data.loc[:, non_zero_var_cols] = normalized_data.astype(data.dtypes[non_zero_var_cols])
        else:
            # Normalize all columns normally
            normalized_data = (data - data_means) / data_stds
            data = normalized_data.astype(data.dtypes)
        
        data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').dropna(axis=1, how='any')
        
        # Guard: require at least 2 rows and 2 columns to compute distances
        if data.shape[0] < 2 or data.shape[1] < 2:
            return None
        
        return data
    
    def _perform_clustering(self, data, n_clusters, method):
        """Perform clustering using specified method."""
        if method == "leiden":
            return self._perform_leiden_clustering(data)
        else:
            return self._perform_hierarchical_clustering(data, n_clusters, method)
    
    def _perform_hierarchical_clustering(self, data, n_clusters, method):
        """Perform hierarchical clustering."""
        # Calculate distance matrix
        distances = pdist(data.values, metric='euclidean')
        
        # Perform linkage
        linkage_matrix = linkage(distances, method=method)
        
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _perform_leiden_clustering(self, data):
        """Perform Leiden clustering."""
        if not _HAVE_LEIDEN:
            raise ImportError("leidenalg and igraph are required for Leiden clustering")
        
        # Calculate distance matrix
        distances = pdist(data.values, metric='euclidean')
        
        # Convert to similarity matrix (invert distances)
        max_dist = np.max(distances)
        similarities = max_dist - distances
        
        # Create graph from similarity matrix
        n = data.shape[0]
        edges = []
        weights = []
        
        # Convert condensed distance matrix to edge list
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[idx] > 0:  # Only add positive similarities
                    edges.append((i, j))
                    weights.append(similarities[idx])
                idx += 1
        
        # Create igraph
        g = ig.Graph(n)
        g.add_edges(edges)
        g.es['weight'] = weights
        
        # Perform Leiden clustering
        if self.resolution_radio.isChecked():
            # Use resolution parameter
            resolution = self.resolution_spinbox.value()
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=resolution,
                seed=42,
            )
        else:
            # Use modularity optimization
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights='weight',
                seed=42,
            )
        
        # Get cluster labels
        cluster_labels = np.array(partition.membership) + 1  # Start from 1
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _create_heatmap(self):
        """Create the heatmap visualization."""
        self.figure.clear()
        
        # Check if clustered_data exists
        if self.clustered_data is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No clustered data available.\nPlease run clustering first.", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Heatmap")
            self.canvas.draw()
            return
        
        # Determine source and prepare data ordering and optional grouping
        source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
        data_to_plot = self.clustered_data.copy()
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
            if hasattr(self, 'heatmap_filter_selection') and self.heatmap_filter_selection:
                wanted_ids = set()
                for cid in sorted(self.clustered_data['cluster'].unique()):
                    name = self._get_cluster_display_name(cid)
                    if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                        wanted_ids.add(cid)
                if wanted_ids:
                    data_to_plot = data_to_plot[data_to_plot['cluster'].isin(sorted(wanted_ids))]
            data_to_plot = data_to_plot.sort_values('cluster')

        # Use seaborn if available, otherwise fall back to matplotlib
        if _HAVE_SEABORN:
            # The seaborn path internally computes source/filtering similarly
            self._create_seaborn_heatmap()
            return
        
        # Create single heatmap plot - no subplots needed
        ax_heatmap = self.figure.add_subplot(111)

        # Prepare data for heatmap (exclude grouping columns)
        feature_cols = self._select_feature_columns(data_to_plot)
        heatmap_data = data_to_plot[feature_cols].values

        # No dendrograms - just show the heatmap data as-is

        # Create heatmap with user-selected colormap
        colormap_name = self._get_colormap_name()
        im = ax_heatmap.imshow(heatmap_data.T, aspect='auto', cmap=colormap_name, interpolation='nearest')

        # Set labels only; tick order should follow row order in clustered_data
        ax_heatmap.set_xlabel('Cells')
        ax_heatmap.set_ylabel('Features')
        ax_heatmap.set_yticks(np.arange(len(feature_cols)))
        ax_heatmap.set_yticklabels(feature_cols, fontsize=6, rotation=0)
        
        # Remove x-axis tick labels (cluster identity shown via color bar instead)
        ax_heatmap.set_xticks([])
        
        
        # Add group color bars along x-axis
        unique_groups = sorted(data_to_plot[group_col].unique())
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
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
        
        
        # (Feature list removed; top-right is optionally used for column dendrogram)

        # Add legend for groups/clusters
        legend_elements = []
        if source == 'Manual Gates':
            for key, color in cluster_color_map.items():
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=str(key)))
        else:
            for key, color in cluster_color_map.items():
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=self._get_cluster_display_name(key)))
        self.figure.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        self.canvas.draw()
    
    def _create_seaborn_heatmap(self):
        """Create heatmap using seaborn clustermap with color bars."""
        try:
            # Check if clustered_data exists
            if self.clustered_data is None:
                self._create_matplotlib_heatmap()
                return
            # Prepare data considering source/filter
            source = self.heatmap_source_combo.currentText() if hasattr(self, 'heatmap_source_combo') else 'Clusters'
            data_to_plot = self.clustered_data.copy()
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
                    for cid in sorted(self.clustered_data['cluster'].unique()):
                        name = self._get_cluster_display_name(cid)
                        if name in self.heatmap_filter_selection or str(cid) in self.heatmap_filter_selection:
                            wanted_ids.add(cid)
                    if wanted_ids:
                        data_to_plot = data_to_plot[data_to_plot['cluster'].isin(sorted(wanted_ids))]
                data_to_plot = data_to_plot.sort_values('cluster')

            feature_cols = self._select_feature_columns(data_to_plot)
            heatmap_data = data_to_plot[feature_cols]
            
            # Store original feature order for y-tick labels
            original_feature_order = list(feature_cols)
            
            # Create group color mapping
            unique_groups = sorted(data_to_plot[group_col].unique())
            cluster_colors = sns.color_palette("Set3", len(unique_groups))
            cluster_color_map = {gid: cluster_colors[i] for i, gid in enumerate(unique_groups)}
            
            # Create color series for color bar
            cluster_colors_series = data_to_plot[group_col].map(cluster_color_map)
            
            # Determine clustering settings based on method
            clustering_type = self.clustering_type.currentText()
            is_leiden = clustering_type == "Leiden"
            
            if is_leiden:
                # For Leiden clustering, disable dendrograms
                row_cluster = False
                col_cluster = False
                linkage_method = None
            else:
                # For hierarchical clustering, use dendrograms
                row_cluster = True
                col_cluster = (self.dendro_mode.currentText() == "Rows and columns")
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
            
            g.ax_heatmap.set_yticks(range(len(feature_names)))
            g.ax_heatmap.set_yticklabels(feature_names, fontsize=6)
            
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
        # Create subplots - simplified layout without cluster size bar
        gs = self.figure.add_gridspec(1, 1, hspace=0.1, wspace=0.1)
        
        # Main heatmap - use full figure area
        ax_heatmap = self.figure.add_subplot(gs[0])
        
        # Use tight layout to maximize plot area
        self.figure.tight_layout(pad=1.0)
        
        # Prepare data for heatmap (exclude cluster column)
        feature_cols = self._select_feature_columns(self.clustered_data)
        heatmap_data = self.clustered_data[feature_cols].values

        # No dendrograms - just show the heatmap data as-is
        
        # Create heatmap with user-selected colormap
        colormap_name = self._get_colormap_name()
        im = ax_heatmap.imshow(heatmap_data.T, aspect='auto', cmap=colormap_name, interpolation='nearest')
        
        # Set labels and ticks
        ax_heatmap.set_xlabel('Cells')
        ax_heatmap.set_ylabel('Features')
        ax_heatmap.set_yticks(np.arange(len(feature_cols)))
        ax_heatmap.set_yticklabels(feature_cols, fontsize=6, rotation=0)
        
        # Remove x-axis tick labels (cluster identity shown via color bar instead)
        ax_heatmap.set_xticks([])
        
        
        # Add cluster color bars along x-axis
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(self.clustered_data['cluster'].unique())))
        cluster_color_map = {cluster_id: cluster_colors[i] for i, cluster_id in enumerate(sorted(self.clustered_data['cluster'].unique()))}
        
        # Create color bar for each cell
        cell_colors = [cluster_color_map[cluster_id] for cluster_id in self.clustered_data['cluster']]
        
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

        # Add legend for groups/clusters
        legend_elements = []
        if source == 'Manual Gates':
            for key, color in cluster_color_map.items():
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=str(key)))
        else:
            for key, color in cluster_color_map.items():
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=self._get_cluster_display_name(key)))
        self.figure.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        self.canvas.draw()
    
    def _add_cluster_legend(self, g, cluster_color_map, source='Clusters'):
        """Add legend to seaborn clustermap using cluster names or manual group labels."""
        legend_elements = []
        for key, color in cluster_color_map.items():
            if source == 'Manual Gates':
                label = str(key)
            else:
                label = self._get_cluster_display_name(key)
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=label))
        g.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    def _run_umap(self):
        """Run UMAP dimensionality reduction analysis."""
        try:
            if not _HAVE_UMAP:
                QtWidgets.QMessageBox.warning(self, "UMAP Not Available", 
                    "UMAP is not installed. Please install umap-learn to use this feature.")
                return
            
            # Get feature selection from user
            available_cols = self._list_available_feature_columns(True)  # Include morphometric features
            from openmcd.ui.dialogs.feature_selector_dialog import FeatureSelectorDialog
            selector = FeatureSelectorDialog(available_cols, self)
            if selector.exec_() != QtWidgets.QDialog.Accepted:
                return
            selected_columns = selector.get_selected_columns()
            
            if not selected_columns:
                QtWidgets.QMessageBox.warning(self, "No Features", "Please select at least one feature for UMAP analysis.")
                return
            
            # Prepare data for UMAP, align with clustered order if available
            if self.clustered_data is not None:
                ordered_index = self.clustered_data.index
                data = self.feature_dataframe.loc[ordered_index, selected_columns].copy()
            else:
                data = self.feature_dataframe[selected_columns].copy()
            
            # Handle missing values and infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(data.median())
            
            if data.empty or data.shape[0] < 2:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for UMAP analysis.")
                return
            
            # Clear canvas before UMAP
            self.figure.clear()
            self.canvas.draw()
            
            # Allow user to choose n_neighbors
            default_n = 15
            max_n = max(2, min(default_n, data.shape[0] - 1))
            # Simple input dialog for n_neighbors with bounds
            n_neighbors, ok = QtWidgets.QInputDialog.getInt(
                self,
                "UMAP n_neighbors",
                f"Set n_neighbors (2–{max(2, data.shape[0]-1)}):",
                value=max_n,
                min=2,
                max=max(2, data.shape[0]-1)
            )
            if not ok:
                return
            # Perform UMAP
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=int(n_neighbors), min_dist=0.1)
            self.umap_embedding = reducer.fit_transform(data.values)
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
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UMAP Error", f"Error during UMAP analysis: {str(e)}")
    
    def _show_heatmap(self):
        """Switch back to heatmap view."""
        if self.clustered_data is not None:
            self._create_heatmap()
        else:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view the heatmap.")

    def _on_view_changed(self, view: str):
        """Switch visualization based on selected view and manage dependencies."""
        # Clear canvas before switching views
        self.figure.clear()
        self.canvas.draw()
        
        self._update_viz_controls_visibility()
        if view == 'Heatmap':
            self._show_heatmap()
        elif view == 'UMAP':
            if getattr(self, 'umap_embedding', None) is None:
                self._run_umap()
            else:
                self._create_umap_plot()
        elif view == 'Stacked Bars':
            self._show_stacked_bars()
        elif view == 'Differential Expression':
            self._show_differential_expression()
        
        # Force canvas refresh after view change
        self.canvas.draw()
        
        # Enable save if there is content
        self.save_plot_btn.setEnabled(True)
        self.save_output_btn.setEnabled(True)

    def _update_viz_controls_visibility(self):
        """Show/hide controls depending on selected view."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        # Color-by visible only for UMAP
        for i in range(self.color_by_combo.count()):
            pass
        if hasattr(self, 'color_by_label'):
            self.color_by_label.setVisible(view == 'UMAP')
        self.color_by_combo.setVisible(view == 'UMAP')
        # Group-by visible only for Stacked Bars
        if hasattr(self, 'group_by_label'):
            self.group_by_label.setVisible(view == 'Stacked Bars')
        self.group_by_combo.setVisible(view == 'Stacked Bars')
        # Colormap visible only for Heatmap and Differential Expression; hidden for UMAP and Stacked Bars
        if hasattr(self, 'colormap_label'):
            self.colormap_label.setVisible(view in ['Heatmap', 'Differential Expression'])
        self.colormap_combo.setVisible(view in ['Heatmap', 'Differential Expression'])
        # Top N visible only for Differential Expression
        if hasattr(self, 'top_n_label'):
            self.top_n_label.setVisible(view == 'Differential Expression')
        self.top_n_spinbox.setVisible(view == 'Differential Expression')
        # Heatmap-only controls
        is_heatmap = view == 'Heatmap'
        if hasattr(self, 'heatmap_source_combo'):
            self.heatmap_source_combo.setVisible(is_heatmap)
        if hasattr(self, 'heatmap_source_label'):
            self.heatmap_source_label.setVisible(is_heatmap)
        if hasattr(self, 'heatmap_filter_btn'):
            self.heatmap_filter_btn.setVisible(is_heatmap)
    
    def _on_colormap_changed(self, _text: str):
        """Handle colormap selection change."""
        # Refresh the current view if it uses colormaps
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view in ['Heatmap', 'Differential Expression']:
            if view == 'Heatmap':
                self._show_heatmap()
            elif view == 'Differential Expression':
                self._show_differential_expression()
    
    def _on_top_n_changed(self, _value: int):
        """Handle top N markers selection change."""
        # Refresh the differential expression view
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
        """Return numeric feature columns to plot, excluding non-numeric/meta columns."""
        exclude_cols = { 'cluster', '__group__', 'cluster_phenotype', 'manual_phenotype' }
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    feature_cols.append(col)
            except Exception:
                continue
        return feature_cols

    def _on_heatmap_source_changed(self, _text: str):
        """Refresh heatmap when the source (Clusters vs Manual Gates) changes."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        if view == 'Heatmap':
            self._show_heatmap()

    def _get_cluster_display_name(self, cluster_id):
        """Return display label for a cluster id, using annotation if available."""
        if isinstance(self.cluster_annotation_map, dict) and cluster_id in self.cluster_annotation_map and self.cluster_annotation_map[cluster_id]:
            return self.cluster_annotation_map[cluster_id]
        return f"Cluster {cluster_id}"

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
            options = sorted(self.clustered_data['cluster'].unique())
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
    
    def _create_umap_plot(self):
        """Create UMAP scatter plot."""
        if self.umap_embedding is None:
            return
        
        self.figure.clear()
        # Use tight layout to maximize plot area
        ax = self.figure.add_subplot(111)
        self.figure.tight_layout(pad=1.0)
        
        # Determine coloring
        color_by = self.color_by_combo.currentText() if hasattr(self, 'color_by_combo') else 'Cluster'
        if color_by == 'Cluster' and self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
            # Align cluster labels to UMAP order
            if hasattr(self, 'umap_index') and self.umap_index is not None:
                cluster_labels_series = self.clustered_data['cluster']
                cluster_labels = cluster_labels_series.reindex(self.umap_index).values
            else:
                cluster_labels = self.clustered_data['cluster'].values
            unique_clusters = sorted(np.unique(cluster_labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
            handles = []
            labels = []
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                sc = ax.scatter(self.umap_embedding[mask, 0], self.umap_embedding[mask, 1],
                                c=[cluster_color_map[cluster_id]],
                                alpha=0.8, s=18, edgecolors='none')
                handles.append(sc)
                labels.append(self._get_cluster_display_name(cluster_id))
            # Place legend inside axes to avoid clipping
            ax.legend(handles, labels, loc='best', frameon=True, fontsize=8)
        elif hasattr(self, 'umap_raw_data') and color_by in getattr(self, 'umap_selected_columns', []):
            # Continuous coloring by selected feature (aligned to UMAP order)
            vals = self.umap_raw_data[color_by].values
            sc = ax.scatter(self.umap_embedding[:, 0], self.umap_embedding[:, 1], c=vals,
                            cmap='viridis', alpha=0.85, s=16, edgecolors='none')
            cbar = self.figure.colorbar(sc, ax=ax)
            cbar.set_label(color_by)
        else:
            # Fallback single color
            ax.scatter(self.umap_embedding[:, 0], self.umap_embedding[:, 1], c='blue', alpha=0.7, s=16, edgecolors='none')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Dimensionality Reduction')
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()

    def _populate_color_by_options(self):
        """Populate the color-by combo with Cluster + used features."""
        if not hasattr(self, 'color_by_combo'):
            return
        current = self.color_by_combo.currentText() if self.color_by_combo.count() > 0 else 'Cluster'
        self.color_by_combo.blockSignals(True)
        self.color_by_combo.clear()
        self.color_by_combo.addItem('Cluster')
        for col in getattr(self, 'umap_selected_columns', []) or []:
            self.color_by_combo.addItem(col)
        # Add phenotype if available
        if hasattr(self, 'clustered_data') and self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
            if self.color_by_combo.findText('Phenotype') < 0:
                self.color_by_combo.addItem('Phenotype')
        # Add manual phenotype if available
        if 'manual_phenotype' in self.feature_dataframe.columns:
            if self.color_by_combo.findText('Manual Phenotype') < 0:
                self.color_by_combo.addItem('Manual Phenotype')
        idx = self.color_by_combo.findText(current)
        if idx >= 0:
            self.color_by_combo.setCurrentIndex(idx)
        self.color_by_combo.blockSignals(False)

    def _on_color_by_changed(self, _text: str):
        if getattr(self, 'umap_embedding', None) is not None:
            self._create_umap_plot()

    def _show_stacked_bars(self):
        """Show stacked bar plots of cluster frequencies per selected group (ROI/condition/slide)."""
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view stacked bars.")
            return
        group_col = self.group_by_combo.currentText() if hasattr(self, 'group_by_combo') and self.group_by_combo.count() > 0 else None
        if not group_col or group_col not in self.feature_dataframe.columns:
            QtWidgets.QMessageBox.warning(self, "No Grouping", "No valid grouping column is available.")
            return

        try:
            # Align metadata to clustered_data order
            meta_series = self.feature_dataframe.loc[self.clustered_data.index, group_col]
            clusters = self.clustered_data['cluster']
            # Build counts per group and cluster
            ct = pd.crosstab(meta_series, clusters).sort_index()
            # Convert to frequencies
            freq = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

            # Prepare colors consistent with other views
            unique_clusters = sorted(clusters.unique())
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}

            # Plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            bottom = np.zeros(len(freq))
            x = np.arange(len(freq))
            for cluster_id in unique_clusters:
                vals = freq.get(cluster_id, pd.Series(0, index=freq.index)).values
                ax.bar(x, vals, bottom=bottom, color=cluster_color_map[cluster_id], label=self._get_cluster_display_name(cluster_id))
                bottom = bottom + vals

            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in freq.index], rotation=45, ha='right')
            ax.set_ylabel('Fraction of cells')
            ax.set_title(f'Cluster composition by {group_col}')
            ax.set_ylim(0, 1)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating stacked bars: {str(e)}")

    def _show_differential_expression(self):
        """Show differential expression heatmap showing top 5 markers per cluster."""
        if self.clustered_data is None or 'cluster' not in self.clustered_data.columns:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view differential expression.")
            return
        
        try:
            # Get numeric feature columns only (exclude cluster/phenotype/text metadata)
            feature_cols = self._select_feature_columns(self.clustered_data)
            
            if not feature_cols:
                QtWidgets.QMessageBox.warning(self, "No Features", "No features available for differential expression analysis.")
                return
            
            # Calculate mean expression per cluster for each feature
            cluster_means = self.clustered_data.groupby('cluster')[feature_cols].mean()
            
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
            
            # Sort clusters for consistent ordering
            sorted_clusters = sorted(differential_scores.index)
            
            for cluster_id in sorted_clusters:
                # Get z-scores for this cluster
                cluster_scores = differential_scores.loc[cluster_id]
                # Sort by z-score (descending) and take top N
                top_n_for_cluster = cluster_scores.nlargest(top_n).index.tolist()
                cluster_top_features[cluster_id] = top_n_for_cluster
                # Add features for this cluster to the ordered list
                top_features.extend(top_n_for_cluster)
            
            if not top_features:
                QtWidgets.QMessageBox.warning(self, "No Features", "No features found for differential expression analysis.")
                return
            
            # Create heatmap data with all top features
            heatmap_data = differential_scores[top_features]
            
            # Create the plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Create heatmap with user-selected colormap
            colormap_name = self._get_colormap_name()
            im = ax.imshow(heatmap_data.T, cmap=colormap_name, aspect='auto', 
                          vmin=-3, vmax=3)  # Limit color scale to ±3 z-scores
            
            # Set labels
            ax.set_xticks(range(len(heatmap_data.index)))
            ax.set_xticklabels([self._get_cluster_display_name(i) for i in heatmap_data.index])
            ax.set_yticks(range(len(heatmap_data.columns)))
            ax.set_yticklabels(heatmap_data.columns, rotation=0)
            
            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Z-score (Differential Expression)', rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(f'Top {top_n} Differential Expression Markers per Cluster')
            ax.set_xlabel('Clusters')
            ax.set_ylabel('Features')
            
            # Add text annotations showing actual z-scores
            # Also highlight the top N markers for each cluster
            for i in range(len(heatmap_data.index)):
                cluster_id = heatmap_data.index[i]
                top_n_for_this_cluster = cluster_top_features[cluster_id]
                
                for j in range(len(heatmap_data.columns)):
                    feature_name = heatmap_data.columns[j]
                    value = heatmap_data.iloc[i, j]
                    
                    # Color text based on background
                    text_color = 'white' if abs(value) > 1.5 else 'black'
                    
                    # Make top N markers for this cluster more prominent
                    fontweight = 'bold'
                    fontsize = 9
                    if feature_name in top_n_for_this_cluster:
                        # Highlight top N markers with larger, bolder text
                        fontweight = 'bold'
                        fontsize = 10
                        # Add a subtle background highlight
                        ax.add_patch(plt.Rectangle((i-0.4, j-0.4), 0.8, 0.8, 
                                                 fill=False, edgecolor='black', 
                                                 linewidth=2, alpha=0.7))
                    
                    ax.text(i, j, f'{value:.2f}', ha='center', va='center', 
                           color=text_color, fontsize=fontsize, fontweight=fontweight)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add explanation text below the colorbar
            explanation_text = (f"Black boxes highlight the top {top_n} markers for each cluster.\n"
                              "Z-scores show how much each cluster differs from the overall mean.")
            # Position text below the colorbar
            ax.text(1.02, -0.15, explanation_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error creating differential expression heatmap: {str(e)}")

    def _open_gating_dialog(self):
        """Open gating rules editor and apply on save."""
        # Allow selection among intensity features by default
        marker_cols = [col for col in self.feature_dataframe.columns
                       if any(col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        dlg = GatingRulesDialog(self.gating_rules, marker_cols, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.gating_rules = dlg.get_rules()
            self._apply_gating_rules()
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
        unique_clusters = sorted(self.clustered_data['cluster'].unique())
        # Build and show dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Annotate Phenotypes")
        v = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        editors = {}
        for cid in unique_clusters:
            le = QtWidgets.QLineEdit()
            if cid in self.cluster_annotation_map:
                le.setText(self.cluster_annotation_map[cid])
            form.addRow(f"Cluster {cid}", le)
            editors[cid] = le
        v.addLayout(form)
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
            d = PhenotypeSuggestionDialog(self, unique_clusters, apply_names, self.llm_phenotype_cache, self.normalization_config)
            d.exec_()
        llm_btn.clicked.connect(open_llm_dialog)

        # Make the dialog wider for better usability
        dlg.resize(500, dlg.sizeHint().height())

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Save mapping from editors
            self.cluster_annotation_map = {
                cid: editors[cid].text().strip() for cid in unique_clusters if editors[cid].text().strip()
            }
            self._apply_cluster_annotations()
            QtWidgets.QMessageBox.information(self, "Annotations Applied", "Cluster annotations have been applied.")

    def _apply_cluster_annotations(self):
        """Apply current annotation map to clustered_data and feature_dataframe as 'cluster_phenotype'."""
        if not self.clustered_data is None and self.cluster_annotation_map:
            # Use backend names for CSV export if available, otherwise use display names
            export_name_map = self.cluster_backend_names if self.cluster_backend_names else self.cluster_annotation_map
            
            # Map on clustered_data using backend names for CSV export
            self.clustered_data['cluster_phenotype'] = self.clustered_data['cluster'].map(export_name_map).fillna('')
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
        
        # Get cluster info
        cluster_info = []
        for cluster_id in sorted(self.clustered_data['cluster'].unique()):
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
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Plot", default,
            "PNG Files (*.png)"
        )
        if not file_path:
            return
        try:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            QtWidgets.QMessageBox.information(self, "Success", f"Plot saved to: {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving plot: {str(e)}")

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
            # Start with the original feature dataframe (all features)
            output_df = self.feature_dataframe.copy()
            
            # Add cluster labels
            if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
                # Align cluster data with feature dataframe by index
                cluster_series = self.clustered_data['cluster'].reindex(output_df.index)
                output_df['cluster'] = cluster_series
            
            # Add cluster phenotype annotations if available
            if self.clustered_data is not None and 'cluster_phenotype' in self.clustered_data.columns:
                phenotype_series = self.clustered_data['cluster_phenotype'].reindex(output_df.index)
                output_df['cluster_phenotype'] = phenotype_series
            
            # Add manual phenotype annotations if available
            if self.clustered_data is not None and 'manual_phenotype' in self.clustered_data.columns:
                manual_series = self.clustered_data['manual_phenotype'].reindex(output_df.index)
                output_df['manual_phenotype'] = manual_series
            
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

# --------------------------
# Cluster Explorer Dialog
# --------------------------
class ClusterExplorerDialog(QtWidgets.QDialog):
    def __init__(self, cluster_info, feature_dataframe, parent=None, label_provider=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Explorer")
        self.setModal(True)
        
        # Set size to 90% of parent window if available
        if parent is not None:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)
        
        self.setMinimumSize(1000, 700)
        self.cluster_info = cluster_info
        self.feature_dataframe = feature_dataframe
        self.current_cluster = None
        self.cell_images = []
        self._label_provider = label_provider
        
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Cluster Explorer")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 16px; }")
        layout.addWidget(title_label)
        
        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Cluster selection
        controls_layout.addWidget(QtWidgets.QLabel("Select Cluster:"))
        self.cluster_combo = QtWidgets.QComboBox()
        for info in self.cluster_info:
            label = self._get_cluster_label(info['cluster_id'])
            self.cluster_combo.addItem(f"{label} ({info['size']} cells)", info)
        self.cluster_combo.currentIndexChanged.connect(self._on_cluster_changed)
        controls_layout.addWidget(self.cluster_combo)
        
        # Channel selection
        controls_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        controls_layout.addWidget(self.channel_combo)
        
        # RGB mode checkbox
        self.rgb_checkbox = QtWidgets.QCheckBox("RGB Mode")
        self.rgb_checkbox.setToolTip("Show RGB composite instead of single channel")
        self.rgb_checkbox.toggled.connect(self._on_rgb_mode_toggled)
        controls_layout.addWidget(self.rgb_checkbox)
        
        # RGB channel selection (initially hidden)
        self.rgb_channels_layout = QtWidgets.QHBoxLayout()
        self.rgb_channels_layout.addWidget(QtWidgets.QLabel("R:"))
        self.rgb_r_combo = QtWidgets.QComboBox()
        self.rgb_channels_layout.addWidget(self.rgb_r_combo)
        
        self.rgb_channels_layout.addWidget(QtWidgets.QLabel("G:"))
        self.rgb_g_combo = QtWidgets.QComboBox()
        self.rgb_channels_layout.addWidget(self.rgb_g_combo)
        
        self.rgb_channels_layout.addWidget(QtWidgets.QLabel("B:"))
        self.rgb_b_combo = QtWidgets.QComboBox()
        self.rgb_channels_layout.addWidget(self.rgb_b_combo)
        
        # Add RGB channel selection to a widget that can be shown/hidden
        self.rgb_channels_widget = QtWidgets.QWidget()
        self.rgb_channels_widget.setLayout(self.rgb_channels_layout)
        self.rgb_channels_widget.hide()
        controls_layout.addWidget(self.rgb_channels_widget)
        
        # Load images button
        self.load_btn = QtWidgets.QPushButton("Load Cell Images")
        self.load_btn.clicked.connect(self._load_cell_images)
        controls_layout.addWidget(self.load_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Image grid
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        layout.addWidget(self.scroll_area)
        
        # Status
        self.status_label = QtWidgets.QLabel("Select a cluster and channel to view cell images")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize
        self._populate_channels()
        self._on_cluster_changed()
    
    def _populate_channels(self):
        """Populate channel combo box with available channels."""
        # Get channels from feature dataframe columns
        # Look for any intensity feature suffix to identify channels
        intensity_suffixes = ['_mean', '_std', '_p10', '_p90', '_integrated', '_frac_pos', '_median', '_mad']
        channels = set()
        
        for col in self.feature_dataframe.columns:
            for suffix in intensity_suffixes:
                if col.endswith(suffix):
                    channel = col[:-len(suffix)]  # Remove the suffix to get channel name
                    channels.add(channel)
                    break  # Found a match, no need to check other suffixes
        
        # Add channels (RGB mode is controlled by checkbox)
        self.channel_combo.addItems(sorted(channels))
        
        # Populate RGB channel combos
        for combo in [self.rgb_r_combo, self.rgb_g_combo, self.rgb_b_combo]:
            combo.addItems(sorted(channels))
    
    def _on_cluster_changed(self):
        """Handle cluster selection change."""
        self.current_cluster = self.cluster_combo.currentData()
        if self.current_cluster:
            label = self._get_cluster_label(self.current_cluster['cluster_id'])
            self.status_label.setText(f"Selected {label} with {self.current_cluster['size']} cells")
    
    def _on_rgb_mode_toggled(self):
        """Handle RGB mode checkbox toggle."""
        if self.rgb_checkbox.isChecked():
            self.rgb_channels_widget.show()
            self.channel_combo.setEnabled(False)
        else:
            self.rgb_channels_widget.hide()
            self.channel_combo.setEnabled(True)
    
    def _load_cell_images(self):
        """Load and display cell images for the selected cluster."""
        if not self.current_cluster:
            return
        
        channel = self.channel_combo.currentText()
        if not channel:
            QtWidgets.QMessageBox.warning(self, "No Channel", "Please select a channel.")
            return
        
        try:
            # Get parent window to access loader and segmentation masks
            parent_window = self.parent()
            if not hasattr(parent_window, 'loader') or not hasattr(parent_window, 'segmentation_masks'):
                QtWidgets.QMessageBox.warning(self, "No Data", "Cannot access image data. Please ensure segmentation masks are loaded.")
                return
            
            # Clear previous images
            self.cell_images = []
            
            # Create image grid
            grid_widget = QtWidgets.QWidget()
            grid_layout = QtWidgets.QGridLayout(grid_widget)
            
            # Get cell data for this cluster
            cluster_cells = self.current_cluster['cells']
            
            # Limit to first 12 cells for performance
            max_cells = min(12, len(cluster_cells))
            crop_size = 30  # 30x30 pixel crop
            
            for i, cell_idx in enumerate(cluster_cells[:max_cells]):
                if i >= max_cells:
                    break
                
                # Get cell data
                cell_data = self.feature_dataframe.iloc[cell_idx]
                acq_id = cell_data['acquisition_id']
                
                # Get mask and image
                if acq_id in parent_window.segmentation_masks:
                    mask = parent_window.segmentation_masks[acq_id]
                    cell_id = int(cell_data['cell_id'])
                    
                    # Create cell mask
                    cell_mask = (mask == cell_id).astype(np.uint8)
                    
                    if np.any(cell_mask):
                        try:
                            # Calculate cell center using regionprops
                            props = regionprops(cell_mask)
                            if props:
                                center_y, center_x = props[0].centroid
                                center_y, center_x = int(center_y), int(center_x)
                                
                                # Define crop boundaries
                                half_crop = crop_size // 2
                                y_start = max(0, center_y - half_crop)
                                y_end = min(mask.shape[0], center_y + half_crop)
                                x_start = max(0, center_x - half_crop)
                                x_end = min(mask.shape[1], center_x + half_crop)
                                
                                # Crop the cell mask
                                cropped_mask = cell_mask[y_start:y_end, x_start:x_end]
                                
                                if self.rgb_checkbox.isChecked():
                                    # Load RGB composite using user-selected channels
                                    rgb_img = self._load_rgb_image(parent_window, acq_id)
                                    if rgb_img is not None:
                                        # Crop RGB image
                                        cropped_rgb = rgb_img[y_start:y_end, x_start:x_end]
                                        # Apply mask
                                        for c in range(3):
                                            cropped_rgb[:, :, c] *= cropped_mask
                                        
                                        # Create image widget with acquisition and cell info
                                        acq_label = self._get_acquisition_label(acq_id)
                                        img_widget = self._create_image_widget(cropped_rgb, f"{acq_label} - Cell {cell_id}", is_rgb=True)
                                        grid_layout.addWidget(img_widget, i // 4, i % 4)
                                        
                                        self.cell_images.append({
                                            'cell_id': cell_id,
                                            'acquisition_id': acq_id,
                                            'image': cropped_rgb
                                        })
                                else:
                                    # Load single channel
                                    channel_img = parent_window.loader.get_image(acq_id, channel)
                                    # Crop channel image
                                    cropped_channel = channel_img[y_start:y_end, x_start:x_end]
                                    # Apply mask
                                    cropped_channel *= cropped_mask
                                    
                                    # Create image widget with acquisition and cell info
                                    acq_label = self._get_acquisition_label(acq_id)
                                    # Include channel/acquisition info in title; do not pass unsupported kwargs
                                    img_widget = self._create_image_widget(cropped_channel, f"{acq_label} - {channel} - Cell {cell_id}", is_rgb=False)
                                    grid_layout.addWidget(img_widget, i // 4, i % 4)
                                    
                                    self.cell_images.append({
                                        'cell_id': cell_id,
                                        'acquisition_id': acq_id,
                                        'image': cropped_channel
                                    })
                            
                        except Exception as e:
                            print(f"Error loading image for cell {cell_id}: {e}")
                            continue
            
            self.scroll_area.setWidget(grid_widget)
            self.status_label.setText(f"Loaded {len(self.cell_images)} cell images for Cluster {self.current_cluster['cluster_id']}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading cell images: {str(e)}")
    
    def _load_rgb_image(self, parent_window, acq_id):
        """Load RGB composite image for an acquisition using user-selected channels."""
        try:
            # Use user-selected channels if RGB mode is enabled
            if self.rgb_checkbox.isChecked():
                r_channel = self.rgb_r_combo.currentText()
                g_channel = self.rgb_g_combo.currentText()
                b_channel = self.rgb_b_combo.currentText()
                
                if r_channel and g_channel and b_channel:
                    # Load the three user-selected channels
                    ch1 = parent_window.loader.get_image(acq_id, r_channel)
                    ch2 = parent_window.loader.get_image(acq_id, g_channel)
                    ch3 = parent_window.loader.get_image(acq_id, b_channel)
                    
                    # Normalize each channel to 0-1 range
                    ch1_norm = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-8)
                    ch2_norm = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-8)
                    ch3_norm = (ch3 - ch3.min()) / (ch3.max() - ch3.min() + 1e-8)
                    
                    # Create RGB composite
                    rgb_img = np.stack([ch1_norm, ch2_norm, ch3_norm], axis=-1)
                    return rgb_img
            else:
                # Fallback to automatic channel detection
                channels = set()
                intensity_suffixes = ['_mean', '_std', '_p10', '_p90', '_integrated', '_frac_pos', '_median', '_mad']
                for col in self.feature_dataframe.columns:
                    for suffix in intensity_suffixes:
                        if col.endswith(suffix):
                            channel = col[:-len(suffix)]  # Remove the suffix to get channel name
                            channels.add(channel)
                            break  # Found a match, no need to check other suffixes
                
                # Try to find RGB channels (common naming patterns)
                rgb_channels = []
                for pattern in ['DAPI', 'FITC', 'TRITC', 'Cy5', 'Hoechst', 'GFP', 'RFP', 'mCherry']:
                    for channel in channels:
                        if pattern.lower() in channel.lower():
                            rgb_channels.append(channel)
                            break
                
                # If we don't have 3 channels, just use the first 3 available
                if len(rgb_channels) < 3:
                    rgb_channels = list(channels)[:3]
                
                if len(rgb_channels) >= 3:
                    # Load the three channels
                    ch1 = parent_window.loader.get_image(acq_id, rgb_channels[0])
                    ch2 = parent_window.loader.get_image(acq_id, rgb_channels[1])
                    ch3 = parent_window.loader.get_image(acq_id, rgb_channels[2])
                    
                    # Normalize each channel to 0-1 range
                    ch1_norm = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-8)
                    ch2_norm = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-8)
                    ch3_norm = (ch3 - ch3.min()) / (ch3.max() - ch3.min() + 1e-8)
                    
                    # Create RGB composite
                    rgb_img = np.stack([ch1_norm, ch2_norm, ch3_norm], axis=-1)
                    return rgb_img
            
        except Exception as e:
            print(f"Error loading RGB image: {e}")
        
        return None
    
    def _get_acquisition_label(self, acq_id):
        """Get a user-friendly label for an acquisition ID."""
        # Try to find the acquisition in the parent window
        parent_window = self.parent()
        if hasattr(parent_window, 'acquisitions'):
            for acq in parent_window.acquisitions:
                if acq.id == acq_id:
                    return acq.name
        return acq_id

    def _get_cluster_label(self, cluster_id):
        """Get annotated cluster name from parent dialog if available."""
        provider = self._label_provider or self.parent()
        if provider is not None and hasattr(provider, '_get_cluster_display_name'):
            try:
                return provider._get_cluster_display_name(cluster_id)
            except Exception:
                pass
        return f"Cluster {cluster_id}"
    
    def _create_image_widget(self, image, title, is_rgb=False):
        """Create a widget to display a cell image."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Create matplotlib figure
        fig = Figure(figsize=(2, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Display image
        if is_rgb:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        
        layout.addWidget(canvas)
        return widget


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
            from PyQt5 import QtWidgets as _QtW
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
            from PyQt5 import QtWidgets as _QtW
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
        self._cache_dict = cache_dict or {}
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

        self.context_edit = QtWidgets.QLineEdit()
        self.context_edit.setPlaceholderText("e.g., human colorectal cancer (optional)")
        form.addRow("Cohort/tissue context:", self.context_edit)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1"])
        form.addRow("Model:", self.model_combo)

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

        layout.addLayout(form)

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
        
        # Check for cached results and display them immediately
        self._check_and_display_cached_results()

        self._suggestions = {}  # cluster_id -> parsed json

    def closeEvent(self, event):
        """Handle dialog closing to preserve cache."""
        # Ensure the current suggestions are cached for future use
        if self._cache_dict is not None and self._suggestions:
            self._cache_dict.update(self._suggestions)
            print(f"[DEBUG] Preserved cache on close for clusters: {list(self._suggestions.keys())}")
        event.accept()

    def _reset_progress_bar(self):
        """Reset the progress bar to its default state."""
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Suggestion")

    def _check_and_display_cached_results(self):
        """Check if we have cached results for the current cluster set and display them."""
        if not self._cache_dict:
            return
            
        # Check if we have cached results for all current clusters
        cached_results = {}
        for cid in self._cluster_ids:
            # Convert cluster ID to int for consistent comparison
            cid_int = int(cid)
            # Check both the original cid and converted int version
            if cid in self._cache_dict:
                cached_results[cid] = self._cache_dict[cid]
            elif cid_int in self._cache_dict:
                cached_results[cid] = self._cache_dict[cid_int]
        
        # Debug: Print cache status
        print(f"[DEBUG] Cache check: {len(cached_results)}/{len(self._cluster_ids)} clusters cached")
        print(f"[DEBUG] Current cluster IDs: {self._cluster_ids}")
        print(f"[DEBUG] Cached cluster IDs: {list(self._cache_dict.keys())}")
        
        # If we have cached results for all clusters, display them
        if cached_results and len(cached_results) == len(self._cluster_ids):
            # Store the cached results in _suggestions so they can be applied
            self._suggestions = cached_results.copy()
            self._render_choices(cached_results)
            self.apply_btn.setEnabled(True)
            # Disable the run button since we already have results
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Results Cached - Re-run to refresh")
            print("[DEBUG] Displaying cached results")
        else:
            print("[DEBUG] No cached results found or incomplete cache")

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
            if self._cache_dict is not None and self._suggestions:
                self._cache_dict.update(self._suggestions)
            QtWidgets.QMessageBox.information(self, "Applied", f"Applied {len(display_name_map)} suggested names.")

    def _debug_validate_payload(self, payload: dict) -> bool:
        try:
            ok = True
            # Basic keys
            if not isinstance(payload, dict):
                print("[DEBUG] Payload is not a dict")
                return False
            if 'input' not in payload:
                print("[DEBUG] Payload missing 'input'")
                ok = False
            if 'model' not in payload:
                print("[DEBUG] Payload missing 'model'")
                ok = False
            # Input structure
            msgs = payload.get('input')
            if not isinstance(msgs, list) or len(msgs) < 2:
                print("[DEBUG] Payload 'input' is not a list of at least 2 messages")
                ok = False
            else:
                # Check roles and content types
                roles = [m.get('role') for m in msgs if isinstance(m, dict)]
                if roles[:2] != ['system', 'user']:
                    print(f"[DEBUG] Unexpected roles in messages: {roles}")
                for m in msgs:
                    content = m.get('content') if isinstance(m, dict) else None
                    if not isinstance(content, list):
                        print("[DEBUG] Message 'content' is not a list")
                        ok = False
                        break
                    for block in content:
                        if not isinstance(block, dict) or block.get('type') != 'input_text' or 'text' not in block:
                            print("[DEBUG] Content block missing type='input_text' or 'text'")
                            ok = False
                            break
            # Ensure user context JSON parses back
            try:
                user_blocks = msgs[1]['content'] if isinstance(msgs, list) and len(msgs) > 1 else []
                user_texts = [b.get('text') for b in user_blocks if isinstance(b, dict) and b.get('type') == 'input_text']
                if user_texts:
                    json.loads(user_texts[0])
            except Exception as e:
                print(f"[DEBUG] User context JSON failed to parse: {e}")
                ok = False
            return ok
        except Exception as e:
            print(f"[DEBUG] Exception during payload validation: {e}")
            return False

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
            # Use k_int as the K parameter for consistency
            self.progress.setFormat("Computing cluster statistics...")
            QtWidgets.QApplication.processEvents()
            stats_per_cluster = self._compute_stats(self._parent_dialog.clustered_data, K=k_int, mode=mode, k_int=k_int, k_morpho=k_morpho)
            results = {}
            total = max(1, len(self._cluster_ids))
            self.progress.setRange(0, total)
            self.progress.setValue(0)
            self.progress.setFormat("Processing clusters with LLM...")
            QtWidgets.QApplication.processEvents()
            for idx, cid in enumerate(self._cluster_ids, start=1):
                context_str = self.context_edit.text().strip() or "IMC panel of single cells"
                payload = self._build_prompt_payload(cid, stats_per_cluster.get(cid, {}), k_int, context_str)
                payload['model'] = self.model_combo.currentText()
                # Validate payload before sending
                self._debug_validate_payload(payload)
                suggestion = self._call_openai(api_key, payload)
                # Validate JSON
                obj = self._validate_json(suggestion, cid)
                if obj is None:
                    # One retry with repair instruction
                    suggestion = self._call_openai(api_key, payload, repair=True)
                    obj = self._validate_json(suggestion, cid)
                if obj is not None:
                    # Store with consistent integer keys
                    cid_int = int(cid)
                    results[cid_int] = obj
                    self._suggestions[cid_int] = obj
                # Update progress
                self.progress.setValue(idx)
                QtWidgets.QApplication.processEvents()
            if results:
                # Cache the results
                if self._cache_dict is not None:
                    self._cache_dict.update(results)
                    print(f"[DEBUG] Cached results for clusters: {list(results.keys())}")
                    print(f"[DEBUG] Total cached clusters: {list(self._cache_dict.keys())}")
                self._render_choices(results)
                self.apply_btn.setEnabled(True)
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

    def _base_marker_name(self, feature_name: str) -> str:
        # Strip trailing suffix like _mean/_median/etc.
        if '_' in feature_name:
            return feature_name.rsplit('_', 1)[0]
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
        cluster_ids = sorted(clustered_df['cluster'].unique())
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
        for cid in sorted(results.keys(), key=lambda x: int(str(x)) if str(x).isdigit() else str(x)):
            obj = results[cid]
            guesses = obj.get('phenotype_guesses') or []

            group_box = QtWidgets.QGroupBox(f"Cluster {cid} – Select phenotype")
            v = QtWidgets.QVBoxLayout(group_box)
            btn_group = QtWidgets.QButtonGroup(group_box)
            btn_group.setExclusive(True)

            # Create a radio per guess and show rationale
            for idx, g in enumerate(guesses):
                name = str(g.get('name', '')).strip() or 'Unknown'
                rb = QtWidgets.QRadioButton(f"{name}")
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
        system_prompt = (
            "You are assisting with IMC cell type suggestions. Use only the provided marker statistics. "
            "Prefer canonical immune/epithelial/stromal names and give exactly 3 plausible phenotypes per cluster. "
            "Consider the range of values across and within clusters for each marker to help determine if the marker is truly unique to the cluster. "
            "Consider if the z-score and mean value is true expression of the marker or if it is due to noise. "
            "Try to give varied phenotypes, rather than the same phenotype with different names. "
            "Focus on different marker combinations to avoid giving the same phenotype with different names. "
            "If uncertain, return \"Unknown\". Output valid JSON exactly matching the given schema. Do not invent markers. "
            "Return valid JSON only and no prose/explanations."
        )
        schema = {
            "cluster_id": str(cid),
            "phenotype_guesses": [ { "name": "", "rationale": "" } ],
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
        # Build Responses API input structure
        input_msgs = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": system_prompt + " Schema: {\\n  \\\"cluster_id\\\": \\\"string\\\",\\n  \\\"phenotype_guesses\\\": [\\n    { \\\"name\\\": \\\"string\\\", \\\"rationale\\\": \\\"string\\\" }\\n  ],\\n  \\\"key_markers_positive\\\": [\\\"string\\\"],\\n  \\\"key_markers_negative\\\": [\\\"string\\\"],\\n  \\\"notes\\\": \\\"string\\\"\\n}"}
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
        
        client = OpenAI(api_key=api_key, timeout=30.0)
        data = payload.copy()
        # Append repair instruction as another system content block if needed
        input_payload = data.get('input')
        if repair and isinstance(input_payload, list):
            input_payload = input_payload + [{"role": "system", "content": [{"type": "input_text", "text": "Return valid JSON only, no prose."}]}]
        # Debug to console: request meta
        try:
            pass
        except Exception:
            pass
        # Responses API call
        try:
            print(f"[DEBUG] Making OpenAI API call to model: {data.get('model', 'gpt-5')}")
            resp = client.responses.create(
                model=data.get('model', 'gpt-5'),
                max_output_tokens=data.get('max_tokens', 2000),
                input=input_payload,
                reasoning={'effort': 'low'}
            )
            print("[DEBUG] OpenAI API call successful")
        except Exception as e:
            error_msg = str(e)
            print(f"[DEBUG] OpenAI API call failed: {error_msg}")
            
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
        # If response looks empty/minimal, dump a truncated raw view to console for debugging
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
            if not isinstance(obj.get('key_markers_positive', []), list):
                obj['key_markers_positive'] = []
            if not isinstance(obj.get('key_markers_negative', []), list):
                obj['key_markers_negative'] = []
            if 'notes' not in obj:
                obj['notes'] = ""
            return obj
        except Exception:
            return None
