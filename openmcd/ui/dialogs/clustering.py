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
    def __init__(self, feature_dataframe, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Clustering Analysis")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        self.feature_dataframe = feature_dataframe
        self.cluster_labels = None
        self.clustered_data = None
        self.umap_embedding = None
        self.cluster_annotation_map = {}
        self.gating_rules = []  # list of dict: {name, logic, conditions: [{column, op, threshold}]}
        
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
        self.view_combo.addItems(["Heatmap", "UMAP", "Stacked Bars"])
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        viz_layout.addWidget(self.view_combo)

        # Color-by control (UMAP only)
        viz_layout.addWidget(QtWidgets.QLabel("Color by:"))
        self.color_by_combo = QtWidgets.QComboBox()
        self.color_by_combo.addItem("Cluster")
        self.color_by_combo.currentTextChanged.connect(self._on_color_by_changed)
        viz_layout.addWidget(self.color_by_combo)

        # Group-by for stacked bars (Stacked Bars only)
        viz_layout.addWidget(QtWidgets.QLabel("Group by:"))
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
            
            # Perform clustering
            self.clustered_data, self.cluster_labels = self._perform_clustering(data, n_clusters, cluster_method)
            
            # Default to heatmap view after clustering
            print("Creating heatmap...")
            self._create_heatmap()
            print("Heatmap creation completed")
            
            # Enable buttons
            self.explore_btn.setEnabled(True)
            self.annotate_btn.setEnabled(True)
            self.save_plot_btn.setEnabled(True)
            # If UMAP was previously run, keep that available
            # Otherwise, selecting UMAP will prompt to run

            # Auto-apply annotations if already loaded for these cluster ids
            if self.cluster_annotation_map:
                self._apply_cluster_annotations()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Clustering Error", f"Error during clustering: {str(e)}")
    
    def _list_available_feature_columns(self, include_morpho):
        marker_cols = [col for col in self.feature_dataframe.columns 
                      if any(suffix in col for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
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
        
        # Extract data
        data = self.feature_dataframe[feature_cols].copy()
        
        # Handle missing/infinite values safely
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median(numeric_only=True))
        
        # Normalize data (z-score) and drop any residual non-finite rows/cols
        data = (data - data.mean()) / data.std(ddof=0)
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
        print(f"Creating heatmap, seaborn available: {_HAVE_SEABORN}")
        self.figure.clear()
        
        # Use seaborn if available, otherwise fall back to matplotlib
        if _HAVE_SEABORN:
            print("Using seaborn implementation")
            self._create_seaborn_heatmap()
            return
        
        print("Using matplotlib implementation")
        # Create single heatmap plot - no subplots needed
        ax_heatmap = self.figure.add_subplot(111)

        # Prepare data for heatmap (exclude cluster column)
        feature_cols = [c for c in self.clustered_data.columns if c != 'cluster']
        heatmap_data = self.clustered_data[feature_cols].values

        # No dendrograms - just show the heatmap data as-is

        # Create heatmap
        im = ax_heatmap.imshow(heatmap_data.T, aspect='auto', cmap='viridis', interpolation='nearest')

        # Set labels only; tick order should follow row order in clustered_data
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
        
        
        # (Feature list removed; top-right is optionally used for column dendrogram)
        
        self.canvas.draw()
    
    def _create_seaborn_heatmap(self):
        """Create heatmap using seaborn clustermap with color bars."""
        try:
            print("Starting seaborn heatmap creation")
            # Prepare data for heatmap (exclude cluster column)
            feature_cols = [c for c in self.clustered_data.columns if c != 'cluster']
            heatmap_data = self.clustered_data[feature_cols]
            print(f"Feature columns: {feature_cols}")
            print(f"Heatmap data shape: {heatmap_data.shape}")
            
            # Create cluster color mapping
            unique_clusters = sorted(self.clustered_data['cluster'].unique())
            cluster_colors = sns.color_palette("Set3", len(unique_clusters))
            cluster_color_map = {cluster_id: cluster_colors[i] for i, cluster_id in enumerate(unique_clusters)}
            
            # Create cluster color series for color bar
            cluster_colors_series = self.clustered_data['cluster'].map(cluster_color_map)
            
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
            
            # Create clustermap with appropriate parameters
            g = sns.clustermap(
                heatmap_data.T,  # Transpose for features as rows, cells as columns
                cmap='viridis',
                row_cluster=row_cluster,
                col_cluster=col_cluster,
                method=linkage_method,
                metric='euclidean',
                cbar_kws={'label': 'Normalized Feature Value'},
                figsize=(10, 6),  # Smaller figure size to fit window better
                col_colors=cluster_colors_series  # This creates the color bar
            )
            
            # Labels – let seaborn manage tick order after clustering; just style
            g.ax_heatmap.set_xlabel('Cells')
            g.ax_heatmap.set_ylabel('Features')
            for lbl in g.ax_heatmap.get_yticklabels():
                lbl.set_fontsize(6)
                lbl.set_rotation(0)
            
            # Add cluster legend
            self._add_cluster_legend(g, cluster_color_map)
            
            # Replace the figure with the seaborn figure
            old_figure = self.figure
            self.figure = g.fig
            self.canvas.figure = self.figure
            
            # Close the old figure to free memory
            plt.close(old_figure)
            
            # Force canvas update
            print("Calling canvas.draw()...")
            self.canvas.draw()
            print("Canvas draw completed")
            print("Seaborn heatmap created successfully")
            
        except Exception as e:
            print(f"Error creating seaborn heatmap: {e}")
            print("Falling back to matplotlib implementation")
            # Fall back to matplotlib implementation
            self._create_matplotlib_heatmap()
    
    def _create_matplotlib_heatmap(self):
        """Fallback heatmap using matplotlib (original implementation)."""
        print("Creating matplotlib heatmap")
        # Create subplots - simplified layout without cluster size bar
        gs = self.figure.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.3)
        
        # Main heatmap
        ax_heatmap = self.figure.add_subplot(gs[1])
        
        # Prepare data for heatmap (exclude cluster column)
        feature_cols = [c for c in self.clustered_data.columns if c != 'cluster']
        heatmap_data = self.clustered_data[feature_cols].values

        # No dendrograms - just show the heatmap data as-is
        
        # Create heatmap
        im = ax_heatmap.imshow(heatmap_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Set labels and ticks
        ax_heatmap.set_xlabel('Cells')
        ax_heatmap.set_ylabel('Features')
        print(f"Setting y-ticks for {len(feature_cols)} features")
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
        
        self.canvas.draw()
        print("Matplotlib heatmap created successfully")
    
    def _add_cluster_legend(self, g, cluster_color_map):
        """Add cluster legend to seaborn clustermap."""
        # Create legend
        legend_elements = []
        for cluster_id, color in cluster_color_map.items():
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=f'Cluster {cluster_id}'))
        
        # Add legend to the figure
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
            
            # Perform UMAP
            print(f"Running UMAP on {data.shape[0]} cells with {data.shape[1]} features...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            self.umap_embedding = reducer.fit_transform(data.values)
            # Persist for coloring
            self.umap_index = data.index.to_list()
            self.umap_selected_columns = list(selected_columns)
            self.umap_raw_data = data.copy()
            
            # Create UMAP plot
            self._create_umap_plot()
            
            # Populate color-by options
            self._populate_color_by_options()
            # Enable save button since a plot is shown
            self.save_plot_btn.setEnabled(True)
            
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
        # Enable save if there is content
        self.save_plot_btn.setEnabled(True)

    def _update_viz_controls_visibility(self):
        """Show/hide controls depending on selected view."""
        view = self.view_combo.currentText() if hasattr(self, 'view_combo') else 'Heatmap'
        # Color-by visible only for UMAP
        for i in range(self.color_by_combo.count()):
            pass
        self.color_by_combo.setVisible(view == 'UMAP')
        # Group-by visible only for Stacked Bars
        self.group_by_combo.setVisible(view == 'Stacked Bars')
    
    def _create_umap_plot(self):
        """Create UMAP scatter plot."""
        if self.umap_embedding is None:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
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
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                ax.scatter(self.umap_embedding[mask, 0], self.umap_embedding[mask, 1],
                          c=[cluster_color_map[cluster_id]], label=f'Cluster {cluster_id}',
                          alpha=0.8, s=18, edgecolors='none')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
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
                ax.bar(x, vals, bottom=bottom, color=cluster_color_map[cluster_id], label=f"Cluster {cluster_id}")
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

    def _open_gating_dialog(self):
        """Open gating rules editor and apply on save."""
        # Allow selection among intensity features by default
        marker_cols = [col for col in self.feature_dataframe.columns
                       if any(suffix in col for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        dlg = GatingRulesDialog(self.gating_rules, marker_cols, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.gating_rules = dlg.get_rules()
            self._apply_gating_rules()
            QtWidgets.QMessageBox.information(self, "Gating Applied", "Manual phenotypes assigned using gating rules.")

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
        # Save/Load inside dialog
        tools = QtWidgets.QHBoxLayout()
        load_btn = QtWidgets.QPushButton("Load…")
        save_btn = QtWidgets.QPushButton("Save…")
        tools.addWidget(load_btn)
        tools.addWidget(save_btn)
        tools.addStretch()
        v.addLayout(tools)
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Apply")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)

        def do_load():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Annotations", "", "CSV Files (*.csv)")
            if not path:
                return
            try:
                df = pd.read_csv(path)
                if not {'cluster_id', 'phenotype'}.issubset(df.columns):
                    raise ValueError("CSV must have columns: cluster_id, phenotype")
                for _, row in df.iterrows():
                    try:
                        cid = int(row['cluster_id'])
                    except Exception:
                        continue
                    name = str(row['phenotype']).strip()
                    if cid in editors:
                        editors[cid].setText(name)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load Error", f"Error loading annotations: {str(e)}")

        def do_save():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Annotations", "cluster_annotations.csv", "CSV Files (*.csv)")
            if not path:
                return
            try:
                rows = [(cid, editors[cid].text().strip()) for cid in unique_clusters if editors[cid].text().strip()]
                df = pd.DataFrame(rows, columns=['cluster_id', 'phenotype'])
                df.to_csv(path, index=False)
                QtWidgets.QMessageBox.information(self, "Saved", f"Annotations saved to: {path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving annotations: {str(e)}")

        load_btn.clicked.connect(do_load)
        save_btn.clicked.connect(do_save)

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
            # Map on clustered_data
            self.clustered_data['cluster_phenotype'] = self.clustered_data['cluster'].map(self.cluster_annotation_map).fillna('')
            # Write back to feature_dataframe aligned by index
            aligned = self.feature_dataframe.reindex(self.clustered_data.index)
            if 'cluster_phenotype' not in self.feature_dataframe.columns:
                self.feature_dataframe['cluster_phenotype'] = ''
            self.feature_dataframe.loc[self.clustered_data.index, 'cluster_phenotype'] = self.clustered_data['cluster_phenotype'].values
            # Update color-by options
            self._populate_color_by_options()
            # If currently showing heatmap or umap, refresh
            if hasattr(self, 'umap_embedding') and self.umap_embedding is not None:
                self._create_umap_plot()
            else:
                self._create_heatmap()

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
        explorer = ClusterExplorerDialog(cluster_info, self.feature_dataframe, self.parent())
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

# --------------------------
# Cluster Explorer Dialog
# --------------------------
class ClusterExplorerDialog(QtWidgets.QDialog):
    def __init__(self, cluster_info, feature_dataframe, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Explorer")
        self.setModal(True)
        self.setMinimumSize(1000, 700)
        self.cluster_info = cluster_info
        self.feature_dataframe = feature_dataframe
        self.current_cluster = None
        self.cell_images = []
        
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
            self.cluster_combo.addItem(f"Cluster {info['cluster_id']} ({info['size']} cells)", info)
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
        channels = set()
        for col in self.feature_dataframe.columns:
            if '_mean' in col:
                channel = col.replace('_mean', '')
                channels.add(channel)
        
        # Add channels (RGB mode is controlled by checkbox)
        self.channel_combo.addItems(sorted(channels))
        
        # Populate RGB channel combos
        for combo in [self.rgb_r_combo, self.rgb_g_combo, self.rgb_b_combo]:
            combo.addItems(sorted(channels))
    
    def _on_cluster_changed(self):
        """Handle cluster selection change."""
        self.current_cluster = self.cluster_combo.currentData()
        if self.current_cluster:
            self.status_label.setText(f"Selected Cluster {self.current_cluster['cluster_id']} with {self.current_cluster['size']} cells")
    
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
                for col in self.feature_dataframe.columns:
                    if '_mean' in col:
                        channel = col.replace('_mean', '')
                        channels.add(channel)
                
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
