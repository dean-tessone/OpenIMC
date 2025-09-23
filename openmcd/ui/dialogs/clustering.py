from typing import List

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
        
        self._create_ui()
        self._setup_plot()
        self._on_clustering_type_changed()  # Initialize UI state
        
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
        
        # Number of clusters
        options_layout.addWidget(QtWidgets.QLabel("Number of clusters:"))
        self.n_clusters = QtWidgets.QSpinBox()
        self.n_clusters.setRange(2, 20)
        self.n_clusters.setValue(5)
        options_layout.addWidget(self.n_clusters)
        
        # Clustering method type
        options_layout.addWidget(QtWidgets.QLabel("Clustering Method:"))
        self.clustering_type = QtWidgets.QComboBox()
        clustering_types = ["Hierarchical"]
        if _HAVE_LEIDEN:
            clustering_types.append("Leiden")
        self.clustering_type.addItems(clustering_types)
        self.clustering_type.setCurrentText("Hierarchical")
        self.clustering_type.currentTextChanged.connect(self._on_clustering_type_changed)
        options_layout.addWidget(self.clustering_type)
        
        # Hierarchical method selection (initially visible)
        self.hierarchical_label = QtWidgets.QLabel("Linkage Method:")
        self.hierarchical_method = QtWidgets.QComboBox()
        self.hierarchical_method.addItems(["ward", "complete", "average", "single"])
        self.hierarchical_method.setCurrentText("ward")
        options_layout.addWidget(self.hierarchical_label)
        options_layout.addWidget(self.hierarchical_method)
        
        # Resolution parameter for Leiden (initially hidden)
        self.resolution_label = QtWidgets.QLabel("Resolution:")
        self.resolution_spinbox = QtWidgets.QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.1, 5.0)
        self.resolution_spinbox.setSingleStep(0.1)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setDecimals(1)
        self.resolution_label.setVisible(False)
        self.resolution_spinbox.setVisible(False)
        options_layout.addWidget(self.resolution_label)
        options_layout.addWidget(self.resolution_spinbox)

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
        
        # Plot area
        plot_group = QtWidgets.QGroupBox("Clustering Results")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        
        # Create matplotlib canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.umap_btn = QtWidgets.QPushButton("UMAP Plot")
        self.umap_btn.clicked.connect(self._run_umap)
        self.umap_btn.setEnabled(True)
        button_layout.addWidget(self.umap_btn)
        
        self.heatmap_btn = QtWidgets.QPushButton("Show Heatmap")
        self.heatmap_btn.clicked.connect(self._show_heatmap)
        self.heatmap_btn.setEnabled(False)
        button_layout.addWidget(self.heatmap_btn)
        
        self.explore_btn = QtWidgets.QPushButton("Explore Clusters")
        self.explore_btn.clicked.connect(self._explore_clusters)
        self.explore_btn.setEnabled(False)
        button_layout.addWidget(self.explore_btn)
        
        self.save_btn = QtWidgets.QPushButton("Save Heatmap")
        self.save_btn.clicked.connect(self._save_heatmap)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        plot_layout.addLayout(button_layout)
        layout.addWidget(plot_group)
        
    def _setup_plot(self):
        """Setup the matplotlib plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Click 'Run Clustering' to generate heatmap", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        self.canvas.draw()
        
    def _on_clustering_type_changed(self):
        """Handle clustering type change to show/hide relevant controls."""
        clustering_type = self.clustering_type.currentText()
        is_leiden = clustering_type == "Leiden"
        is_hierarchical = clustering_type == "Hierarchical"
        
        # Show/hide resolution parameter for Leiden
        self.resolution_label.setVisible(is_leiden)
        self.resolution_spinbox.setVisible(is_leiden)
        
        # Show/hide hierarchical method selection
        self.hierarchical_label.setVisible(is_hierarchical)
        self.hierarchical_method.setVisible(is_hierarchical)
        
        # Show/hide dendrogram controls for hierarchical methods
        self.dendro_label.setVisible(is_hierarchical)
        self.dendro_mode.setVisible(is_hierarchical)
        
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
            
            # Create heatmap
            print("Creating heatmap...")
            self._create_heatmap()
            print("Heatmap creation completed")
            
            # Enable buttons
            self.explore_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            # Enable heatmap button if UMAP was run
            if self.umap_embedding is not None:
                self.heatmap_btn.setEnabled(True)
            
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
        resolution = self.resolution_spinbox.value()
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, 
                                           resolution_parameter=resolution, seed=42)
        
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
            
            # Set labels and feature names
            g.ax_heatmap.set_xlabel('Cells')
            g.ax_heatmap.set_ylabel('Features')
            
            # Ensure all feature names are shown with small font
            print(f"Setting y-ticks for {len(feature_cols)} features")
            g.ax_heatmap.set_yticks(np.arange(len(feature_cols)))
            g.ax_heatmap.set_yticklabels(feature_cols, fontsize=6, rotation=0)
            g.ax_heatmap.set_ylim(-0.5, len(feature_cols) - 0.5)
            
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
            
            # Prepare data for UMAP
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
            
            # Create UMAP plot
            self._create_umap_plot()
            
            # Enable heatmap button if clustering was done
            if self.clustered_data is not None:
                self.heatmap_btn.setEnabled(True)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "UMAP Error", f"Error during UMAP analysis: {str(e)}")
    
    def _show_heatmap(self):
        """Switch back to heatmap view."""
        if self.clustered_data is not None:
            self._create_heatmap()
        else:
            QtWidgets.QMessageBox.warning(self, "No Clustering", "Please run clustering first to view the heatmap.")
    
    def _create_umap_plot(self):
        """Create UMAP scatter plot."""
        if self.umap_embedding is None:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Determine coloring
        if self.clustered_data is not None and 'cluster' in self.clustered_data.columns:
            # Color by cluster
            cluster_labels = self.clustered_data['cluster'].values
            unique_clusters = sorted(np.unique(cluster_labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
            cluster_color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
            
            # Create scatter plot with cluster colors
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                ax.scatter(self.umap_embedding[mask, 0], self.umap_embedding[mask, 1], 
                          c=[cluster_color_map[cluster_id]], label=f'Cluster {cluster_id}', 
                          alpha=0.7, s=20)
            
            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # No clustering - use single color
            ax.scatter(self.umap_embedding[:, 0], self.umap_embedding[:, 1], 
                      c='blue', alpha=0.7, s=20)
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Dimensionality Reduction')
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
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
    
    def _save_heatmap(self):
        """Save the heatmap as an image."""
        if self.figure is None:
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Heatmap", "cell_clustering_heatmap.png", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(self, "Success", f"Heatmap saved to: {file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving heatmap: {str(e)}")

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
                                    img_widget = self._create_image_widget(cropped_channel, f"{acq_label} - Cell {cell_id}", is_rgb=False, channel=channel, acq_id=acq_id)
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
