from typing import List

import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist


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
        
        self._create_ui()
        self._setup_plot()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Cell Clustering Analysis")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 16px; }")
        layout.addWidget(title_label)
        
        # Options panel
        options_group = QtWidgets.QGroupBox("Clustering Options")
        options_layout = QtWidgets.QHBoxLayout(options_group)
        
        # Aggregation method
        options_layout.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.agg_method = QtWidgets.QComboBox()
        self.agg_method.addItems(["mean", "median"])
        self.agg_method.setCurrentText("mean")
        options_layout.addWidget(self.agg_method)
        
        # Include morphometric features
        self.include_morpho = QtWidgets.QCheckBox("Include morphometric features")
        self.include_morpho.setChecked(True)
        options_layout.addWidget(self.include_morpho)
        
        # Number of clusters
        options_layout.addWidget(QtWidgets.QLabel("Number of clusters:"))
        self.n_clusters = QtWidgets.QSpinBox()
        self.n_clusters.setRange(2, 20)
        self.n_clusters.setValue(5)
        options_layout.addWidget(self.n_clusters)
        
        # Clustering method
        options_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.cluster_method = QtWidgets.QComboBox()
        self.cluster_method.addItems(["ward", "complete", "average", "single"])
        self.cluster_method.setCurrentText("ward")
        options_layout.addWidget(self.cluster_method)
        
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
        ax.set_title("Cell Clustering Analysis")
        ax.text(0.5, 0.5, "Click 'Run Clustering' to generate heatmap", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        self.canvas.draw()
        
    def _run_clustering(self):
        """Run the clustering analysis."""
        try:
            # Get options
            agg_method = self.agg_method.currentText()
            include_morpho = self.include_morpho.isChecked()
            n_clusters = self.n_clusters.value()
            cluster_method = self.cluster_method.currentText()
            
            # Prepare data
            data = self._prepare_clustering_data(agg_method, include_morpho)
            
            if data is None or data.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for clustering.")
                return
            
            # Perform clustering
            self.clustered_data, self.cluster_labels = self._perform_clustering(data, n_clusters, cluster_method)
            
            # Create heatmap
            self._create_heatmap()
            
            # Enable buttons
            self.explore_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Clustering Error", f"Error during clustering: {str(e)}")
    
    def _prepare_clustering_data(self, agg_method, include_morpho):
        """Prepare data for clustering."""
        # Get marker columns (intensity features)
        marker_cols = [col for col in self.feature_dataframe.columns 
                      if any(suffix in col for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        
        # Get morphometric columns if requested
        morpho_cols = []
        if include_morpho:
            morpho_cols = [col for col in self.feature_dataframe.columns 
                          if col in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity', 
                                   'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um', 
                                   'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count']]
        
        # Combine all feature columns
        feature_cols = marker_cols + morpho_cols
        
        if not feature_cols:
            return None
        
        # Extract data
        data = self.feature_dataframe[feature_cols].copy()
        
        # Handle missing values
        data = data.fillna(data.median())
        
        # Normalize data
        data = (data - data.mean()) / data.std()
        
        return data
    
    def _perform_clustering(self, data, n_clusters, method):
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
    
    def _create_heatmap(self):
        """Create the heatmap visualization."""
        self.figure.clear()
        
        # Create subplots
        gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.3, wspace=0.3)
        
        # Main heatmap
        ax_heatmap = self.figure.add_subplot(gs[1, 0])
        
        # Prepare data for heatmap (exclude cluster column)
        heatmap_data = self.clustered_data.drop('cluster', axis=1).values
        
        # Create heatmap
        im = ax_heatmap.imshow(heatmap_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Set labels
        ax_heatmap.set_xlabel('Cells')
        ax_heatmap.set_ylabel('Features')
        ax_heatmap.set_title('Cell Clustering Heatmap')
        
        # Add cluster boundaries
        cluster_boundaries = []
        current_cluster = self.clustered_data['cluster'].iloc[0]
        for i, cluster in enumerate(self.clustered_data['cluster']):
            if cluster != current_cluster:
                cluster_boundaries.append(i)
                current_cluster = cluster
        
        # Draw cluster boundaries
        for boundary in cluster_boundaries:
            ax_heatmap.axvline(x=boundary, color='red', linewidth=2, alpha=0.7)
        
        # Colorbar
        cbar = self.figure.colorbar(im, ax=ax_heatmap, shrink=0.8)
        cbar.set_label('Normalized Feature Value')
        
        # Dendrogram
        ax_dendro = self.figure.add_subplot(gs[0, 0])
        distances = pdist(self.clustered_data.drop('cluster', axis=1).values, metric='euclidean')
        linkage_matrix = linkage(distances, method=self.cluster_method.currentText())
        dendrogram(linkage_matrix, ax=ax_dendro, orientation='top', color_threshold=0)
        ax_dendro.set_title('Hierarchical Clustering Dendrogram')
        ax_dendro.set_xlabel('')
        ax_dendro.set_ylabel('Distance')
        
        # Cluster size bar
        ax_cluster = self.figure.add_subplot(gs[1, 1])
        cluster_sizes = self.clustered_data['cluster'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
        ax_cluster.barh(range(len(cluster_sizes)), cluster_sizes.values, color=colors)
        ax_cluster.set_yticks(range(len(cluster_sizes)))
        ax_cluster.set_yticklabels([f'Cluster {i+1}' for i in cluster_sizes.index])
        ax_cluster.set_xlabel('Number of Cells')
        ax_cluster.set_title('Cluster Sizes')
        
        # Feature labels
        ax_features = self.figure.add_subplot(gs[0, 1])
        feature_names = self.clustered_data.drop('cluster', axis=1).columns
        ax_features.text(0.5, 0.5, f'Features ({len(feature_names)}):\n' + 
                        '\n'.join(feature_names[:10]) + ('...' if len(feature_names) > 10 else ''),
                        ha='center', va='center', transform=ax_features.transAxes, fontsize=8)
        ax_features.set_title('Feature List')
        ax_features.axis('off')
        
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
        
        self.channel_combo.addItems(sorted(channels))
    
    def _on_cluster_changed(self):
        """Handle cluster selection change."""
        self.current_cluster = self.cluster_combo.currentData()
        if self.current_cluster:
            self.status_label.setText(f"Selected Cluster {self.current_cluster['cluster_id']} with {self.current_cluster['size']} cells")
    
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
            
            # Limit to first 20 cells for performance
            max_cells = min(20, len(cluster_cells))
            
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
                        # Load channel image
                        try:
                            channel_img = parent_window.loader.get_image(acq_id, channel)
                            
                            # Apply mask to image
                            masked_img = channel_img * cell_mask
                            
                            # Create image widget
                            img_widget = self._create_image_widget(masked_img, f"Cell {cell_id}")
                            grid_layout.addWidget(img_widget, i // 4, i % 4)
                            
                            self.cell_images.append({
                                'cell_id': cell_id,
                                'acquisition_id': acq_id,
                                'image': masked_img
                            })
                            
                        except Exception as e:
                            print(f"Error loading image for cell {cell_id}: {e}")
                            continue
            
            self.scroll_area.setWidget(grid_widget)
            self.status_label.setText(f"Loaded {len(self.cell_images)} cell images for Cluster {self.current_cluster['cluster_id']}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading cell images: {str(e)}")
    
    def _create_image_widget(self, image, title):
        """Create a widget to display a cell image."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Create matplotlib figure
        fig = Figure(figsize=(2, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Display image
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        
        layout.addWidget(canvas)
        return widget
