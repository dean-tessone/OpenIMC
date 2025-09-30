from typing import Optional, Dict, Any, Tuple, List

import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from scipy.spatial import cKDTree
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
try:
    from scipy import sparse as sp
    _HAVE_SPARSE = True
except Exception:
    _HAVE_SPARSE = False


class SpatialAnalysisDialog(QtWidgets.QDialog):
    def __init__(self, feature_dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial Analysis")
        self.setMinimumSize(900, 650)

        self.feature_dataframe = feature_dataframe
        self.edge_df: Optional[pd.DataFrame] = None
        self.adj_matrix = None
        self.metadata: Dict[str, Any] = {}
        self.neighborhood_df: Optional[pd.DataFrame] = None
        self.cluster_summary_df: Optional[pd.DataFrame] = None
        self.enrichment_df: Optional[pd.DataFrame] = None
        self.distance_df: Optional[pd.DataFrame] = None
        self.ripley_df: Optional[pd.DataFrame] = None

        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Parameters group
        params_group = QtWidgets.QGroupBox("Spatial Graph Construction")
        params_layout = QtWidgets.QGridLayout(params_group)

        self.graph_mode_combo = QtWidgets.QComboBox()
        self.graph_mode_combo.addItems(["kNN", "Radius"])
        self.graph_mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        self.k_spin = QtWidgets.QSpinBox()
        self.k_spin.setRange(1, 64)
        self.k_spin.setValue(6)
        
        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 500.0)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setValue(20.0)

        params_layout.addWidget(QtWidgets.QLabel("Mode:"), 0, 0)
        params_layout.addWidget(self.graph_mode_combo, 0, 1)
        
        self.k_label = QtWidgets.QLabel("k:")
        params_layout.addWidget(self.k_label, 0, 2)
        params_layout.addWidget(self.k_spin, 0, 3)
        
        self.radius_label = QtWidgets.QLabel("Radius (µm):")
        params_layout.addWidget(self.radius_label, 0, 4)
        params_layout.addWidget(self.radius_spin, 0, 5)
        
        # Initially show kNN controls, hide radius
        self._on_mode_changed()

        layout.addWidget(params_group)

        # Actions
        action_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run Analysis")
        self.export_btn = QtWidgets.QPushButton("Export Results…")
        self.export_btn.setEnabled(False)
        action_row.addWidget(self.run_btn)
        action_row.addWidget(self.export_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        # Results tabs with matplotlib canvases
        self.tabs = QtWidgets.QTabWidget()
        
        # Neighborhood Composition tab
        self.neighborhood_tab = QtWidgets.QWidget()
        neighborhood_layout = QtWidgets.QVBoxLayout(self.neighborhood_tab)
        self.neighborhood_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        neighborhood_layout.addWidget(self.neighborhood_canvas)
        self.tabs.addTab(self.neighborhood_tab, "Neighborhood Composition")
        
        # Pairwise Enrichment tab
        self.enrichment_tab = QtWidgets.QWidget()
        enrichment_layout = QtWidgets.QVBoxLayout(self.enrichment_tab)
        self.enrichment_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        enrichment_layout.addWidget(self.enrichment_canvas)
        self.tabs.addTab(self.enrichment_tab, "Pairwise Enrichment")
        
        # Distance Distributions tab
        self.distance_tab = QtWidgets.QWidget()
        distance_layout = QtWidgets.QVBoxLayout(self.distance_tab)
        self.distance_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        distance_layout.addWidget(self.distance_canvas)
        self.tabs.addTab(self.distance_tab, "Distance Distributions")
        
        # Ripley K/L tab
        self.ripley_tab = QtWidgets.QWidget()
        ripley_layout = QtWidgets.QVBoxLayout(self.ripley_tab)
        self.ripley_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        ripley_layout.addWidget(self.ripley_canvas)
        self.tabs.addTab(self.ripley_tab, "Ripley K/L")
        
        layout.addWidget(self.tabs, 1)

        # Wire signals
        self.run_btn.clicked.connect(self._run_analysis)
        self.export_btn.clicked.connect(self._export_results)

    def _on_mode_changed(self):
        """Handle mode change to show/hide relevant controls."""
        mode = self.graph_mode_combo.currentText()
        is_knn = mode == "kNN"
        
        # Show/hide k controls
        self.k_label.setVisible(is_knn)
        self.k_spin.setVisible(is_knn)
        
        # Show/hide radius controls  
        self.radius_label.setVisible(not is_knn)
        self.radius_spin.setVisible(not is_knn)

    def _run_analysis(self):
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(self, "No Data", "Feature dataframe is empty.")
            return
        required_cols = {"acquisition_id", "cell_id", "centroid_x", "centroid_y"}
        missing = [c for c in required_cols if c not in self.feature_dataframe.columns]
        if missing:
            QtWidgets.QMessageBox.critical(self, "Missing columns", f"Missing required columns: {', '.join(missing)}")
            return

        mode = self.graph_mode_combo.currentText()
        k = int(self.k_spin.value())
        radius_um = float(self.radius_spin.value())

        try:
            edge_records: List[Tuple[str, int, int, float]] = []
            rows = []
            cols = []
            data = []

            parent = self.parent() if hasattr(self, 'parent') else None

            # Process per ROI/acquisition to respect ROI boundaries
            for roi_id, roi_df in self.feature_dataframe.groupby('acquisition_id'):
                roi_df = roi_df.dropna(subset=["centroid_x", "centroid_y"])  # ensure valid coordinates
                if roi_df.empty:
                    continue
                coords_px = roi_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
                cell_ids = roi_df["cell_id"].astype(int).to_numpy()

                # pixel size in µm for this ROI
                pixel_size_um = 1.0
                try:
                    if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                        pixel_size_um = float(parent._get_pixel_size_um(roi_id))  # type: ignore[attr-defined]
                except Exception:
                    pixel_size_um = 1.0

                tree = cKDTree(coords_px)

                if mode == "kNN":
                    # Query k+1 (including self), exclude self idx 0
                    query_k = min(k + 1, max(2, len(coords_px)))
                    dists, idxs = tree.query(coords_px, k=query_k)
                    
                    # Handle scalar case (when only 1 point or k=1)
                    if np.isscalar(dists):
                        dists = np.array([[dists]])
                        idxs = np.array([[idxs]])
                    # Ensure 2D for array case
                    elif dists.ndim == 1:
                        dists = dists[:, None]
                        idxs = idxs[:, None]
                    for i in range(len(coords_px)):
                        src_global = int(cell_ids[i])
                        for j in range(1, min(dists.shape[1], k + 1)):
                            nbr_idx = int(idxs[i, j])
                            if nbr_idx < 0 or nbr_idx >= len(coords_px):
                                continue
                            dst_global = int(cell_ids[nbr_idx])
                            # undirected, canonical order
                            a, b = (src_global, dst_global) if src_global < dst_global else (dst_global, src_global)
                            dist_um = float(dists[i, j]) * pixel_size_um
                            edge_records.append((str(roi_id), a, b, dist_um))
                else:
                    # Radius graph: convert radius µm to pixels
                    radius_px = radius_um / max(pixel_size_um, 1e-12)
                    pairs = tree.query_pairs(r=radius_px)
                    for i, j in pairs:
                        a_id = int(cell_ids[int(i)])
                        b_id = int(cell_ids[int(j)])
                        a, b = (a_id, b_id) if a_id < b_id else (b_id, a_id)
                        dist_um = float(np.linalg.norm(coords_px[int(i)] - coords_px[int(j)])) * pixel_size_um
                        edge_records.append((str(roi_id), a, b, dist_um))

                # Build adjacency for this ROI (optional)
                if _HAVE_SPARSE and len(cell_ids) > 0:
                    id_to_pos = {int(cid): idx for idx, cid in enumerate(cell_ids)}
                    for (r_roi, a, b, _d) in edge_records[-len(edge_records):]:
                        if r_roi != str(roi_id):
                            continue
                        ia = id_to_pos.get(int(a))
                        ib = id_to_pos.get(int(b))
                        if ia is None or ib is None:
                            continue
                        rows.extend([ia, ib])
                        cols.extend([ib, ia])
                        data.extend([1.0, 1.0])

            # Remove duplicate edges across ROIs if any (shouldn't occur due to grouping)
            if edge_records:
                edge_df = pd.DataFrame(edge_records, columns=["roi_id", "cell_id_A", "cell_id_B", "distance_um"])
                edge_df = edge_df.drop_duplicates()
                self.edge_df = edge_df
            else:
                self.edge_df = pd.DataFrame(columns=["roi_id", "cell_id_A", "cell_id_B", "distance_um"])

            if _HAVE_SPARSE and rows:
                # Note: adjacency here is last ROI processed; building a global block-diagonal matrix would require indexing over all ROIs
                self.adj_matrix = sp.coo_matrix((np.array(data), (np.array(rows), np.array(cols))))
            else:
                self.adj_matrix = None

            # Step 2: Neighborhood Composition Analysis
            self._compute_neighborhood_composition()
            
            # Step 3: Pairwise Interaction Enrichment
            self._compute_pairwise_enrichment()
            
            # Step 4: Distance Distribution Analysis
            self._compute_distance_distributions()
            
            # Step 5: Ripley K/L Functions
            self._compute_ripley_functions()

            # Metadata
            self.metadata = {
                "mode": mode,
                "k": k,
                "radius_um": radius_um,
                "num_edges": int(len(self.edge_df)),
            }

            QtWidgets.QMessageBox.information(self, "Spatial Analysis", f"Completed analysis with {len(self.edge_df)} edges.")
            self.export_btn.setEnabled(True)
            
            # Update visualizations
            self._update_neighborhood_plot()
            self._update_enrichment_plot()
            self._update_distance_plot()
            self._update_ripley_plot()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Spatial Graph Error", str(e))

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
            
            # Export adjacency matrix
            if _HAVE_SPARSE and self.adj_matrix is not None:
                from scipy.sparse import save_npz
                save_npz(os.path.join(out_dir, "adjacency.npz"), self.adj_matrix.tocsr())
            
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

    def _compute_neighborhood_composition(self):
        """Compute neighborhood composition for each cell."""
        if self.edge_df is None or self.edge_df.empty:
            return
            
        # Check if cluster information is available
        cluster_col = None
        for col in ['cluster', 'cluster_id', 'cluster_phenotype']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
                
        if cluster_col is None:
            # No clustering data available, create dummy clusters
            self.feature_dataframe = self.feature_dataframe.copy()
            self.feature_dataframe['cluster'] = 0
            cluster_col = 'cluster'
        
        # Get unique clusters
        unique_clusters = sorted(self.feature_dataframe[cluster_col].unique())
        n_clusters = len(unique_clusters)
        
        # Initialize neighborhood composition dataframe
        neighborhood_data = []
        
        # Process each ROI separately
        for roi_id, roi_df in self.feature_dataframe.groupby('acquisition_id'):
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)]
            
            if roi_edges.empty:
                continue
                
            # Create adjacency mapping for this ROI
            cell_to_neighbors = {}
            for _, edge in roi_edges.iterrows():
                cell_a, cell_b = int(edge['cell_id_A']), int(edge['cell_id_B'])
                if cell_a not in cell_to_neighbors:
                    cell_to_neighbors[cell_a] = []
                if cell_b not in cell_to_neighbors:
                    cell_to_neighbors[cell_b] = []
                cell_to_neighbors[cell_a].append(cell_b)
                cell_to_neighbors[cell_b].append(cell_a)
            
            # Compute neighborhood composition for each cell in this ROI
            for _, cell_row in roi_df.iterrows():
                cell_id = int(cell_row['cell_id'])
                cell_cluster = cell_row[cluster_col]
                
                # Initialize composition vector
                composition = {f'frac_cluster_{i}': 0.0 for i in range(n_clusters)}
                
                if cell_id in cell_to_neighbors:
                    neighbors = cell_to_neighbors[cell_id]
                    if neighbors:
                        # Get neighbor cluster counts
                        neighbor_clusters = []
                        for neighbor_id in neighbors:
                            neighbor_row = roi_df[roi_df['cell_id'] == neighbor_id]
                            if not neighbor_row.empty:
                                neighbor_cluster = neighbor_row.iloc[0][cluster_col]
                                neighbor_clusters.append(neighbor_cluster)
                        
                        # Calculate fractions
                        total_neighbors = len(neighbor_clusters)
                        for cluster in unique_clusters:
                            cluster_count = neighbor_clusters.count(cluster)
                            cluster_idx = unique_clusters.index(cluster)
                            composition[f'frac_cluster_{cluster_idx}'] = cluster_count / total_neighbors
                
                # Add cell information
                row_data = {
                    'cell_id': cell_id,
                    'roi_id': roi_id,
                    'cluster_id': cell_cluster,
                }
                row_data.update(composition)
                neighborhood_data.append(row_data)
        
        self.neighborhood_df = pd.DataFrame(neighborhood_data)
        
        # Compute per-cluster summary
        if not self.neighborhood_df.empty:
            cluster_summary_data = []
            for cluster in unique_clusters:
                cluster_cells = self.neighborhood_df[self.neighborhood_df['cluster_id'] == cluster]
                if not cluster_cells.empty:
                    summary_row = {'cluster_id': cluster}
                    for i in range(n_clusters):
                        col_name = f'frac_cluster_{i}'
                        if col_name in cluster_cells.columns:
                            summary_row[f'avg_frac_cluster_{i}'] = cluster_cells[col_name].mean()
                    cluster_summary_data.append(summary_row)
            
            self.cluster_summary_df = pd.DataFrame(cluster_summary_data)

    def _update_neighborhood_plot(self):
        """Update the neighborhood composition visualization."""
        if self.neighborhood_df is None or self.neighborhood_df.empty:
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
            cluster_labels = [f'Cluster {int(c)}' for c in unique_clusters]
            neighbor_labels = [f'Cluster {i}' for i in range(n_clusters)]
            
            # Create composition matrix
            composition_matrix = np.zeros((len(unique_clusters), n_clusters))
            for i, cluster in enumerate(unique_clusters):
                cluster_row = self.cluster_summary_df[self.cluster_summary_df['cluster_id'] == cluster]
                if not cluster_row.empty:
                    for j in range(n_clusters):
                        col_name = f'avg_frac_cluster_{j}'
                        if col_name in cluster_row.columns:
                            composition_matrix[i, j] = cluster_row.iloc[0][col_name]
            
            # Heatmap
            im = ax1.imshow(composition_matrix, cmap='viridis', aspect='auto')
            ax1.set_xticks(range(n_clusters))
            ax1.set_xticklabels(neighbor_labels, rotation=45)
            ax1.set_yticks(range(len(unique_clusters)))
            ax1.set_yticklabels(cluster_labels)
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
            for i, col in enumerate(frac_cols):
                plot_data.append(self.neighborhood_df[col].values)
                plot_labels.append(f'Cluster {i}')
            
            ax2.boxplot(plot_data, labels=plot_labels)
            ax2.set_xlabel('Neighbor Cluster')
            ax2.set_ylabel('Fraction')
            ax2.set_title('Distribution of Neighbor Fractions')
            ax2.tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        self.neighborhood_canvas.draw()

    def _compute_pairwise_enrichment(self):
        """Compute pairwise interaction enrichment analysis."""
        if self.edge_df is None or self.edge_df.empty:
            return
            
        # Check if cluster information is available
        cluster_col = None
        for col in ['cluster', 'cluster_id', 'cluster_phenotype']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
                
        if cluster_col is None:
            return  # Skip if no clustering data
        
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
            
            # Count cells per cluster
            cluster_counts = roi_df[cluster_col].value_counts().to_dict()
            total_cells = len(roi_df)
            
            # Count observed edges between cluster pairs
            observed_edges = {}
            for _, edge in roi_edges.iterrows():
                cell_a, cell_b = int(edge['cell_id_A']), int(edge['cell_id_B'])
                
                # Get cluster assignments
                cluster_a = None
                cluster_b = None
                
                cell_a_row = roi_df[roi_df['cell_id'] == cell_a]
                cell_b_row = roi_df[roi_df['cell_id'] == cell_b]
                
                if not cell_a_row.empty:
                    cluster_a = cell_a_row.iloc[0][cluster_col]
                if not cell_b_row.empty:
                    cluster_b = cell_b_row.iloc[0][cluster_col]
                
                if cluster_a is not None and cluster_b is not None:
                    # Create canonical pair (smaller cluster first)
                    pair = tuple(sorted([cluster_a, cluster_b]))
                    observed_edges[pair] = observed_edges.get(pair, 0) + 1
            
            # Calculate expected edges and enrichment for each cluster pair
            for i, cluster_a in enumerate(unique_clusters):
                for j, cluster_b in enumerate(unique_clusters):
                    if j < i:  # Avoid duplicates
                        continue
                        
                    pair = (cluster_a, cluster_b)
                    observed = observed_edges.get(pair, 0)
                    
                    # Calculate expected edges under random distribution
                    if cluster_a == cluster_b:
                        # Within-cluster edges
                        n_a = cluster_counts[cluster_a]
                        expected = (n_a * (n_a - 1)) / 2  # All possible pairs within cluster
                    else:
                        # Between-cluster edges
                        n_a = cluster_counts[cluster_a]
                        n_b = cluster_counts[cluster_b]
                        expected = n_a * n_b  # All possible pairs between clusters
                    
                    # Calculate Z-score (simplified - assumes Poisson distribution)
                    if expected > 0:
                        z_score = (observed - expected) / np.sqrt(expected)
                        # Simple p-value approximation (two-tailed)
                        p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
                    else:
                        z_score = 0.0
                        p_value = 1.0
                    
                    enrichment_data.append({
                        'roi_id': roi_id,
                        'cluster_A': cluster_a,
                        'cluster_B': cluster_b,
                        'observed_edges': observed,
                        'expected_mean': expected,
                        'z_score': z_score,
                        'p_value': p_value
                    })
        
        self.enrichment_df = pd.DataFrame(enrichment_data)

    def _normal_cdf(self, x):
        """Approximate normal CDF using error function."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))

    def _update_enrichment_plot(self):
        """Update the pairwise enrichment visualization."""
        if self.enrichment_df is None or self.enrichment_df.empty:
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
        
        # Plot 1: Z-score heatmap
        im1 = ax1.imshow(enrichment_matrix, cmap='RdBu_r', aspect='auto', 
                        vmin=-3, vmax=3)
        ax1.set_xticks(range(n_clusters))
        ax1.set_xticklabels([f'Cluster {int(c)}' for c in all_clusters], rotation=45)
        ax1.set_yticks(range(n_clusters))
        ax1.set_yticklabels([f'Cluster {int(c)}' for c in all_clusters])
        ax1.set_xlabel('Cluster B')
        ax1.set_ylabel('Cluster A')
        ax1.set_title('Pairwise Interaction Enrichment (Z-scores)')
        
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
        """Compute distance distribution analysis for nearest neighbors."""
        if self.edge_df is None or self.edge_df.empty:
            return
            
        # Check if cluster information is available
        cluster_col = None
        for col in ['cluster', 'cluster_id', 'cluster_phenotype']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
                
        if cluster_col is None:
            return  # Skip if no clustering data
        
        distance_data = []
        
        # Process each ROI separately
        for roi_id, roi_df in self.feature_dataframe.groupby('acquisition_id'):
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)]
            
            if roi_edges.empty:
                continue
                
            # Get unique clusters in this ROI
            unique_clusters = sorted(roi_df[cluster_col].unique())
            
            # For each cell, find nearest neighbor of each cluster type
            for _, cell_row in roi_df.iterrows():
                cell_id = int(cell_row['cell_id'])
                cell_cluster = cell_row[cluster_col]
                
                # Get all edges for this cell
                cell_edges = roi_edges[
                    (roi_edges['cell_id_A'] == cell_id) | 
                    (roi_edges['cell_id_B'] == cell_id)
                ]
                
                if cell_edges.empty:
                    continue
                
                # Find nearest neighbor for each cluster type
                for target_cluster in unique_clusters:
                    min_distance = float('inf')
                    nearest_cell_id = None
                    
                    for _, edge in cell_edges.iterrows():
                        # Determine which cell is the neighbor
                        if edge['cell_id_A'] == cell_id:
                            neighbor_id = int(edge['cell_id_B'])
                        else:
                            neighbor_id = int(edge['cell_id_A'])
                        
                        # Check if neighbor belongs to target cluster
                        neighbor_row = roi_df[roi_df['cell_id'] == neighbor_id]
                        if not neighbor_row.empty:
                            neighbor_cluster = neighbor_row.iloc[0][cluster_col]
                            if neighbor_cluster == target_cluster:
                                distance = edge['distance_um']
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_cell_id = neighbor_id
                    
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
        
        # Plot 2: Violin plot by cluster pair
        if len(all_clusters) >= 2:
            # Create violin plot data
            violin_data = []
            violin_labels = []
            
            for cell_cluster in all_clusters[:4]:  # Limit to first 4 clusters for readability
                for target_cluster in all_clusters[:4]:
                    pair_data = self.distance_df[
                        (self.distance_df['cell_A_cluster'] == cell_cluster) &
                        (self.distance_df['nearest_B_cluster'] == target_cluster)
                    ]['nearest_B_dist_um'].values
                    
                    if len(pair_data) > 0:
                        violin_data.append(pair_data)
                        violin_labels.append(f'{int(cell_cluster)}→{int(target_cluster)}')
            
            if violin_data:
                ax2.violinplot(violin_data, positions=range(len(violin_data)))
                ax2.set_xticks(range(len(violin_labels)))
                ax2.set_xticklabels(violin_labels, rotation=45)
                ax2.set_ylabel('Distance (µm)')
                ax2.set_title('Distance Distributions by Cluster Pair')
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: CDF curves for each cluster
        for cluster in all_clusters[:5]:  # Limit to first 5 clusters
            cluster_distances = self.distance_df[
                self.distance_df['cell_A_cluster'] == cluster
            ]['nearest_B_dist_um'].values
            
            if len(cluster_distances) > 0:
                sorted_distances = np.sort(cluster_distances)
                cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
                ax3.plot(sorted_distances, cumulative, label=f'Cluster {int(cluster)}', linewidth=2)
        
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
            ax4.set_xticklabels(summary_df['ROI'], rotation=45)
            ax4.set_ylabel('Distance (µm)')
            ax4.set_title('Mean Distances by ROI')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.distance_canvas.draw()

    def _compute_ripley_functions(self):
        """Compute Ripley K and L functions for clustering/dispersion analysis."""
        if self.edge_df is None or self.edge_df.empty:
            return
            
        # Check if cluster information is available
        cluster_col = None
        for col in ['cluster', 'cluster_id', 'cluster_phenotype']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
                
        if cluster_col is None:
            return  # Skip if no clustering data
        
        ripley_data = []
        
        # Process each ROI separately
        for roi_id, roi_df in self.feature_dataframe.groupby('acquisition_id'):
            roi_edges = self.edge_df[self.edge_df['roi_id'] == str(roi_id)]
            
            if roi_edges.empty:
                continue
                
            # Get unique clusters in this ROI
            unique_clusters = sorted(roi_df[cluster_col].unique())
            
            # Get pixel size for this ROI
            parent = self.parent() if hasattr(self, 'parent') else None
            pixel_size_um = 1.0
            try:
                if parent is not None and hasattr(parent, '_get_pixel_size_um'):
                    pixel_size_um = float(parent._get_pixel_size_um(roi_id))  # type: ignore[attr-defined]
            except Exception:
                pixel_size_um = 1.0
            
            # Convert coordinates to micrometers
            coords_um = roi_df[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
            
            # Define radius range (in micrometers)
            max_radius = min(np.max(coords_um) - np.min(coords_um)) * 0.4  # 40% of ROI size
            radius_steps = np.linspace(1.0, max_radius, 20)  # 20 radius steps
            
            # Compute Ripley functions for each cluster
            for cluster in unique_clusters:
                cluster_cells = roi_df[roi_df[cluster_col] == cluster]
                if len(cluster_cells) < 2:
                    continue
                    
                cluster_coords = cluster_cells[["centroid_x", "centroid_y"]].to_numpy() * pixel_size_um
                n_points = len(cluster_coords)
                
                # Estimate ROI area (convex hull approximation)
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(coords_um)
                    roi_area = hull.volume  # For 2D, volume is area
                except Exception:
                    # Fallback: bounding box area
                    roi_area = (np.max(coords_um[:, 0]) - np.min(coords_um[:, 0])) * \
                              (np.max(coords_um[:, 1]) - np.min(coords_um[:, 1]))
                
                # Compute K function for this cluster
                for r in radius_steps:
                    # Count points within radius r
                    k_sum = 0
                    for i, point in enumerate(cluster_coords):
                        distances = np.sqrt(np.sum((cluster_coords - point)**2, axis=1))
                        # Exclude the point itself
                        within_radius = np.sum((distances <= r) & (distances > 0))
                        k_sum += within_radius
                    
                    # Normalize by point density
                    if n_points > 1:
                        k_obs = k_sum / (n_points * (n_points - 1))
                    else:
                        k_obs = 0
                    
                    # Expected K under complete spatial randomness (CSR)
                    k_exp = np.pi * r**2 / roi_area
                    
                    # L function
                    l_obs = np.sqrt(k_obs / np.pi) - r
                    l_exp = 0  # Expected L under CSR
                    
                    ripley_data.append({
                        'roi_id': roi_id,
                        'cell_type': cluster,
                        'r_um': r,
                        'K_obs': k_obs,
                        'K_exp': k_exp,
                        'L_obs': l_obs,
                        'L_exp': l_exp
                    })
        
        self.ripley_df = pd.DataFrame(ripley_data)

    def _update_ripley_plot(self):
        """Update the Ripley K/L functions visualization."""
        if self.ripley_df is None or self.ripley_df.empty:
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
                ax1.plot(cluster_data['r_um'], cluster_data['K_obs'], 
                        label=f'Cluster {int(cluster)} (Observed)', linewidth=2)
                ax1.plot(cluster_data['r_um'], cluster_data['K_exp'], 
                        '--', alpha=0.7, label=f'Cluster {int(cluster)} (Expected)')
        
        ax1.set_xlabel('Radius (µm)')
        ax1.set_ylabel('K(r)')
        ax1.set_title('Ripley K Function')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: L function curves
        for cluster in unique_clusters:
            cluster_data = self.ripley_df[self.ripley_df['cell_type'] == cluster]
            if not cluster_data.empty:
                ax2.plot(cluster_data['r_um'], cluster_data['L_obs'], 
                        label=f'Cluster {int(cluster)} (Observed)', linewidth=2)
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



