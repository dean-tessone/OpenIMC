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
Worker functions for spatial analysis using multiprocessing.
Can be used by both CLI and GUI.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def permutation_worker(args):
    """
    Worker function for computing permutations for a single cluster pair.
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing:
            - roi_edges: DataFrame with edges for this ROI
            - roi_df: DataFrame with cells for this ROI
            - cluster_col: Name of cluster column
            - cluster_a: First cluster ID
            - cluster_b: Second cluster ID
            - pair: Tuple of (cluster_a, cluster_b) in sorted order
            - observed: Observed edge count for this pair
            - n_perm: Number of permutations
            - seed: Random seed
            
    Returns:
        Dictionary with enrichment statistics for this cluster pair
    """
    roi_edges, roi_df, cluster_col, cluster_a, cluster_b, pair, observed, n_perm, seed = args
    
    # Convert roi_edges to list of tuples for faster iteration
    edge_list = [(int(row['cell_id_A']), int(row['cell_id_B'])) 
                 for _, row in roi_edges.iterrows()]
    
    # Get cluster values as array for shuffling
    cluster_values = roi_df[cluster_col].values.copy()
    cell_ids = roi_df['cell_id'].values
    
    permuted_counts = []
    for perm_idx in range(n_perm):
        # Use a different seed for each permutation to ensure reproducibility
        np.random.seed(seed + perm_idx)
        # Shuffle cluster labels
        shuffled_clusters = cluster_values.copy()
        np.random.shuffle(shuffled_clusters)
        
        # Create temporary mapping
        temp_cell_to_cluster = dict(zip(cell_ids, shuffled_clusters))
        
        # Count edges for this permutation
        perm_count = 0
        for cell_a, cell_b in edge_list:
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
    
    return {
        'cluster_A': cluster_a,
        'cluster_B': cluster_b,
        'observed_edges': observed,
        'expected_mean': expected_mean,
        'expected_std': expected_std,
        'z_score': z_score,
        'p_value': p_value,
        'n_permutations': n_perm
    }


def distance_distribution_worker(args):
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


def neighborhood_composition_worker(args):
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


def ripley_worker(args):
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
            count = np.sum(within_radius)
            
            # Simple edge correction: if point is near boundary, weight by area
            # For now, use simple correction (can be improved)
            k_sum += count
        
        # K(r) = (1 / lambda) * average count
        k_value = (k_sum / n_points) / lambda_density if lambda_density > 0 else 0
        
        # L(r) = sqrt(K(r) / pi) - r
        l_value = np.sqrt(k_value / np.pi) - r
        
        ripley_data.append({
            'roi_id': roi_id,
            'cluster': cluster,
            'radius_um': r,
            'k_value': k_value,
            'l_value': l_value,
            'n_points': n_points
        })
    
    return ripley_data

