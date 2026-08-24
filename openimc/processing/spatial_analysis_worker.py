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

from openimc.ui.cluster_utils import canonicalize_cluster_id, sort_cluster_values


def _count_cluster_pairs(label_codes: np.ndarray, edge_sources: np.ndarray, edge_targets: np.ndarray, n_clusters: int) -> np.ndarray:
    """Count undirected cluster-cluster edge frequencies for a labeling."""
    edge_codes_a = label_codes[edge_sources]
    edge_codes_b = label_codes[edge_targets]
    pair_low = np.minimum(edge_codes_a, edge_codes_b)
    pair_high = np.maximum(edge_codes_a, edge_codes_b)
    pair_indices = (pair_low * n_clusters + pair_high).astype(np.int64, copy=False)
    counts = np.bincount(pair_indices, minlength=n_clusters * n_clusters)
    return counts.reshape(n_clusters, n_clusters)


def roi_enrichment_worker(args):
    """
    Worker function for computing enrichment for an entire ROI (all cluster pairs).
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        args: Tuple containing:
            - roi_id: ROI identifier
            - roi_df: DataFrame with cells for this ROI
            - roi_edges: DataFrame with edges for this ROI
            - cluster_col: Name of cluster column
            - n_perm: Number of permutations
            - seed: Random seed
            
    Returns:
        List of dictionaries with enrichment statistics for all cluster pairs in this ROI
    """
    roi_id, roi_df, roi_edges, cluster_col, n_perm, seed = args

    if int(n_perm) < 1:
        raise ValueError("n_perm must be at least 1")

    results = []

    if roi_df.empty or roi_edges.empty or cluster_col not in roi_df.columns:
        return results

    roi_cells = roi_df.loc[roi_df[cluster_col].notna(), ['cell_id', cluster_col]].copy()
    roi_cells.loc[:, cluster_col] = roi_cells[cluster_col].map(canonicalize_cluster_id)
    roi_cells = roi_cells.loc[roi_cells[cluster_col].notna()].copy()
    unique_clusters = sort_cluster_values(roi_cells[cluster_col].unique(), canonical=True)
    if len(unique_clusters) < 2:
        return results

    cluster_to_code = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
    cell_ids = roi_cells['cell_id'].astype(int).to_numpy()
    cluster_codes = roi_cells[cluster_col].map(cluster_to_code).to_numpy(dtype=np.int32, copy=False)
    cell_id_to_pos = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}

    edge_pairs = (
        roi_edges[['cell_id_A', 'cell_id_B']]
        .dropna()
        .astype(int)
        .to_numpy()
    )
    edge_sources = []
    edge_targets = []
    for cell_a, cell_b in edge_pairs:
        pos_a = cell_id_to_pos.get(int(cell_a))
        pos_b = cell_id_to_pos.get(int(cell_b))
        if pos_a is None or pos_b is None:
            continue
        edge_sources.append(pos_a)
        edge_targets.append(pos_b)

    if not edge_sources:
        return results

    edge_sources = np.asarray(edge_sources, dtype=np.int32)
    edge_targets = np.asarray(edge_targets, dtype=np.int32)

    n_clusters = len(unique_clusters)
    total_cells = int(cluster_codes.size)
    total_edges = int(edge_sources.size)
    if total_edges == 0:
        return results

    observed_counts = _count_cluster_pairs(cluster_codes, edge_sources, edge_targets, n_clusters)
    cluster_sizes = np.bincount(cluster_codes, minlength=n_clusters)
    pair_rows, pair_cols = np.triu_indices(n_clusters)
    observed_vector = observed_counts[pair_rows, pair_cols]
    permuted_vectors = np.empty((n_perm, len(pair_rows)), dtype=np.int32)

    for perm_idx in range(n_perm):
        shuffled_codes = cluster_codes.copy()
        rng = np.random.RandomState(seed + perm_idx)
        rng.shuffle(shuffled_codes)
        permuted_counts = _count_cluster_pairs(shuffled_codes, edge_sources, edge_targets, n_clusters)
        permuted_vectors[perm_idx] = permuted_counts[pair_rows, pair_cols]

    expected_means = permuted_vectors.mean(axis=0)
    expected_stds = permuted_vectors.std(axis=0)

    denominator = total_cells * (total_cells - 1)
    for pair_idx, (row_idx, col_idx) in enumerate(zip(pair_rows, pair_cols)):
        cluster_a = unique_clusters[row_idx]
        cluster_b = unique_clusters[col_idx]
        observed = int(observed_vector[pair_idx])
        n_a = int(cluster_sizes[row_idx])
        n_b = int(cluster_sizes[col_idx])

        if denominator <= 0:
            expected = 0.0
        elif row_idx == col_idx:
            expected = (n_a * (n_a - 1) / denominator) * total_edges
        else:
            expected = (2 * n_a * n_b / denominator) * total_edges

        expected_mean = float(expected_means[pair_idx])
        expected_std = float(expected_stds[pair_idx])

        if expected_std > 0:
            z_score = float((observed - expected_mean) / expected_std)
            n_extreme = int(np.count_nonzero(
                np.abs(permuted_vectors[:, pair_idx] - expected_mean)
                >= abs(observed - expected_mean)
            ))
            # The observed labeling is one additional member of the null
            # distribution; the correction prevents impossible zero p-values.
            p_value = float((n_extreme + 1) / (n_perm + 1))
        else:
            z_score = 0.0
            p_value = 1.0

        results.append({
            'roi_id': str(roi_id),
            'cluster_A': cluster_a,
            'cluster_B': cluster_b,
            'observed': observed,
            'expected': float(expected),
            'p_value': p_value,
            'z_score': z_score,
            'n_permutations': n_perm
        })

    return results


def permutation_worker(args):
    """
    Worker function for computing permutations for a single cluster pair.
    This function must be at module level to be picklable for multiprocessing.
    Kept for backward compatibility but roi_enrichment_worker is preferred.
    
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
    import os
    roi_edges, roi_df, cluster_col, cluster_a, cluster_b, pair, observed, n_perm, seed = args

    if int(n_perm) < 1:
        raise ValueError("n_perm must be at least 1")
    
    worker_pid = os.getpid()
    
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
        permuted_counts_array = np.asarray(permuted_counts)
        n_extreme = int(np.count_nonzero(
            np.abs(permuted_counts_array - expected_mean) >= abs(observed - expected_mean)
        ))
        p_value = (n_extreme + 1) / (n_perm + 1)
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

    if roi_df.empty or cluster_col not in roi_df.columns:
        return distance_data

    roi_cells = roi_df.loc[
        roi_df[cluster_col].notna(),
        ['cell_id', 'centroid_x', 'centroid_y', cluster_col],
    ].copy()
    roi_cells.loc[:, cluster_col] = roi_cells[cluster_col].map(canonicalize_cluster_id)
    roi_cells = roi_cells.loc[roi_cells[cluster_col].notna()].copy()
    if roi_cells.empty:
        return distance_data

    coords_um = roi_cells[["centroid_x", "centroid_y"]].to_numpy(dtype=float) * float(pixel_size_um)
    cell_ids = roi_cells["cell_id"].astype(int).to_numpy()
    cell_clusters = roi_cells[cluster_col].to_numpy()
    unique_clusters = sort_cluster_values(roi_cells[cluster_col].unique(), canonical=True)

    if len(unique_clusters) == 0:
        return distance_data

    cluster_targets = {}
    for cluster in unique_clusters:
        cluster_indices = np.flatnonzero(cell_clusters == cluster)
        if cluster_indices.size == 0:
            continue
        cluster_targets[cluster] = {
            'indices': cluster_indices,
            'cell_ids': cell_ids[cluster_indices],
            'tree': cKDTree(coords_um[cluster_indices]),
        }

    for target_cluster in unique_clusters:
        target_info = cluster_targets.get(target_cluster)
        if target_info is None:
            continue

        nearest_distances, nearest_indices = target_info['tree'].query(coords_um, k=1)
        nearest_distances = np.asarray(nearest_distances, dtype=float)
        nearest_indices = np.asarray(nearest_indices)
        nearest_cell_ids = target_info['cell_ids'][nearest_indices].astype(np.int64, copy=False)
        valid_mask = np.isfinite(nearest_distances)

        same_cluster_mask = cell_clusters == target_cluster
        if np.any(same_cluster_mask):
            if len(target_info['cell_ids']) < 2:
                valid_mask[same_cluster_mask] = False
            else:
                same_coords = coords_um[same_cluster_mask]
                same_cell_ids = cell_ids[same_cluster_mask]
                same_distances, same_indices = target_info['tree'].query(same_coords, k=2)
                same_distances = np.atleast_2d(same_distances).astype(float, copy=False)
                same_indices = np.atleast_2d(same_indices)
                same_neighbor_ids = target_info['cell_ids'][same_indices].astype(np.int64, copy=False)
                first_is_other = same_neighbor_ids[:, 0] != same_cell_ids
                replacement_col = np.where(first_is_other, 0, 1)
                row_positions = np.arange(len(same_cell_ids))
                same_nearest_distances = same_distances[row_positions, replacement_col]
                same_nearest_cell_ids = same_neighbor_ids[row_positions, replacement_col]
                nearest_distances[same_cluster_mask] = same_nearest_distances
                nearest_cell_ids[same_cluster_mask] = same_nearest_cell_ids
                valid_mask[same_cluster_mask] = (
                    np.isfinite(same_nearest_distances)
                    & (same_nearest_cell_ids != same_cell_ids)
                )

        valid_indices = np.flatnonzero(valid_mask)
        distance_data.extend(
            {
                'roi_id': roi_id,
                'cell_A_id': int(cell_ids[idx]),
                'cell_A_cluster': cell_clusters[idx],
                'nearest_B_cluster': target_cluster,
                'nearest_B_dist_um': float(nearest_distances[idx]),
                'nearest_B_cell_id': int(nearest_cell_ids[idx]),
            }
            for idx in valid_indices
        )

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
    
    # Compute bounding box for isotropic edge correction
    bbox_min = np.min(coords_um, axis=0)  # [x_min, y_min]
    bbox_max = np.max(coords_um, axis=0)  # [x_max, y_max]
    
    # Compute K function for this cluster
    for r in radius_steps:
        # Count points within radius r with isotropic edge correction
        k_sum = 0.0
        for i, point in enumerate(cluster_coords):
            distances = np.sqrt(np.sum((cluster_coords - point)**2, axis=1))
            # Exclude the point itself
            within_radius = np.where((distances <= r) & (distances > 0))[0]
            
            for j_idx in within_radius:
                dist_ij = distances[j_idx]
                # Isotropic edge correction: fraction of circle of radius dist_ij
                # centered at point that lies within the rectangular ROI
                # Uses Ripley's isotropic correction for rectangular windows
                dx_min = point[0] - bbox_min[0]
                dx_max = bbox_max[0] - point[0]
                dy_min = point[1] - bbox_min[1]
                dy_max = bbox_max[1] - point[1]
                
                # Count how many boundary edges are closer than dist_ij
                theta = 2 * np.pi  # full circle
                for d_edge in [dx_min, dx_max, dy_min, dy_max]:
                    if d_edge < dist_ij:
                        # Subtract the arc outside the boundary
                        theta -= 2 * np.arccos(np.clip(d_edge / dist_ij, -1, 1))
                
                # Clamp to avoid division by zero
                weight = max(theta / (2 * np.pi), 0.01)
                k_sum += 1.0 / weight
        
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
