"""
Command-line interface for OpenIMC batch processing.

This module provides CLI commands for HPC/batch processing without the GUI.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

# Import processing modules
from openimc.data.mcd_loader import MCDLoader
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.processing.export_worker import process_channel_for_export
from openimc.processing.feature_worker import extract_features_for_acquisition, _apply_denoise_to_channel
from openimc.processing.watershed_worker import watershed_segmentation
from openimc.ui.utils import arcsinh_normalize, percentile_clip_normalize, combine_channels

# Try to import Cellpose (optional)
try:
    from cellpose import models
    _HAVE_CELLPOSE = True
except ImportError:
    _HAVE_CELLPOSE = False


def load_data(input_path: str):
    """Load data from MCD file or OME-TIFF directory."""
    input_path = Path(input_path)
    
    if input_path.is_file() and input_path.suffix.lower() in ['.mcd', '.mcdx']:
        # Load MCD file
        loader = MCDLoader()
        loader.open(str(input_path))
        return loader, 'mcd'
    elif input_path.is_dir():
        # Load OME-TIFF directory
        loader = OMETIFFLoader()
        loader.open(str(input_path))
        return loader, 'ometiff'
    else:
        raise ValueError(f"Input path must be an MCD file or directory containing OME-TIFF files: {input_path}")


def parse_denoise_settings(denoise_json: Optional[str]) -> Dict:
    """Parse denoise settings from JSON string or file."""
    if not denoise_json:
        return {}
    
    # Check if it's a file path
    if os.path.isfile(denoise_json):
        with open(denoise_json, 'r') as f:
            return json.load(f)
    
    # Try to parse as JSON string
    try:
        return json.loads(denoise_json)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON for denoise settings: {denoise_json}")


def preprocess_command(args):
    """Preprocess images: denoising and arcsinh scaling, export to OME-TIFF."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_data(args.input)
    
    try:
        acquisitions = loader.list_acquisitions()
        print(f"Found {len(acquisitions)} acquisition(s)")
        
        # Parse denoise settings
        denoise_settings = parse_denoise_settings(args.denoise_settings) if args.denoise_settings else {}
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            # Get all channels
            channels = loader.get_channels(acq.id)
            img_stack = loader.get_all_channels(acq.id)
            
            print(f"  Image shape: {img_stack.shape}, Channels: {len(channels)}")
            
            # Process each channel
            processed_channels = []
            for i, channel_name in enumerate(channels):
                channel_img = img_stack[..., i] if img_stack.ndim == 3 else img_stack
                
                # Apply denoising if configured
                denoise_source = "custom" if channel_name in denoise_settings else "none"
                channel_denoise = denoise_settings.get(channel_name, {})
                
                # Process channel
                processed = process_channel_for_export(
                    channel_img, channel_name, denoise_source,
                    {channel_name: channel_denoise} if channel_denoise else {},
                    "arcsinh" if args.arcsinh else "None",
                    args.arcsinh_cofactor if args.arcsinh else 10.0,
                    (1.0, 99.0),
                    None  # viewer_denoise_func not used in CLI
                )
                
                processed_channels.append(processed)
            
            # Stack channels
            processed_stack = np.stack(processed_channels, axis=-1)
            
            # Save as OME-TIFF
            output_filename = f"{acq.name}.ome.tif"
            if acq.well:
                output_filename = f"{acq.name}_well_{acq.well}.ome.tif"
            output_path = output_dir / output_filename
            
            # Create OME metadata
            metadata = {
                'axes': 'YXS' if processed_stack.ndim == 3 else 'YX',
                'Channel': {'Name': channels}
            }
            
            print(f"  Saving to: {output_path}")
            tifffile.imwrite(
                str(output_path),
                processed_stack,
                metadata=metadata,
                ome=True
            )
        
        print(f"\n✓ Preprocessing complete! Output saved to: {output_dir}")
        
    finally:
        loader.close()


def segment_command(args):
    """Segment cells using Cellpose or watershed method."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_data(args.input)
    
    try:
        acquisitions = loader.list_acquisitions()
        
        # Get acquisition (use first if not specified)
        if args.acquisition:
            acq = next((a for a in acquisitions if a.id == args.acquisition or a.name == args.acquisition), None)
            if not acq:
                raise ValueError(f"Acquisition '{args.acquisition}' not found")
            acquisitions = [acq]
        
        # Parse denoise settings
        denoise_settings = parse_denoise_settings(args.denoise_settings) if args.denoise_settings else {}
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            channels = loader.get_channels(acq.id)
            img_stack = loader.get_all_channels(acq.id)
            
            # Parse channel lists
            nuclear_channels = args.nuclear_channels.split(',') if args.nuclear_channels else []
            nuclear_channels = [ch.strip() for ch in nuclear_channels]
            cyto_channels = args.cytoplasm_channels.split(',') if args.cytoplasm_channels else []
            cyto_channels = [ch.strip() for ch in cyto_channels]
            
            # Parse weights if provided
            nuclear_weights = None
            if args.nuclear_weights:
                try:
                    nuclear_weights = [float(w.strip()) for w in args.nuclear_weights.split(',')]
                except ValueError:
                    raise ValueError(f"Invalid nuclear weights format: {args.nuclear_weights}")
            
            cyto_weights = None
            if args.cyto_weights:
                try:
                    cyto_weights = [float(w.strip()) for w in args.cyto_weights.split(',')]
                except ValueError:
                    raise ValueError(f"Invalid cyto weights format: {args.cyto_weights}")
            
            # Validate channels
            missing_nuclear = [ch for ch in nuclear_channels if ch not in channels]
            missing_cyto = [ch for ch in cyto_channels if ch not in channels]
            if missing_nuclear:
                raise ValueError(f"Nuclear channels not found: {missing_nuclear}")
            if missing_cyto and args.method != 'watershed':
                raise ValueError(f"Cytoplasm channels not found: {missing_cyto}")
            
            # Run segmentation
            if args.method == 'cellpose':
                if not _HAVE_CELLPOSE:
                    raise ImportError("Cellpose not installed. Install with: pip install cellpose")
                
                # Preprocess channels exactly like GUI: load individually, denoise, normalize, then combine
                # Build preprocessing config
                preprocessing_config = {
                    'nuclear_channels': nuclear_channels,
                    'cyto_channels': cyto_channels,
                    'nuclear_combo_method': args.nuclear_fusion_method,
                    'cyto_combo_method': args.cyto_fusion_method,
                    'nuclear_weights': nuclear_weights,
                    'cyto_weights': cyto_weights,
                    'normalization_method': 'arcsinh' if args.arcsinh else 'None',
                    'arcsinh_cofactor': args.arcsinh_cofactor if args.arcsinh else 10.0,
                    'percentile_params': (1.0, 99.0)
                }
                
                # Load and preprocess nuclear channels (exactly like GUI _preprocess_channels_for_segmentation)
                nuclear_imgs = []
                for channel in nuclear_channels:
                    img = loader.get_image(acq.id, channel)
                    # Apply denoising if custom settings provided
                    if denoise_settings and channel in denoise_settings:
                        img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
                    # Apply normalization if configured
                    if preprocessing_config['normalization_method'] == 'arcsinh':
                        img = arcsinh_normalize(img, cofactor=preprocessing_config['arcsinh_cofactor'])
                    elif preprocessing_config['normalization_method'] == 'percentile_clip':
                        p_low, p_high = preprocessing_config['percentile_params']
                        img = percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
                    nuclear_imgs.append(img)
                
                # Combine nuclear channels
                nuclear_combo_method = preprocessing_config['nuclear_combo_method']
                nuclear_weights_list = preprocessing_config['nuclear_weights']
                nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights_list)
                
                # Load and preprocess cytoplasm channels
                cyto_img = None
                if cyto_channels:
                    cyto_imgs = []
                    for channel in cyto_channels:
                        img = loader.get_image(acq.id, channel)
                        # Apply denoising if custom settings provided
                        if denoise_settings and channel in denoise_settings:
                            img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
                        # Apply normalization if configured
                        if preprocessing_config['normalization_method'] == 'arcsinh':
                            img = arcsinh_normalize(img, cofactor=preprocessing_config['arcsinh_cofactor'])
                        elif preprocessing_config['normalization_method'] == 'percentile_clip':
                            p_low, p_high = preprocessing_config['percentile_params']
                            img = percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
                        cyto_imgs.append(img)
                    
                    # Combine cytoplasm channels
                    cyto_combo_method = preprocessing_config['cyto_combo_method']
                    cyto_weights_list = preprocessing_config['cyto_weights']
                    cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights_list)
                
                # Prepare input images for Cellpose
                if args.model == 'nuclei':
                    # For nuclei model, use only nuclear channel
                    images = [nuclear_img]
                    channels_cp = [0, 0]  # [cytoplasm, nucleus] - both are nuclear channel
                else:  # cyto3
                    # For cyto3 model, use both channels
                    if cyto_img is None:
                        cyto_img = nuclear_img  # Fallback to nuclear channel
                    images = [cyto_img, nuclear_img]
                    channels_cp = [0, 1]  # [cytoplasm, nucleus]
                
                # Initialize Cellpose model
                model = models.Cellpose(model_type=args.model, gpu=args.gpu_id is not None, device=args.gpu_id)
                
                print(f"  Running Cellpose segmentation (model: {args.model})...")
                masks, flows, styles, diams = model.eval(
                    images,
                    diameter=args.diameter,
                    flow_threshold=args.flow_threshold,
                    cellprob_threshold=args.cellprob_threshold,
                    channels=channels_cp
                )
                mask = masks[0]
                
            elif args.method == 'watershed':
                print(f"  Running watershed segmentation...")
                mask = watershed_segmentation(
                    img_stack, channels, nuclear_channels, cyto_channels,
                    denoise_settings=denoise_settings if denoise_settings else None,
                    normalization_method="arcsinh" if args.arcsinh else "None",
                    arcsinh_cofactor=args.arcsinh_cofactor if args.arcsinh else 10.0,
                    min_cell_area=args.min_cell_area,
                    max_cell_area=args.max_cell_area,
                    compactness=args.compactness
                )
            else:
                raise ValueError(f"Unknown segmentation method: {args.method}")
            
            # Save mask
            output_filename = f"{acq.name}_segmentation.tif"
            if acq.well:
                output_filename = f"{acq.name}_well_{acq.well}_segmentation.tif"
            output_path = output_dir / output_filename
            
            print(f"  Saving segmentation mask to: {output_path}")
            tifffile.imwrite(str(output_path), mask.astype(np.uint32), compression='lzw')
            
            # Also save as numpy array for easier loading
            np.save(str(output_path).replace('.tif', '.npy'), mask)
            
            print(f"  ✓ Segmentation complete: {np.max(mask)} cells detected")
        
        print(f"\n✓ Segmentation complete! Output saved to: {output_dir}")
        
    finally:
        loader.close()


def extract_features_command(args):
    """Extract features from segmented cells."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_data(args.input)
    
    try:
        acquisitions = loader.list_acquisitions()
        
        # Get acquisition
        if args.acquisition:
            acq = next((a for a in acquisitions if a.id == args.acquisition or a.name == args.acquisition), None)
            if not acq:
                raise ValueError(f"Acquisition '{args.acquisition}' not found")
            acquisitions = [acq]
        
        # Load segmentation mask(s) - can be a directory or single file
        mask_path = Path(args.mask)
        masks_dict = {}
        
        if mask_path.is_dir():
            # Directory of masks - load masks for each acquisition
            print(f"Loading masks from directory: {mask_path}")
            for mask_file in sorted(mask_path.glob('*.tif')) + sorted(mask_path.glob('*.tiff')) + sorted(mask_path.glob('*.npy')):
                # Try to match mask filename to acquisition name
                mask_name = mask_file.stem
                # Try to find matching acquisition
                for acq in acquisitions:
                    if acq.name in mask_name or acq.id in mask_name:
                        if mask_file.suffix == '.npy':
                            masks_dict[acq.id] = np.load(str(mask_file))
                        else:
                            masks_dict[acq.id] = tifffile.imread(str(mask_file))
                        print(f"  Loaded mask for {acq.name}: {mask_file.name}")
                        break
        else:
            # Single mask file - use for all acquisitions
            print(f"Loading mask from: {mask_path}")
            if mask_path.suffix == '.npy':
                mask = np.load(str(mask_path))
            else:
                mask = tifffile.imread(str(mask_path))
            # Use same mask for all acquisitions
            for acq in acquisitions:
                masks_dict[acq.id] = mask
        
        # Parse denoise settings
        denoise_settings = parse_denoise_settings(args.denoise_settings) if args.denoise_settings else {}
        
        # Build feature selection dict
        selected_features = {}
        # If neither specified, use defaults (both True)
        if not args.morphological and not args.intensity:
            args.morphological = True
            args.intensity = True
        
        if args.morphological:
            # Add all morphological features
            selected_features.update({
                'area_um2': True,
                'perimeter_um': True,
                'equivalent_diameter_um': True,
                'eccentricity': True,
                'solidity': True,
                'extent': True,
                'circularity': True,
                'major_axis_len_um': True,
                'minor_axis_len_um': True,
                'aspect_ratio': True,
                'bbox_area_um2': True,
                'touches_border': True,
                'holes_count': True,
                'centroid_x': True,
                'centroid_y': True
            })
        if args.intensity:
            # Add all intensity features
            selected_features.update({
                'mean': True,
                'median': True,
                'std': True,
                'mad': True,
                'p10': True,
                'p90': True,
                'integrated': True,
                'frac_pos': True
            })
        
        all_features = []
        
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            # Get mask for this acquisition
            if acq.id not in masks_dict:
                print(f"  Warning: No mask found for acquisition {acq.name}, skipping")
                continue
            
            mask = masks_dict[acq.id]
            
            channels = loader.get_channels(acq.id)
            img_stack = loader.get_all_channels(acq.id)
            
            # Prepare acquisition info
            acq_info = {
                'channels': channels,
                'channel_metals': acq.channel_metals,
                'channel_labels': acq.channel_labels
            }
            
            # Extract features
            features_df = extract_features_for_acquisition(
                acq.id,
                mask,
                selected_features,
                acq_info,
                acq.name,
                img_stack,
                args.arcsinh,
                args.arcsinh_cofactor if args.arcsinh else 10.0,
                "custom" if denoise_settings else "None",
                denoise_settings,
                acq.source_file
            )
            
            # Add acquisition info
            features_df['acquisition_id'] = acq.id
            features_df['acquisition_name'] = acq.name
            if acq.well:
                features_df['well'] = acq.well
            
            all_features.append(features_df)
        
        # Combine all features
        if len(all_features) > 1:
            combined_features = pd.concat(all_features, ignore_index=True)
        else:
            combined_features = all_features[0]
        
        # Save to CSV
        output_path = Path(args.output)
        print(f"\nSaving features to: {output_path}")
        combined_features.to_csv(output_path, index=False)
        
        print(f"✓ Feature extraction complete! Extracted {len(combined_features)} cells")
        
    finally:
        loader.close()


def cluster_command(args):
    """Perform clustering on feature data."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Select columns for clustering
    if args.columns:
        cluster_columns = [col.strip() for col in args.columns.split(',')]
    else:
        # Auto-detect: exclude non-feature columns (matching GUI)
        exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'cell_id'}
        cluster_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    # Validate columns
    missing = [col for col in cluster_columns if col not in features_df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    
    # Prepare data exactly like GUI _prepare_clustering_data
    data = features_df[cluster_columns].copy()
    
    # Handle missing/infinite values safely (matching GUI)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median(numeric_only=True))
    
    # Apply scaling (matching GUI _apply_scaling)
    if args.scaling == 'zscore':
        # Z-score normalization: (x - mean) / std
        data_means = data.mean()
        data_stds = data.std(ddof=0)
        
        # Handle columns with zero variance or NaN std/mean
        zero_var_cols = (data_stds == 0) | data_stds.isna() | data_means.isna()
        if zero_var_cols.any():
            # Set zero variance/NaN columns to 0 (centered but not scaled)
            data.loc[:, zero_var_cols] = 0
            non_zero_var_cols = ~zero_var_cols
            if non_zero_var_cols.any():
                normalized_data = (data.loc[:, non_zero_var_cols] - data_means[non_zero_var_cols]) / data_stds[non_zero_var_cols]
                data.loc[:, non_zero_var_cols] = normalized_data
        else:
            # Normalize all columns
            data = (data - data_means) / data_stds
    elif args.scaling == 'mad':
        # MAD (Median Absolute Deviation) scaling: (x - median) / MAD
        data_medians = data.median()
        
        # Calculate MAD for each column
        mad_values = {}
        for col in data.columns:
            col_data = data[col].values
            median_val = data_medians[col]
            if pd.isna(median_val):
                mad_values[col] = 0.0
            else:
                mad = np.median(np.abs(col_data - median_val))
                mad_values[col] = 0.0 if pd.isna(mad) else mad
        
        mad_series = pd.Series(mad_values)
        
        # Handle columns with zero MAD or NaN
        zero_mad_cols = (mad_series == 0) | mad_series.isna() | data_medians.isna()
        if zero_mad_cols.any():
            data.loc[:, zero_mad_cols] = 0
            non_zero_mad_cols = ~zero_mad_cols
            if non_zero_mad_cols.any():
                for col in data.columns[non_zero_mad_cols]:
                    data[col] = (data[col] - data_medians[col]) / mad_series[col]
        else:
            for col in data.columns:
                data[col] = (data[col] - data_medians[col]) / mad_series[col]
    
    # Handle any infinities that might have been introduced
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Drop any residual non-finite rows/cols (matching GUI)
    data = data.dropna(axis=0, how='any').dropna(axis=1, how='any')
    
    # Guard: require at least 2 rows and 2 columns
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Insufficient data for clustering. Need at least 2 rows and 2 columns after cleaning.")
    
    # Store original indices to map back
    original_indices = data.index
    data_values = data.values
    
    # Perform clustering
    print(f"Running {args.method} clustering...")
    
    if args.method == 'hierarchical':
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        
        # Calculate distance matrix and linkage (matching GUI)
        distances = pdist(data_values, metric='euclidean')
        linkage_matrix = linkage(distances, method=args.linkage)
        
        # Get cluster labels
        if args.n_clusters is None:
            raise ValueError("--n-clusters is required for hierarchical clustering")
        cluster_labels = fcluster(linkage_matrix, args.n_clusters, criterion='maxclust')
        
    elif args.method == 'leiden':
        import igraph as ig
        from scipy.spatial.distance import pdist
        import leidenalg
        
        # Calculate distance matrix (matching GUI exactly)
        distances = pdist(data_values, metric='euclidean')
        
        # Convert to similarity matrix (invert distances) - matching GUI
        max_dist = np.max(distances)
        similarities = max_dist - distances
        
        # Create graph from similarity matrix (matching GUI exactly)
        n = data_values.shape[0]
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
        
        # Run Leiden clustering (matching GUI)
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=args.resolution,
            seed=42,
        )
        cluster_labels = np.array(partition.membership) + 1  # Start from 1 (matching GUI)
        
    elif args.method == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples
        )
        cluster_labels = clusterer.fit_predict(data_values)
        # HDBSCAN uses -1 for noise, convert to 1-based (matching GUI)
        cluster_labels = cluster_labels + 1  # -1 becomes 0, others become 1-based
    
    else:
        raise ValueError(f"Unknown clustering method: {args.method}")
    
    # Map cluster labels back to original dataframe indices
    # Create a series with cluster labels for the cleaned data
    cluster_series = pd.Series(cluster_labels, index=original_indices)
    
    # Add cluster labels to original dataframe (NaN for rows that were dropped)
    features_df['cluster'] = cluster_series
    # Fill NaN with 0 (noise/unassigned) if needed
    features_df['cluster'] = features_df['cluster'].fillna(0).astype(int)
    
    # Save output
    output_path = Path(args.output)
    print(f"Saving clustered features to: {output_path}")
    features_df.to_csv(output_path, index=False)
    
    print(f"✓ Clustering complete! Found {len(set(cluster_labels))} clusters")
    if -1 in cluster_labels:
        n_noise = sum(cluster_labels == -1)
        print(f"  ({n_noise} cells marked as noise)")


def spatial_command(args):
    """Perform spatial analysis on feature data (matching GUI workflow)."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Check for required columns
    required_cols = ['centroid_x', 'centroid_y']
    missing = [col for col in required_cols if col not in features_df.columns]
    if missing:
        raise ValueError(f"Required columns for spatial analysis: {missing}")
    
    # Get pixel size (default to 1.0 µm if not available)
    pixel_size_um = getattr(args, 'pixel_size_um', 1.0)
    
    # Build spatial graph per ROI if acquisition_id is present (matching GUI)
    if 'acquisition_id' in features_df.columns:
        print("Building spatial graph per ROI (acquisition)...")
        edge_records = []
        
        for roi_id, roi_df in features_df.groupby('acquisition_id'):
            roi_df = roi_df.dropna(subset=["centroid_x", "centroid_y"])
            if roi_df.empty:
                continue
            
            coords_px = roi_df[["centroid_x", "centroid_y"]].values
            cell_ids = roi_df["cell_id"].values if 'cell_id' in roi_df.columns else roi_df.index.values
            
            # Use cKDTree for efficient spatial queries (matching GUI)
            from scipy.spatial import cKDTree
            tree = cKDTree(coords_px)
            
            # Convert radius from pixels to micrometers if needed
            # CLI uses pixels, but we'll store in micrometers to match GUI
            radius_px = args.radius
            
            # Build edges using kNN within radius (matching GUI)
            roi_edges_set = set()
            query_k = min(args.k_neighbors + 1, max(2, len(coords_px)))
            dists, idxs = tree.query(coords_px, k=query_k)
            
            # Handle scalar case
            if np.isscalar(dists):
                dists = np.array([[dists]])
                idxs = np.array([[idxs]])
            elif dists.ndim == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
            
            for i in range(len(coords_px)):
                src_cell_id = int(cell_ids[i])
                for j in range(1, min(dists.shape[1], args.k_neighbors + 1)):
                    nbr_idx = int(idxs[i, j])
                    if nbr_idx < 0 or nbr_idx >= len(coords_px):
                        continue
                    dst_cell_id = int(cell_ids[nbr_idx])
                    dist_px = float(dists[i, j])
                    dist_um = dist_px * pixel_size_um
                    
                    # Only include edges within radius
                    if dist_px <= radius_px:
                        # Create canonical edge (smaller cell_id first)
                        edge_key = (min(src_cell_id, dst_cell_id), max(src_cell_id, dst_cell_id))
                        if edge_key not in roi_edges_set:
                            roi_edges_set.add(edge_key)
                            edge_records.append({
                                'roi_id': str(roi_id),
                                'cell_id_A': src_cell_id,
                                'cell_id_B': dst_cell_id,
                                'distance_um': dist_um
                            })
        
        # Create edges dataframe (matching GUI format)
        edges_df = pd.DataFrame(edge_records)
    else:
        # Single ROI or no ROI grouping - build graph globally
        print("Building spatial graph (single ROI)...")
        coords = features_df[['centroid_x', 'centroid_y']].dropna().values
        
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        
        # Build edges using kNN within radius
        edge_records = []
        edge_set = set()
        query_k = min(args.k_neighbors + 1, max(2, len(coords)))
        dists, idxs = tree.query(coords, k=query_k)
        
        # Handle scalar case
        if np.isscalar(dists):
            dists = np.array([[dists]])
            idxs = np.array([[idxs]])
        elif dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        
        for i in range(len(coords)):
            for j in range(1, min(dists.shape[1], args.k_neighbors + 1)):
                nbr_idx = int(idxs[i, j])
                if nbr_idx < 0 or nbr_idx >= len(coords):
                    continue
                dist_px = float(dists[i, j])
                dist_um = dist_px * pixel_size_um
                
                if dist_px <= args.radius:
                    edge_key = (min(i, nbr_idx), max(i, nbr_idx))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edge_records.append({
                            'source': i,
                            'target': nbr_idx,
                            'distance': dist_px,  # Keep in pixels for compatibility
                            'distance_um': dist_um
                        })
        
        edges_df = pd.DataFrame(edge_records)
    
    # Save edges
    output_path = Path(args.output)
    print(f"Saving spatial graph edges to: {output_path}")
    edges_df.to_csv(output_path, index=False)
    
    print(f"✓ Spatial analysis complete! Found {len(edges_df)} edges")
    
    # Optionally detect communities
    if args.detect_communities:
        print("Detecting spatial communities...")
        import igraph as ig
        
        # Build graph from edges
        # Map cell IDs to indices if needed
        if 'cell_id_A' in edges_df.columns and 'cell_id' in features_df.columns:
            # Create mapping from cell_id to index
            cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(features_df['cell_id'].values)}
            edge_list = []
            weights = []
            for _, e in edges_df.iterrows():
                cell_a = int(e['cell_id_A'])
                cell_b = int(e['cell_id_B'])
                if cell_a in cell_id_to_idx and cell_b in cell_id_to_idx:
                    edge_list.append((cell_id_to_idx[cell_a], cell_id_to_idx[cell_b]))
                    dist_um = e.get('distance_um', e.get('distance', 1.0) * pixel_size_um)
                    weights.append(1.0 / (dist_um + 1e-6))
            g = ig.Graph(len(features_df))
            g.add_edges(edge_list)
            g.es['weight'] = weights
        else:
            # Use index-based edges
            g = ig.Graph()
            g.add_vertices(len(features_df))
            edge_list = []
            weights = []
            for _, e in edges_df.iterrows():
                source = int(e.get('source', e.get('cell_id_A', 0)))
                target = int(e.get('target', e.get('cell_id_B', 0)))
                edge_list.append((source, target))
                dist_um = e.get('distance_um', e.get('distance', 1.0) * pixel_size_um)
                weights.append(1.0 / (dist_um + 1e-6))
            g.add_edges(edge_list)
            g.es['weight'] = weights
        
        # Run community detection
        import leidenalg
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        communities = partition.membership
        
        # Map community labels back to dataframe
        if 'cell_id_A' in edges_df.columns and 'cell_id' in features_df.columns:
            # Map from graph vertex index to cell_id, then to dataframe index
            idx_to_cell_id = {idx: cell_id for idx, cell_id in enumerate(features_df['cell_id'].values)}
            community_series = pd.Series(index=features_df.index, dtype=int)
            for vertex_idx, community in enumerate(communities):
                if vertex_idx < len(features_df):
                    community_series.iloc[vertex_idx] = community
            features_df['spatial_community'] = community_series
        else:
            # Direct mapping (vertex index = dataframe index)
            features_df['spatial_community'] = communities[:len(features_df)]
        
        # Save with communities
        community_output = output_path.parent / (output_path.stem + '_with_communities.csv')
        features_df.to_csv(community_output, index=False)
        print(f"  Saved communities to: {community_output}")


def cluster_figures_command(args):
    """Generate cluster visualization figures."""
    print(f"Loading clustered features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    if 'cluster' not in features_df.columns:
        raise ValueError("Features file must contain 'cluster' column. Run clustering first.")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = args.font_size
    plt.rcParams['figure.dpi'] = args.dpi
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure: UMAP embedding with clusters
    print("Generating UMAP embedding...")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        
        # Select feature columns
        exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'centroid_x', 'centroid_y'}
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        embedding = reducer.fit_transform(features_df[feature_cols].values)
        
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=features_df['cluster'], cmap='tab20', s=10, alpha=0.6)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Cluster Visualization (UMAP)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        
        output_path = output_dir / 'cluster_umap.png'
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    except ImportError:
        print("  Warning: UMAP not available, skipping UMAP plot")
    
    # Create heatmap of cluster means
    print("Generating cluster heatmap...")
    exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'centroid_x', 'centroid_y'}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    cluster_means = features_df.groupby('cluster')[feature_cols].mean()
    
    fig, ax = plt.subplots(figsize=(max(args.width, 12), max(args.height, 8)))
    sns.heatmap(cluster_means.T, annot=False, cmap='viridis', ax=ax, cbar_kws={'label': 'Mean value'})
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Feature')
    ax.set_title('Cluster Mean Feature Values')
    plt.tight_layout()
    
    output_path = output_dir / 'cluster_heatmap.png'
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    print(f"\n✓ Cluster figures saved to: {output_dir}")


def spatial_figures_command(args):
    """Generate spatial analysis visualization figures."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    required_cols = ['centroid_x', 'centroid_y']
    missing = [col for col in required_cols if col not in features_df.columns]
    if missing:
        raise ValueError(f"Required columns for spatial figures: {missing}")
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Set style
    plt.rcParams['font.size'] = args.font_size
    plt.rcParams['figure.dpi'] = args.dpi
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create spatial scatter plot
    print("Generating spatial scatter plot...")
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    
    if 'cluster' in features_df.columns:
        scatter = ax.scatter(
            features_df['centroid_x'], features_df['centroid_y'],
            c=features_df['cluster'], cmap='tab20', s=10, alpha=0.6
        )
        ax.set_title('Spatial Distribution by Cluster')
        plt.colorbar(scatter, ax=ax, label='Cluster')
    else:
        ax.scatter(features_df['centroid_x'], features_df['centroid_y'], s=10, alpha=0.6)
        ax.set_title('Spatial Distribution of Cells')
    
    ax.set_xlabel('X coordinate (pixels)')
    ax.set_ylabel('Y coordinate (pixels)')
    ax.set_aspect('equal')
    plt.tight_layout()
    
    output_path = output_dir / 'spatial_distribution.png'
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    # Create spatial graph visualization if edges file provided
    if args.edges:
        print("Generating spatial graph visualization...")
        edges_df = pd.read_csv(args.edges)
        
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        
        # Plot cells
        if 'cluster' in features_df.columns:
            scatter = ax.scatter(
                features_df['centroid_x'], features_df['centroid_y'],
                c=features_df['cluster'], cmap='tab20', s=20, alpha=0.8, zorder=2
            )
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            ax.scatter(features_df['centroid_x'], features_df['centroid_y'], s=20, alpha=0.8, zorder=2)
        
        # Plot edges
        for _, edge in edges_df.iterrows():
            source_idx = int(edge['source'])
            target_idx = int(edge['target'])
            ax.plot(
                [features_df.loc[source_idx, 'centroid_x'], features_df.loc[target_idx, 'centroid_x']],
                [features_df.loc[source_idx, 'centroid_y'], features_df.loc[target_idx, 'centroid_y']],
                'k-', alpha=0.1, linewidth=0.5, zorder=1
            )
        
        ax.set_xlabel('X coordinate (pixels)')
        ax.set_ylabel('Y coordinate (pixels)')
        ax.set_title('Spatial Graph')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        output_path = output_dir / 'spatial_graph.png'
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    print(f"\n✓ Spatial figures saved to: {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='OpenIMC CLI for batch processing without GUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess images with denoising and arcsinh scaling
  openimc preprocess input.mcd output/ --arcsinh --arcsinh-cofactor 10.0

  # Segment cells using Cellpose (cytoplasm channels optional for cyto3)
  openimc segment input.mcd output/ --method cellpose --nuclear-channels DAPI --model cyto3 --gpu-id 0

  # Segment cells using Watershed (requires both channels)
  openimc segment input.mcd output/ --method watershed --nuclear-channels DNA1 --cytoplasm-channels CK8_CK18

  # Extract features (mask can be directory or single file)
  openimc extract-features input.mcd output/features.csv --mask output/masks/ --morphological --intensity

  # Cluster cells (Leiden uses resolution, not n-clusters)
  openimc cluster features.csv clustered_features.csv --method leiden --resolution 1.0
  openimc cluster features.csv clustered_features.csv --method hierarchical --n-clusters 10
  openimc cluster features.csv clustered_features.csv --method hdbscan --min-cluster-size 10

  # Spatial analysis (--radius is required)
  openimc spatial features.csv edges.csv --radius 50 --k-neighbors 10 --detect-communities

  # Generate figures
  openimc cluster-figures clustered_features.csv output/figures/ --dpi 300 --font-size 12
  openimc spatial-figures features.csv output/figures/ --edges edges.csv --dpi 300
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess images (denoising, arcsinh, export to OME-TIFF)')
    preprocess_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    preprocess_parser.add_argument('output', help='Output directory for processed OME-TIFF files')
    preprocess_parser.add_argument('--arcsinh', action='store_true', help='Apply arcsinh normalization')
    preprocess_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='Arcsinh cofactor (default: 10.0)')
    preprocess_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    preprocess_parser.set_defaults(func=preprocess_command)
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment cells (Cellpose or watershed)')
    segment_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    segment_parser.add_argument('output', help='Output directory for segmentation masks')
    segment_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (uses first if not specified)')
    segment_parser.add_argument('--method', choices=['cellpose', 'watershed'], default='cellpose', help='Segmentation method')
    segment_parser.add_argument('--nuclear-channels', type=str, required=True, help='Comma-separated list of nuclear channel names')
    segment_parser.add_argument('--cytoplasm-channels', type=str, help='Comma-separated list of cytoplasm channel names (for cyto3 model)')
    segment_parser.add_argument('--nuclear-fusion-method', choices=['single', 'mean', 'weighted', 'max', 'pca1'], default='mean', help='Method to combine nuclear channels (default: mean)')
    segment_parser.add_argument('--cyto-fusion-method', choices=['single', 'mean', 'weighted', 'max', 'pca1'], default='mean', help='Method to combine cytoplasm channels (default: mean)')
    segment_parser.add_argument('--nuclear-weights', type=str, help='Comma-separated weights for nuclear channels (e.g., "0.5,0.3,0.2")')
    segment_parser.add_argument('--cyto-weights', type=str, help='Comma-separated weights for cytoplasm channels (e.g., "0.5,0.3,0.2")')
    segment_parser.add_argument('--model', choices=['cyto3', 'nuclei'], default='cyto3', help='Cellpose model type')
    segment_parser.add_argument('--diameter', type=int, help='Cell diameter in pixels (Cellpose)')
    segment_parser.add_argument('--flow-threshold', type=float, default=0.4, help='Flow threshold (Cellpose, default: 0.4)')
    segment_parser.add_argument('--cellprob-threshold', type=float, default=0.0, help='Cell probability threshold (Cellpose, default: 0.0)')
    segment_parser.add_argument('--gpu-id', type=int, help='GPU ID to use (Cellpose)')
    segment_parser.add_argument('--min-cell-area', type=int, default=100, help='Minimum cell area in pixels (watershed, default: 100)')
    segment_parser.add_argument('--max-cell-area', type=int, default=10000, help='Maximum cell area in pixels (watershed, default: 10000)')
    segment_parser.add_argument('--compactness', type=float, default=0.01, help='Watershed compactness (default: 0.01)')
    segment_parser.add_argument('--arcsinh', action='store_true', help='Apply arcsinh normalization before segmentation')
    segment_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='Arcsinh cofactor (default: 10.0)')
    segment_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    segment_parser.set_defaults(func=segment_command)
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract-features', help='Extract features from segmented cells')
    extract_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    extract_parser.add_argument('output', help='Output CSV file path')
    extract_parser.add_argument('--mask', type=str, required=True, help='Path to segmentation mask directory or single mask file (.tif, .tiff, or .npy). If directory, masks are matched to acquisitions by filename.')
    extract_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (uses first if not specified)')
    extract_parser.add_argument('--morphological', action='store_true', help='Extract morphological features')
    extract_parser.add_argument('--intensity', action='store_true', help='Extract intensity features')
    extract_parser.add_argument('--arcsinh', action='store_true', help='Apply arcsinh normalization before feature extraction')
    extract_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='Arcsinh cofactor (default: 10.0)')
    extract_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    extract_parser.set_defaults(func=extract_features_command)
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Perform clustering on feature data')
    cluster_parser.add_argument('features', help='Input CSV file with features')
    cluster_parser.add_argument('output', help='Output CSV file with cluster labels')
    cluster_parser.add_argument('--method', choices=['hierarchical', 'leiden', 'hdbscan'], default='leiden', help='Clustering method')
    cluster_parser.add_argument('--n-clusters', type=int, help='Number of clusters (required for hierarchical, not used for leiden/hdbscan)')
    cluster_parser.add_argument('--columns', type=str, help='Comma-separated list of columns to use for clustering (auto-detect if not specified)')
    cluster_parser.add_argument('--scaling', choices=['none', 'zscore', 'mad'], default='zscore', help='Feature scaling method (zscore or mad, matching GUI)')
    cluster_parser.add_argument('--linkage', choices=['ward', 'complete', 'average'], default='ward', help='Linkage method for hierarchical clustering')
    cluster_parser.add_argument('--resolution', type=float, default=1.0, help='Resolution parameter for Leiden clustering (default: 1.0)')
    cluster_parser.add_argument('--min-cluster-size', type=int, default=10, help='Minimum cluster size (hdbscan, default: 10)')
    cluster_parser.add_argument('--min-samples', type=int, default=5, help='Minimum samples (hdbscan, default: 5)')
    cluster_parser.set_defaults(func=cluster_command)
    
    # Spatial command
    spatial_parser = subparsers.add_parser('spatial', help='Perform spatial analysis on feature data')
    spatial_parser.add_argument('features', help='Input CSV file with features (must contain centroid_x, centroid_y)')
    spatial_parser.add_argument('output', help='Output CSV file with spatial graph edges')
    spatial_parser.add_argument('--radius', type=float, required=True, help='Maximum distance for edges (pixels)')
    spatial_parser.add_argument('--k-neighbors', type=int, default=10, help='k for k-nearest neighbors (default: 10)')
    spatial_parser.add_argument('--pixel-size-um', type=float, default=1.0, help='Pixel size in micrometers (default: 1.0, used for distance_um conversion)')
    spatial_parser.add_argument('--detect-communities', action='store_true', help='Also detect spatial communities')
    spatial_parser.set_defaults(func=spatial_command)
    
    # Cluster figures command
    cluster_figures_parser = subparsers.add_parser('cluster-figures', help='Generate cluster visualization figures')
    cluster_figures_parser.add_argument('features', help='Input CSV file with clustered features')
    cluster_figures_parser.add_argument('output', help='Output directory for figures')
    cluster_figures_parser.add_argument('--dpi', type=int, default=300, help='Figure DPI (default: 300)')
    cluster_figures_parser.add_argument('--font-size', type=float, default=10.0, help='Font size in points (default: 10.0)')
    cluster_figures_parser.add_argument('--width', type=float, default=8.0, help='Figure width in inches (default: 8.0)')
    cluster_figures_parser.add_argument('--height', type=float, default=6.0, help='Figure height in inches (default: 6.0)')
    cluster_figures_parser.set_defaults(func=cluster_figures_command)
    
    # Spatial figures command
    spatial_figures_parser = subparsers.add_parser('spatial-figures', help='Generate spatial analysis visualization figures')
    spatial_figures_parser.add_argument('features', help='Input CSV file with features (must contain centroid_x, centroid_y)')
    spatial_figures_parser.add_argument('output', help='Output directory for figures')
    spatial_figures_parser.add_argument('--edges', type=str, help='Optional CSV file with spatial graph edges')
    spatial_figures_parser.add_argument('--dpi', type=int, default=300, help='Figure DPI (default: 300)')
    spatial_figures_parser.add_argument('--font-size', type=float, default=10.0, help='Font size in points (default: 10.0)')
    spatial_figures_parser.add_argument('--width', type=float, default=8.0, help='Figure width in inches (default: 8.0)')
    spatial_figures_parser.add_argument('--height', type=float, default=6.0, help='Figure height in inches (default: 6.0)')
    spatial_figures_parser.set_defaults(func=spatial_figures_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run the command
    try:
        args.func(args)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

