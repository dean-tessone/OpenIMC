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
import yaml

# Import processing modules
from openimc.data.mcd_loader import MCDLoader
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.processing.export_worker import process_channel_for_export
from openimc.processing.feature_worker import extract_features_for_acquisition, _apply_denoise_to_channel
from openimc.processing.watershed_worker import watershed_segmentation
from openimc.processing.batch_correction import apply_combat_correction, apply_harmony_correction, detect_batch_variable
from openimc.processing.spillover_correction import load_spillover
from openimc.ui.utils import arcsinh_normalize, percentile_clip_normalize, channelwise_minmax_normalize, combine_channels

# Import core functions
from openimc.core import (
    load_mcd,
    parse_denoise_settings as parse_denoise_settings_core,
    preprocess,
    segment,
    extract_features,
    cluster,
    build_spatial_graph,
    batch_correction,
    pixel_correlation,
    qc_analysis,
    spillover_correction,
    generate_spillover_matrix,
    deconvolution,
    spatial_enrichment,
    spatial_distance_distribution,
    build_spatial_graph_anndata,
    spatial_neighborhood_enrichment,
    spatial_cooccurrence,
    spatial_autocorrelation,
    spatial_ripley,
    export_anndata
)

# Try to import Cellpose (optional)
try:
    from cellpose import models
    _HAVE_CELLPOSE = True
except ImportError:
    _HAVE_CELLPOSE = False

# Try to import CellSAM (optional)
try:
    from cellSAM import get_model, cellsam_pipeline
    _HAVE_CELLSAM = True
except ImportError:
    _HAVE_CELLSAM = False


# Wrapper functions for backward compatibility
def load_data(input_path: str, channel_format: str = 'CHW'):
    """Load data from MCD file or OME-TIFF directory.
    
    This is a wrapper around openimc.core.load_mcd for backward compatibility.
    
    Args:
        input_path: Path to MCD file or OME-TIFF directory
        channel_format: Format for OME-TIFF files ('CHW' or 'HWC'), default is 'CHW'
    """
    return load_mcd(input_path, channel_format)


def parse_denoise_settings(denoise_json: Optional[str]) -> Dict:
    """Parse denoise settings from JSON string or file.
    
    This is a wrapper around openimc.core.parse_denoise_settings for backward compatibility.
    """
    return parse_denoise_settings_core(denoise_json)


def preprocess_command(args):
    """Preprocess images: denoising and export to OME-TIFF.
    
    Note: arcsinh normalization is not applied to exported images.
    Only denoising is applied. Arcsinh transform should be applied on extracted intensity features.
    """
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_mcd(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
    try:
        acquisitions = loader.list_acquisitions()
        print(f"Found {len(acquisitions)} acquisition(s)")
        
        # Parse denoise settings
        denoise_settings = parse_denoise_settings(args.denoise_settings) if args.denoise_settings else {}
        
        # Create output directory
        output_dir = Path(args.output)
        
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            # Get image shape for info
            channels = loader.get_channels(acq.id)
            img_stack = loader.get_all_channels(acq.id)
            print(f"  Image shape: {img_stack.shape}, Channels: {len(channels)}")
            
            # Use core preprocess function
            output_path = preprocess(
                loader=loader,
                acquisition=acq,
                output_dir=output_dir,
                denoise_settings=denoise_settings,
                normalization_method="None",  # No normalization applied to exported images
                arcsinh_cofactor=10.0,  # Unused but kept for function signature
                percentile_params=(1.0, 99.0),
                viewer_denoise_func=None  # Not used in CLI
            )
            
            print(f"  Saving to: {output_path}")
        
        print(f"\n✓ Preprocessing complete! Output saved to: {output_dir}")
        
    finally:
        loader.close()


def segment_command(args):
    """Segment cells using DeepCell CellSAM, Cellpose, or watershed method."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_mcd(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
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
        
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
                
            # Use core segment function
            mask = segment(
                loader=loader,
                acquisition=acq,
                method=args.method,
                nuclear_channels=nuclear_channels,
                cyto_channels=cyto_channels if cyto_channels else None,
                output_dir=output_dir,
                denoise_settings=denoise_settings,
                normalization_method='arcsinh' if args.arcsinh else 'None',
                arcsinh_cofactor=args.arcsinh_cofactor if args.arcsinh else 10.0,
                percentile_params=(1.0, 99.0),
                nuclear_combo_method=args.nuclear_fusion_method,
                cyto_combo_method=args.cyto_fusion_method,
                nuclear_weights=nuclear_weights,
                cyto_weights=cyto_weights,
                # Cellpose parameters
                cellpose_model=args.model,
                diameter=args.diameter,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                gpu_id=args.gpu_id,
                # CellSAM parameters
                deepcell_api_key=args.deepcell_api_key,
                bbox_threshold=args.bbox_threshold,
                use_wsi=args.use_wsi,
                low_contrast_enhancement=args.low_contrast_enhancement,
                gauge_cell_size=args.gauge_cell_size,
                # Watershed parameters
                min_cell_area=args.min_cell_area,
                max_cell_area=args.max_cell_area,
                compactness=args.compactness
            )
            
            print(f"  ✓ Segmentation complete: {np.max(mask)} cells detected")
        
        print(f"\n✓ Segmentation complete! Output saved to: {output_dir}")
        
    finally:
        loader.close()


def extract_features_command(args):
    """Extract features from segmented cells."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_mcd(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
    try:
        acquisitions = loader.list_acquisitions()
        
        # Get acquisition
        if args.acquisition:
            acq = next((a for a in acquisitions if a.id == args.acquisition or a.name == args.acquisition), None)
            if not acq:
                raise ValueError(f"Acquisition '{args.acquisition}' not found")
            acquisitions = [acq]
        
        # Parse denoise settings
        denoise_settings = parse_denoise_settings(args.denoise_settings) if args.denoise_settings else {}
        
        # Build feature selection flags
        # If neither specified, use defaults (both True)
        morphological = args.morphological if args.morphological or args.intensity else True
        intensity = args.intensity if args.morphological or args.intensity else True
        
        # Use core extract_features function
        combined_features = extract_features(
            loader=loader,
            acquisitions=acquisitions,
            mask_path=args.mask,
            output_path=args.output,
            morphological=morphological,
            intensity=intensity,
            denoise_settings=denoise_settings,
            arcsinh=args.arcsinh,
            arcsinh_cofactor=args.arcsinh_cofactor if args.arcsinh else 10.0,
            spillover_config=None,  # CLI doesn't support spillover correction yet
            excluded_channels=None  # CLI doesn't support channel exclusion yet
        )
        
        print(f"✓ Feature extraction complete! Extracted {len(combined_features)} cells")
        
    finally:
        loader.close()


def cluster_command(args):
    """Perform clustering on feature data."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Parse columns if provided
    columns = None
    if args.columns:
        columns = [col.strip() for col in args.columns.split(',')]
    
    # Use core cluster function
    result_df = cluster(
        features_df=features_df,
        method=args.method,
        columns=columns,
        scaling=args.scaling,
        output_path=args.output,
        # Hierarchical parameters
        n_clusters=args.n_clusters,
        linkage=args.linkage,
        # Leiden/Louvain parameters
        resolution=args.resolution,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        metric=args.metric,
        # K-means parameters
        n_init=args.n_init,
        # HDBSCAN parameters
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_method=args.cluster_selection_method,
        hdbscan_metric=args.hdbscan_metric
    )
    
    # Count clusters
    unique_clusters = result_df['cluster'].unique()
    n_clusters_found = len([c for c in unique_clusters if c > 0])  # Exclude 0 (noise/unassigned)
    n_noise = (result_df['cluster'] == 0).sum()
    
    print(f"✓ Clustering complete! Found {n_clusters_found} clusters")
    if n_noise > 0:
        print(f"  ({n_noise} cells marked as noise/unassigned)")


def spatial_command(args):
    """Perform spatial analysis on feature data (matching GUI workflow)."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Get pixel size (default to 1.0 µm if not available)
    pixel_size_um = getattr(args, 'pixel_size_um', 1.0)
    
    # Get method from args (defaults to 'kNN' from argparse)
    method = args.method
    
    # Validate radius is provided for Radius method
    if method == 'Radius' and args.radius is None:
        raise ValueError("--radius is required for Radius method")
            
    # Use core build_spatial_graph function
    print(f"Building spatial graph using {method} method...")
    edges_df, features_with_communities = build_spatial_graph(
        features_df=features_df,
        method=method,
        k_neighbors=args.k_neighbors,
        radius=args.radius if method == 'Radius' else None,
        pixel_size_um=pixel_size_um,
        roi_column=None,  # Auto-detect from dataframe
        detect_communities=args.detect_communities,
        community_seed=args.seed,
        output_path=args.output
    )
    
    print(f"✓ Spatial analysis complete! Found {len(edges_df)} edges")
    
    if features_with_communities is not None:
        print(f"  Communities detected and saved")


def batch_correction_command(args):
    """Apply batch correction to feature data."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Use core batch_correction function
    corrected_df = batch_correction(
        features_df=features_df,
        method=args.method,
        batch_var=args.batch_var,
        features=args.columns.split(',') if args.columns else None,
        output_path=args.output,
        covariates=args.covariates.split(',') if args.covariates else None,
        n_clusters=args.n_clusters,
        sigma=args.sigma,
        theta=args.theta,
        lambda_reg=args.lambda_reg,
        max_iter=args.max_iter,
        pca_variance=args.pca_variance
    )
    
    print(f"✓ Batch correction complete! Corrected {len(corrected_df)} cells")


def pixel_correlation_command(args):
    """Compute pixel-level correlations between markers."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_mcd(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
    try:
        acquisitions = loader.list_acquisitions()
        
        # Get acquisition
        if args.acquisition:
            acq = next((a for a in acquisitions if a.id == args.acquisition or a.name == args.acquisition), None)
            if not acq:
                raise ValueError(f"Acquisition '{args.acquisition}' not found")
            acquisitions = [acq]
        else:
            acquisitions = [acquisitions[0]]  # Use first if not specified
        
        # Parse channels
        channels = args.channels.split(',') if args.channels else None
        if channels:
            channels = [ch.strip() for ch in channels]
        else:
            # Get all channels from first acquisition
            channels = loader.get_channels(acquisitions[0].id)
        
        # Load mask if provided
        mask = None
        if args.mask:
            import tifffile
            mask = tifffile.imread(args.mask)
        
        # Compute correlations for each acquisition
        all_results = []
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            results_df = pixel_correlation(
                loader=loader,
                acquisition=acq,
                channels=channels,
                mask=mask,
                multiple_testing_correction=args.multiple_testing_correction
            )
            
            if not results_df.empty:
                results_df['acquisition_id'] = acq.id
                results_df['acquisition_name'] = acq.name
                all_results.append(results_df)
        
        # Combine results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
    
            # Save output
            output_path = Path(args.output)
            combined_results.to_csv(output_path, index=False)
            print(f"\n✓ Pixel correlation complete! Results saved to: {output_path}")
        else:
            print("\n✗ No correlations computed")
    
    finally:
        loader.close()


def qc_analysis_command(args):
    """Perform quality control analysis."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_mcd(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
    try:
        acquisitions = loader.list_acquisitions()
        
        # Get acquisition
        if args.acquisition:
            acq = next((a for a in acquisitions if a.id == args.acquisition or a.name == args.acquisition), None)
            if not acq:
                raise ValueError(f"Acquisition '{args.acquisition}' not found")
            acquisitions = [acq]
        else:
            acquisitions = [acquisitions[0]]  # Use first if not specified
        
        # Parse channels
        channels = args.channels.split(',') if args.channels else None
        if channels:
            channels = [ch.strip() for ch in channels]
        else:
            # Get all channels from first acquisition
            channels = loader.get_channels(acquisitions[0].id)
        
        # Load mask if provided (for cell-level analysis)
        mask = None
        if args.mask:
            import tifffile
            mask = tifffile.imread(args.mask)
        
        # Run QC analysis for each acquisition
        all_results = []
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            results_df = qc_analysis(
                loader=loader,
                acquisition=acq,
                channels=channels,
                mode=args.mode,
                mask=mask
            )
            
            if not results_df.empty:
                all_results.append(results_df)
        
        # Combine results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Save output
            output_path = Path(args.output)
            combined_results.to_csv(output_path, index=False)
            print(f"\n✓ QC analysis complete! Results saved to: {output_path}")
        else:
            print("\n✗ No QC results computed")
    
    finally:
        loader.close()


def spillover_correction_command(args):
    """Apply spillover correction to feature data."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Use core spillover_correction function
    corrected_df = spillover_correction(
        features_df=features_df,
        spillover_matrix=args.spillover_matrix,
        method=args.method,
        arcsinh_cofactor=args.arcsinh_cofactor,
        channel_map=None,  # Could add CLI arg for this
        output_path=args.output
    )
    
    print(f"✓ Spillover correction complete! Corrected {len(corrected_df)} cells")


def generate_spillover_matrix_command(args):
    """Generate spillover matrix from single-stain control MCD file."""
    print(f"Processing MCD file: {args.input}")
            
    # Parse donor mapping if provided
    donor_map = None
    if args.donor_map:
        import json
        if os.path.exists(args.donor_map):
            with open(args.donor_map, 'r') as f:
                donor_map = json.load(f)
        else:
            # Try to parse as JSON string
            donor_map = json.loads(args.donor_map)
    
    # Use core generate_spillover_matrix function
    S_df, qc_df = generate_spillover_matrix(
        mcd_path=args.input,
        donor_label_per_acq=donor_map,
        cap=args.cap,
        aggregate=args.aggregate,
        output_path=args.output
    )
    
    print(f"✓ Spillover matrix generation complete!")
    print(f"  Matrix shape: {S_df.shape}")
    print(f"  Saved to: {args.output}")


def deconvolution_command(args):
    """Apply deconvolution to high resolution IMC images."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_mcd(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
    try:
        acquisitions = loader.list_acquisitions()
        
        # Get acquisition
        if args.acquisition:
            acq = next((a for a in acquisitions if a.id == args.acquisition or a.name == args.acquisition), None)
            if not acq:
                raise ValueError(f"Acquisition '{args.acquisition}' not found")
            acquisitions = [acq]
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply deconvolution to each acquisition
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            output_path = deconvolution(
                loader=loader,
                acquisition=acq,
                output_dir=output_dir,
                x0=args.x0,
                iterations=args.iterations,
                output_format=args.output_format
            )
            
            print(f"  Saved to: {output_path}")
        
        print(f"\n✓ Deconvolution complete! Output saved to: {output_dir}")
    
    finally:
        loader.close()


def spatial_enrichment_command(args):
    """Compute pairwise spatial enrichment between clusters."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    print(f"Loading edges from: {args.edges}")
    edges_df = pd.read_csv(args.edges)
    
    # Use core spatial_enrichment function
    enrichment_df = spatial_enrichment(
        features_df=features_df,
        edges_df=edges_df,
        cluster_column=args.cluster_column,
        n_permutations=args.n_permutations,
        seed=args.seed,
        roi_column=args.roi_column,
        output_path=args.output
    )
    
    print(f"✓ Spatial enrichment complete! Found {len(enrichment_df)} cluster pairs")


def spatial_distance_command(args):
    """Compute distance distributions between clusters."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    print(f"Loading edges from: {args.edges}")
    edges_df = pd.read_csv(args.edges)
    
    # Use core spatial_distance_distribution function
    distance_df = spatial_distance_distribution(
        features_df=features_df,
        edges_df=edges_df,
        cluster_column=args.cluster_column,
        roi_column=args.roi_column,
        output_path=args.output
    )
    
    print(f"✓ Distance distribution complete! Found {len(distance_df)} cluster pairs")


def spatial_anndata_command(args):
    """Build spatial graph using AnnData/squidpy approach."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    pixel_size_um = getattr(args, 'pixel_size_um', 1.0)
    method = args.method
    
    if method == 'Radius' and args.radius is None:
        raise ValueError("--radius is required for Radius method")
    
    print(f"Building spatial graph using {method} method (AnnData/squidpy)...")
    anndata_dict = build_spatial_graph_anndata(
        features_df=features_df,
        method=method,
        k_neighbors=args.k_neighbors,
        radius=args.radius if method == 'Radius' else None,
        pixel_size_um=pixel_size_um,
        roi_column=args.roi_column,
        roi_id=args.roi_id,
        seed=args.seed
    )
    
    print(f"✓ Spatial graph built for {len(anndata_dict)} ROI(s)")
    
    # Export if output path provided
    if args.output:
        export_anndata(anndata_dict, args.output, combined=args.combined)
        print(f"✓ AnnData exported to: {args.output}")


def spatial_nhood_enrichment_command(args):
    """Run neighborhood enrichment analysis."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    pixel_size_um = getattr(args, 'pixel_size_um', 1.0)
    
    # Build graph first
    print("Building spatial graph...")
    anndata_dict = build_spatial_graph_anndata(
        features_df=features_df,
        method=args.method,
        k_neighbors=args.k_neighbors,
        radius=args.radius if args.method == 'Radius' else None,
        pixel_size_um=pixel_size_um,
        roi_column=args.roi_column,
        roi_id=args.roi_id,
        seed=args.seed
    )
    
    if not anndata_dict:
        raise ValueError("Failed to build spatial graph. Check your data and parameters.")
    
    # Run neighborhood enrichment
    print("Running neighborhood enrichment...")
    results = spatial_neighborhood_enrichment(
        anndata_dict=anndata_dict,
        cluster_key=args.cluster_column,
        aggregation=args.aggregation
    )
    
    print(f"✓ Neighborhood enrichment complete for {len(results['results'])} ROI(s)")
    
    # Export if output path provided
    if args.output:
        export_anndata(results['results'], args.output, combined=args.combined)
        print(f"✓ Results exported to: {args.output}")


def spatial_cooccurrence_command(args):
    """Run co-occurrence analysis."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    pixel_size_um = getattr(args, 'pixel_size_um', 1.0)
    
    # Parse interval
    interval = [float(x.strip()) for x in args.interval.split(',')]
    if len(interval) < 2:
        raise ValueError("Co-occurrence requires at least 2 distances in interval")
    
    # Build graph first
    print("Building spatial graph...")
    anndata_dict = build_spatial_graph_anndata(
        features_df=features_df,
        method=args.method,
        k_neighbors=args.k_neighbors,
        radius=args.radius if args.method == 'Radius' else None,
        pixel_size_um=pixel_size_um,
        roi_column=args.roi_column,
        roi_id=args.roi_id,
        seed=args.seed
    )
    
    if not anndata_dict:
        raise ValueError("Failed to build spatial graph. Check your data and parameters.")
    
    # Run co-occurrence
    print("Running co-occurrence analysis...")
    results = spatial_cooccurrence(
        anndata_dict=anndata_dict,
        cluster_key=args.cluster_column,
        interval=interval,
        reference_cluster=args.reference_cluster
    )
    
    print(f"✓ Co-occurrence complete for {len(results)} ROI(s)")
    
    # Export if output path provided
    if args.output:
        export_anndata(results, args.output, combined=args.combined)
        print(f"✓ Results exported to: {args.output}")


def spatial_autocorr_command(args):
    """Run spatial autocorrelation (Moran's I) analysis."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    pixel_size_um = getattr(args, 'pixel_size_um', 1.0)
    
    # Parse markers
    markers = None
    if args.markers and args.markers.lower() != 'all':
        markers = [m.strip() for m in args.markers.split(',')]
    
    # Build graph first
    print("Building spatial graph...")
    anndata_dict = build_spatial_graph_anndata(
        features_df=features_df,
        method=args.method,
        k_neighbors=args.k_neighbors,
        radius=args.radius if args.method == 'Radius' else None,
        pixel_size_um=pixel_size_um,
        roi_column=args.roi_column,
        roi_id=args.roi_id,
        seed=args.seed
    )
    
    if not anndata_dict:
        raise ValueError("Failed to build spatial graph. Check your data and parameters.")
    
    # Run autocorrelation
    print("Running spatial autocorrelation...")
    results = spatial_autocorrelation(
        anndata_dict=anndata_dict,
        markers=markers,
        aggregation=args.aggregation
    )
    
    print(f"✓ Spatial autocorrelation complete for {len(results['results'])} ROI(s)")
    
    # Export if output path provided
    if args.output:
        export_anndata(results['results'], args.output, combined=args.combined)
        print(f"✓ Results exported to: {args.output}")


def spatial_ripley_command(args):
    """Run Ripley function analysis."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    pixel_size_um = getattr(args, 'pixel_size_um', 1.0)
    
    # Build graph first
    print("Building spatial graph...")
    anndata_dict = build_spatial_graph_anndata(
        features_df=features_df,
        method=args.method,
        k_neighbors=args.k_neighbors,
        radius=args.radius if args.method == 'Radius' else None,
        pixel_size_um=pixel_size_um,
        roi_column=args.roi_column,
        roi_id=args.roi_id,
        seed=args.seed
    )
    
    if not anndata_dict:
        raise ValueError("Failed to build spatial graph. Check your data and parameters.")
    
    # Run Ripley
    print(f"Running Ripley {args.mode} function...")
    results = spatial_ripley(
        anndata_dict=anndata_dict,
        cluster_key=args.cluster_column,
        mode=args.mode,
        max_dist=args.max_dist
    )
    
    print(f"✓ Ripley analysis complete for {len(results)} ROI(s)")
    
    # Export if output path provided
    if args.output:
        export_anndata(results, args.output, combined=args.combined)
        print(f"✓ Results exported to: {args.output}")


def export_anndata_command(args):
    """Export AnnData objects from H5AD file(s)."""
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required. Install with: pip install anndata")
    
    # Load AnnData
    if args.input.endswith('.h5ad'):
        # Single file
        adata = ad.read_h5ad(args.input)
        anndata_dict = {'combined': adata}
    else:
        # Directory of files
        input_path = Path(args.input)
        anndata_dict = {}
        for h5ad_file in input_path.glob('*.h5ad'):
            roi_id = h5ad_file.stem.replace('anndata_roi_', '')
            anndata_dict[roi_id] = ad.read_h5ad(str(h5ad_file))
    
    # Export
    export_anndata(anndata_dict, args.output, combined=args.combined)
    print(f"✓ AnnData exported to: {args.output}")


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
        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        
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


def workflow_command(args):
    """Execute a complete workflow from a YAML configuration file.
    
    Supports all OpenIMC functions:
    - preprocessing: Denoising and export to OME-TIFF
    - deconvolution: High resolution deconvolution
    - segmentation: Cell segmentation (CellSAM, Cellpose, Watershed, Ilastik)
    - feature_extraction: Extract features from segmented cells
    - batch_correction: Batch correction (Harmony, ComBat)
    - pixel_correlation: Pixel-level correlation analysis
    - qc_analysis: Quality control analysis
    - clustering: Cell clustering
    - spatial_analysis: Spatial analysis
    
    Each step can specify:
    - enabled: true/false
    - input: path to input file/directory (optional, uses previous step output if not specified)
    - output: path to output file/directory (optional, uses default location if not specified)
    """
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading workflow configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get base paths (optional - steps can specify their own)
    input_path = Path(config.get('input', '.')) if config.get('input') else None
    output_base = Path(config.get('output', 'workflow_output'))
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Track intermediate outputs for chaining steps
    workflow_state = {
        'input': input_path,
        'output_base': output_base,
        'preprocessing_output': None,
        'deconvolution_output': None,
        'segmentation_output': None,
        'features_output': None,
        'batch_corrected_output': None,
    }
    
    # Helper function to get input path for a step
    def get_step_input(step_config, default_input, step_name):
        """Get input path for a step, using config or default."""
        if 'input' in step_config:
            return Path(step_config['input'])
        elif default_input:
            return default_input
        else:
            raise ValueError(f"{step_name} requires 'input' path in config or previous step output")
    
    # Helper function to get output path for a step
    def get_step_output(step_config, default_output, step_name):
        """Get output path for a step, using config or default."""
        if 'output' in step_config:
            return Path(step_config['output'])
        else:
            return default_output
    
    # Step: Preprocessing (should come first)
    if 'preprocessing' in config and config['preprocessing'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: PREPROCESSING")
        print("="*60)
        
        prep_config = config['preprocessing']
        prep_input = get_step_input(prep_config, workflow_state['input'], 'Preprocessing')
        prep_output = get_step_output(prep_config, output_base / 'preprocessed', 'Preprocessing')
        prep_output.mkdir(parents=True, exist_ok=True)
        
        class PreprocessArgs:
            pass
        prep_args = PreprocessArgs()
        prep_args.input = str(prep_input)
        prep_args.output = str(prep_output)
        prep_args.channel_format = prep_config.get('channel_format', config.get('channel_format', 'CHW'))
        prep_args.denoise_settings = prep_config.get('denoise_settings')
        prep_args.arcsinh = prep_config.get('arcsinh', False)
        prep_args.arcsinh_cofactor = prep_config.get('arcsinh_cofactor', 10.0)
        
        preprocess_command(prep_args)
        workflow_state['preprocessing_output'] = prep_output
        workflow_state['input'] = prep_output  # Update input for next steps
    
    # Step: Deconvolution
    if 'deconvolution' in config and config['deconvolution'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: DECONVOLUTION")
        print("="*60)
        
        deconv_config = config['deconvolution']
        deconv_input = get_step_input(deconv_config, workflow_state['input'], 'Deconvolution')
        deconv_output = get_step_output(deconv_config, output_base / 'deconvolved', 'Deconvolution')
        deconv_output.mkdir(parents=True, exist_ok=True)
        
        from openimc.processing.deconvolution_worker import deconvolve_acquisition
        
        # Determine loader type
        loader_type = 'mcd' if str(deconv_input).endswith(('.mcd', '.mcdx')) else 'ometiff'
        
        # Get acquisitions
        loader, _ = load_data(str(deconv_input), channel_format=config.get('channel_format', 'CHW'))
        try:
            acquisitions = loader.list_acquisitions()
            if deconv_config.get('acquisition'):
                acq = next((a for a in acquisitions if a.id == deconv_config['acquisition'] or a.name == deconv_config['acquisition']), None)
                if not acq:
                    raise ValueError(f"Acquisition '{deconv_config['acquisition']}' not found")
                acquisitions = [acq]
            
            for acq in acquisitions:
                print(f"  Deconvolving acquisition: {acq.name} (ID: {acq.id})")
                channels = loader.get_channels(acq.id)
                output_path = deconvolve_acquisition(
                    data_path=str(deconv_input),
                    acq_id=acq.id,
                    output_dir=str(deconv_output),
                    x0=deconv_config.get('x0', 7.0),
                    iterations=deconv_config.get('iterations', 4),
                    output_format=deconv_config.get('output_format', 'float'),
                    channel_names=channels,
                    source_file_path=acq.source_file,
                    unique_acq_id=acq.id,
                    loader_type=loader_type,
                    channel_format=config.get('channel_format', 'CHW'),
                    well_name=acq.well
                )
                print(f"  ✓ Saved: {output_path}")
        finally:
            loader.close()
        
        workflow_state['deconvolution_output'] = deconv_output
        workflow_state['input'] = deconv_output  # Update input for next steps
    
    # Step: Segmentation (if configured)
    if 'segmentation' in config and config['segmentation'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: SEGMENTATION")
        print("="*60)
        
        seg_config = config['segmentation']
        seg_input = get_step_input(seg_config, workflow_state['input'], 'Segmentation')
        seg_output = get_step_output(seg_config, output_base / 'segmentation', 'Segmentation')
        seg_output.mkdir(parents=True, exist_ok=True)
        
        # Create a mock args object for segment_command
        class SegmentArgs:
            pass
        
        seg_args = SegmentArgs()
        seg_args.input = str(seg_input)
        seg_args.output = str(seg_output)
        seg_args.channel_format = seg_config.get('channel_format', config.get('channel_format', 'CHW'))
        seg_args.acquisition = seg_config.get('acquisition')
        seg_args.method = seg_config.get('method', 'cellsam')
        seg_args.nuclear_channels = ','.join(seg_config.get('nuclear_channels', []))
        seg_args.cytoplasm_channels = ','.join(seg_config.get('cytoplasm_channels', [])) if seg_config.get('cytoplasm_channels') else None
        seg_args.nuclear_fusion_method = seg_config.get('nuclear_fusion_method', 'mean')
        seg_args.cyto_fusion_method = seg_config.get('cyto_fusion_method', 'mean')
        seg_args.nuclear_weights = ','.join(map(str, seg_config.get('nuclear_weights', []))) if seg_config.get('nuclear_weights') else None
        seg_args.cyto_weights = ','.join(map(str, seg_config.get('cyto_weights', []))) if seg_config.get('cyto_weights') else None
        seg_args.model = seg_config.get('model', 'cyto3')
        seg_args.diameter = seg_config.get('diameter')
        seg_args.flow_threshold = seg_config.get('flow_threshold', 0.4)
        seg_args.cellprob_threshold = seg_config.get('cellprob_threshold', 0.0)
        seg_args.gpu_id = seg_config.get('gpu_id')
        seg_args.min_cell_area = seg_config.get('min_cell_area', 100)
        seg_args.max_cell_area = seg_config.get('max_cell_area', 10000)
        seg_args.compactness = seg_config.get('compactness', 0.01)
        seg_args.deepcell_api_key = seg_config.get('deepcell_api_key') or os.environ.get("DEEPCELL_ACCESS_TOKEN")
        seg_args.bbox_threshold = seg_config.get('bbox_threshold', 0.4)
        seg_args.use_wsi = seg_config.get('use_wsi', False)
        seg_args.low_contrast_enhancement = seg_config.get('low_contrast_enhancement', False)
        seg_args.gauge_cell_size = seg_config.get('gauge_cell_size', False)
        seg_args.arcsinh = seg_config.get('arcsinh', False)
        seg_args.arcsinh_cofactor = seg_config.get('arcsinh_cofactor', 10.0)
        
        # Handle denoise settings
        denoise_settings = seg_config.get('denoise_settings')
        if denoise_settings:
            if isinstance(denoise_settings, str):
                seg_args.denoise_settings = denoise_settings
            else:
                # Save to temporary JSON file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(denoise_settings, f)
                    seg_args.denoise_settings = f.name
        else:
            seg_args.denoise_settings = None
        
        segment_command(seg_args)
        workflow_state['segmentation_output'] = seg_output
    
    # Step: Feature Extraction (if configured)
    if 'feature_extraction' in config and config['feature_extraction'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: FEATURE EXTRACTION")
        print("="*60)
        
        feat_config = config['feature_extraction']
        feat_input = get_step_input(feat_config, workflow_state['input'], 'Feature Extraction')
        
        # Determine mask path
        if 'mask' in feat_config:
            mask_path = Path(feat_config['mask'])
        elif workflow_state['segmentation_output']:
            mask_path = workflow_state['segmentation_output']
        else:
            raise ValueError("Feature extraction requires either 'mask' path in config or segmentation to be run first")
        
        features_output = get_step_output(feat_config, output_base / 'features.csv', 'Feature Extraction')
        
        # Create a mock args object for extract_features_command
        class ExtractArgs:
            pass
        
        extract_args = ExtractArgs()
        extract_args.input = str(feat_input)
        extract_args.output = str(features_output)
        extract_args.channel_format = feat_config.get('channel_format', config.get('channel_format', 'CHW'))
        extract_args.mask = str(mask_path)
        extract_args.acquisition = feat_config.get('acquisition')
        extract_args.morphological = feat_config.get('morphological', True)
        extract_args.intensity = feat_config.get('intensity', True)
        extract_args.arcsinh = feat_config.get('arcsinh', False)
        extract_args.arcsinh_cofactor = feat_config.get('arcsinh_cofactor', 10.0)
        
        # Handle denoise settings
        denoise_settings = feat_config.get('denoise_settings')
        if denoise_settings:
            if isinstance(denoise_settings, str):
                extract_args.denoise_settings = denoise_settings
            else:
                # Save to temporary JSON file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(denoise_settings, f)
                    extract_args.denoise_settings = f.name
        else:
            extract_args.denoise_settings = None
        
        # Handle spillover correction
        spillover_config = feat_config.get('spillover_correction')
        if spillover_config and spillover_config.get('enabled', False):
            # We need to modify extract_features_for_acquisition to accept spillover config
            # For now, we'll extract features first, then apply spillover correction
            print("  Note: Spillover correction will be applied after feature extraction")
            extract_args._spillover_config = spillover_config
        else:
            extract_args._spillover_config = None
        
        extract_features_command(extract_args)
        
        # Apply spillover correction if configured
        if extract_args._spillover_config:
            print("\n  Applying spillover correction...")
            features_df = pd.read_csv(features_output)
            
            spillover_file = extract_args._spillover_config.get('matrix_file')
            if not spillover_file:
                raise ValueError("spillover_correction.matrix_file must be specified")
            
            spillover_matrix = load_spillover(spillover_file)
            spillover_method = extract_args._spillover_config.get('method', 'nnls')
            
            # Get channel names from feature columns (intensity features end with _mean, _median, etc.)
            # We need to identify which columns are intensity features
            intensity_feature_types = ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated']
            channel_names = set()
            for col in features_df.columns:
                for feat_type in intensity_feature_types:
                    if col.endswith(f'_{feat_type}'):
                        channel_name = col[:-len(f'_{feat_type}')]
                        channel_names.add(channel_name)
                        break
            
            if not channel_names:
                print("  Warning: No intensity features found for spillover correction")
            else:
                # Apply spillover correction to each intensity feature type separately
                # This matches the approach in feature_worker.py
                from openimc.processing.spillover_correction import compensate_counts
                
                for feature_type in intensity_feature_types:
                    # Extract columns for this feature type across all channels
                    feature_cols = [f"{ch_name}_{feature_type}" for ch_name in channel_names 
                                   if f"{ch_name}_{feature_type}" in features_df.columns]
                    
                    if not feature_cols:
                        continue
                    
                    # Create a temporary DataFrame with cells x channels for this feature type
                    feature_data = features_df[feature_cols].copy()
                    # Rename columns to match channel names (remove the feature_type suffix)
                    channel_map = {col: col.replace(f"_{feature_type}", "") for col in feature_cols}
                    feature_data.rename(columns=channel_map, inplace=True)
                    
                    # Apply spillover correction
                    comp_data, _ = compensate_counts(
                        feature_data,
                        spillover_matrix,
                        method=spillover_method,
                        strict_align=False,
                        return_all_channels=True
                    )
                    
                    # Rename columns back and update features_df
                    comp_data.rename(columns={ch: f"{ch}_{feature_type}" for ch in comp_data.columns}, inplace=True)
                    for col in comp_data.columns:
                        if col in features_df.columns:
                            features_df[col] = comp_data[col].values
                
                # Save updated features
                features_df.to_csv(features_output, index=False)
                print(f"  ✓ Spillover correction applied to all intensity features and saved to {features_output}")
        
        workflow_state['features_output'] = features_output
    
    # Step: Batch Correction (if configured)
    if 'batch_correction' in config and config['batch_correction'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: BATCH CORRECTION")
        print("="*60)
        
        batch_config = config['batch_correction']
        
        # Determine input features path
        if 'input_features' in batch_config:
            features_path = Path(batch_config['input_features'])
        elif workflow_state['features_output']:
            features_path = workflow_state['features_output']
        else:
            raise ValueError("Batch correction requires either 'input_features' path in config or feature extraction to be run first")
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        batch_output = get_step_output(batch_config, output_base / 'features_batch_corrected.csv', 'Batch Correction')
        
        # Load features
        print(f"Loading features from: {features_path}")
        features_df = pd.read_csv(features_path)
        
        # Determine batch variable
        batch_var = batch_config.get('batch_variable')
        if not batch_var:
            batch_var = detect_batch_variable(features_df)
            if not batch_var:
                raise ValueError("Could not detect batch variable. Please specify 'batch_variable' in config.")
            print(f"  Auto-detected batch variable: {batch_var}")
        else:
            if batch_var not in features_df.columns:
                raise ValueError(f"Batch variable '{batch_var}' not found in features")
        
        # Determine features to correct
        feature_columns = batch_config.get('features')
        if not feature_columns:
            # Auto-detect: exclude non-feature columns
            exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'cell_id', 
                          'centroid_x', 'centroid_y', 'source_file', 'source_well', batch_var}
            feature_columns = [col for col in features_df.columns if col not in exclude_cols]
            print(f"  Auto-detected {len(feature_columns)} features for correction")
        else:
            # Validate specified features
            missing = [f for f in feature_columns if f not in features_df.columns]
            if missing:
                raise ValueError(f"Features not found: {missing}")
        
        # Apply batch correction
        method = batch_config.get('method', 'harmony')
        print(f"  Applying {method} batch correction...")
        
        if method == 'combat':
            covariates = batch_config.get('covariates')
            corrected_df = apply_combat_correction(
                features_df,
                batch_var,
                feature_columns,
                covariates=covariates
            )
        elif method == 'harmony':
            corrected_df = apply_harmony_correction(
                features_df,
                batch_var,
                feature_columns,
                n_clusters=batch_config.get('n_clusters', 30),
                sigma=batch_config.get('sigma', 0.1),
                theta=batch_config.get('theta', 2.0),
                lambda_reg=batch_config.get('lambda_reg', 1.0),
                max_iter=batch_config.get('max_iter', 10),
                pca_variance=batch_config.get('pca_variance', 0.9)
            )
        else:
            raise ValueError(f"Unknown batch correction method: {method}")
        
        # Save corrected features
        print(f"  Saving corrected features to: {batch_output}")
        corrected_df.to_csv(batch_output, index=False)
        print(f"  ✓ Batch correction complete! Output saved to: {batch_output}")
        workflow_state['batch_corrected_output'] = batch_output
    
    # Step: Pixel Correlation
    if 'pixel_correlation' in config and config['pixel_correlation'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: PIXEL CORRELATION")
        print("="*60)
        
        corr_config = config['pixel_correlation']
        corr_input = get_step_input(corr_config, workflow_state['input'], 'Pixel Correlation')
        corr_output = get_step_output(corr_config, output_base / 'pixel_correlation.csv', 'Pixel Correlation')
        
        # This would require implementing a CLI version of pixel correlation
        # For now, we'll note that this needs to be implemented
        print("  Note: Pixel correlation CLI implementation needed")
        print(f"  Would analyze: {corr_input}")
        print(f"  Would save to: {corr_output}")
    
    # Step: QC Analysis
    if 'qc_analysis' in config and config['qc_analysis'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: QC ANALYSIS")
        print("="*60)
        
        qc_config = config['qc_analysis']
        qc_input = get_step_input(qc_config, workflow_state['input'], 'QC Analysis')
        qc_output = get_step_output(qc_config, output_base / 'qc_analysis.csv', 'QC Analysis')
        
        # This would require implementing a CLI version of QC analysis
        # For now, we'll note that this needs to be implemented
        print("  Note: QC analysis CLI implementation needed")
        print(f"  Would analyze: {qc_input}")
        print(f"  Would save to: {qc_output}")
        if qc_config.get('mask'):
            print(f"  Using mask: {qc_config['mask']}")
    
    # Step: Clustering (if configured)
    if 'clustering' in config and config['clustering'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: CLUSTERING")
        print("="*60)
        
        cluster_config = config['clustering']
        
        # Determine input features path
        if 'input_features' in cluster_config:
            features_path = Path(cluster_config['input_features'])
        elif workflow_state['batch_corrected_output']:
            features_path = workflow_state['batch_corrected_output']
        elif workflow_state['features_output']:
            features_path = workflow_state['features_output']
        else:
            raise ValueError("Clustering requires either 'input_features' path in config or feature extraction to be run first")
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        cluster_output = get_step_output(cluster_config, output_base / 'clustered_features.csv', 'Clustering')
        
        class ClusterArgs:
            pass
        cluster_args = ClusterArgs()
        cluster_args.features = str(features_path)
        cluster_args.output = str(cluster_output)
        cluster_args.method = cluster_config.get('method', 'leiden')
        cluster_args.n_clusters = cluster_config.get('n_clusters')
        cluster_args.columns = ','.join(cluster_config.get('columns', [])) if cluster_config.get('columns') else None
        cluster_args.scaling = cluster_config.get('scaling', 'zscore')
        cluster_args.linkage = cluster_config.get('linkage', 'ward')
        cluster_args.resolution = cluster_config.get('resolution', 1.0)
        cluster_args.min_cluster_size = cluster_config.get('min_cluster_size', 10)
        cluster_args.min_samples = cluster_config.get('min_samples', 5)
        cluster_args.seed = cluster_config.get('seed', 42)
        
        cluster_command(cluster_args)
    
    # Step: Spatial Analysis (if configured)
    if 'spatial_analysis' in config and config['spatial_analysis'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: SPATIAL ANALYSIS")
        print("="*60)
        
        spatial_config = config['spatial_analysis']
        
        # Determine input features path
        if 'input_features' in spatial_config:
            features_path = Path(spatial_config['input_features'])
        elif workflow_state['batch_corrected_output']:
            features_path = workflow_state['batch_corrected_output']
        elif workflow_state['features_output']:
            features_path = workflow_state['features_output']
        else:
            raise ValueError("Spatial analysis requires either 'input_features' path in config or feature extraction to be run first")
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        spatial_output = get_step_output(spatial_config, output_base / 'spatial_edges.csv', 'Spatial Analysis')
        
        class SpatialArgs:
            pass
        spatial_args = SpatialArgs()
        spatial_args.features = str(features_path)
        spatial_args.output = str(spatial_output)
        spatial_args.radius = spatial_config.get('radius')
        if not spatial_args.radius:
            raise ValueError("spatial_analysis.radius is required")
        spatial_args.k_neighbors = spatial_config.get('k_neighbors', 10)
        spatial_args.pixel_size_um = spatial_config.get('pixel_size_um', 1.0)
        spatial_args.detect_communities = spatial_config.get('detect_communities', False)
        spatial_args.seed = spatial_config.get('seed', 42)
        
        spatial_command(spatial_args)
    
    print("\n" + "="*60)
    print("✓ WORKFLOW COMPLETE")
    print("="*60)
    print(f"Output directory: {output_base}")


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

  # Run complete workflow from config file
  openimc workflow config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess images (denoising, export to OME-TIFF). Note: arcsinh normalization is not applied to exported images.')
    preprocess_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    preprocess_parser.add_argument('output', help='Output directory for processed OME-TIFF files')
    preprocess_parser.add_argument('--arcsinh', action='store_true', help='(Deprecated) Arcsinh normalization is not applied to exported images. Use during feature extraction instead.')
    preprocess_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='(Deprecated) Arcsinh cofactor (default: 10.0). Not used for export.')
    preprocess_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    preprocess_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    preprocess_parser.set_defaults(func=preprocess_command)
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment cells (DeepCell CellSAM, Cellpose, or watershed)')
    segment_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    segment_parser.add_argument('output', help='Output directory for segmentation masks')
    segment_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    segment_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (uses first if not specified)')
    segment_parser.add_argument('--method', choices=['cellsam', 'cellpose', 'watershed'], default='cellsam', help='Segmentation method (default: cellsam)')
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
    # DeepCell CellSAM parameters
    segment_parser.add_argument('--deepcell-api-key', type=str, help='DeepCell API key (CellSAM). Can also be set via DEEPCELL_ACCESS_TOKEN environment variable')
    segment_parser.add_argument('--bbox-threshold', type=float, default=0.4, help='Bbox threshold for CellSAM (default: 0.4, lower for faint cells: 0.01-0.1)')
    segment_parser.add_argument('--use-wsi', action='store_true', help='Use WSI mode for CellSAM (for ROIs with >500 cells, increases processing time)')
    segment_parser.add_argument('--low-contrast-enhancement', action='store_true', help='Enable low contrast enhancement for CellSAM (for poor contrast images)')
    segment_parser.add_argument('--gauge-cell-size', action='store_true', help='Enable gauge cell size for CellSAM (runs twice: estimates error, then returns mask)')
    segment_parser.add_argument('--arcsinh', action='store_true', help='Apply arcsinh normalization before segmentation')
    segment_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='Arcsinh cofactor (default: 10.0)')
    segment_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    segment_parser.set_defaults(func=segment_command)
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract-features', help='Extract features from segmented cells')
    extract_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    extract_parser.add_argument('output', help='Output CSV file path')
    extract_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    extract_parser.add_argument('--mask', type=str, required=True, help='Path to segmentation mask directory or single mask file (.tif, .tiff, or .npy). If directory, masks are matched to acquisitions by filename.')
    extract_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (uses first if not specified)')
    extract_parser.add_argument('--morphological', action='store_true', help='Extract morphological features')
    extract_parser.add_argument('--intensity', action='store_true', help='Extract intensity features')
    extract_parser.add_argument('--arcsinh', action='store_true', help='Apply arcsinh transformation to extracted intensity features (mean, median, std, etc.), not to raw images')
    extract_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='Arcsinh cofactor (default: 10.0)')
    extract_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    extract_parser.set_defaults(func=extract_features_command)
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Perform clustering on feature data')
    cluster_parser.add_argument('features', help='Input CSV file with features')
    cluster_parser.add_argument('output', help='Output CSV file with cluster labels')
    cluster_parser.add_argument('--method', choices=['hierarchical', 'leiden', 'louvain', 'kmeans', 'hdbscan'], default='leiden', help='Clustering method')
    cluster_parser.add_argument('--n-clusters', type=int, help='Number of clusters (required for hierarchical and kmeans, not used for leiden/louvain/hdbscan)')
    cluster_parser.add_argument('--columns', type=str, help='Comma-separated list of columns to use for clustering (auto-detect if not specified)')
    cluster_parser.add_argument('--scaling', choices=['none', 'zscore', 'mad'], default='zscore', help='Feature scaling method (zscore or mad, matching GUI)')
    cluster_parser.add_argument('--linkage', choices=['ward', 'complete', 'average'], default='ward', help='Linkage method for hierarchical clustering')
    cluster_parser.add_argument('--resolution', type=float, default=1.0, help='Resolution parameter for Leiden clustering (default: 1.0)')
    cluster_parser.add_argument('--n-neighbors', type=int, default=15, help='Number of neighbors for k-NN graph (Leiden/Louvain only, default: 15)')
    cluster_parser.add_argument('--metric', choices=['euclidean', 'manhattan', 'cosine'], default='euclidean', help='Distance metric for k-NN graph (Leiden/Louvain only, default: euclidean)')
    cluster_parser.add_argument('--n-init', type=int, default=10, help='Number of initializations for K-means (default: 10)')
    cluster_parser.add_argument('--min-cluster-size', type=int, default=10, help='Minimum cluster size (hdbscan, default: 10)')
    cluster_parser.add_argument('--min-samples', type=int, default=5, help='Minimum samples (hdbscan, default: 5)')
    cluster_parser.add_argument('--cluster-selection-method', choices=['eom', 'leaf'], default='eom', help='Cluster selection method for HDBSCAN (default: eom)')
    cluster_parser.add_argument('--hdbscan-metric', choices=['euclidean', 'manhattan'], default='euclidean', help='Distance metric for HDBSCAN (default: euclidean)')
    cluster_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    cluster_parser.set_defaults(func=cluster_command)
    
    # Spatial command
    spatial_parser = subparsers.add_parser('spatial', help='Perform spatial analysis on feature data')
    spatial_parser.add_argument('features', help='Input CSV file with features (must contain centroid_x, centroid_y)')
    spatial_parser.add_argument('output', help='Output CSV file with spatial graph edges')
    spatial_parser.add_argument('--method', choices=['kNN', 'Radius', 'Delaunay'], default='kNN', help='Graph construction method (default: kNN)')
    spatial_parser.add_argument('--radius', type=float, help='Maximum distance for edges in pixels (required for Radius method)')
    spatial_parser.add_argument('--k-neighbors', type=int, default=10, help='k for k-nearest neighbors (default: 10, used for kNN method)')
    spatial_parser.add_argument('--pixel-size-um', type=float, default=1.0, help='Pixel size in micrometers (default: 1.0, used for distance_um conversion)')
    spatial_parser.add_argument('--detect-communities', action='store_true', help='Also detect spatial communities')
    spatial_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    spatial_parser.set_defaults(func=spatial_command)
    
    # Spatial AnnData command (build graph)
    spatial_anndata_parser = subparsers.add_parser('spatial-anndata', help='Build spatial graph using AnnData/squidpy approach')
    spatial_anndata_parser.add_argument('features', help='Input CSV file with features (must contain centroid_x, centroid_y)')
    spatial_anndata_parser.add_argument('--output', type=str, help='Output H5AD file or directory for AnnData objects')
    spatial_anndata_parser.add_argument('--method', choices=['kNN', 'Radius', 'Delaunay'], default='kNN', help='Graph construction method (default: kNN)')
    spatial_anndata_parser.add_argument('--radius', type=float, help='Maximum distance for edges in micrometers (required for Radius method)')
    spatial_anndata_parser.add_argument('--k-neighbors', type=int, default=20, help='k for k-nearest neighbors (default: 20, used for kNN method)')
    spatial_anndata_parser.add_argument('--pixel-size-um', type=float, default=1.0, help='Pixel size in micrometers (default: 1.0)')
    spatial_anndata_parser.add_argument('--roi-column', type=str, help='Column name for ROI grouping (auto-detected if not specified)')
    spatial_anndata_parser.add_argument('--roi-id', type=str, help='Specific ROI ID to process (processes all if not specified)')
    spatial_anndata_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    spatial_anndata_parser.add_argument('--combined', action='store_true', help='Export as single combined file (default: separate files per ROI)')
    spatial_anndata_parser.set_defaults(func=spatial_anndata_command)
    
    # Spatial neighborhood enrichment command
    nhood_parser = subparsers.add_parser('spatial-nhood-enrichment', help='Run neighborhood enrichment analysis using squidpy')
    nhood_parser.add_argument('features', help='Input CSV file with features')
    nhood_parser.add_argument('--output', type=str, help='Output H5AD file or directory for results')
    nhood_parser.add_argument('--method', choices=['kNN', 'Radius', 'Delaunay'], default='kNN', help='Graph construction method (default: kNN)')
    nhood_parser.add_argument('--radius', type=float, help='Maximum distance for edges in micrometers (required for Radius method)')
    nhood_parser.add_argument('--k-neighbors', type=int, default=20, help='k for k-nearest neighbors (default: 20)')
    nhood_parser.add_argument('--pixel-size-um', type=float, default=1.0, help='Pixel size in micrometers (default: 1.0)')
    nhood_parser.add_argument('--roi-column', type=str, help='Column name for ROI grouping (auto-detected if not specified)')
    nhood_parser.add_argument('--roi-id', type=str, help='Specific ROI ID to process (processes all if not specified)')
    nhood_parser.add_argument('--cluster-column', type=str, default='cluster', help='Column name containing cluster labels (default: cluster)')
    nhood_parser.add_argument('--aggregation', choices=['mean', 'sum'], default='mean', help='Aggregation method for multiple ROIs (default: mean)')
    nhood_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    nhood_parser.add_argument('--combined', action='store_true', help='Export as single combined file (default: separate files per ROI)')
    nhood_parser.set_defaults(func=spatial_nhood_enrichment_command)
    
    # Spatial co-occurrence command
    cooccur_parser = subparsers.add_parser('spatial-cooccurrence', help='Run co-occurrence analysis using squidpy')
    cooccur_parser.add_argument('features', help='Input CSV file with features')
    cooccur_parser.add_argument('--output', type=str, help='Output H5AD file or directory for results')
    cooccur_parser.add_argument('--method', choices=['kNN', 'Radius', 'Delaunay'], default='kNN', help='Graph construction method (default: kNN)')
    cooccur_parser.add_argument('--radius', type=float, help='Maximum distance for edges in micrometers (required for Radius method)')
    cooccur_parser.add_argument('--k-neighbors', type=int, default=20, help='k for k-nearest neighbors (default: 20)')
    cooccur_parser.add_argument('--pixel-size-um', type=float, default=1.0, help='Pixel size in micrometers (default: 1.0)')
    cooccur_parser.add_argument('--roi-column', type=str, help='Column name for ROI grouping (auto-detected if not specified)')
    cooccur_parser.add_argument('--roi-id', type=str, help='Specific ROI ID to process (processes all if not specified)')
    cooccur_parser.add_argument('--cluster-column', type=str, default='cluster', help='Column name containing cluster labels (default: cluster)')
    cooccur_parser.add_argument('--interval', type=str, default='10,20,30,50,100', help='Comma-separated distances in micrometers (default: 10,20,30,50,100)')
    cooccur_parser.add_argument('--reference-cluster', type=str, help='Optional reference cluster for co-occurrence')
    cooccur_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    cooccur_parser.add_argument('--combined', action='store_true', help='Export as single combined file (default: separate files per ROI)')
    cooccur_parser.set_defaults(func=spatial_cooccurrence_command)
    
    # Spatial autocorrelation command
    autocorr_parser = subparsers.add_parser('spatial-autocorr', help='Run spatial autocorrelation (Moran\'s I) analysis using squidpy')
    autocorr_parser.add_argument('features', help='Input CSV file with features')
    autocorr_parser.add_argument('--output', type=str, help='Output H5AD file or directory for results')
    autocorr_parser.add_argument('--method', choices=['kNN', 'Radius', 'Delaunay'], default='kNN', help='Graph construction method (default: kNN)')
    autocorr_parser.add_argument('--radius', type=float, help='Maximum distance for edges in micrometers (required for Radius method)')
    autocorr_parser.add_argument('--k-neighbors', type=int, default=20, help='k for k-nearest neighbors (default: 20)')
    autocorr_parser.add_argument('--pixel-size-um', type=float, default=1.0, help='Pixel size in micrometers (default: 1.0)')
    autocorr_parser.add_argument('--roi-column', type=str, help='Column name for ROI grouping (auto-detected if not specified)')
    autocorr_parser.add_argument('--roi-id', type=str, help='Specific ROI ID to process (processes all if not specified)')
    autocorr_parser.add_argument('--markers', type=str, default='all', help='Comma-separated list of markers to analyze, or "all" (default: all)')
    autocorr_parser.add_argument('--aggregation', choices=['mean', 'sum'], default='mean', help='Aggregation method for multiple ROIs (default: mean)')
    autocorr_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    autocorr_parser.add_argument('--combined', action='store_true', help='Export as single combined file (default: separate files per ROI)')
    autocorr_parser.set_defaults(func=spatial_autocorr_command)
    
    # Spatial Ripley command
    ripley_parser = subparsers.add_parser('spatial-ripley', help='Run Ripley function analysis using squidpy')
    ripley_parser.add_argument('features', help='Input CSV file with features')
    ripley_parser.add_argument('--output', type=str, help='Output H5AD file or directory for results')
    ripley_parser.add_argument('--method', choices=['kNN', 'Radius', 'Delaunay'], default='kNN', help='Graph construction method (default: kNN)')
    ripley_parser.add_argument('--radius', type=float, help='Maximum distance for edges in micrometers (required for Radius method)')
    ripley_parser.add_argument('--k-neighbors', type=int, default=20, help='k for k-nearest neighbors (default: 20)')
    ripley_parser.add_argument('--pixel-size-um', type=float, default=1.0, help='Pixel size in micrometers (default: 1.0)')
    ripley_parser.add_argument('--roi-column', type=str, help='Column name for ROI grouping (auto-detected if not specified)')
    ripley_parser.add_argument('--roi-id', type=str, help='Specific ROI ID to process (processes all if not specified)')
    ripley_parser.add_argument('--cluster-column', type=str, default='cluster', help='Column name containing cluster labels (default: cluster)')
    ripley_parser.add_argument('--mode', choices=['F', 'G', 'L'], default='L', help='Ripley function mode (default: L)')
    ripley_parser.add_argument('--max-dist', type=float, default=50.0, help='Maximum distance in micrometers (default: 50.0)')
    ripley_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    ripley_parser.add_argument('--combined', action='store_true', help='Export as single combined file (default: separate files per ROI)')
    ripley_parser.set_defaults(func=spatial_ripley_command)
    
    # Export AnnData command
    export_anndata_parser = subparsers.add_parser('export-anndata', help='Export AnnData objects to H5AD file(s)')
    export_anndata_parser.add_argument('input', help='Input H5AD file or directory containing H5AD files')
    export_anndata_parser.add_argument('output', help='Output H5AD file (if combined) or directory (if separate)')
    export_anndata_parser.add_argument('--combined', action='store_true', help='Export as single combined file (default: separate files per ROI)')
    export_anndata_parser.set_defaults(func=export_anndata_command)
    
    # Batch correction command
    batch_parser = subparsers.add_parser('batch-correction', help='Apply batch correction to feature data')
    batch_parser.add_argument('features', help='Input CSV file with features')
    batch_parser.add_argument('output', help='Output CSV file with corrected features')
    batch_parser.add_argument('--method', choices=['combat', 'harmony'], default='harmony', help='Batch correction method (default: harmony)')
    batch_parser.add_argument('--batch-var', type=str, help='Column name containing batch identifiers (auto-detected if not specified)')
    batch_parser.add_argument('--columns', type=str, help='Comma-separated list of feature columns to correct (auto-detected if not specified)')
    batch_parser.add_argument('--covariates', type=str, help='Comma-separated list of covariate columns (ComBat only)')
    batch_parser.add_argument('--n-clusters', type=int, default=30, help='Number of Harmony clusters (default: 30)')
    batch_parser.add_argument('--sigma', type=float, default=0.1, help='Width of soft kmeans clusters for Harmony (default: 0.1)')
    batch_parser.add_argument('--theta', type=float, default=2.0, help='Diversity clustering penalty parameter for Harmony (default: 2.0)')
    batch_parser.add_argument('--lambda-reg', type=float, default=1.0, help='Regularization parameter for Harmony (default: 1.0)')
    batch_parser.add_argument('--max-iter', type=int, default=10, help='Maximum iterations for Harmony (default: 10)')
    batch_parser.add_argument('--pca-variance', type=float, default=0.9, help='Proportion of variance to retain in PCA for Harmony (default: 0.9)')
    batch_parser.set_defaults(func=batch_correction_command)
    
    # Pixel correlation command
    pixel_corr_parser = subparsers.add_parser('pixel-correlation', help='Compute pixel-level correlations between markers')
    pixel_corr_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    pixel_corr_parser.add_argument('output', help='Output CSV file with correlation results')
    pixel_corr_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    pixel_corr_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (uses first if not specified)')
    pixel_corr_parser.add_argument('--channels', type=str, help='Comma-separated list of channels to analyze (uses all if not specified)')
    pixel_corr_parser.add_argument('--mask', type=str, help='Path to segmentation mask file (optional, for within-cell analysis)')
    pixel_corr_parser.add_argument('--multiple-testing-correction', type=str, choices=['bonferroni', 'fdr_bh', 'fdr_by'], help='Multiple testing correction method')
    pixel_corr_parser.set_defaults(func=pixel_correlation_command)
    
    # QC analysis command
    qc_parser = subparsers.add_parser('qc-analysis', help='Perform quality control analysis')
    qc_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    qc_parser.add_argument('output', help='Output CSV file with QC metrics')
    qc_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    qc_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (uses first if not specified)')
    qc_parser.add_argument('--channels', type=str, help='Comma-separated list of channels to analyze (uses all if not specified)')
    qc_parser.add_argument('--mode', choices=['pixel', 'cell'], default='pixel', help='Analysis mode (default: pixel)')
    qc_parser.add_argument('--mask', type=str, help='Path to segmentation mask file (required for cell mode)')
    qc_parser.set_defaults(func=qc_analysis_command)
    
    # Spillover correction command
    spillover_parser = subparsers.add_parser('spillover-correction', help='Apply spillover correction to feature data')
    spillover_parser.add_argument('features', help='Input CSV file with features')
    spillover_parser.add_argument('spillover_matrix', help='Path to spillover matrix CSV file')
    spillover_parser.add_argument('output', help='Output CSV file with corrected features')
    spillover_parser.add_argument('--method', choices=['nnls', 'pgd'], default='pgd', help='Compensation method (default: pgd)')
    spillover_parser.add_argument('--arcsinh-cofactor', type=float, help='Optional cofactor for arcsinh transformation')
    spillover_parser.set_defaults(func=spillover_correction_command)
    
    # Generate spillover matrix command
    spillover_gen_parser = subparsers.add_parser('generate-spillover-matrix', help='Generate spillover matrix from single-stain control MCD file')
    spillover_gen_parser.add_argument('input', help='Input MCD file with single-stain controls')
    spillover_gen_parser.add_argument('output', help='Output CSV file for spillover matrix')
    spillover_gen_parser.add_argument('--donor-map', type=str, help='JSON file or string mapping acquisition IDs to donor channel names')
    spillover_gen_parser.add_argument('--cap', type=float, default=0.3, help='Maximum spillover coefficient (default: 0.3)')
    spillover_gen_parser.add_argument('--aggregate', choices=['median', 'mean'], default='median', help='Aggregation method (default: median)')
    spillover_gen_parser.set_defaults(func=generate_spillover_matrix_command)
    
    # Deconvolution command
    deconv_parser = subparsers.add_parser('deconvolution', help='Apply deconvolution to high resolution IMC images')
    deconv_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    deconv_parser.add_argument('output', help='Output directory for deconvolved images')
    deconv_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    deconv_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (processes all if not specified)')
    deconv_parser.add_argument('--x0', type=float, default=7.0, help='Parameter for kernel calculation (default: 7.0)')
    deconv_parser.add_argument('--iterations', type=int, default=4, help='Number of Richardson-Lucy iterations (default: 4)')
    deconv_parser.add_argument('--output-format', choices=['float', 'uint16'], default='float', help='Output format (default: float)')
    deconv_parser.set_defaults(func=deconvolution_command)
    
    # Spatial enrichment command
    enrichment_parser = subparsers.add_parser('spatial-enrichment', help='Compute pairwise spatial enrichment between clusters')
    enrichment_parser.add_argument('features', help='Input CSV file with features and cluster labels')
    enrichment_parser.add_argument('edges', help='Input CSV file with spatial graph edges')
    enrichment_parser.add_argument('output', help='Output CSV file with enrichment results')
    enrichment_parser.add_argument('--cluster-column', type=str, default='cluster', help='Column name containing cluster labels (default: cluster)')
    enrichment_parser.add_argument('--n-permutations', type=int, default=100, help='Number of permutations for null distribution (default: 100)')
    enrichment_parser.add_argument('--roi-column', type=str, help='Column name for ROI grouping (auto-detected if not specified)')
    enrichment_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    enrichment_parser.set_defaults(func=spatial_enrichment_command)
    
    # Spatial distance command
    distance_parser = subparsers.add_parser('spatial-distance', help='Compute distance distributions between clusters')
    distance_parser.add_argument('features', help='Input CSV file with features and cluster labels')
    distance_parser.add_argument('edges', help='Input CSV file with spatial graph edges')
    distance_parser.add_argument('output', help='Output CSV file with distance distribution results')
    distance_parser.add_argument('--cluster-column', type=str, default='cluster', help='Column name containing cluster labels (default: cluster)')
    distance_parser.add_argument('--roi-column', type=str, help='Column name for ROI grouping (auto-detected if not specified)')
    distance_parser.set_defaults(func=spatial_distance_command)
    
    # Cluster figures command
    cluster_figures_parser = subparsers.add_parser('cluster-figures', help='Generate cluster visualization figures')
    cluster_figures_parser.add_argument('features', help='Input CSV file with clustered features')
    cluster_figures_parser.add_argument('output', help='Output directory for figures')
    cluster_figures_parser.add_argument('--dpi', type=int, default=300, help='Figure DPI (default: 300)')
    cluster_figures_parser.add_argument('--font-size', type=float, default=10.0, help='Font size in points (default: 10.0)')
    cluster_figures_parser.add_argument('--width', type=float, default=8.0, help='Figure width in inches (default: 8.0)')
    cluster_figures_parser.add_argument('--height', type=float, default=6.0, help='Figure height in inches (default: 6.0)')
    cluster_figures_parser.add_argument('--seed', type=int, default=42, help='Random seed for UMAP reproducibility (default: 42)')
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
    
    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Execute a complete workflow from a YAML configuration file')
    workflow_parser.add_argument('config', help='Path to YAML configuration file')
    workflow_parser.set_defaults(func=workflow_command)
    
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

