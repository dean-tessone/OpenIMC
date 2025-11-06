"""
Comprehensive logging system for OpenIMC.

This module provides structured logging for all user actions that produce outputs,
designed to support creation of methods sections for scientific papers.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class MethodsLogger:
    """
    Logger for recording all analysis operations with detailed parameters.
    
    Logs are appended to a file to preserve history across sessions.
    Each entry is a structured JSON object with timestamp and operation details.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_file: Path to log file. If None, uses default location.
        """
        if log_file is None:
            # Default to logs directory in the OpenIMC project folder
            # Find the project root by looking for this file's location
            current_file = Path(__file__).resolve()
            # Navigate from openimc/utils/logger.py to project root
            project_root = current_file.parent.parent.parent
            log_dir = project_root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(log_dir / "methods_log.jsonl")
        else:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = log_file
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log file exists and write header if new file."""
        if not os.path.exists(self.log_file):
            # Write a header comment (JSON Lines format doesn't support comments,
            # but we can add a metadata entry)
            with open(self.log_file, 'w') as f:
                metadata = {
                    "type": "log_metadata",
                    "timestamp": datetime.now().isoformat(),
                    "description": "OpenIMC Methods Log - This file records all analysis operations",
                    "format": "JSON Lines (one JSON object per line)"
                }
                f.write(json.dumps(metadata) + "\n")
    
    def _write_entry(self, entry_type: str, operation: str, parameters: Dict[str, Any], 
                    acquisitions: Optional[List[str]] = None, 
                    output_path: Optional[str] = None,
                    notes: Optional[str] = None,
                    source_file: Optional[str] = None):
        """
        Write a log entry.
        
        Args:
            entry_type: Type of operation (e.g., "segmentation", "feature_extraction")
            operation: Specific operation name (e.g., "cellpose", "watershed")
            parameters: Dictionary of parameters used
            acquisitions: List of acquisition IDs affected
            output_path: Path to output file if applicable
            notes: Additional notes or comments
            source_file: Name of the source file being analyzed
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": entry_type,
            "operation": operation,
            "parameters": parameters,
            "acquisitions": acquisitions if acquisitions else [],
            "output_path": output_path,
            "notes": notes,
            "source_file": source_file
        }
        
        # Append to log file (thread-safe write)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + "\n")
    
    def log_segmentation(self, method: str, parameters: Dict[str, Any], 
                        acquisitions: List[str], **kwargs):
        """
        Log a segmentation operation.
        
        Args:
            method: Segmentation method ("cellpose", "watershed", "ilastik")
            parameters: Dictionary of method-specific parameters
            acquisitions: List of acquisition IDs segmented
            **kwargs: Additional fields (output_path, notes, etc.)
        """
        self._write_entry(
            entry_type="segmentation",
            operation=method,
            parameters=parameters,
            acquisitions=acquisitions,
            output_path=kwargs.get("output_path"),
            notes=kwargs.get("notes"),
            source_file=kwargs.get("source_file")
        )
    
    def log_feature_extraction(self, parameters: Dict[str, Any], 
                              acquisitions: List[str], 
                              features_extracted: List[str],
                              **kwargs):
        """
        Log a feature extraction operation.
        
        Args:
            parameters: Extraction parameters (normalization, denoising, etc.)
            acquisitions: List of acquisition IDs processed
            features_extracted: List of feature names extracted
            **kwargs: Additional fields (output_path, notes, etc.)
        """
        params = parameters.copy()
        params["features_extracted"] = features_extracted
        
        self._write_entry(
            entry_type="feature_extraction",
            operation="extract_features",
            parameters=params,
            acquisitions=acquisitions,
            output_path=kwargs.get("output_path"),
            notes=kwargs.get("notes"),
            source_file=kwargs.get("source_file")
        )
    
    def log_clustering(self, method: str, parameters: Dict[str, Any], 
                      features_used: List[str], 
                      n_clusters: Optional[int] = None,
                      **kwargs):
        """
        Log a clustering operation.
        
        Args:
            method: Clustering method ("hierarchical", "leiden", etc.)
            parameters: Clustering parameters
            features_used: List of features used for clustering
            n_clusters: Number of clusters (if applicable)
            **kwargs: Additional fields (output_path, notes, etc.)
        """
        params = parameters.copy()
        params["features_used"] = features_used
        if n_clusters is not None:
            params["n_clusters"] = n_clusters
        
        self._write_entry(
            entry_type="clustering",
            operation=method,
            parameters=params,
            acquisitions=kwargs.get("acquisitions", []),
            output_path=kwargs.get("output_path"),
            notes=kwargs.get("notes"),
            source_file=kwargs.get("source_file")
        )
    
    def log_class_annotation(self, annotation_map: Dict[Any, str], 
                            method: Optional[str] = None,
                            **kwargs):
        """
        Log a class annotation operation.
        
        Args:
            annotation_map: Dictionary mapping cluster IDs to phenotype names
            method: Annotation method ("manual", "llm_suggestion", etc.)
            **kwargs: Additional fields (notes, etc.)
        """
        params = {
            "annotation_map": {str(k): str(v) for k, v in annotation_map.items()},
            "method": method or "manual"
        }
        
        self._write_entry(
            entry_type="class_annotation",
            operation="assign_phenotypes",
            parameters=params,
            acquisitions=kwargs.get("acquisitions", []),
            notes=kwargs.get("notes"),
            source_file=kwargs.get("source_file")
        )
    
    def log_gating(self, gating_rules: List[Dict[str, Any]], 
                  **kwargs):
        """
        Log a gating operation.
        
        Args:
            gating_rules: List of gating rule dictionaries
            **kwargs: Additional fields (output_path, notes, etc.)
        """
        self._write_entry(
            entry_type="gating",
            operation="manual_gating",
            parameters={"rules": gating_rules},
            acquisitions=kwargs.get("acquisitions", []),
            output_path=kwargs.get("output_path"),
            notes=kwargs.get("notes"),
            source_file=kwargs.get("source_file")
        )
    
    def log_spatial_analysis(self, analysis_type: str, 
                            parameters: Dict[str, Any],
                            **kwargs):
        """
        Log a spatial analysis operation.
        
        Args:
            analysis_type: Type of analysis ("graph_construction", "pairwise_enrichment", 
                          "distance_distribution", "ripley_k", "neighborhood_composition", 
                          "community_detection")
            parameters: Analysis-specific parameters
            **kwargs: Additional fields (acquisitions, output_path, notes, etc.)
        """
        self._write_entry(
            entry_type="spatial_analysis",
            operation=analysis_type,
            parameters=parameters,
            acquisitions=kwargs.get("acquisitions", []),
            output_path=kwargs.get("output_path"),
            notes=kwargs.get("notes"),
            source_file=kwargs.get("source_file")
        )
    
    def log_export(self, export_type: str, parameters: Dict[str, Any],
                  output_path: str, **kwargs):
        """
        Log an export operation.
        
        Args:
            export_type: Type of export ("csv", "tiff", "png", etc.)
            parameters: Export parameters
            output_path: Path to exported file
            **kwargs: Additional fields (acquisitions, notes, etc.)
        """
        params = parameters.copy()
        params["export_type"] = export_type
        
        self._write_entry(
            entry_type="export",
            operation="export_data",
            parameters=params,
            acquisitions=kwargs.get("acquisitions", []),
            output_path=output_path,
            notes=kwargs.get("notes"),
            source_file=kwargs.get("source_file")
        )
    
    def log_spillover_matrix(self, parameters: Dict[str, Any],
                            donor_mapping: Dict[str, Any],
                            **kwargs):
        """
        Log a spillover matrix generation operation.
        
        Args:
            parameters: Computation parameters (cap, aggregate, p_low, p_high, channel_field)
            donor_mapping: Dictionary mapping acquisitions to donor channels
            **kwargs: Additional fields (acquisitions, output_path, notes, etc.)
        """
        params = parameters.copy()
        params["donor_mapping"] = donor_mapping
        
        self._write_entry(
            entry_type="spillover_matrix",
            operation="generate_spillover_matrix",
            parameters=params,
            acquisitions=kwargs.get("acquisitions", []),
            output_path=kwargs.get("output_path"),
            notes=kwargs.get("notes"),
            source_file=kwargs.get("source_file")
        )
    
    def get_log_file_path(self) -> str:
        """Get the path to the log file."""
        return self.log_file


# Global logger instance
_logger_instance: Optional[MethodsLogger] = None


def get_logger(log_file: Optional[str] = None) -> MethodsLogger:
    """
    Get or create the global logger instance.
    
    Args:
        log_file: Optional path to log file (only used on first call)
    
    Returns:
        MethodsLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = MethodsLogger(log_file)
    return _logger_instance


def set_log_file(log_file: str):
    """
    Set a custom log file path (creates new logger instance).
    
    Args:
        log_file: Path to log file
    """
    global _logger_instance
    _logger_instance = MethodsLogger(log_file)

