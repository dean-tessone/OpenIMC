"""
Batch Correction Processing Functions

This module provides batch correction implementations using Combat and Harmony.
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np

# Optional imports
try:
    from combat.pycombat import pycombat
    _HAVE_COMBAT = True
except ImportError:
    _HAVE_COMBAT = False

try:
    from harmonypy import harmonize
    _HAVE_HARMONY = True
except ImportError:
    _HAVE_HARMONY = False


def apply_combat_correction(
    data: pd.DataFrame,
    batch_var: str,
    features: List[str],
    covariates: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply Combat batch correction to feature data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Feature dataframe with batch variable and features
    batch_var : str
        Column name containing batch identifiers
    features : List[str]
        List of feature column names to correct
    covariates : Optional[List[str]]
        Optional list of covariate column names
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with corrected features
    """
    if not _HAVE_COMBAT:
        raise ImportError("Combat is not installed. Install with: pip install combat")
    
    # Create a copy to avoid modifying original
    corrected_data = data.copy()
    
    # Extract feature matrix (samples x features)
    feature_matrix = data[features].values.T  # Combat expects features x samples
    
    # Get batch labels
    batch_labels = data[batch_var].astype(str).values
    
    # Prepare covariates if provided
    covar_matrix = None
    if covariates:
        # Convert categorical covariates to numeric
        covar_data = data[covariates].copy()
        for col in covariates:
            if covar_data[col].dtype == 'object':
                covar_data[col] = pd.Categorical(covar_data[col]).codes
        covar_matrix = covar_data.values
    
    # Apply Combat correction
    try:
        corrected_matrix = pycombat(feature_matrix, batch_labels, covar=covar_matrix)
    except Exception as e:
        raise RuntimeError(f"Combat correction failed: {str(e)}")
    
    # Update corrected features in dataframe
    # Transpose back to samples x features
    corrected_matrix = corrected_matrix.T
    for i, feature in enumerate(features):
        corrected_data[feature] = corrected_matrix[:, i]
    
    return corrected_data


def apply_harmony_correction(
    data: pd.DataFrame,
    batch_var: str,
    features: List[str],
    n_clusters: int = 30,
    sigma: float = 0.1,
    theta: float = 2.0,
    lambda_reg: float = 1.0,
    max_iter: int = 10
) -> pd.DataFrame:
    """
    Apply Harmony batch correction to feature data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Feature dataframe with batch variable and features
    batch_var : str
        Column name containing batch identifiers
    features : List[str]
        List of feature column names to correct
    n_clusters : int
        Number of Harmony clusters (default: 30)
    sigma : float
        Width of soft kmeans clusters (default: 0.1)
    theta : float
        Diversity clustering penalty parameter (default: 2.0)
    lambda_reg : float
        Regularization parameter (default: 1.0)
    max_iter : int
        Maximum number of iterations (default: 10)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with corrected features
    """
    if not _HAVE_HARMONY:
        raise ImportError("Harmony is not installed. Install with: pip install harmonypy")
    
    # Create a copy to avoid modifying original
    corrected_data = data.copy()
    
    # Extract feature matrix (samples x features)
    feature_matrix = data[features].values
    
    # Get batch labels
    batch_labels = data[batch_var].astype(str).values
    
    # Apply Harmony correction
    try:
        # Harmony expects data as numpy array (samples x features)
        # and batch labels as array or list
        corrected_matrix = harmonize(
            feature_matrix,
            batch_labels,
            n_clusters=n_clusters,
            sigma=sigma,
            theta=theta,
            lambda_reg=lambda_reg,
            max_iter=max_iter
        )
    except Exception as e:
        raise RuntimeError(f"Harmony correction failed: {str(e)}")
    
    # Update corrected features in dataframe
    for i, feature in enumerate(features):
        corrected_data[feature] = corrected_matrix[:, i]
    
    return corrected_data


def detect_batch_variable(data: pd.DataFrame) -> Optional[str]:
    """
    Detect which batch variable is available in the dataframe.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Feature dataframe
    
    Returns:
    --------
    Optional[str]
        Name of available batch variable, or None
    """
    # Prefer source_file, then acquisition_id
    if 'source_file' in data.columns:
        return 'source_file'
    elif 'acquisition_id' in data.columns:
        return 'acquisition_id'
    return None


def validate_batch_correction_inputs(
    data: pd.DataFrame,
    batch_var: str,
    features: List[str]
) -> None:
    """
    Validate inputs for batch correction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Feature dataframe
    batch_var : str
        Batch variable column name
    features : List[str]
        List of feature column names
    
    Raises:
    -------
    ValueError
        If inputs are invalid
    """
    if data is None or data.empty:
        raise ValueError("Dataframe is empty")
    
    if batch_var not in data.columns:
        raise ValueError(f"Batch variable '{batch_var}' not found in dataframe")
    
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Check that batch variable has multiple batches
    unique_batches = data[batch_var].nunique()
    if unique_batches < 2:
        raise ValueError(
            f"Batch variable '{batch_var}' has only {unique_batches} unique value(s). "
            f"At least 2 batches are required for batch correction."
        )
    
    # Check for missing values in features
    for feature in features:
        if data[feature].isna().any():
            raise ValueError(
                f"Feature '{feature}' contains missing values. "
                f"Please handle missing values before batch correction."
            )

