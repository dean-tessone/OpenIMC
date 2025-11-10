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
    from harmonypy import run_harmony
    _HAVE_HARMONY = True
except ImportError:
    _HAVE_HARMONY = False

try:
    import bbknn
    _HAVE_BBKNN = True
except ImportError:
    _HAVE_BBKNN = False


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
    # This preserves ALL columns from the original data - only selected features will be corrected
    corrected_data = data.copy()
    
    # Validate input data before processing
    # Check for features with zero or very low variance (these can cause issues with Combat)
    features_to_correct = []
    features_to_skip = []
    
    for feature in features:
        feature_values = data[feature].values
        # Check for constant or near-constant features
        if np.var(feature_values) < 1e-10:
            features_to_skip.append(feature)
            continue
        # Check for features with too many zeros (can cause division issues)
        zero_ratio = np.sum(feature_values == 0) / len(feature_values)
        if zero_ratio > 0.95:  # More than 95% zeros
            features_to_skip.append(feature)
            continue
        features_to_correct.append(feature)
    
    if not features_to_correct:
        raise ValueError(
            "No valid features to correct. All selected features have zero variance or are mostly zeros. "
            "Please select different features or check your data."
        )
    
    if features_to_skip:
        import warnings
        warnings.warn(
            f"Skipping {len(features_to_skip)} features with zero variance or too many zeros: "
            f"{', '.join(features_to_skip[:5])}{'...' if len(features_to_skip) > 5 else ''}"
        )
    
    # Extract feature matrix as DataFrame (Combat expects features x samples as DataFrame)
    # Select only valid features
    feature_df = data[features_to_correct].T  # Transpose to features x samples
    
    # Get batch labels
    batch_labels = data[batch_var].astype(str).values
    
    # Prepare covariates if provided
    covar_df = None
    if covariates:
        # Convert categorical covariates to numeric
        covar_data = data[covariates].copy()
        for col in covariates:
            if covar_data[col].dtype == 'object':
                covar_data[col] = pd.Categorical(covar_data[col]).codes
        covar_df = covar_data
    
    # Apply Combat correction
    try:
        # pycombat expects DataFrame (features x samples), batch labels, and optional covar DataFrame
        corrected_df = pycombat(feature_df, batch_labels, covar=covar_df)
        
        # Validate that corrected_df has the expected structure
        if corrected_df is None:
            raise RuntimeError("pycombat returned None")
        
        # Check if corrected_df is a DataFrame
        if not isinstance(corrected_df, pd.DataFrame):
            raise RuntimeError(f"pycombat returned unexpected type: {type(corrected_df)}")
        
        # Check dimensions
        if corrected_df.shape != feature_df.shape:
            raise RuntimeError(
                f"Shape mismatch: expected {feature_df.shape}, got {corrected_df.shape}"
            )
        
        # Check for NaN or infinite values (warn but don't fail - these might be valid in some cases)
        nan_count = corrected_df.isna().sum().sum()
        inf_count = np.isinf(corrected_df.values).sum()
        if nan_count > 0:
            import warnings
            warnings.warn(
                f"pycombat returned {nan_count} NaN values. "
                f"This may indicate issues with the input data or batch structure."
            )
        if inf_count > 0:
            import warnings
            warnings.warn(
                f"pycombat returned {inf_count} infinite values. "
                f"This may indicate issues with the input data or batch structure."
            )
            
    except Exception as e:
        raise RuntimeError(f"Combat correction failed: {str(e)}")
    
    # Update corrected features in dataframe
    # Transpose back to samples x features
    # The corrected_df has features as rows (index) and samples as columns
    # After transpose, rows are samples and columns are features (in the same order as original)
    corrected_transposed = corrected_df.T
    
    # Validate the transposed structure
    if corrected_transposed.shape[0] != len(data):
        raise RuntimeError(
            f"Row count mismatch after transpose: expected {len(data)}, "
            f"got {corrected_transposed.shape[0]}"
        )
    
    # Ensure the feature columns are in the correct order
    # The index of corrected_df should match the feature names
    # Use the index of corrected_df to match features (since features are rows in corrected_df)
    for i, feature in enumerate(features_to_correct):
        # The feature should be in the index of corrected_df (before transpose)
        # After transpose, it should be in the columns
        if feature in corrected_transposed.columns:
            values = corrected_transposed[feature].values.copy()  # Make a copy to ensure we have the values
            # Validate values before assignment
            if len(values) != len(data):
                raise RuntimeError(
                    f"Value length mismatch for feature '{feature}': "
                    f"expected {len(data)}, got {len(values)}"
                )
            # Direct assignment using .loc to ensure proper indexing
            corrected_data.loc[:, feature] = values
        elif feature in corrected_df.index:
            # Feature is in the index of corrected_df, get it by index name
            values = corrected_df.loc[feature].values.copy()
            if len(values) != len(data):
                raise RuntimeError(
                    f"Value length mismatch for feature '{feature}': "
                    f"expected {len(data)}, got {len(values)}"
                )
            corrected_data.loc[:, feature] = values
        else:
            # Fallback: use position if column name doesn't match
            # This handles cases where column names might have changed
            if i < len(corrected_transposed.columns):
                values = corrected_transposed.iloc[:, i].values.copy()
                if len(values) != len(data):
                    raise RuntimeError(
                        f"Value length mismatch for feature '{feature}' (by position): "
                        f"expected {len(data)}, got {len(values)}"
                    )
                corrected_data.loc[:, feature] = values
            else:
                raise ValueError(
                    f"Feature '{feature}' not found in corrected data. "
                    f"Available columns in transposed: {list(corrected_transposed.columns)}. "
                    f"Available index in corrected_df: {list(corrected_df.index)}. "
                    f"Expected {len(features)} features, got {len(corrected_transposed.columns)} columns."
                )
    
    # Handle NaN values: replace with original values if correction produced NaN
    # This can happen when Combat encounters numerical issues
    for feature in features_to_correct:
        if corrected_data[feature].isna().any():
            nan_mask = corrected_data[feature].isna()
            nan_count = nan_mask.sum()
            
            # If ALL values are NaN, keep the entire feature as original (correction failed)
            if nan_count == len(corrected_data):
                import warnings
                warnings.warn(
                    f"Feature '{feature}' produced all NaN values after correction. "
                    f"Keeping original values for this feature. "
                    f"This indicates numerical issues with this feature or batch structure."
                )
                corrected_data[feature] = data[feature].values
            else:
                # Replace only NaN values with original values
                corrected_data.loc[nan_mask, feature] = data.loc[nan_mask, feature].values
                import warnings
                warnings.warn(
                    f"Feature '{feature}' had {nan_count} NaN values after correction. "
                    f"These have been replaced with original values. "
                    f"This may indicate numerical issues with this feature or batch structure."
                )
    
    # Keep skipped features with their original values (they weren't corrected)
    for feature in features_to_skip:
        # These features were not corrected, so they keep their original values
        pass
    
    # Note: All other features (not in the 'features' list) are automatically preserved
    # in corrected_data because we started with data.copy(). They remain unchanged.
    
    # Validate that all original columns are preserved
    if set(corrected_data.columns) != set(data.columns):
        missing_cols = set(data.columns) - set(corrected_data.columns)
        raise RuntimeError(
            f"Some columns were lost during batch correction: {missing_cols}. "
            f"This should not happen."
        )
    
    # Final validation: check that corrected features have valid values
    # (NaN values have already been handled above)
    for feature in features_to_correct:
        if corrected_data[feature].isna().any():
            # This shouldn't happen after our NaN handling, but check anyway
            remaining_nan = corrected_data[feature].isna().sum()
            if remaining_nan > 0:
                import warnings
                warnings.warn(
                    f"Feature '{feature}' still has {remaining_nan} NaN values after handling. "
                    f"This may indicate an issue with the data."
                )
        if (corrected_data[feature] == 0).all():
            # This might be valid, but log a warning
            import warnings
            warnings.warn(f"All values for feature '{feature}' are zero after correction")
    
    return corrected_data


def apply_harmony_correction(
    data: pd.DataFrame,
    batch_var: str,
    features: List[str],
    n_clusters: int = 30,
    sigma: float = 0.1,
    theta: float = 2.0,
    lambda_reg: float = 1.0,
    max_iter: int = 10,
    pca_variance: float = 0.9
) -> pd.DataFrame:
    """
    Apply Harmony batch correction to feature data.
    
    Harmony operates in PCA space to reduce dimensionality and noise.
    The corrected data is then transformed back to the original feature space.
    
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
    pca_variance : float
        Proportion of variance to retain in PCA (default: 0.9, i.e., 90%)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with corrected features
    """
    if not _HAVE_HARMONY:
        raise ImportError("Harmony is not installed. Install with: pip install harmonypy")
    
    # Import sklearn for PCA
    from sklearn.decomposition import PCA
    
    # Create a copy to avoid modifying original
    # This preserves ALL columns from the original data - only selected features will be corrected
    corrected_data = data.copy()
    
    # Extract feature matrix (samples x features)
    feature_matrix = data[features].values
    
    # Perform PCA on features
    # Determine number of components needed to retain specified variance
    try:
        # First, fit PCA with all components to get explained variance
        pca_full = PCA()
        pca_full.fit(feature_matrix)
        
        # Calculate cumulative explained variance
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Find number of components needed to retain specified variance
        n_components = np.searchsorted(cumsum_variance, pca_variance) + 1
        
        # Ensure we have at least 1 component and at most min(n_samples-1, n_features)
        n_components = max(1, min(n_components, feature_matrix.shape[0] - 1, feature_matrix.shape[1]))
        
        # Fit PCA with the determined number of components
        pca = PCA(n_components=n_components, random_state=0)
        pca_data = pca.fit_transform(feature_matrix)
        
        # Create metadata DataFrame with batch variable
        meta_data = pd.DataFrame({batch_var: data[batch_var].astype(str)})
        
        # Apply Harmony correction in PCA space
        harmony_result = run_harmony(
            pca_data,
            meta_data,
            vars_use=[batch_var],
            nclust=n_clusters,
            sigma=sigma,
            theta=theta,
            lamb=lambda_reg,
            max_iter_harmony=max_iter
        )
        
        # run_harmony returns a Harmony object with Z_corr attribute
        # Z_corr is the corrected data in PCA space (samples x n_components)
        corrected_pca = harmony_result.Z_corr.T
        
        # Transform back to feature space
        corrected_matrix = pca.inverse_transform(corrected_pca)
        
    except Exception as e:
        raise RuntimeError(f"Harmony correction failed: {str(e)}")
    
    # Update corrected features in dataframe
    for i, feature in enumerate(features):
        corrected_data[feature] = corrected_matrix[:, i]
    
    # Note: All other features (not in the 'features' list) are automatically preserved
    # in corrected_data because we started with data.copy(). They remain unchanged.
    
    # Validate that all original columns are preserved
    if set(corrected_data.columns) != set(data.columns):
        missing_cols = set(data.columns) - set(corrected_data.columns)
        raise RuntimeError(
            f"Some columns were lost during batch correction: {missing_cols}. "
            f"This should not happen."
        )
    
    return corrected_data


def apply_bbknn_correction(
    data: pd.DataFrame,
    batch_var: str,
    features: List[str],
    n_pcs: int = 50,
    neighbors_within_batch: int = 3,
    trim: int = 0,
    n_trees: int = 10,
    use_annoy: bool = True
) -> pd.DataFrame:
    """
    Apply BBKNN (Batch Balanced KNN) batch correction to feature data.
    
    BBKNN corrects batch effects by computing a balanced k-nearest neighbor graph
    across batches. This function performs PCA on the features, applies BBKNN correction
    in PCA space, and transforms back to feature space.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Feature dataframe with batch variable and features
    batch_var : str
        Column name containing batch identifiers
    features : List[str]
        List of feature column names to correct
    n_pcs : int
        Number of principal components to use (default: 50)
    neighbors_within_batch : int
        Number of neighbors to use within each batch (default: 3)
    trim : int
        Trim parameter for BBKNN (default: 0)
    n_trees : int
        Number of trees for approximate nearest neighbor search (default: 10)
    use_annoy : bool
        Whether to use Annoy for approximate nearest neighbor search (default: True)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with corrected features
    """
    if not _HAVE_BBKNN:
        raise ImportError("BBKNN is not installed. Install with: pip install bbknn")
    
    # Import sklearn for PCA
    from sklearn.decomposition import PCA
    
    # Create a copy to avoid modifying original
    corrected_data = data.copy()
    
    # Extract feature matrix (samples x features)
    feature_matrix = data[features].values
    
    # Get batch labels
    batch_labels = data[batch_var].astype(str).values
    
    # Perform PCA on features
    # Limit n_pcs to min(n_samples, n_features)
    n_pcs = min(n_pcs, feature_matrix.shape[0] - 1, feature_matrix.shape[1])
    
    try:
        pca = PCA(n_components=n_pcs, random_state=0)
        pca_data = pca.fit_transform(feature_matrix)
        
        # Apply BBKNN correction
        # BBKNN works by computing a balanced k-nearest neighbor graph
        # We'll use it to get corrected PCA coordinates
        # Create a temporary AnnData-like structure for BBKNN
        # BBKNN expects data in PCA space and modifies the neighborhood graph
        # We'll use it to compute corrected coordinates
        
        # For BBKNN, we need to work with the PCA space
        # BBKNN's main function modifies the neighborhood graph, but we can
        # use it to get corrected embeddings by computing a corrected graph
        # and then using it to smooth the PCA coordinates
        
        # Create a simple approach: use BBKNN to compute corrected graph,
        # then use graph-based smoothing to correct the PCA coordinates
        import anndata as ad
        
        # Create AnnData object
        adata = ad.AnnData(pca_data)
        adata.obs[batch_var] = batch_labels
        
        # Apply BBKNN
        bbknn.bbknn(
            adata,
            batch_key=batch_var,
            neighbors_within_batch=neighbors_within_batch,
            trim=trim,
            n_trees=n_trees,
            use_annoy=use_annoy
        )
        
        # Get corrected PCA coordinates by using the graph for smoothing
        # We'll use the graph Laplacian to smooth the original PCA coordinates
        # This applies graph-based smoothing that respects batch boundaries
        
        # Get the connectivity matrix
        connectivities = adata.obsp['connectivities']
        
        # Normalize the connectivity matrix
        row_sums = np.array(connectivities.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_connectivities = connectivities.multiply(1.0 / row_sums[:, np.newaxis])
        
        # Smooth the PCA coordinates using the graph
        # This applies a graph-based smoothing that respects batch boundaries
        corrected_pca = normalized_connectivities.dot(pca_data)
        
        # Transform back to feature space
        # Use the inverse PCA transform
        corrected_matrix = pca.inverse_transform(corrected_pca)
        
    except ImportError as e:
        # If anndata is not available
        raise ImportError(
            f"BBKNN requires anndata package. "
            f"Install with: pip install anndata bbknn. "
            f"Original error: {str(e)}"
        )
    except Exception as e:
        raise RuntimeError(f"BBKNN correction failed: {str(e)}")
    
    # Update corrected features in dataframe
    for i, feature in enumerate(features):
        corrected_data[feature] = corrected_matrix[:, i]
    
    # Validate that all original columns are preserved
    if set(corrected_data.columns) != set(data.columns):
        missing_cols = set(data.columns) - set(corrected_data.columns)
        raise RuntimeError(
            f"Some columns were lost during batch correction: {missing_cols}. "
            f"This should not happen."
        )
    
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

