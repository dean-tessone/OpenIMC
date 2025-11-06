"""
Spillover correction for Imaging Mass Cytometry.

CATALYST-like spillover compensation for IMC (Python).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Literal, Optional, Tuple, Dict

try:
    from scipy.optimize import nnls as _scipy_nnls
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

MethodT = Literal["nnls", "pgd"]  # CATALYST uses NNLS; PGD is a fast approximate NNLS.


# ----------------------------- I/O + alignment -----------------------------

def load_spillover(spillover_csv: str) -> pd.DataFrame:
    """
    Load a square spillover matrix CSV with row/col names = channel names.
    Diagonal ~ 1.0; off-diagonals are spill fractions.
    """
    S = pd.read_csv(spillover_csv, index_col=0)
    if S.shape[0] != S.shape[1]:
        raise ValueError("Spillover matrix must be square.")
    if not (S.index.equals(S.columns)):
        # keep, but warn in logs if you have a logger; we rely on names anyway
        pass
    return S.astype(float)


def align_channels(
    X: pd.DataFrame,
    S: pd.DataFrame,
    channel_map: Optional[Dict[str, str]] = None,
    strict: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Align feature matrix X (cells x channels) and spillover S (channels x channels) by names.
    Optionally accept a channel_map to rename columns in X -> S names.
    Returns: (X_aligned, S_sub, mask_present) where mask_present marks channels used.
    """
    Xc = X.copy()
    if channel_map:
        Xc = Xc.rename(columns=channel_map)

    # Intersect channels present in both
    common = [c for c in S.columns if c in Xc.columns]
    missing_in_X = [c for c in S.columns if c not in Xc.columns]
    extra_in_X = [c for c in Xc.columns if c not in S.columns]

    if strict and (missing_in_X or extra_in_X):
        raise ValueError(f"Channel mismatch. Missing in X: {missing_in_X}; Extra in X: {extra_in_X}")

    if len(common) == 0:
        raise ValueError("No overlapping channels between X and spillover matrix.")

    # Subset & reorder
    X_sub = Xc[common]
    S_sub = S.loc[common, common]

    mask_present = np.array([c in common for c in S.columns], dtype=bool)
    return X_sub, S_sub, mask_present


# ----------------------------- Core solvers --------------------------------

def _nnls_batch_scipy(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Row-wise NNLS using SciPy (exact; slower)."""
    N, C = Y.shape
    X = np.empty_like(Y)
    # Column scaling improves conditioning
    colnorm = np.linalg.norm(S, axis=0)
    colnorm[colnorm == 0] = 1.0
    S_scaled = S / colnorm
    for i in range(N):
        z, _ = _scipy_nnls(S_scaled, Y[i])
        X[i] = z / colnorm
    return X


def _nnls_batch_pgd(
    Y: np.ndarray,
    S: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-6,
    init: Literal["zeros", "obs"] = "zeros"
) -> np.ndarray:
    """
    Batched projected gradient descent for NNLS:
      minimize 0.5 || S Z - Y ||_F^2 s.t. Z >= 0
    Very fast, typically close to NNLS.
    """
    N, C = Y.shape
    # Lipschitz constant for gradient: L = ||S||_2^2
    svals = np.linalg.svd(S, compute_uv=False)
    L = (svals[0] ** 2) if svals.size else 1.0
    alpha = 1.0 / (L + 1e-12)

    if init == "zeros":
        Z = np.zeros_like(Y)
    elif init == "obs":
        reg = 1e-6 * np.eye(C)
        Z = (Y @ np.linalg.inv(S + reg).T).clip(min=0.0)
    else:
        raise ValueError("init must be 'zeros' or 'obs'.")

    prev_obj = np.inf
    for k in range(max_iter):
        R = (Z @ S.T) - Y      # (N,C), this is (S Z - Y) in row-space
        grad = R @ S           # ∇ = (S Z - Y) S^T
        Z -= alpha * grad
        np.maximum(Z, 0.0, out=Z)

        if (k & 9) == 9:       # check every ~10 steps
            obj = 0.5 * np.sum(R * R)
            if abs(prev_obj - obj) <= tol * max(prev_obj, 1.0):
                break
            prev_obj = obj

    return Z


def compensate_counts(
    X_counts: pd.DataFrame,
    S: pd.DataFrame,
    method: MethodT = "nnls",
    arcsinh_cofactor: Optional[float] = None,
    channel_map: Optional[Dict[str, str]] = None,
    strict_align: bool = False,
    return_all_channels: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    CATALYST-like compensation on *raw counts*.

    Parameters
    ----------
    X_counts : DataFrame (cells x channels) raw (linear) counts.
    S        : DataFrame (channels x channels) spillover matrix.
    method   : "nnls" (SciPy) or "pgd" (fast NNLS).
    arcsinh_cofactor : if provided, also return arcsinh(comp_counts / cofactor).
    channel_map : optional {X_col_name -> S_channel_name} mapping.
    strict_align : if True, raise on channel mismatches.
    return_all_channels : if True, reattach non-overlap channels (unchanged) to output.

    Returns
    -------
    (comp_counts_df, comp_asinh_df or None)
    """
    # Align channels and get numeric arrays
    X_sub, S_sub, mask_present = align_channels(X_counts, S, channel_map, strict=strict_align)
    Y = X_sub.values.astype(np.float64, copy=False)  # (N, C)
    M = S_sub.values.astype(np.float64, copy=False)  # (C, C)

    if method == "nnls":
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy not available; use method='pgd'.")
        X_comp = _nnls_batch_scipy(Y, M)
    elif method == "pgd":
        X_comp = _nnls_batch_pgd(Y, M, max_iter=300, tol=1e-6, init="obs")
    else:
        raise ValueError("method must be 'nnls' or 'pgd'.")

    # Clamp tiny negatives from numeric noise
    X_comp[X_comp < 0] = 0.0

    comp = pd.DataFrame(X_comp, index=X_sub.index, columns=X_sub.columns)

    if return_all_channels and (len(X_counts.columns) != len(X_sub.columns)):
        # Reattach untouched channels (those not in S)
        untouched = [c for c in X_counts.columns if c not in X_sub.columns]
        for c in untouched:
            comp[c] = X_counts[c].values
        # Preserve original column order
        comp = comp[X_counts.columns]

    comp_asinh = None
    if arcsinh_cofactor is not None:
        c = float(arcsinh_cofactor)
        comp_asinh = comp.copy()
        comp_asinh.loc[:, X_sub.columns] = np.arcsinh(comp.loc[:, X_sub.columns].values / c)

    return comp, comp_asinh


# ----------------------------- Image helper -----------------------------

def compensate_image_counts(
    img_counts: np.ndarray,  # H x W x C (raw counts)
    S: pd.DataFrame,
    channel_order: Iterable[str],
    method: MethodT = "pgd"
) -> np.ndarray:
    """
    Apply compensation in the image domain (counts). Usually for QC; quant should use per-cell tables.
    
    Only channels present in both the image and spillover matrix will be compensated.
    Channels not in the spillover matrix will be left unchanged.
    """
    if img_counts.ndim != 3:
        raise ValueError("img_counts must be HxWxC.")
    H, W, C = img_counts.shape
    channel_order = list(channel_order)
    if len(channel_order) != C:
        raise ValueError("channel_order must list exactly C channel names.")

    # Find channels that are in both the image and spillover matrix
    common_channels = [ch for ch in channel_order if ch in S.columns and ch in S.index]
    
    if len(common_channels) == 0:
        print(f"[spillover] WARNING: No overlapping channels between image and spillover matrix. Skipping compensation.")
        return img_counts.astype(np.float32)
    
    # Create output array (copy to avoid modifying input)
    result = img_counts.astype(np.float64, copy=True)
    
    # Get indices of common channels in the original order
    common_indices = [channel_order.index(ch) for ch in common_channels]
    
    # Align to S for common channels only
    S_sub = S.loc[common_channels, common_channels]
    Y = img_counts[:, :, common_indices].reshape(-1, len(common_channels)).astype(np.float64, copy=False)
    M = S_sub.values.astype(np.float64, copy=False)

    if method == "nnls":
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy not available; use method='pgd'.")
        X_comp = _nnls_batch_scipy(Y, M)
    elif method == "pgd":
        X_comp = _nnls_batch_pgd(Y, M, max_iter=200, tol=1e-6, init="obs")
    else:
        raise ValueError("method must be 'nnls' or 'pgd'.")

    X_comp[X_comp < 0] = 0.0
    
    # Write compensated values back to result array
    result[:, :, common_indices] = X_comp.reshape(H, W, len(common_channels))
    
    return result.astype(np.float32)

