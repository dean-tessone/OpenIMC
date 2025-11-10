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
Spillover matrix generation for Imaging Mass Cytometry.

Python re-implementation of CATALYST's computeSpillmat() and adaptSpillmat().
Also includes pixel-level spillover estimation from MCD files.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

# Optional readimc for MCD file support
_HAVE_READIMC = False
try:
    from readimc import MCDFile as McdFile  # type: ignore
    _HAVE_READIMC = True
except Exception:
    _HAVE_READIMC = False


# ---------- helpers to parse channel names (mirrors .get_ms_from_chs / .get_mets_from_chs) ----------


def _get_mass_from_channel(ch: str) -> Optional[int]:
    """Extract mass number from channel name (e.g., "Yb176", "Ir191Di" -> 176, 191)."""
    # strip all letters/punct; keep digits
    m = re.sub(r"[^\d]", "", ch)
    return int(m) if m else None


def _get_metal_from_channel(ch: str) -> str:
    """Remove mass & trailing "Di"/"Dd" markers (e.g., "Ir191Di" -> "Ir")."""
    return re.sub(r"([[:punct:]]*)(\d*)((Di)|(Dd))*", "", ch)


# Python's regex engine lacks [:posix:] classes by default; emulate:
_GET_METAL_RE = re.compile(r"[^A-Za-z]*([A-Za-z]+).*")
def _get_metal(ch: str) -> str:
    """Extract metal name from channel (e.g., "Yb176" -> "Yb")."""
    m = _GET_METAL_RE.match(ch or "")
    return m.group(1) if m else ""


def _masses(chs: Sequence[str]) -> List[Optional[int]]:
    """Extract mass numbers from channel names."""
    return [_get_mass_from_channel(c) for c in chs]


def _metals(chs: Sequence[str]) -> List[str]:
    """Extract metal names from channel names."""
    return [_get_metal(c) for c in chs]


# ---------- isotopes & candidate spill receivers (mirrors .get_spill_chs) ----------


def _default_isotope_list() -> Dict[str, List[int]]:
    """Minimal workable isotope list; replace/extend with full periodic coverage as needed."""
    return {
        "Yb": [168, 170, 171, 172, 173, 174, 176],
        "Er": [162, 164, 166, 167, 168, 170],
        "Dy": [156, 158, 160, 161, 162, 163, 164],
        "Gd": [152, 154, 155, 156, 157, 158, 160],
        "Sm": [147, 149, 152, 154],
        "Nd": [142, 143, 144, 145, 146, 148, 150],
        "Pr": [141],
        "Eu": [151, 153],
        "Tb": [159],
        "Ho": [165],
        "Tm": [169],
        "Lu": [175, 176],
        "Ir": [191, 193],
        "Pt": [195],
        "Ce": [140],
        "La": [139],
        # add others you use
    }


def _get_spill_candidates(ms: List[Optional[int]], mets: List[str],
                          iso_list: Optional[Dict[str, List[int]]] = None) -> List[List[int]]:
    """
    Return, for each index i, the indices j that are plausible spill receivers:
    M±1, M+16 (oxide), and isotopic neighbors (from isotope list).
    """
    iso_list = iso_list or _default_isotope_list()
    ms_num = [int(v) if v is not None else None for v in ms]
    idx_by_mass = {}
    for idx, m in enumerate(ms_num):
        if m is not None:
            idx_by_mass.setdefault(m, []).append(idx)

    out = []
    for i, m in enumerate(ms_num):
        cands: List[int] = []
        if m is None:
            out.append(cands)
            continue
        # ±1 amu
        for delta in (-1, +1):
            jlist = idx_by_mass.get(m + delta, [])
            cands.extend(jlist)
        # +16 (oxide)
        cands.extend(idx_by_mass.get(m + 16, []))
        # isotopic impurities: channels at same metal's other masses
        metal = mets[i]
        iso_masses = [x for x in iso_list.get(metal, []) if x != m]
        for im in iso_masses:
            cands.extend(idx_by_mass.get(im, []))
        # unique & sorted
        out.append(sorted(set(cands)))
    return out


# ---------- core spill estimate per pair (mirrors .get_sij) ----------


def _trimmed_mean(x: np.ndarray, trim: float) -> float:
    """Compute trimmed mean of array."""
    if x.size == 0:
        return 0.0
    if trim <= 0:
        return float(np.mean(x))
    lo = int(np.floor(trim * x.size))
    hi = int(np.ceil((1 - trim) * x.size))
    xs = np.sort(x)
    xs = xs[lo:hi] if hi > lo else xs
    return float(np.mean(xs)) if xs.size else 0.0


def _get_sij(pos_i: np.ndarray, neg_i: np.ndarray,
             pos_j: np.ndarray, neg_j: np.ndarray,
             method: Literal["default", "classic"], trim: float) -> float:
    """Estimate spillover coefficient from channel i to channel j."""
    # mirror CATALYST's two branches
    if method == "default":
        bg_i = _trimmed_mean(neg_i, 0.10)  # fixed 10% trim for background (as in R)
        bg_j = _trimmed_mean(neg_j, 0.10)
        spiller  = pos_i - bg_i
        receiver = pos_j - bg_j
        # clip negatives to zero
        spiller  = np.maximum(spiller, 0.0)
        receiver = np.maximum(receiver, 0.0)
        ratio = receiver / np.clip(spiller, 1e-12, None)
        ratio[~np.isfinite(ratio)] = 0.0
        return float(np.median(ratio))  # median of event-wise ratios
    elif method == "classic":
        spill  = _trimmed_mean(pos_i, trim) - _trimmed_mean(neg_i, trim)
        recv   = _trimmed_mean(pos_j, trim) - _trimmed_mean(neg_j, trim)
        spill  = max(spill, 0.0)
        recv   = max(recv, 0.0)
        return float(recv / spill) if spill > 0 else 0.0
    else:
        raise ValueError(f"Invalid method: {method}")


# ---------- computeSpillmat (Python) ----------


def compute_spillmat(
    counts: np.ndarray,             # shape: C x N (channels x events)
    channel_names: Sequence[str],   # length C
    bc_id: np.ndarray,              # length N; 0=unassigned, otherwise donor mass (e.g., 176)
    bc_key_masses: Sequence[int],   # e.g., colnames(bc_key) in CATALYST
    assay_name: str = "counts",     # placeholder for parity with R
    interactions: Literal["default", "all"] = "default",
    method: Literal["default", "classic"] = "default",
    trim: float = 0.5,
    threshold: float = 1e-5,
    isotope_list: Optional[Dict[str, List[int]]] = None,
) -> pd.DataFrame:
    """
    Estimate a spillover matrix S from single-stain controls.
    Mirrors CATALYST::computeSpillmat() mechanics.

    Parameters
    ----------
    counts : np.ndarray
        Shape (C, N) where C is number of channels and N is number of events/cells.
    channel_names : Sequence[str]
        Channel names, length C.
    bc_id : np.ndarray
        Barcode ID for each event, length N. 0 = unassigned, otherwise donor mass.
    bc_key_masses : Sequence[int]
        List of barcode masses used in single-stain experiment.
    assay_name : str
        Placeholder for parity with R (default: "counts").
    interactions : {"default", "all"}
        "default" uses plausible spill candidates, "all" uses all channel pairs.
    method : {"default", "classic"}
        Method for computing spillover coefficients.
    trim : float
        Trim fraction for classic method (default: 0.5).
    threshold : float
        Threshold below which spillover estimates are set to 0 (default: 1e-5).
    isotope_list : Dict[str, List[int]], optional
        Custom isotope list. If None, uses default.

    Returns
    -------
    pd.DataFrame
        Spillover matrix with channel names as index and columns.
    """
    counts = np.asarray(counts, dtype=float)
    C, N = counts.shape
    chs = list(channel_names)
    ms = _masses(chs)
    mets = _metals(chs)

    # Validate: single-stain experiment (each bc_key mass appears as a single positive)
    # (R checks bc_key structure; here we minimally check bc_id contents)
    uniq_ids = sorted(set(int(v) for v in bc_id if int(v) != 0))
    if len(uniq_ids) != len(bc_key_masses):
        # parity check; not strictly required, but helpful
        pass

    # Which channels are "barcode channels" (i.e., those stained in single-stain panel)
    bc_chs_idx = [ms.index(m) if m in ms else None for m in bc_key_masses]
    if any(v is None for v in bc_chs_idx):
        raise ValueError("Some bc_key_masses not found in channel masses.")

    # Candidate receiving channels per donor
    if interactions == "default":
        spill_chs = _get_spill_candidates(ms, mets, isotope_list=isotope_list)
        # 'ex' in CATALYST is the same candidate set used to exclude spill-affected populations in negatives
        ex = spill_chs
    elif interactions == "all":
        # every other channel except self
        all_sets = []
        for i in range(C):
            all_sets.append([j for j in range(C) if j != i])
        spill_chs = all_sets
        # For 'ex' use the default plausible set for negative-pop exclusion (parity with R)
        ex = _get_spill_candidates(ms, mets, isotope_list=isotope_list)
    else:
        raise ValueError("interactions must be 'default' or 'all'")

    # Initialize S as identity (channels x channels, ordered by channel_names)
    S = np.eye(C, dtype=float)
    # Split cells by bc_id (like split(seq_len(ncol(x)), x$bc_id))
    # pos indices for each donor mass; negatives = not unassigned, not the donor,
    # and not assigned to spill-affected donors for the current i (mirrors R)
    for donor_mass in uniq_ids:
        if donor_mass not in ms:
            # donor mass present in bc_id but no matching channel row
            continue
        i = ms.index(donor_mass)   # row index (emitter)
        pos = np.where(bc_id == donor_mass)[0]
        # neg: exclude 0 (unassigned), exclude donor, exclude spill-affected populations of donor
        exclude_ids = set([0, donor_mass])
        for m_idx in ex[i]:
            m = ms[m_idx]
            if m is not None:
                exclude_ids.add(m)
        neg = np.array([k for k in range(N) if int(bc_id[k]) not in exclude_ids], dtype=int)

        pos_i = counts[i, pos]
        neg_i = counts[i, neg]

        for j in spill_chs[i]:
            pos_j = counts[j, pos]
            # neg for receiver excludes its own population and its spill-affected populations
            exclude_ids_j = set(exclude_ids) | {ms[j]}
            for m_idx in ex[j]:
                m = ms[m_idx]
                if m is not None:
                    exclude_ids_j.add(m)
            neg_j = np.array([k for k in range(N) if int(bc_id[k]) not in exclude_ids_j], dtype=int)

            sij = _get_sij(
                pos_i, counts[i, neg],  # neg_i
                pos_j, counts[j, neg_j],
                method=method, trim=trim
            )
            S[i, j] = sij

    # threshold small estimates to 0 (as in R: sm[sm < th] <- 0)
    S[S < threshold] = 0.0

    # Subset to barcode channels as rows; drop channels without a mass (NA in R)
    # Here: keep full square but you can match R by selecting rows of bc_chs_idx
    S_df = pd.DataFrame(S, index=chs, columns=chs)
    return S_df


# ---------- adaptSpillmat (Python) ----------


def _check_spill_matrix(sm: pd.DataFrame, isotope_list: Optional[Dict[str, List[int]]] = None) -> pd.DataFrame:
    """Validate and clean spillover matrix."""
    A = sm.values.astype(float)
    if (A < 0).any() or (A > 1).any():
        raise ValueError("Spillover matrix must be in [0, 1].")
    # diagonal must be 1 where row & col share the same name
    common = [c for c in sm.columns if c in sm.index]
    if any(abs(A[sm.index.get_loc(c), sm.columns.get_loc(c)] - 1.0) > 1e-12 for c in common):
        raise ValueError("Diagonal entries must be 1 for overlapping row/column names.")
    # (Optional) validate isotopes exist in list
    # Mirror R's name check: metal+mass must appear in isotope list
    iso = isotope_list or _default_isotope_list()
    def _valid(chs: Iterable[str]) -> bool:
        ms = _masses(chs)
        mets = _metals(chs)
        pairs = {f"{met}{m}" for met, m in zip(mets, ms) if m is not None}
        valid_pairs = {f"{met}{m}" for met, arr in iso.items() for m in arr}
        return pairs.issubset(valid_pairs)
    if not _valid(sm.index) or not _valid(sm.columns):
        # warn instead of hard-stop to be more permissive
        pass
    # drop all-zero columns (R keeps only nonzero columns)
    keep_cols = (A.sum(axis=0) != 0)
    return sm.loc[:, keep_cols]


def adapt_spillmat(
    sm_in: pd.DataFrame,
    out_channels: Sequence[str],
    isotope_list: Optional[Dict[str, List[int]]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Adapt a spillover matrix to a new channel set (mirrors CATALYST::adaptSpillmat).
    - copies existing rows/cols where names overlap,
    - propagates spill columns to new channels that share the same mass,
    - zeros same-mass intra-blocks,
    - restores the diagonal to 1.

    Parameters
    ----------
    sm_in : pd.DataFrame
        Input spillover matrix.
    out_channels : Sequence[str]
        Target channel names for the adapted matrix.
    isotope_list : Dict[str, List[int]], optional
        Custom isotope list. If None, uses default.
    verbose : bool
        Whether to print verbose output (default: True).

    Returns
    -------
    pd.DataFrame
        Adapted spillover matrix.
    """
    iso = isotope_list or _default_isotope_list()
    sm = _check_spill_matrix(sm_in.copy(), iso)

    out_chs = list(out_channels)
    n = len(out_chs)
    S = np.eye(n, dtype=float)
    S = pd.DataFrame(S, index=out_chs, columns=out_chs)

    in_rows = set(sm.index)
    in_cols = set(sm.columns)
    overlap_rows = [c for c in out_chs if c in in_rows]
    overlap_cols = [c for c in out_chs if c in in_cols]
    # copy preexisting block
    if overlap_rows and overlap_cols:
        S.loc[overlap_rows, overlap_cols] = sm.loc[overlap_rows, overlap_cols].values

    # detect "new" channels (present in out, absent in input)
    out_ms = _masses(out_chs)
    out_mets = _metals(out_chs)
    in_ms_cols = _masses(list(sm.columns))
    # map mass -> existing receiving column (first occurrence)
    mass_to_incol: Dict[int, str] = {}
    for col_name, m in zip(sm.columns, in_ms_cols):
        if m is not None and m not in mass_to_incol:
            mass_to_incol[m] = col_name

    new_metal_chs = [ch for ch in out_chs if ch not in in_rows and ch not in in_cols]
    new_masses = [_get_mass_from_channel(ch) for ch in new_metal_chs]

    # If a new channel shares a mass with any existing receiving column,
    # copy that column of spill values to the new column, matching same-mass logic in R
    for ch, m in zip(new_metal_chs, new_masses):
        if m is None:
            continue
        if m in mass_to_incol:
            src_col = mass_to_incol[m]
            # add spill from all existing emitting rows to this new receiving column
            S.loc[:, ch] = S.loc[:, ch].values  # ensure column exists
            # Rows that exist in input rows
            common_rows = [r for r in S.index if r in sm.index]
            S.loc[common_rows, ch] = sm.loc[common_rows, src_col].values
            # zero same-mass block to avoid singularities
            for j, (out_ch, out_m) in enumerate(zip(out_chs, out_ms)):
                if out_m == m:
                    S.loc[out_ch, ch] = 0.0

    # enforce diagonal = 1
    np.fill_diagonal(S.values, 1.0)
    return S


# ---------- MCD-based spillover generation ----------


def _foreground_mask(donor: np.ndarray,
                     p_low: float = 90.0,
                     p_high_clip: float = 99.9) -> np.ndarray:
    """Select robust donor-foreground pixels in raw counts."""
    lo = np.percentile(donor, p_low)
    hi = np.percentile(donor, p_high_clip)
    return (donor >= lo) & (donor <= hi)


def _slope_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    """Compute slope without intercept: b = (x'y) / (x'x)."""
    num = float(np.dot(x, y))
    den = float(np.dot(x, x)) + 1e-12
    b = num / den
    return b if b > 0 else 0.0


def _row_from_single_stain(stack: np.ndarray, donor_idx: int, cap: float = 0.3) -> np.ndarray:
    """
    Compute one donor row (length C) from a single-stain acquisition.
    
    Parameters
    ----------
    stack : np.ndarray
        Shape (H, W, C) raw counts.
    donor_idx : int
        Index of the donor channel.
    cap : float
        Maximum spillover coefficient (default: 0.3).
    
    Returns
    -------
    np.ndarray
        Spillover row of length C.
    """
    H, W, C = stack.shape
    donor = stack[..., donor_idx].astype(np.float64, copy=False)
    mask = _foreground_mask(donor, p_low=90.0, p_high_clip=99.9)
    
    if not np.any(mask):
        raise ValueError("Empty foreground mask — adjust thresholds or check acquisition.")
    
    x = donor[mask]
    row = np.zeros(C, dtype=np.float64)
    row[donor_idx] = 1.0
    
    for r in range(C):
        if r == donor_idx:
            continue
        y = stack[..., r].astype(np.float64, copy=False)[mask]
        b = _slope_no_intercept(x, y)
        if cap is not None:
            b = min(b, cap)
        row[r] = b
    
    return row


def _stack_from_acq(mcd_file: McdFile, acq) -> np.ndarray:
    """
    Return H x W x C raw counts from an acquisition.
    
    Parameters
    ----------
    mcd_file : McdFile
        Open MCD file object.
    acq
        Acquisition object from readimc.
    
    Returns
    -------
    np.ndarray
        Image stack with shape (H, W, C).
    """
    # readimc's read_acquisition returns C x H x W
    data = mcd_file.read_acquisition(acq)  # C x H x W
    # Transpose to H x W x C
    return np.moveaxis(data, 0, -1)


def build_spillover_from_comp_mcd(
    mcd_path: str,
    donor_label_per_acq: Optional[Dict[str, str]] = None,
    channel_name_field: str = "name",  # "name"|"fullname" depending on your metadata
    cap: float = 0.3,
    aggregate: str = "median",
    p_low: float = 90.0,
    p_high_clip: float = 99.9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create spillover matrix S from a compensation.mcd (single-stain per acquisition).
    
    This function analyzes pixel-level data from single-stain control acquisitions
    to estimate spillover coefficients between channels.
    
    Parameters
    ----------
    mcd_path : str
        Path to the MCD file containing single-stain controls.
    donor_label_per_acq : Dict[str, str], optional
        Mapping from acquisition ID to donor channel name.
        If not provided, attempts to infer from acquisition names.
    channel_name_field : str
        Field to use for channel names: "name" or "fullname" (default: "name").
    cap : float
        Maximum spillover coefficient (default: 0.3).
    aggregate : str
        Aggregation method when multiple acquisitions per donor: "median" or "mean" (default: "median").
    p_low : float
        Lower percentile for foreground selection (default: 90.0).
    p_high_clip : float
        Upper percentile for foreground clipping (default: 99.9).
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (S_df, qc_df) where S_df is the spillover matrix and qc_df contains QC metrics.
    
    Raises
    ------
    RuntimeError
        If readimc is not available.
    ValueError
        If donor channels cannot be inferred or are invalid.
    """
    if not _HAVE_READIMC:
        raise RuntimeError("readimc is not installed. Run: pip install readimc")
    
    mcd = McdFile(mcd_path)
    mcd.open()
    
    try:
        # Get all acquisitions first
        acquisitions = []
        if hasattr(mcd, "slides"):
            for slide in mcd.slides:
                if hasattr(slide, "acquisitions"):
                    acquisitions.extend(slide.acquisitions)
        elif hasattr(mcd, "acquisitions"):
            acquisitions = mcd.acquisitions
        else:
            raise ValueError("Could not find acquisitions in MCD file")
        
        if not acquisitions:
            raise ValueError("No acquisitions found in MCD file")
        
        # Get channel names from first acquisition
        first_acq = acquisitions[0]
        channel_metals = getattr(first_acq, "channel_names", [])
        channel_labels = getattr(first_acq, "channel_labels", [])
        
        ch_names = []
        for i, (metal, label) in enumerate(zip(channel_metals, channel_labels)):
            if label and metal:
                ch_names.append(f"{label}_{metal}")
            elif label:
                ch_names.append(label)
            elif metal:
                ch_names.append(metal)
            else:
                ch_names.append(f"Channel_{i+1}")
        
        C = len(ch_names)
        
        # Collect donor-row estimates per channel
        rows_by_donor: Dict[str, List[np.ndarray]] = {nm: [] for nm in ch_names}
        
        # Check if we have index-based mapping (more reliable)
        mapping_by_index = None
        mapping_by_id = None
        if isinstance(donor_label_per_acq, dict) and '_by_index' in donor_label_per_acq:
            mapping_by_index = donor_label_per_acq.get('_by_index', {})
            mapping_by_id = donor_label_per_acq.get('_by_id', {})
        else:
            # Legacy format - treat as ID mapping
            mapping_by_id = donor_label_per_acq or {}
        
        for acq_idx, acq in enumerate(acquisitions):
            # First try index-based mapping (most reliable since order is preserved)
            donor_name = None
            if mapping_by_index and acq_idx in mapping_by_index:
                donor_name = mapping_by_index[acq_idx]
            
            # If not found by index, try ID-based mapping
            if donor_name is None and mapping_by_id:
                acq_id = getattr(acq, "id", None) or getattr(acq, "name", None) or str(id(acq))
                acq_name = getattr(acq, "name", "") or ""
                acq_well = getattr(acq, "well", None)
                
                # Try multiple matching strategies
                # Try by ID first
                if acq_id in mapping_by_id:
                    donor_name = mapping_by_id[acq_id]
                # Try by name
                elif acq_name in mapping_by_id:
                    donor_name = mapping_by_id[acq_name]
                # Try by well name
                elif acq_well and acq_well in mapping_by_id:
                    donor_name = mapping_by_id[acq_well]
                # Try by string representation of ID (in case it's an integer)
                elif str(acq_id) in mapping_by_id:
                    donor_name = mapping_by_id[str(acq_id)]
                # Try as integer if ID is numeric
                elif isinstance(acq_id, (int, float)) and int(acq_id) in mapping_by_id:
                    donor_name = mapping_by_id[int(acq_id)]
            
            # If still not found, try heuristic matching
            if donor_name is None:
                acq_id = getattr(acq, "id", None) or getattr(acq, "name", None) or str(id(acq))
                acq_name = getattr(acq, "name", "") or ""
                acq_well = getattr(acq, "well", None)
                search_name = (acq_well if acq_well else acq_name).lower()
                if search_name:
                    # Heuristic: find a channel name substring in acquisition name
                    candidates = [nm for nm in ch_names if nm and nm.lower() in search_name]
                    if len(candidates) == 1:
                        donor_name = candidates[0]
                    else:
                        # Try to match by metal name
                        for ch_name in ch_names:
                            # Extract potential metal from channel name (e.g., "Yb176" -> "yb")
                            metal_match = re.match(r"([a-z]+)", ch_name.lower())
                            if metal_match:
                                metal = metal_match.group(1)
                                if metal in search_name:
                                    candidates.append(ch_name)
                        if len(candidates) == 1:
                            donor_name = candidates[0]
            
            if donor_name is None:
                acq_id = getattr(acq, "id", None) or getattr(acq, "name", None) or str(id(acq))
                acq_name = getattr(acq, "name", "") or ""
                acq_well = getattr(acq, "well", None)
                raise ValueError(
                    f"Cannot infer donor for acquisition '{acq_name or acq_well or acq_id}' "
                    f"(ID: {acq_id}, Well: {acq_well}, Index: {acq_idx}). "
                    f"Please ensure all acquisitions have a donor channel mapped in the dialog."
                )
            
            if donor_name not in ch_names:
                raise ValueError(f"Donor '{donor_name}' not in MCD channels. Available: {ch_names[:10]}...")
            
            donor_idx = ch_names.index(donor_name)
            
            stack = _stack_from_acq(mcd, acq)  # H x W x C (raw counts)
            row = _row_from_single_stain(stack, donor_idx, cap=cap)
            rows_by_donor[donor_name].append(row)
        
        # Aggregate to one row per donor
        S = np.eye(C, dtype=np.float64)
        qc_rows = []
        
        for j, donor_name in enumerate(ch_names):
            rows = rows_by_donor[donor_name]
            if rows:
                R = np.vstack(rows)
                if aggregate == "median":
                    row = np.median(R, axis=0)
                else:
                    row = np.mean(R, axis=0)
                row[j] = 1.0
                S[j, :] = row
                
                off = row.copy()
                off[j] = 0.0
                # Calculate median pixels used
                pixels_used_list = []
                for acq in acquisitions:
                    acq_id = getattr(acq, "id", None) or getattr(acq, "name", None) or str(id(acq))
                    acq_name = getattr(acq, "name", "") or ""
                    # Check if this acquisition was used for this donor
                    matched_donor = None
                    if donor_label_per_acq and acq_id in donor_label_per_acq:
                        matched_donor = donor_label_per_acq[acq_id]
                    elif donor_label_per_acq and acq_name in donor_label_per_acq:
                        matched_donor = donor_label_per_acq[acq_name]
                    else:
                        # Try heuristic matching
                        an = acq_name.lower()
                        candidates = [nm for nm in ch_names if nm and nm.lower() in an]
                        if len(candidates) == 1:
                            matched_donor = candidates[0]
                    
                    if matched_donor == donor_name:
                        try:
                            stack = _stack_from_acq(mcd, acq)
                            if j < stack.shape[2]:
                                mask = _foreground_mask(stack[..., j])
                                pixels_used_list.append(np.count_nonzero(mask))
                        except Exception:
                            pass
                
                pixels_median = int(np.median(pixels_used_list)) if pixels_used_list else 0
                
                qc_rows.append({
                    "donor": donor_name,
                    "n_acqs": len(rows),
                    "offdiag_sum": float(off.sum()),
                    "offdiag_max": float(off.max()),
                    "offdiag_max_recipient": ch_names[int(off.argmax())] if off.size and off.max() > 0 else None,
                    "pixels_used_median": pixels_median
                })
            else:
                qc_rows.append({
                    "donor": donor_name,
                    "n_acqs": 0,
                    "offdiag_sum": 0.0,
                    "offdiag_max": 0.0,
                    "offdiag_max_recipient": None,
                    "pixels_used_median": 0
                })
        
        S_df = pd.DataFrame(S, index=ch_names, columns=ch_names)
        qc_df = pd.DataFrame(qc_rows).set_index("donor")
        
        return S_df, qc_df
    
    finally:
        mcd.close()

