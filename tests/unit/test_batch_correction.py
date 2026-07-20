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

"""Unit tests for batch correction helpers and Harmony integration glue."""

import importlib.metadata
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openimc.processing.batch_correction as bc


@pytest.fixture
def sample_batch_df():
    """Create a small feature table with two batches."""
    rng = np.random.default_rng(7)
    n_cells = 80
    return pd.DataFrame(
        {
            "cell_id": np.arange(1, n_cells + 1),
            "source_file": np.where(np.arange(n_cells) < n_cells // 2, "file_a", "file_b"),
            "CD3_mean": rng.normal(loc=1.0, scale=0.2, size=n_cells),
            "CD4_mean": rng.normal(loc=2.0, scale=0.3, size=n_cells),
            "area_um2": rng.uniform(30, 200, size=n_cells),
            "patient_age": np.where(np.arange(n_cells) < n_cells // 2, 54, 61),
            "note": ["meta"] * n_cells,
        }
    )


@pytest.mark.unit
def test_get_feature_columns_from_dataframe_uses_table_columns(sample_batch_df):
    """Feature inference should rely on dataframe columns (not image channels)."""
    features = bc.get_feature_columns_from_dataframe(sample_batch_df, batch_var="source_file")

    assert "CD3_mean" in features
    assert "CD4_mean" in features
    assert "area_um2" in features
    assert "source_file" not in features
    assert "cell_id" not in features
    assert "note" not in features
    assert "NOT_IN_TABLE_mean" not in features


@pytest.mark.integration
def test_real_harmony_020_batch_correction(sample_batch_df):
    """Run OpenIMC's real Harmony path against the release-pinned backend."""
    assert importlib.metadata.version("harmonypy") == "0.2.0"

    corrected = bc.apply_harmony_correction(
        data=sample_batch_df,
        batch_var="source_file",
        features=["CD3_mean", "CD4_mean", "area_um2"],
        n_clusters=2,
        max_iter=3,
        pca_variance=0.95,
    )

    assert corrected.shape == sample_batch_df.shape
    assert corrected.columns.tolist() == sample_batch_df.columns.tolist()
    assert corrected[["cell_id", "source_file", "patient_age", "note"]].equals(
        sample_batch_df[["cell_id", "source_file", "patient_age", "note"]]
    )
    corrected_values = corrected[["CD3_mean", "CD4_mean", "area_um2"]].to_numpy()
    original_values = sample_batch_df[["CD3_mean", "CD4_mean", "area_um2"]].to_numpy()
    assert np.isfinite(corrected_values).all()
    assert not np.allclose(corrected_values, original_values)


def _run_harmony_with_zcorr(monkeypatch, sample_batch_df, zcorr_builder):
    monkeypatch.setattr(bc, "_HAVE_HARMONY", True)

    def fake_run_harmony(pca_data, meta_data, **kwargs):
        return SimpleNamespace(Z_corr=zcorr_builder(pca_data))

    monkeypatch.setattr(bc, "run_harmony", fake_run_harmony, raising=False)
    return bc.apply_harmony_correction(
        data=sample_batch_df,
        batch_var="source_file",
        features=["CD3_mean", "CD4_mean", "area_um2"],
        n_clusters=5,
        max_iter=3,
        pca_variance=0.95,
    )


@pytest.mark.unit
def test_apply_harmony_correction_accepts_sample_component_orientation(monkeypatch, sample_batch_df):
    """Harmony output in (n_samples, n_components) orientation should be accepted."""
    corrected = _run_harmony_with_zcorr(monkeypatch, sample_batch_df, lambda pca_data: pca_data.copy())

    assert corrected.shape == sample_batch_df.shape
    assert np.isfinite(corrected[["CD3_mean", "CD4_mean", "area_um2"]].to_numpy()).all()


@pytest.mark.unit
def test_apply_harmony_correction_accepts_component_sample_orientation(monkeypatch, sample_batch_df):
    """Harmony output in (n_components, n_samples) orientation should be accepted."""
    corrected = _run_harmony_with_zcorr(monkeypatch, sample_batch_df, lambda pca_data: pca_data.T.copy())

    assert corrected.shape == sample_batch_df.shape
    assert np.isfinite(corrected[["CD3_mean", "CD4_mean", "area_um2"]].to_numpy()).all()


@pytest.mark.unit
def test_apply_harmony_correction_rejects_unexpected_shape(monkeypatch, sample_batch_df):
    """Unexpected Harmony output shape should raise a clear error."""
    monkeypatch.setattr(bc, "_HAVE_HARMONY", True)

    def fake_run_harmony(pca_data, meta_data, **kwargs):
        return SimpleNamespace(Z_corr=np.zeros((3, 3)))

    monkeypatch.setattr(bc, "run_harmony", fake_run_harmony, raising=False)

    with pytest.raises(RuntimeError, match="unexpected shape"):
        bc.apply_harmony_correction(
            data=sample_batch_df,
            batch_var="source_file",
            features=["CD3_mean", "CD4_mean", "area_um2"],
            n_clusters=5,
            max_iter=3,
            pca_variance=0.95,
        )


@pytest.mark.unit
def test_apply_harmony_correction_defers_import_until_use(monkeypatch, sample_batch_df):
    """Harmony import errors should surface only when Harmony correction is requested."""
    monkeypatch.setattr(bc, "run_harmony", None, raising=False)
    monkeypatch.setattr(bc, "_HAVE_HARMONY", True)
    monkeypatch.setattr(
        bc,
        "_get_run_harmony",
        lambda: (_ for _ in ()).throw(ImportError("torch DLL failed")),
    )

    with pytest.raises(ImportError, match="torch DLL failed"):
        bc.apply_harmony_correction(
            data=sample_batch_df,
            batch_var="source_file",
            features=["CD3_mean", "CD4_mean", "area_um2"],
            n_clusters=5,
            max_iter=3,
            pca_variance=0.95,
        )
