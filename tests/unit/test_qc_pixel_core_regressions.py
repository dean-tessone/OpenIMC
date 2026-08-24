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

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from openimc.core import (
    _qc_min_background_std,
    aggregate_qc_results,
    pixel_correlation,
    qc_analysis,
)
from openimc.data.mcd_loader import AcquisitionInfo


class _DummyLoader:
    def __init__(self, stack, channels):
        self._stack = stack
        self._channels = list(channels)

    def get_all_channels(self, _acq_id):
        return self._stack

    def get_channels(self, _acq_id):
        return list(self._channels)

    def get_image(self, _acq_id, channel):
        channel_index = self._channels.index(channel)
        return self._stack[:, :, channel_index]


def _build_dummy_acquisition(channels):
    return AcquisitionInfo(
        id="ROI_1",
        name="ROI_1",
        well="A1",
        size=(4, 4),
        channels=list(channels),
        channel_metals=[""] * len(channels),
        channel_labels=[""] * len(channels),
        metadata={},
        source_file="dummy",
    )


def test_qc_analysis_applies_denoise_settings_before_metrics(monkeypatch):
    channels = ["A", "B"]
    stack = np.stack(
        [
            np.arange(16, dtype=np.float32).reshape(4, 4),
            np.ones((4, 4), dtype=np.float32) * 5.0,
        ],
        axis=-1,
    )
    loader = _DummyLoader(stack, channels)
    acquisition = _build_dummy_acquisition(channels)

    calls = []

    def _fake_denoise(img, channel, settings):
        calls.append((channel, settings))
        return img + (3.0 if channel == "A" else 0.0)

    monkeypatch.setattr("openimc.core._apply_denoise_to_channel", _fake_denoise)

    qc_df = qc_analysis(
        loader=loader,
        acquisition=acquisition,
        channels=channels,
        mode="pixel",
        denoise_settings={"A": {"hot": {"method": "median3"}}},
    )

    assert calls == [("A", {"hot": {"method": "median3"}})]
    mean_a = qc_df.loc[qc_df["channel"] == "A", "intensity_mean"].iloc[0]
    mean_b = qc_df.loc[qc_df["channel"] == "B", "intensity_mean"].iloc[0]
    assert mean_a == np.mean(stack[:, :, 0] + 3.0)
    assert mean_b == np.mean(stack[:, :, 1])


def test_qc_analysis_cell_mode_uses_cell_pixels_for_signal_metrics():
    channels = ["A"]
    channel_a = np.array(
        [
            [12.0, 12.0, 12.0, 1.0],
            [12.0, 12.0, 12.0, 2.0],
            [12.0, 12.0, 12.0, 3.0],
            [4.0, 5.0, 6.0, 2.0],
        ],
        dtype=np.float32,
    )
    stack = np.stack([channel_a], axis=-1)
    loader = _DummyLoader(stack, channels)
    acquisition = _build_dummy_acquisition(channels)
    mask = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )

    qc_df = qc_analysis(
        loader=loader,
        acquisition=acquisition,
        channels=channels,
        mode="cell",
        mask=mask,
        cell_signal_method="all_cell_mean",
    )

    assert len(qc_df) == 1

    signal_pixels = channel_a[mask > 0]
    background_pixels = channel_a[mask == 0]
    expected_signal_mean = float(np.mean(signal_pixels))
    expected_signal_std = float(np.std(signal_pixels))
    expected_background_mean = float(np.mean(background_pixels))
    expected_background_std = float(np.std(background_pixels))
    expected_snr = (expected_signal_mean - expected_background_mean) / expected_background_std

    row = qc_df.iloc[0]
    assert row["signal_mean"] == pytest.approx(expected_signal_mean)
    assert row["signal_std"] == pytest.approx(expected_signal_std)
    assert row["background_mean"] == pytest.approx(expected_background_mean)
    assert row["background_std"] == pytest.approx(expected_background_std)
    assert row["snr"] == pytest.approx(expected_snr)
    assert row["cell_signal_method"] == "all_cell_mean"
    assert row["n_signal_pixels"] == signal_pixels.size
    assert row["n_signal_cells"] == 2
    assert row["signal_fraction"] == pytest.approx(1.0)
    assert row["n_total_pixels"] == channel_a.size
    assert row["n_cell_pixels"] == np.count_nonzero(mask > 0)
    assert row["n_background_pixels"] == np.count_nonzero(mask == 0)


def test_qc_analysis_cell_positive_pixels_focuses_on_marker_positive_pixels():
    channels = ["A"]
    channel_a = np.array(
        [
            [12.0, 12.0, 12.0, 1.0],
            [12.0, 12.0, 12.0, 2.0],
            [12.0, 12.0, 12.0, 3.0],
            [4.0, 5.0, 6.0, 2.0],
        ],
        dtype=np.float32,
    )
    stack = np.stack([channel_a], axis=-1)
    loader = _DummyLoader(stack, channels)
    acquisition = _build_dummy_acquisition(channels)
    mask = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )

    legacy_df = qc_analysis(
        loader=loader,
        acquisition=acquisition,
        channels=channels,
        mode="cell",
        mask=mask,
        cell_signal_method="all_cell_mean",
    )
    positive_df = qc_analysis(
        loader=loader,
        acquisition=acquisition,
        channels=channels,
        mode="cell",
        mask=mask,
        cell_signal_method="positive_pixels",
        positive_threshold_sd=2.0,
    )

    row = positive_df.iloc[0]
    background_pixels = channel_a[mask == 0]
    expected_threshold = float(
        np.mean(background_pixels) + 2.0 * _qc_min_background_std(
            float(np.mean(background_pixels)),
            float(np.std(background_pixels)),
            float(np.min(channel_a)),
            float(np.max(channel_a)),
        )
    )

    assert row["cell_signal_method"] == "positive_pixels"
    assert row["signal_threshold"] == pytest.approx(expected_threshold)
    assert np.isnan(row["signal_quantile"])
    assert row["n_signal_pixels"] == 9
    assert row["n_signal_cells"] == 1
    assert row["signal_mean"] == pytest.approx(12.0)
    assert row["signal_fraction"] == pytest.approx(0.9)
    assert row["signal_coverage_pct"] == pytest.approx((9 / 16) * 100.0)
    assert row["snr"] > legacy_df.iloc[0]["snr"]


def test_qc_analysis_cell_upper_quantile_ignores_marker_negative_cells():
    channels = ["A"]
    channel_a = np.array(
        [
            [12.0, 12.0, 0.0, 2.0, 2.0],
            [12.0, 12.0, 0.0, 2.0, 2.0],
            [1.0, 2.0, 1.0, 2.0, 1.0],
            [2.0, 2.0, 0.0, 2.0, 2.0],
            [2.0, 2.0, 0.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 0, 4, 4],
            [3, 3, 0, 4, 4],
        ],
        dtype=np.int32,
    )
    stack = np.stack([channel_a], axis=-1)
    loader = _DummyLoader(stack, channels)
    acquisition = _build_dummy_acquisition(channels)

    legacy_df = qc_analysis(
        loader=loader,
        acquisition=acquisition,
        channels=channels,
        mode="cell",
        mask=mask,
        cell_signal_method="all_cell_mean",
    )
    quantile_df = qc_analysis(
        loader=loader,
        acquisition=acquisition,
        channels=channels,
        mode="cell",
        mask=mask,
        cell_signal_method="upper_quantile",
        upper_quantile=0.75,
    )

    row = quantile_df.iloc[0]
    assert row["cell_signal_method"] == "upper_quantile"
    assert np.isnan(row["signal_threshold"])
    assert row["signal_quantile"] == pytest.approx(0.75)
    assert row["n_signal_cells"] == 1
    assert row["n_signal_pixels"] == 4
    assert row["signal_mean"] == pytest.approx(12.0)
    assert row["signal_fraction"] == pytest.approx(0.25)
    assert row["snr"] > legacy_df.iloc[0]["snr"]


def test_qc_analysis_cell_positive_pixels_handles_no_detected_signal():
    channels = ["A"]
    channel_a = np.array(
        [
            [2.0, 2.0, 2.0, 4.0],
            [2.0, 2.0, 2.0, 5.0],
            [2.0, 2.0, 2.0, 6.0],
            [7.0, 8.0, 9.0, 2.0],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )
    stack = np.stack([channel_a], axis=-1)
    loader = _DummyLoader(stack, channels)
    acquisition = _build_dummy_acquisition(channels)

    qc_df = qc_analysis(
        loader=loader,
        acquisition=acquisition,
        channels=channels,
        mode="cell",
        mask=mask,
        cell_signal_method="positive_pixels",
        positive_threshold_sd=2.0,
    )

    row = qc_df.iloc[0]
    background_pixels = channel_a[mask == 0]
    assert row["snr"] == 0.0
    assert row["n_signal_pixels"] == 0
    assert row["n_signal_cells"] == 0
    assert row["signal_fraction"] == 0.0
    assert row["signal_coverage_pct"] == 0.0
    assert row["signal_mean"] == pytest.approx(float(np.mean(background_pixels)))
    assert np.isfinite(row["signal_threshold"])


def test_pixel_correlation_vectorized_path_matches_manual_pairwise_spearman():
    channels = ["A", "B", "C"]
    channel_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    channel_b = np.array([[4, 3], [2, 1]], dtype=np.float32)
    channel_c = np.array([[1, 3], [2, 4]], dtype=np.float32)
    stack = np.stack([channel_a, channel_b, channel_c], axis=-1)
    loader = _DummyLoader(stack, channels)
    acquisition = _build_dummy_acquisition(channels)

    corr_df = pixel_correlation(loader=loader, acquisition=acquisition, channels=channels, mask=None)

    assert len(corr_df) == 3
    assert corr_df["n_pixels"].nunique() == 1
    assert corr_df["n_pixels"].iloc[0] == 4

    manual = {}
    flat_channels = {channel: stack[:, :, idx].reshape(-1) for idx, channel in enumerate(channels)}
    for i, ch1 in enumerate(channels):
        for ch2 in channels[i + 1:]:
            corr_value, p_value = spearmanr(flat_channels[ch1], flat_channels[ch2])
            manual[(ch1, ch2)] = (corr_value, p_value)

    for _, row in corr_df.iterrows():
        pair = (row["marker1"], row["marker2"])
        expected_corr, expected_p = manual[pair]
        assert row["correlation"] == expected_corr
        assert row["p_value"] == expected_p


def test_pixel_correlation_uses_pairwise_complete_pixels():
    channels = ["A", "B", "C"]
    channel_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    channel_b = np.array([[2, 4], [6, 8]], dtype=np.float32)
    channel_c = np.array([[1, np.nan], [2, 3]], dtype=np.float32)
    stack = np.stack([channel_a, channel_b, channel_c], axis=-1)
    loader = _DummyLoader(stack, channels)
    acquisition = _build_dummy_acquisition(channels)

    corr_df = pixel_correlation(loader, acquisition, channels)
    counts = {
        (row.marker1, row.marker2): row.n_pixels
        for row in corr_df.itertuples(index=False)
    }

    assert counts[("A", "B")] == 4
    assert counts[("A", "C")] == 3
    assert counts[("B", "C")] == 3
    assert corr_df.loc[
        (corr_df["marker1"] == "A") & (corr_df["marker2"] == "B"),
        "correlation",
    ].iloc[0] == pytest.approx(1.0)


def test_qc_summary_pools_pixels_before_calculating_snr():
    per_roi = pd.DataFrame(
        {
            "channel": ["Marker", "Marker"],
            "mode": ["cell", "cell"],
            "cell_signal_method": ["all_cell_mean", "all_cell_mean"],
            "snr": [9.0, 1.0],  # Deliberately not the pooled result.
            "signal_mean": [10.0, 20.0],
            "signal_std": [2.0, 4.0],
            "background_mean": [1.0, 3.0],
            "background_std": [1.0, 2.0],
            "mean_intensity": [4.0, 12.0],
            "std_intensity": [3.0, 6.0],
            "median_intensity": [2.0, 8.0],
            "min_intensity": [0.0, 0.0],
            "max_intensity": [30.0, 40.0],
            "n_signal_pixels": [100, 300],
            "n_background_pixels": [200, 100],
            "n_cell_pixels": [150, 350],
            "n_total_pixels": [350, 450],
            "n_cells": [10, 20],
        }
    )

    summary = aggregate_qc_results(per_roi).iloc[0]

    expected_signal_mean = (100 * 10.0 + 300 * 20.0) / 400
    expected_background_mean = (200 * 1.0 + 100 * 3.0) / 300
    expected_background_second_moment = (
        200 * (1.0 ** 2 + 1.0 ** 2) + 100 * (2.0 ** 2 + 3.0 ** 2)
    ) / 300
    expected_background_std = np.sqrt(
        expected_background_second_moment - expected_background_mean ** 2
    )
    expected_snr = (
        expected_signal_mean - expected_background_mean
    ) / expected_background_std

    assert summary["n_rois"] == 2
    assert summary["n_signal_pixels"] == 400
    assert summary["n_background_pixels"] == 300
    assert summary["signal_mean"] == pytest.approx(expected_signal_mean)
    assert summary["background_mean"] == pytest.approx(expected_background_mean)
    assert summary["background_std"] == pytest.approx(expected_background_std)
    assert summary["signal_minus_background"] == pytest.approx(
        expected_signal_mean - expected_background_mean
    )
    assert summary["snr"] == pytest.approx(expected_snr)
    assert summary["snr"] != pytest.approx(per_roi["snr"].mean())


def test_qc_analysis_rejects_cell_mask_without_background():
    channels = ["A"]
    stack = np.ones((4, 4, 1), dtype=np.float32)
    loader = _DummyLoader(stack, channels)
    acquisition = _build_dummy_acquisition(channels)

    with pytest.raises(ValueError, match="non-cell pixels"):
        qc_analysis(
            loader,
            acquisition,
            channels,
            mode="cell",
            mask=np.ones((4, 4), dtype=np.uint16),
        )
