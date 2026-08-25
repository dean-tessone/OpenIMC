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
from copy import deepcopy
from types import SimpleNamespace
from PyQt5 import QtCore, QtWidgets

from openimc.ui.dialogs import clustering as clustering_module
from openimc.ui.dialogs import simple_spatial_analysis as simple_spatial_module
from openimc.data.mcd_loader import AcquisitionInfo
from openimc.ui.dialogs.clustering import CellClusteringDialog
from openimc.ui.dialogs.qc_analysis_dialog import QCAnalysisDialog
from openimc.ui.dialogs.simple_spatial_analysis import SimpleSpatialAnalysisDialog
from openimc.ui.figure_layout import measure_figure_text_overflow
from openimc.ui.utils import combine_pvalues_fisher


def _build_clustered_dataframe(
    n_clusters: int = 6,
    cells_per_cluster: int = 10,
    n_features: int = 12,
) -> pd.DataFrame:
    rows = []
    for cluster_id in range(1, n_clusters + 1):
        for offset in range(cells_per_cluster):
            row = {
                'cell_id': (cluster_id - 1) * cells_per_cluster + offset + 1,
                'centroid_x': float(cluster_id * 12 + offset),
                'centroid_y': float(cluster_id * 8 + offset),
                'acquisition_id': 'ROI_1',
                'cluster': cluster_id,
            }
            for feature_idx in range(n_features):
                row[f'marker_{feature_idx}_mean'] = float(cluster_id * 2.0 + feature_idx + (offset * 0.02))
            rows.append(row)
    return pd.DataFrame(rows)


def _build_spatial_dataframe(n_clusters: int = 14) -> pd.DataFrame:
    rows = []
    for cluster_id in range(1, n_clusters + 1):
        rows.append(
            {
                'cell_id': cluster_id,
                'centroid_x': float(cluster_id * 7),
                'centroid_y': float(cluster_id * 5),
                'acquisition_id': 'ROI_1',
                'cluster': cluster_id,
            }
        )
    return pd.DataFrame(rows)


def _build_enrichment_df(n_clusters: int = 14) -> pd.DataFrame:
    rows = []
    for cluster_a in range(1, n_clusters + 1):
        for cluster_b in range(1, n_clusters + 1):
            rows.append(
                {
                    'roi_id': 'ROI_1',
                    'cluster_A': cluster_a,
                    'cluster_B': cluster_b,
                    'z_score': float((cluster_a - cluster_b) / 3.0),
                    'p_value': 0.01 if cluster_a == cluster_b else 0.20,
                }
            )
    return pd.DataFrame(rows)


def _build_distance_df(source_clusters=(1, 2, 3), target_clusters=(1, 2, 3), samples_per_pair: int = 6) -> pd.DataFrame:
    rows = []
    cell_id = 1
    for source_cluster in source_clusters:
        for offset in range(samples_per_pair):
            source_cell_id = cell_id
            cell_id += 1
            for target_cluster in target_clusters:
                rows.append(
                    {
                        'roi_id': 'ROI_1',
                        'cell_A_id': source_cell_id,
                        'cell_A_cluster': source_cluster,
                        'nearest_B_cluster': target_cluster,
                        'nearest_B_dist_um': float((source_cluster * 2.0) + target_cluster + (offset * 0.25)),
                        'nearest_B_cell_id': int((target_cluster * 100) + offset + 1),
                    }
                )
    return pd.DataFrame(rows)


def _build_cooccurrence_array(n_clusters: int = 14, n_distances: int = 3) -> np.ndarray:
    occ = np.zeros((n_clusters, n_clusters, n_distances), dtype=float)
    for i in range(n_clusters):
        for j in range(n_clusters):
            for k in range(n_distances):
                occ[i, j, k] = ((i + 1) * 0.03) + ((j + 1) * 0.015) + (k * 0.05)
    return occ


def _build_clustering_dialog(
    qtbot,
    n_clusters: int = 6,
    n_features: int = 12,
    parent: QtWidgets.QWidget | None = None,
) -> CellClusteringDialog:
    df = _build_clustered_dataframe(n_clusters=n_clusters, n_features=n_features)
    if parent is not None:
        qtbot.addWidget(parent)
    dialog = CellClusteringDialog(df, clustered_cells_dataframe=df.copy(), parent=parent)
    qtbot.addWidget(dialog)
    dialog.feature_label_map = {
        f'marker_{idx}_mean': f'Feature label {idx} with a deliberately long display name'
        for idx in range(n_features)
    }
    dialog.cluster_annotation_map = {
        cluster_id: f'Cluster {cluster_id} with a deliberately long display name'
        for cluster_id in range(1, n_clusters + 1)
    }
    dialog.resize(1220, 900)
    dialog.show()
    qtbot.wait(120)
    return dialog




def test_clustering_run_logs_pca_metadata_from_core_result(qtbot, monkeypatch):
    df = _build_clustered_dataframe(n_clusters=3, cells_per_cluster=8, n_features=5)
    dialog = CellClusteringDialog(df)
    qtbot.addWidget(dialog)
    selected_features = ['marker_0_mean', 'marker_1_mean', 'marker_2_mean']

    from openimc.ui.dialogs import feature_selector_dialog as feature_selector_module

    class FakeFeatureSelector:
        def __init__(self, _available_cols, _parent):
            pass

        def set_filter_settings(self, _settings):
            pass

        def set_selected_features(self, _features):
            pass

        def exec_(self):
            return QtWidgets.QDialog.Accepted

        def get_selected_columns(self):
            return list(selected_features)

        def get_filter_settings(self):
            return {}

    captured_kwargs = {}
    pca_metadata = {
        'feature_representation': 'principal_components',
        'use_pca': True,
        'pca_selection_mode': 'variance',
        'pca_requested_variance': 0.90,
        'pca_requested_n_components': None,
        'pca_n_components_retained': 2,
        'pca_variance_retained': 0.934,
        'pca_input_feature_count': 3,
    }

    def fake_run_core(cluster_kwargs, cluster_method):
        captured_kwargs.update(cluster_kwargs)
        assert cluster_method == 'leiden'
        result = cluster_kwargs['features_df'].copy()
        result['cluster'] = np.resize(np.array([1, 2, 3], dtype=int), len(result))
        result.attrs['pca_metadata'] = pca_metadata
        return result

    captured_log = {}

    class FakeLogger:
        def log_clustering(self, **kwargs):
            captured_log.update(kwargs)

    def fail_critical(_parent, _title, message):
        raise AssertionError(message)

    monkeypatch.setattr(feature_selector_module, 'FeatureSelectorDialog', FakeFeatureSelector)
    monkeypatch.setattr(dialog, '_run_core_cluster_with_progress', fake_run_core)
    monkeypatch.setattr(dialog, '_create_heatmap', lambda: None)
    monkeypatch.setattr(clustering_module, 'get_logger', lambda: FakeLogger())
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical', fail_critical)

    dialog.use_pca_checkbox.setChecked(True)
    dialog.pca_mode_combo.setCurrentIndex(dialog.pca_mode_combo.findData('variance'))
    dialog.pca_variance_spinbox.setValue(90.0)
    dialog._run_clustering()

    assert captured_kwargs['use_pca'] is True
    assert captured_kwargs['pca_mode'] == 'variance'
    assert captured_kwargs['pca_variance'] == pytest.approx(0.90)
    assert captured_kwargs['columns'] == selected_features
    assert dialog.selected_display_features == selected_features

    params = captured_log['parameters']
    assert params['feature_representation'] == 'principal_components'
    assert params['pca_selection_mode'] == 'variance'
    assert params['pca_requested_variance'] == pytest.approx(0.90)
    assert params['pca_n_components_retained'] == 2
    assert params['pca_variance_retained'] == pytest.approx(0.934)
    assert params['pca_input_feature_count'] == 3
    assert captured_log['features_used'] == selected_features
    assert dialog.last_clustering_params['pca_n_components_retained'] == 2


def _build_simple_spatial_dialog(qtbot, n_clusters: int = 14) -> SimpleSpatialAnalysisDialog:
    df = _build_spatial_dataframe(n_clusters=n_clusters)
    dialog = SimpleSpatialAnalysisDialog(df, clustered_cells_dataframe=df.copy())
    qtbot.addWidget(dialog)
    dialog.cluster_annotation_map = {
        cluster_id: f'Cluster {cluster_id} with a very long descriptive spatial label'
        for cluster_id in range(1, n_clusters + 1)
    }
    dialog.resize(1280, 920)
    dialog.show()
    qtbot.wait(120)
    return dialog


class _QCTestParent(QtWidgets.QWidget):
    def __init__(self, with_masks: bool = True):
        super().__init__()
        self.qc_results_cache = {}
        self._saved_qc_ui_state = {}
        self.segmentation_masks = {"ROI_1": np.ones((4, 4), dtype=np.uint32)} if with_masks else {}
        self.acquisitions = [
            AcquisitionInfo(
                id="ROI_1",
                name="ROI 1",
                well="A1",
                size=(4, 4),
                channels=["Marker A", "Marker B"],
                channel_metals=["", ""],
                channel_labels=["", ""],
                metadata={},
                source_file="dummy.mcd",
            )
        ]

    def _get_qc_file_set_id(self):
        return "file_set_1"

    def _get_acquisition_info(self, acq_id):
        for acq in self.acquisitions:
            if acq.id == acq_id:
                return acq
        return None


def _build_qc_dialog(qtbot, parent=None) -> QCAnalysisDialog:
    if parent is not None:
        qtbot.addWidget(parent)
    dialog = QCAnalysisDialog(parent)
    qtbot.addWidget(dialog)
    if hasattr(dialog, 'qc_settings_dialog'):
        qtbot.addWidget(dialog.qc_settings_dialog)
    dialog.resize(1100, 760)
    dialog.show()
    qtbot.wait(120)
    return dialog


def _legend_ncols(legend) -> int:
    return int(
        getattr(
            legend,
            '_ncols',
            getattr(legend, '_ncol', 1),
        ) or 1
    )


def _select_color_by(dialog: CellClusteringDialog, label: str) -> None:
    dialog.color_by_listwidget.clearSelection()
    for idx in range(dialog.color_by_listwidget.count()):
        item = dialog.color_by_listwidget.item(idx)
        item_label = item.data(QtCore.Qt.UserRole) or item.text()
        if item_label == label or item.text() == label:
            item.setSelected(True)
            return
    raise AssertionError(f"Could not find color-by option {label!r}")


def test_differential_expression_axis_flip_transposes_labels_and_highlights(qtbot):
    dialog = _build_clustering_dialog(qtbot, n_clusters=6, n_features=12)
    dialog.top_n_spinbox.setValue(3)

    dialog._show_differential_expression()
    ax = dialog.figure.axes[0]
    text_count_with_values = len(ax.texts)

    assert ax.get_xlabel() == 'Clusters'
    assert ax.get_ylabel() == 'Features'
    assert any('Cluster 1' in tick.get_text() for tick in ax.get_xticklabels())
    assert any('Feature label' in tick.get_text() for tick in ax.get_yticklabels())
    assert max(patch.get_x() for patch in ax.patches) < 5.5
    default_patch_span = (
        max(patch.get_x() for patch in ax.patches),
        max(patch.get_y() for patch in ax.patches),
    )
    assert text_count_with_values > 10

    dialog.de_flip_axes_checkbox.setChecked(True)
    dialog._show_differential_expression()
    ax = dialog.figure.axes[0]

    assert ax.get_xlabel() == 'Features'
    assert ax.get_ylabel() == 'Clusters'
    assert any('Feature label' in tick.get_text() for tick in ax.get_xticklabels())
    assert any('Cluster 1' in tick.get_text() for tick in ax.get_yticklabels())
    assert max(patch.get_x() for patch in ax.patches) == pytest.approx(default_patch_span[1])
    assert max(patch.get_y() for patch in ax.patches) == pytest.approx(default_patch_span[0])

    dialog.de_show_values_checkbox.setChecked(False)
    dialog._show_differential_expression()
    ax = dialog.figure.axes[0]

    assert len(ax.texts) < text_count_with_values / 3


def test_heatmap_and_cluster_map_default_to_zscore_scaling(qtbot, monkeypatch):
    dialog = _build_clustering_dialog(qtbot, n_clusters=6, n_features=12)
    scaling_calls = []

    def _record_scaling(_self, data, scaling_method):
        scaling_calls.append((_self._active_view_name, scaling_method))
        return data.copy()

    monkeypatch.setattr(CellClusteringDialog, "_apply_scaling", _record_scaling)

    assert dialog.heatmap_scaling_combo.currentText() == "Z-score"

    dialog._show_heatmap()
    dialog._show_cluster_map()

    assert ('Heatmap', 'zscore') in scaling_calls
    assert ('Cluster Map', 'zscore') in scaling_calls


def test_heatmap_annotation_bars_show_cluster_and_custom_patient_labels(qtbot):
    df = _build_clustered_dataframe(n_clusters=4, n_features=6)
    df['source_file'] = [
        f'/tmp/patient_{idx % 2}.ome.tiff'
        for idx in range(len(df))
    ]

    dialog = CellClusteringDialog(df, clustered_cells_dataframe=df.copy())
    qtbot.addWidget(dialog)
    dialog.patient_annotation_enabled = True
    dialog.patient_annotation_column = 'source_file'
    dialog.patient_legend_label = 'Sample'
    dialog.resize(1220, 900)
    dialog.show()
    qtbot.wait(120)

    dialog._show_heatmap()

    annotation_bar_texts = [
        text
        for axis in dialog.figure.axes
        if axis.images and len(axis.get_xticks()) == 0 and len(axis.get_yticks()) == 0
        for text in axis.texts
        if text.get_text()
    ]
    annotation_bar_labels = {text.get_text() for text in annotation_bar_texts}
    overflow = measure_figure_text_overflow(dialog.figure)

    assert 'Cluster' in annotation_bar_labels
    assert 'Sample' in annotation_bar_labels
    assert all(text.get_rotation() == 0 for text in annotation_bar_texts)
    assert max(overflow.values()) <= 0.04


def test_qc_snr_plot_uses_symlog_and_keeps_nonpositive_snr_points(qtbot):
    dialog = _build_qc_dialog(qtbot)
    dialog.qc_results_aggregated = pd.DataFrame(
        {
            'channel': ['Marker A', 'Marker B', 'Marker C'],
            'mean_intensity': [1.0, 2.0, 3.0],
            'signal_mean': [10.0, 25.0, 80.0],
            'snr': [0.7, -0.2, 3.5],
        }
    )

    dialog._plot_snr_vs_intensity()
    ax = dialog.snr_intensity_canvas.figure.axes[0]

    assert ax.get_xscale() == 'log'
    assert ax.get_yscale() == 'symlog'
    scatter_points = ax.collections[0].get_offsets()
    assert len(scatter_points) == 3
    assert np.allclose(scatter_points[:, 0], [10.0, 25.0, 80.0])
    assert 'symlog' in ax.get_ylabel().lower()
    assert 'foreground mean intensity' in ax.get_xlabel().lower()
    assert 'foreground mean intensity' in ax.get_title().lower()
    threshold_lines = [line for line in ax.lines if line.get_linestyle() == '--']
    assert any(np.allclose(line.get_ydata(), [3.0, 3.0]) for line in threshold_lines)
    legend = ax.get_legend()
    assert legend is not None
    assert any('3.0' in text.get_text() for text in legend.get_texts())


def test_qc_cnr_plot_is_distinct_and_uses_its_own_threshold(qtbot):
    dialog = _build_qc_dialog(qtbot)
    dialog.cnr_threshold_spin.setValue(2.5)
    dialog.qc_results_aggregated = pd.DataFrame(
        {
            'channel': ['Marker A', 'Marker B'],
            'signal_mean': [12.0, 30.0],
            'snr': [6.0, 10.0],
            'cnr': [4.0, 7.0],
        }
    )

    dialog._plot_snr_vs_intensity()
    cnr_ax = dialog.snr_intensity_canvas.figure.axes[1]

    scatter_points = cnr_ax.collections[0].get_offsets()
    assert np.allclose(scatter_points[:, 1], [4.0, 7.0])
    assert 'cnr' in cnr_ax.get_ylabel().lower()
    assert 'cnr' in cnr_ax.get_title().lower()
    threshold_lines = [line for line in cnr_ax.lines if line.get_linestyle() == '--']
    assert any(np.allclose(line.get_ydata(), [2.5, 2.5]) for line in threshold_lines)


def test_qc_cell_signal_controls_toggle_by_mode_and_method(qtbot):
    parent = _QCTestParent(with_masks=True)
    dialog = _build_qc_dialog(qtbot, parent)
    dialog.qc_settings_dialog.show()
    qtbot.wait(50)

    assert dialog.analysis_mode == "cell"
    assert dialog.cell_signal_group.isVisibleTo(dialog.qc_settings_dialog)
    assert dialog.get_cell_signal_method() == "positive_pixels"
    assert dialog.positive_threshold_sd_spin.isVisible()
    assert not dialog.upper_quantile_spin.isVisible()

    dialog.cell_signal_method_combo.setCurrentIndex(dialog.cell_signal_method_combo.findData("upper_quantile"))
    qtbot.wait(50)
    assert not dialog.positive_threshold_sd_spin.isVisible()
    assert dialog.upper_quantile_spin.isVisible()

    dialog.mode_combo.setCurrentIndex(0)
    qtbot.wait(50)
    assert dialog.analysis_mode == "pixel"
    assert dialog.cell_signal_group.isHidden()


def test_qc_restores_cell_signal_ui_state(qtbot):
    parent = _QCTestParent(with_masks=True)
    parent._saved_qc_ui_state = {
        "analysis_mode": "Cell-level",
        "snr_threshold": 5.0,
        "cnr_threshold": 6.0,
        "cell_signal_method": "upper_quantile",
        "positive_threshold_sd": 3.0,
        "upper_quantile_percent": 95.0,
    }

    dialog = _build_qc_dialog(qtbot, parent)

    assert dialog.analysis_mode == "cell"
    assert dialog.snr_threshold_spin.value() == pytest.approx(5.0)
    assert dialog.cnr_threshold_spin.value() == pytest.approx(6.0)
    assert dialog.get_cell_signal_method() == "upper_quantile"
    assert dialog.positive_threshold_sd_spin.value() == pytest.approx(3.0)
    assert dialog.upper_quantile_spin.value() == pytest.approx(95.0)


def test_qc_settings_summary_reflects_current_popup_state(qtbot):
    parent = _QCTestParent(with_masks=True)
    dialog = _build_qc_dialog(qtbot, parent)

    summary = dialog.settings_summary_label.text()
    assert "All Acquisitions" in summary
    assert "Cell-level" in summary
    assert "SNR/CNR thresholds: 3.0/3.0" in summary
    assert "Positive pixels" in summary

    dialog.cell_signal_method_combo.setCurrentIndex(dialog.cell_signal_method_combo.findData("upper_quantile"))
    dialog.upper_quantile_spin.setValue(95.0)
    dialog.snr_threshold_spin.setValue(4.5)
    dialog.denoise_source_combo.setCurrentText("Viewer")
    qtbot.wait(50)

    summary = dialog.settings_summary_label.text()
    assert "SNR/CNR thresholds: 4.5/3.0" in summary
    assert "Top 95.0% cells" in summary
    assert "Denoising: Viewer" in summary


def test_qc_cache_key_includes_cell_signal_settings(qtbot):
    parent = _QCTestParent(with_masks=True)
    dialog = _build_qc_dialog(qtbot, parent)
    dialog.cell_signal_method_combo.setCurrentIndex(dialog.cell_signal_method_combo.findData("positive_pixels"))
    dialog.positive_threshold_sd_spin.setValue(2.0)

    dialog.qc_results = pd.DataFrame(
        {
            'channel': ['Marker A'],
            'cell_signal_method': ['positive_pixels'],
            'snr': [3.0],
            'mean_intensity': [10.0],
            'coverage_pct': [25.0],
            'signal_coverage_pct': [12.5],
        }
    )
    dialog.qc_results_aggregated = dialog.qc_results.copy()
    dialog.qc_results_aggregated['n_rois'] = [1]
    dialog._save_results_to_cache()

    dialog.qc_results = None
    dialog.qc_results_aggregated = None
    dialog.cell_signal_method_combo.setCurrentIndex(dialog.cell_signal_method_combo.findData("upper_quantile"))
    dialog.upper_quantile_spin.setValue(95.0)
    dialog._restore_cached_results()

    assert dialog.qc_results is None
    assert dialog.qc_results_aggregated is None


def test_qc_summary_and_titles_reflect_cell_signal_method(qtbot):
    parent = _QCTestParent(with_masks=True)
    dialog = _build_qc_dialog(qtbot, parent)
    dialog.qc_results = pd.DataFrame(
        {
            'channel': ['Marker A', 'Marker B'],
            'cell_signal_method': ['positive_pixels', 'positive_pixels'],
            'snr': [1.5, 3.0],
            'mean_intensity': [10.0, 25.0],
            'coverage_pct': [40.0, 42.0],
            'signal_coverage_pct': [12.5, 18.0],
            'signal_fraction': [0.31, 0.43],
        }
    )
    dialog.qc_results_aggregated = dialog.qc_results.copy()
    dialog.qc_results_aggregated['n_rois'] = [1, 1]
    dialog._update_summary_table()
    dialog._plot_snr_vs_intensity()

    headers = [dialog.summary_table.horizontalHeaderItem(i).text() for i in range(dialog.summary_table.columnCount())]
    ax = dialog.snr_intensity_canvas.figure.axes[0]

    assert 'signal_coverage_pct' in headers
    assert 'Positive pixels above background' in dialog.summary_method_label.text()
    assert 'Positive pixels above background' in ax.get_title()


def test_qc_distribution_plot_uses_configured_snr_threshold(qtbot):
    dialog = _build_qc_dialog(qtbot)
    dialog.snr_threshold_spin.setValue(4.0)
    dialog.qc_results = pd.DataFrame(
        {
            'channel': ['Marker A', 'Marker A', 'Marker B', 'Marker B'],
            'snr': [0.5, 1.2, 4.5, 6.0],
            'mean_intensity': [1.0, 1.5, 2.0, 2.5],
            'signal_mean': [10.0, 11.0, 30.0, 32.0],
            'coverage_pct': [20.0, 22.0, 35.0, 36.0],
        }
    )

    dialog._plot_distributions()
    ax = dialog.distribution_canvas.figure.axes[0]
    intensity_ax = dialog.distribution_canvas.figure.axes[1]

    threshold_lines = [line for line in ax.lines if line.get_linestyle() == '--']
    assert any(np.allclose(line.get_ydata(), [4.0, 4.0]) for line in threshold_lines)
    legend = ax.get_legend()
    assert legend is not None
    assert any('4.0' in text.get_text() for text in legend.get_texts())
    assert 'foreground mean intensity' in intensity_ax.get_ylabel().lower()
    assert 'foreground intensity distribution' in intensity_ax.get_title().lower()


def test_qc_threshold_spin_repositions_reference_lines_without_rerun(qtbot):
    dialog = _build_qc_dialog(qtbot)
    dialog.qc_results = pd.DataFrame(
        {
            'channel': ['Marker A', 'Marker A', 'Marker B', 'Marker B'],
            'snr': [1.0, 1.5, 4.0, 5.0],
            'cnr': [0.5, 0.8, 2.5, 3.0],
            'mean_intensity': [2.0, 2.5, 3.0, 3.5],
            'signal_mean': [10.0, 12.0, 20.0, 24.0],
            'coverage_pct': [20.0, 22.0, 35.0, 36.0],
        }
    )
    dialog.qc_results_aggregated = pd.DataFrame(
        {
            'channel': ['Marker A', 'Marker B'],
            'snr': [1.25, 4.5],
            'cnr': [0.65, 2.75],
            'mean_intensity': [2.25, 3.25],
            'signal_mean': [11.0, 22.0],
            'coverage_pct': [21.0, 35.5],
        }
    )

    dialog._update_plots()
    dialog.snr_threshold_spin.setValue(6.5)
    dialog.cnr_threshold_spin.setValue(5.5)

    snr_ax = dialog.snr_intensity_canvas.figure.axes[0]
    cnr_ax = dialog.snr_intensity_canvas.figure.axes[1]
    distribution_ax = dialog.distribution_canvas.figure.axes[0]

    snr_threshold_lines = [line for line in snr_ax.lines if line.get_linestyle() == '--']
    cnr_threshold_lines = [line for line in cnr_ax.lines if line.get_linestyle() == '--']
    distribution_threshold_lines = [line for line in distribution_ax.lines if line.get_linestyle() == '--']

    assert any(np.allclose(line.get_ydata(), [6.5, 6.5]) for line in snr_threshold_lines)
    assert any(np.allclose(line.get_ydata(), [5.5, 5.5]) for line in cnr_threshold_lines)
    assert any(np.allclose(line.get_ydata(), [6.5, 6.5]) for line in distribution_threshold_lines)


def test_qc_cell_snr_plot_labels_selected_signal_intensity(qtbot):
    parent = _QCTestParent(with_masks=True)
    dialog = _build_qc_dialog(qtbot, parent)
    dialog.qc_results_aggregated = pd.DataFrame(
        {
            'channel': ['Marker A'],
            'mean_intensity': [1.0],
            'signal_mean': [12.0],
            'snr': [3.5],
            'cell_signal_method': ['positive_pixels'],
        }
    )

    dialog._plot_snr_vs_intensity()
    ax = dialog.snr_intensity_canvas.figure.axes[0]

    assert 'selected in-cell signal only' in ax.get_xlabel().lower()


def test_differential_expression_deduplicates_markers_and_hides_boxes_on_toggle(qtbot):
    dialog = _build_clustering_dialog(qtbot, n_clusters=5, n_features=8)
    dialog.top_n_spinbox.setValue(2)
    dialog.selected_display_features = [f'marker_{idx}_mean' for idx in range(8)]

    dialog._show_differential_expression()
    ax = dialog.figure.axes[0]
    feature_labels = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]

    assert len(feature_labels) == len(set(feature_labels))
    assert len(feature_labels) < (
        dialog.clustered_data['cluster'].nunique() * dialog.top_n_spinbox.value()
    )
    assert len(ax.patches) > 0

    dialog.de_show_boxes_checkbox.setChecked(False)
    dialog._show_differential_expression()
    ax = dialog.figure.axes[0]

    assert len(ax.patches) == 0


def test_differential_expression_helper_note_is_gui_only_when_saving(qtbot, monkeypatch):
    dialog = _build_clustering_dialog(qtbot, n_clusters=4, n_features=6)
    dialog.view_combo.setCurrentText('Differential Expression')
    dialog._show_differential_expression()

    captured = {}

    def _fake_save(figure, default_filename, parent):
        captured['texts'] = [
            text.get_text()
            for axis in figure.axes
            for text in axis.texts
        ]
        return False

    monkeypatch.setattr(clustering_module, 'save_figure_with_options', _fake_save)
    dialog._save_current_plot()

    assert not any('Black boxes highlight' in text for text in captured['texts'])
    assert any(
        'Black boxes highlight' in text.get_text()
        for axis in dialog.figure.axes
        for text in axis.texts
    )


def test_tsne_legend_checkbox_controls_categorical_legend(qtbot):
    dialog = _build_clustering_dialog(qtbot, n_clusters=5, n_features=6)
    dialog.tsne_embedding = np.column_stack(
        [
            np.linspace(-2.0, 2.0, len(dialog.clustered_data)),
            np.linspace(2.0, -2.0, len(dialog.clustered_data)),
        ]
    )
    dialog.tsne_index = dialog.clustered_data.index
    dialog._populate_color_by_options()
    _select_color_by(dialog, 'Cluster')

    dialog.show_legend_checkbox.setChecked(True)
    dialog._create_tsne_plot()
    assert dialog.figure.axes[0].get_legend() is not None

    dialog.show_legend_checkbox.setChecked(False)
    dialog._create_tsne_plot()
    assert dialog.figure.axes[0].get_legend() is None


def test_violin_one_vs_others_only_compares_selected_reference_cluster(qtbot):
    dialog = _build_clustering_dialog(qtbot, n_clusters=4, n_features=4)
    dialog.selected_markers = ['marker_0_mean']
    dialog.plot_type_combo.setCurrentText('Violin Plot')
    dialog.stats_test_checkbox.setChecked(True)
    dialog.stats_mode_combo.setCurrentText('One vs Others')
    dialog._update_stats_cluster_combo()
    dialog.stats_cluster_combo.setCurrentIndex(1)

    dialog._show_boxplot_violin()

    reference_cluster = dialog.stats_cluster_combo.currentText()
    results = dialog.statistical_results['marker_0_mean']
    assert results
    assert all(reference_cluster in (cluster1, cluster2) for cluster1, cluster2, *_ in results)
    assert "Kruskal-Wallis" in dialog._build_statistical_summary_text('marker_0_mean')
    assert any("Kruskal-Wallis" in text.get_text() for text in dialog.figure.axes[0].texts)


def test_violin_invalid_reference_cluster_skips_stats_instead_of_falling_back(qtbot):
    dialog = _build_clustering_dialog(qtbot, n_clusters=4, n_features=4)
    dialog.selected_markers = ['marker_0_mean']
    dialog.plot_type_combo.setCurrentText('Violin Plot')
    dialog.stats_test_checkbox.setChecked(True)
    dialog.stats_mode_combo.setCurrentText('One vs Others')
    dialog.stats_cluster_combo.clear()

    dialog._show_boxplot_violin()

    assert dialog.statistical_results['marker_0_mean'] == []
    assert any(
        'Reference cluster is unavailable' in text.get_text()
        for axis in dialog.figure.axes
        for text in axis.texts
    )


def test_simple_pairwise_enrichment_hides_dense_annotations_and_stays_in_canvas(qtbot):
    dialog = _build_simple_spatial_dialog(qtbot, n_clusters=14)
    dialog.enrichment_df = _build_enrichment_df(14)

    dialog._update_enrichment_plot()
    fig = dialog.enrichment_canvas.figure
    ax = fig.axes[0]
    overflow = measure_figure_text_overflow(fig)

    assert "font-weight: 700" in dialog.advanced_analysis_btn.styleSheet()
    assert max(overflow.values()) <= 0.03
    assert len(ax.texts) == 0


def test_simple_enrichment_finalize_updates_plot_and_save_button_without_tab_switch(qtbot, monkeypatch):
    dialog = _build_simple_spatial_dialog(qtbot, n_clusters=6)
    dialog.edge_df = pd.DataFrame({'cell_id_A': [1, 2], 'cell_id_B': [2, 3], 'roi_id': ['ROI_1', 'ROI_1']})
    dialog._update_tab_states()

    monkeypatch.setattr(
        simple_spatial_module,
        'spatial_enrichment',
        lambda **_kwargs: _build_enrichment_df(6),
    )

    def _sync_runner(*, task, finalize, **_kwargs):
        result = task()
        finalize(result, None)
        return result

    monkeypatch.setattr(simple_spatial_module, 'run_blocking_task_with_progress_then_finalize', _sync_runner)

    dialog._run_enrichment_analysis()
    qtbot.wait(40)

    assert dialog.enrichment_analysis_run
    assert dialog.enrichment_save_btn.isEnabled()
    assert dialog.enrichment_canvas.figure.axes


def test_simple_distance_finalize_updates_plot_and_save_button_without_tab_switch(qtbot, monkeypatch):
    dialog = _build_simple_spatial_dialog(qtbot, n_clusters=6)
    dialog.edge_df = pd.DataFrame({'cell_id_A': [1, 2], 'cell_id_B': [2, 3], 'roi_id': ['ROI_1', 'ROI_1']})
    dialog._update_tab_states()

    monkeypatch.setattr(
        simple_spatial_module,
        'spatial_distance_distribution',
        lambda **_kwargs: _build_distance_df(source_clusters=(1, 2, 3), target_clusters=(1, 2, 3)),
    )

    def _sync_runner(*, task, finalize, **_kwargs):
        result = task()
        finalize(result, None)
        return result

    monkeypatch.setattr(simple_spatial_module, 'run_blocking_task_with_progress_then_finalize', _sync_runner)

    dialog._run_distance_analysis()
    qtbot.wait(40)

    assert dialog.distance_analysis_run
    assert dialog.distance_save_btn.isEnabled()
    assert dialog.distance_canvas.figure.axes
    assert dialog.distance_cluster_list.count() == 3


def test_simple_enrichment_export_adds_roi_level_adjusted_pvalues(qtbot):
    dialog = _build_simple_spatial_dialog(qtbot, n_clusters=4)
    dialog.enrichment_df = pd.DataFrame(
        [
            {'roi_id': 'ROI_1', 'cluster_A': 1, 'cluster_B': 1, 'z_score': 2.1, 'p_value': 0.01},
            {'roi_id': 'ROI_1', 'cluster_A': 1, 'cluster_B': 2, 'z_score': 1.3, 'p_value': 0.04},
            {'roi_id': 'ROI_2', 'cluster_A': 1, 'cluster_B': 1, 'z_score': 2.4, 'p_value': 0.02},
            {'roi_id': 'ROI_2', 'cluster_A': 1, 'cluster_B': 2, 'z_score': 1.0, 'p_value': 0.03},
        ]
    )

    export_df = dialog._prepare_enrichment_export_df()

    assert 'p_value_adjusted' in export_df.columns
    roi_1_adjusted = export_df.loc[export_df['roi_id'] == 'ROI_1', 'p_value_adjusted'].to_numpy()
    roi_2_adjusted = export_df.loc[export_df['roi_id'] == 'ROI_2', 'p_value_adjusted'].to_numpy()
    np.testing.assert_allclose(roi_1_adjusted, np.array([0.02, 0.04]))
    np.testing.assert_allclose(roi_2_adjusted, np.array([0.03, 0.03]))
    assert export_df['cluster_A_label'].iloc[0].startswith('Cluster 1')


def test_simple_distance_plot_uses_faceted_horizontal_boxplots_and_fits_canvas(qtbot):
    dialog = _build_simple_spatial_dialog(qtbot, n_clusters=6)
    dialog.distance_df = _build_distance_df(source_clusters=(1, 2), target_clusters=(1, 2, 3))
    dialog.distance_analysis_run = True
    dialog._populate_distance_cluster_list()
    dialog.distance_show_self_pairs_check.setChecked(False)

    for idx in range(dialog.distance_cluster_list.count()):
        dialog.distance_cluster_list.item(idx).setSelected(idx < 2)

    dialog._update_distance_plot()
    qtbot.wait(40)
    fig = dialog.distance_canvas.figure
    overflow = measure_figure_text_overflow(fig)

    assert len(fig.axes) == 2
    assert all('From Cluster' in axis.get_title() for axis in fig.axes)
    assert max(overflow.values()) <= 0.04


def test_simple_spatial_label_customization_syncs_with_parent_and_color_labels(qtbot, monkeypatch):
    parent = QtWidgets.QWidget()
    parent.clustering_dialog = SimpleNamespace(
        cluster_annotation_map={1: 'Myeloid niche', 2: 'Lymphoid niche'},
        feature_label_map={'marker_signal_mean': 'Signal intensity'},
        filter_settings=None,
    )
    qtbot.addWidget(parent)

    df = pd.DataFrame(
        {
            'cell_id': [1, 2],
            'centroid_x': [0.0, 1.0],
            'centroid_y': [0.0, 1.0],
            'acquisition_id': ['ROI_1', 'ROI_1'],
            'cluster': [1, 2],
            'marker_signal_mean': [0.1, 0.9],
            'marker_signal_std': [0.01, 0.03],
        }
    )
    dialog = SimpleSpatialAnalysisDialog(df, clustered_cells_dataframe=df.copy(), parent=parent)
    qtbot.addWidget(dialog)

    assert dialog.cluster_annotation_map[1] == 'Myeloid niche'
    assert dialog.feature_label_map['marker_signal_mean'] == 'Signal intensity'
    combo_labels = [dialog.spatial_color_combo.itemText(i) for i in range(dialog.spatial_color_combo.count())]
    assert 'Signal intensity' in combo_labels
    assert 'marker_signal_std' not in [dialog.spatial_color_combo.itemData(i) for i in range(dialog.spatial_color_combo.count())]

    monkeypatch.setattr(
        simple_spatial_module,
        'edit_feature_label_map',
        lambda *args, **kwargs: {'marker_signal_mean': 'Re-labeled signal'},
    )
    monkeypatch.setattr(
        simple_spatial_module,
        'edit_cluster_annotation_map',
        lambda *args, **kwargs: {1: 'Updated cluster 1', 2: 'Updated cluster 2'},
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, 'information', lambda *args, **kwargs: QtWidgets.QMessageBox.Ok)

    dialog._open_feature_labels_dialog()
    dialog._open_cluster_labels_dialog()

    assert dialog.feature_label_map['marker_signal_mean'] == 'Re-labeled signal'
    assert dialog.cluster_annotation_map[1] == 'Updated cluster 1'
    assert parent.clustering_dialog.feature_label_map['marker_signal_mean'] == 'Re-labeled signal'
    assert parent.clustering_dialog.cluster_annotation_map[1] == 'Updated cluster 1'


def test_simple_spatial_visualization_uses_colorbar_for_continuous_features_and_legend_for_clusters(qtbot):
    df = pd.DataFrame(
        {
            'cell_id': [1, 2, 3],
            'centroid_x': [0.0, 1.0, 2.0],
            'centroid_y': [0.0, 0.5, 1.0],
            'acquisition_id': ['ROI_1', 'ROI_1', 'ROI_1'],
            'cluster': ['1', '10', '2'],
            'marker_signal_mean': [0.1, 0.5, 0.9],
            'marker_signal_std': [0.01, 0.02, 0.03],
        }
    )
    dialog = SimpleSpatialAnalysisDialog(df, clustered_cells_dataframe=df.copy())
    qtbot.addWidget(dialog)
    dialog.resize(1100, 820)
    dialog.show()
    qtbot.wait(40)

    dialog.cluster_annotation_map = {1: 'Cluster one', 2: 'Cluster two', 10: 'Cluster ten'}
    dialog._apply_cluster_annotations_to_dataframes()
    dialog._populate_spatial_color_options()
    combo_data = [dialog.spatial_color_combo.itemData(idx) for idx in range(dialog.spatial_color_combo.count())]
    assert 'marker_signal_mean' in combo_data
    assert 'marker_signal_std' not in combo_data

    feature_index = next(
        idx for idx in range(dialog.spatial_color_combo.count())
        if dialog.spatial_color_combo.itemData(idx) == 'marker_signal_mean'
    )
    dialog.spatial_color_combo.setCurrentIndex(feature_index)
    feature_cache = dialog._build_spatial_visualization_cache_data('ROI_1', color_option='marker_signal_mean', show_edges=False)
    dialog._render_spatial_visualization('ROI_1', feature_cache)

    assert dialog.spatial_color_combo.completer().filterMode() == QtCore.Qt.MatchContains
    assert dialog.spatial_viz_canvas.figure.axes[0].get_legend() is None
    assert len(dialog.spatial_viz_canvas.figure.axes) == 2

    cluster_index = next(
        idx for idx in range(dialog.spatial_color_combo.count())
        if dialog.spatial_color_combo.itemData(idx) == 'cluster'
    )
    dialog.spatial_color_combo.setCurrentIndex(cluster_index)
    cluster_cache = dialog._build_spatial_visualization_cache_data('ROI_1', color_option='cluster', show_edges=False)
    dialog._render_spatial_visualization('ROI_1', cluster_cache)

    legend = dialog.spatial_viz_canvas.figure.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.texts] == ['Cluster one', 'Cluster two', 'Cluster ten']


def test_simple_spatial_visualization_prefers_single_column_legend_when_height_allows(qtbot):
    dialog = _build_simple_spatial_dialog(qtbot, n_clusters=14)

    cluster_index = next(
        idx for idx in range(dialog.spatial_color_combo.count())
        if dialog.spatial_color_combo.itemData(idx) == 'cluster'
    )
    dialog.spatial_color_combo.setCurrentIndex(cluster_index)
    cluster_cache = dialog._build_spatial_visualization_cache_data(
        'ROI_1',
        color_option='cluster',
        show_edges=False,
    )
    dialog._render_spatial_visualization('ROI_1', cluster_cache)
    qtbot.wait(60)

    figure = dialog.spatial_viz_canvas.figure
    legend = figure.axes[0].get_legend()
    overflow = measure_figure_text_overflow(figure)

    assert legend is not None
    assert _legend_ncols(legend) == 1
    assert max(overflow.values()) <= 0.035


def test_phenotype_suggestion_dialog_computes_stats_without_parent_sorted_helper(qtbot):
    df = pd.DataFrame(
        {
            'cell_id': [1, 2, 3, 4, 5, 6],
            'centroid_x': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'centroid_y': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            'acquisition_id': ['ROI_1'] * 6,
            'cluster': ['10', '10', '2', '2', '1', '1'],
            'CD3_mean': [0.9, 0.8, 0.2, 0.1, 0.3, 0.25],
            'Area': [10.0, 11.0, 20.0, 21.0, 15.0, 16.0],
        }
    )
    parent_dialog = CellClusteringDialog(df, clustered_cells_dataframe=df.copy())
    qtbot.addWidget(parent_dialog)

    suggestion_dialog = clustering_module.PhenotypeSuggestionDialog(
        parent_dialog,
        ['10', '2', '1'],
        lambda *args, **kwargs: None,
        {},
        parent_dialog.normalization_config,
    )
    qtbot.addWidget(suggestion_dialog)

    stats = suggestion_dialog._compute_stats(df, mode="Both", k_int=1, k_morpho=1)

    assert list(stats.keys()) == ['1', '2', '10']


def _skip_if_squidpy_unavailable():
    from openimc.ui.dialogs import advanced_spatial_analysis as advanced_module

    if not (advanced_module._HAVE_SQUIDPY or advanced_module._HAVE_SQUIDPY_LOCAL):
        if not advanced_module._ensure_local_squidpy():
            pytest.skip("squidpy not available")
    return advanced_module.AdvancedSpatialAnalysisDialog


def _build_advanced_dialog(qtbot, n_clusters: int = 14):
    AdvancedSpatialAnalysisDialog = _skip_if_squidpy_unavailable()
    df = _build_spatial_dataframe(n_clusters=n_clusters)
    dialog = AdvancedSpatialAnalysisDialog(df, clustered_cells_dataframe=df.copy())
    qtbot.addWidget(dialog)
    dialog.cluster_annotation_map = {
        cluster_id: f'Cluster {cluster_id} with a very long advanced spatial label'
        for cluster_id in range(1, n_clusters + 1)
    }
    dialog.resize(1280, 920)
    dialog.show()
    qtbot.wait(120)
    return dialog


class _TempAnnData:
    def __init__(
        self,
        uns,
        categories,
        cluster_key='cluster',
        *,
        var_names=None,
        X=None,
        obsm=None,
        obsp=None,
    ):
        self.uns = deepcopy(uns)
        obs = pd.DataFrame({cluster_key: pd.Categorical(categories, categories=categories)})
        obs.index = [str(category) for category in categories]
        self.obs = obs
        self.var_names = np.asarray(var_names if var_names is not None else [], dtype=object)
        self.X = X if X is not None else np.empty((len(categories), 0))
        self.obsm = obsm or {}
        self.obsp = obsp or {}

    def copy(self):
        cluster_key = self.obs.columns[0]
        categories = list(self.obs[cluster_key].cat.categories) if hasattr(self.obs[cluster_key], 'cat') else self.obs[cluster_key].tolist()
        return _TempAnnData(
            deepcopy(self.uns),
            categories,
            cluster_key=cluster_key,
            var_names=np.array(self.var_names, copy=True),
            X=np.array(self.X, copy=True),
            obsm={key: np.array(value, copy=True) for key, value in self.obsm.items()},
            obsp={key: np.array(value, copy=True) if hasattr(value, 'shape') else deepcopy(value) for key, value in self.obsp.items()},
        )


def test_advanced_neighborhood_enrichment_compacts_dense_labels(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=14)
    categories = list(range(1, 15))
    matrix = np.fromfunction(lambda i, j: (i - j) / 2.0, (14, 14), dtype=float)
    adata = _TempAnnData({'nhood_enrichment': {'zscore': matrix}}, categories)

    dialog._plot_sq_nhood_enrichment(adata)
    fig = dialog.sq_nhood_canvas.figure
    ax = fig.axes[0]
    overflow = measure_figure_text_overflow(fig)

    assert max(overflow.values()) <= 0.03
    assert not any('Cell annotations hidden for readability' in text.get_text() for text in ax.texts)


def test_advanced_nhood_export_dataframe_contains_adjusted_pvalues(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    categories = [1, 2]
    matrix = np.array([[2.5, -1.0], [-1.5, 3.0]], dtype=float)
    pvalue_matrix = np.array([[0.01, 0.25], [0.25, 0.02]], dtype=float)
    adjusted_matrix = np.array([[0.04, 0.25], [0.25, 0.04]], dtype=float)
    adata = _TempAnnData(
        {
            'nhood_enrichment': {
                'zscore': matrix,
                'pvalue': pvalue_matrix,
                'pvalue_fdr_bh': adjusted_matrix,
            }
        },
        categories,
    )
    adata._significant_counts = np.array([[1, 0], [0, 2]], dtype=int)

    export_df = dialog._build_sq_nhood_export_df(
        adata,
        roi_label='All ROIs',
        cluster_key='cluster',
        aggregation_method='mean',
    )

    assert len(export_df) == 4
    assert {'p_value', 'p_value_adjusted', 'significant_roi_count', 'aggregation_method', 'p_value_source'}.issubset(export_df.columns)
    np.testing.assert_allclose(export_df['p_value'].to_numpy(), pvalue_matrix.reshape(-1))
    np.testing.assert_allclose(export_df['p_value_adjusted'].to_numpy(), adjusted_matrix.reshape(-1))
    assert set(export_df['p_value_source']) == {'permutation'}


def test_advanced_cooccurrence_heatmap_compacts_dense_labels(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=14)
    dialog.sq_cooccur_plot_type_combo.setCurrentText("Heatmap")
    dialog.sq_cooccur_interval = [10.0, 20.0, 30.0]
    categories = list(range(1, 15))
    occ = _build_cooccurrence_array(14, 3)
    adata = _TempAnnData(
        {'co_occurrence': {'occ': occ, 'interval': [0.0, 10.0, 20.0, 30.0]}},
        categories,
    )

    dialog._plot_sq_cooccurrence(adata)
    fig = dialog.sq_cooccur_canvas.figure
    ax = fig.axes[0]
    overflow = measure_figure_text_overflow(fig)

    assert max(overflow.values()) <= 0.03
    assert not any('Cell annotations hidden for readability' in text.get_text() for text in ax.texts)


def test_advanced_cooccurrence_heatmap_preserves_typed_distance_selection(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    dialog.sq_cooccur_plot_type_combo.setCurrentText("Heatmap")
    assert isinstance(dialog.sq_cooccur_distance_spin, QtWidgets.QDoubleSpinBox)
    dialog.sq_cooccur_interval = [10.0, 20.0, 30.0]
    dialog.sq_cooccur_distance_spin.setValue(20.0)

    categories = [1, 2, 3, 4]
    occ = _build_cooccurrence_array(4, 3)
    adata = _TempAnnData(
        {'co_occurrence': {'occ': occ, 'interval': [0.0, 10.0, 20.0, 30.0]}},
        categories,
    )

    dialog._plot_sq_cooccurrence(adata)
    ax = dialog.sq_cooccur_canvas.figure.axes[0]

    assert ax.get_title() == 'Co-occurrence Analysis at 20 µm'
    assert dialog.sq_cooccur_distance_spin.value() == pytest.approx(20.0)
    assert dialog.sq_cooccur_heatmap_distance == pytest.approx(20.0)


def test_advanced_cooccurrence_reference_cluster_defaults_to_one_without_extra_plot(qtbot, monkeypatch):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    categories = [1, 2, 3, 4]
    adata = _TempAnnData(
        {'co_occurrence': {'occ': _build_cooccurrence_array(4, 3), 'interval': [0.0, 10.0, 20.0, 30.0]}},
        categories,
    )
    dialog.anndata_cache = {'ROI_1': adata}

    plot_calls = []
    monkeypatch.setattr(dialog, '_plot_sq_cooccurrence', lambda _adata: plot_calls.append(_adata))

    dialog._update_cooccur_ref_cluster_combo(adata, preserve_selection=True)

    assert plot_calls == []
    assert dialog.sq_cooccur_ref_cluster_combo.currentData() == 1


@pytest.mark.parametrize(
    ("plot_type", "selected_cluster", "expected_reference_cluster"),
    [
        ("Heatmap", 2, None),
        ("Curves", 2, 2),
        ("Curves", None, None),
    ],
)
def test_advanced_cooccurrence_run_respects_plot_type_and_cluster_selection(
    qtbot,
    monkeypatch,
    plot_type,
    selected_cluster,
    expected_reference_cluster,
):
    from openimc.ui.dialogs import advanced_spatial_analysis as advanced_module

    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    categories = [1, 2, 3, 4]
    base_adata = _TempAnnData(
        {},
        categories,
        obsp={'spatial_connectivities': np.eye(4, dtype=float)},
    )
    dialog.spatial_graph_built = True
    dialog.anndata_cache = {'ROI_1': base_adata}
    dialog._update_cooccur_ref_cluster_combo(base_adata, preserve_selection=False)
    dialog.sq_cooccur_plot_type_combo.setCurrentText(plot_type)

    if selected_cluster is None:
        dialog.sq_cooccur_ref_cluster_combo.setCurrentIndex(0)
    else:
        for idx in range(dialog.sq_cooccur_ref_cluster_combo.count()):
            if dialog.sq_cooccur_ref_cluster_combo.itemData(idx) == selected_cluster:
                dialog.sq_cooccur_ref_cluster_combo.setCurrentIndex(idx)
                break

    captured = {}

    def _fake_cooccurrence(*, anndata_dict, cluster_key, interval, reference_cluster):
        captured['reference_cluster'] = reference_cluster
        result_adata = base_adata.copy()
        result_adata.uns['co_occurrence'] = {
            'occ': _build_cooccurrence_array(len(categories), len(interval)),
            'interval': [0.0, *interval],
        }
        return {'ROI_1': result_adata}

    monkeypatch.setattr(advanced_module, 'spatial_cooccurrence', _fake_cooccurrence)
    monkeypatch.setattr(
        advanced_module,
        'run_blocking_task_with_progress',
        lambda **kwargs: kwargs['task'](),
    )
    monkeypatch.setattr(advanced_module.QtWidgets.QMessageBox, 'information', lambda *args, **kwargs: None)
    monkeypatch.setattr(dialog, '_plot_sq_cooccurrence', lambda _adata: None)

    dialog._run_sq_cooccurrence()

    assert captured['reference_cluster'] == expected_reference_cluster


def test_advanced_cooccurrence_export_dataframe_is_long_form(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    categories = [1, 2]
    occ = _build_cooccurrence_array(2, 3)
    adata = _TempAnnData(
        {'co_occurrence': {'occ': occ, 'interval': [0.0, 10.0, 20.0, 30.0]}},
        categories,
    )

    export_df = dialog._build_sq_cooccur_export_df(
        adata,
        roi_label='ROI_1',
        cluster_key='cluster',
    )

    assert len(export_df) == 12
    assert export_df['distance_um'].tolist()[:3] == [10.0, 20.0, 30.0]
    assert {'source_cluster', 'target_cluster', 'co_occurrence_score'}.issubset(export_df.columns)


def test_advanced_cooccurrence_curves_draw_lines_without_interval_shading(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=6)
    dialog.sq_cooccur_plot_type_combo.setCurrentText("Curves")
    dialog.sq_cooccur_ref_cluster_combo.clear()
    dialog.sq_cooccur_ref_cluster_combo.addItem("Cluster 2", 2)
    dialog.sq_cooccur_ref_cluster_combo.setCurrentIndex(0)
    categories = list(range(1, 7))
    occ = _build_cooccurrence_array(6, 3)
    adata = _TempAnnData(
        {'co_occurrence': {'occ': occ, 'interval': [0.0, 10.0, 20.0, 30.0]}},
        categories,
    )

    dialog._plot_sq_cooccurrence(adata)

    assert len(dialog.sq_cooccur_canvas.figure.axes) == 1
    assert all(
        collection.__class__.__name__ != 'FillBetweenPolyCollection'
        for axis in dialog.sq_cooccur_canvas.figure.axes
        for collection in axis.collections
    )


def test_advanced_autocorr_bar_plot_orders_largest_values_at_top(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    adata = _TempAnnData(
        {'moranI': {'I': np.array([0.15, 0.72, -0.10, 0.41]), 'pval_norm': np.array([0.2, 0.001, 0.7, 0.03]), 'var_names': ['A', 'B', 'C', 'D']}},
        [1, 2, 3, 4],
    )
    dialog.sq_autocorr_topk_spin.setValue(4)

    dialog._plot_sq_autocorrelation(adata)
    ax = dialog.sq_autocorr_canvas.figure.axes[0]
    tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]

    assert tick_labels[0] == 'B'
    assert tick_labels[-1] == 'C'


def test_advanced_autocorr_export_dataframe_adds_adjusted_pvalues(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    adata = _TempAnnData(
        {
            'moranI': {
                'I': np.array([0.72, 0.41]),
                'pval_norm': np.array([0.01, 0.04]),
                'pval_sim': np.array([0.001, 0.03]),
                'pval_sim_fdr_bh': np.array([0.002, 0.03]),
                'var_names': ['A', 'B'],
            }
        },
        [1, 2],
    )

    export_df = dialog._build_sq_autocorr_export_df(
        adata,
        roi_label='ROI_1',
    )

    assert {'feature', 'feature_label', 'moran_i', 'p_value', 'p_value_adjusted', 'p_value_source'}.issubset(export_df.columns)
    np.testing.assert_allclose(export_df['p_value'].to_numpy(), np.array([0.001, 0.03]))
    np.testing.assert_allclose(export_df['p_value_adjusted'].to_numpy(), np.array([0.002, 0.03]))
    assert set(export_df['p_value_source']) == {'pval_sim'}


def test_advanced_autocorr_visualizations_coerce_object_values_without_crashing(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    adata = _TempAnnData(
        {'moranI': {'I': np.array([0.25]), 'pval_norm': np.array([0.01]), 'var_names': ['marker_a']}},
        [1, 2, 3, 4],
        var_names=['marker_a'],
        X=np.array([[1.0], [2.0], [3.5], [4.5]], dtype=object),
        obsm={'spatial': np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 0.6]], dtype=float)},
        obsp={'spatial_connectivities': np.eye(4, dtype=float)},
    )
    dialog.anndata_cache = {'ROI_1': adata}
    dialog.sq_autocorr_roi_combo.clear()
    dialog.sq_autocorr_roi_combo.addItem('ROI_1', 'ROI_1')
    dialog.sq_autocorr_roi_combo.setCurrentIndex(0)
    dialog.sq_autocorr_var_combo.clear()
    dialog.sq_autocorr_var_combo.addItem('marker_a', 'marker_a')
    dialog.sq_autocorr_var_combo.setCurrentIndex(0)

    dialog.sq_autocorr_viz_type_combo.setCurrentText("Moran Scatter Plot")
    dialog._plot_sq_autocorr_visualization()
    assert dialog.sq_autocorr_canvas.figure.axes

    dialog.sq_autocorr_viz_type_combo.setCurrentText("Spatial Map")
    dialog._plot_sq_autocorr_visualization()
    assert dialog.sq_autocorr_canvas.figure.axes


def test_advanced_spatial_transfers_feature_labels_to_autocorr_selector(qtbot):
    parent = QtWidgets.QWidget()
    parent.clustering_dialog = SimpleNamespace(
        cluster_annotation_map={1: 'Tumor core', 2: 'Stroma', 3: 'T cells', 4: 'B cells'},
        feature_label_map={'marker_a': 'Marker A display'},
        filter_settings=None,
    )
    qtbot.addWidget(parent)

    AdvancedSpatialAnalysisDialog = _skip_if_squidpy_unavailable()
    df = pd.DataFrame(
        {
            'cell_id': [1, 2, 3, 4],
            'centroid_x': [0.0, 1.0, 2.0, 3.0],
            'centroid_y': [0.0, 1.0, 2.0, 3.0],
            'acquisition_id': ['ROI_1'] * 4,
            'cluster': [1, 2, 3, 4],
            'marker_a': [0.1, 0.2, 0.3, 0.4],
        }
    )
    dialog = AdvancedSpatialAnalysisDialog(df, clustered_cells_dataframe=df.copy(), parent=parent)
    qtbot.addWidget(dialog)
    dialog.anndata_cache = {
        'ROI_1': _TempAnnData(
            {'moranI': {'I': np.array([0.4]), 'pval_norm': np.array([0.002]), 'var_names': ['marker_a']}},
            [1, 2, 3, 4],
            var_names=['marker_a'],
        )
    }
    dialog.sq_autocorr_roi_combo.clear()
    dialog.sq_autocorr_roi_combo.addItem('ROI_1', 'ROI_1')
    dialog.sq_autocorr_roi_combo.setCurrentIndex(0)

    dialog._update_autocorr_var_combo()

    assert dialog._get_cluster_display_name(1) == 'Tumor core'
    assert dialog.sq_autocorr_var_combo.itemText(0).startswith('Marker A display')


def test_advanced_ripley_curves_draw_lines_without_interval_shading(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    categories = list(range(1, 5))
    stat_df = pd.DataFrame(
        {
            'bins': [10.0, 20.0, 30.0] * 4,
            'stats': np.linspace(0.2, 1.4, 12),
            'cluster': np.repeat(categories, 3),
        }
    )
    sims_df = pd.DataFrame(
        {
            'bins': [10.0, 20.0, 30.0] * 3,
            'stats': np.linspace(0.1, 0.6, 9),
        }
    )
    adata = _TempAnnData({'ripley': {'L_stat': stat_df, 'sims_stat': sims_df}}, categories)
    dialog.sq_ripley_mode_combo.setCurrentText('L')

    dialog._plot_sq_ripley(adata, 'cluster')

    assert all(
        collection.__class__.__name__ != 'FillBetweenPolyCollection'
        for axis in dialog.sq_ripley_canvas.figure.axes
        for collection in axis.collections
    )
    assert any(line.get_linestyle() == '--' for line in dialog.sq_ripley_canvas.figure.axes[0].lines)


def test_advanced_ripley_export_dataframe_merges_simulation_summary(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    categories = [1, 2]
    stat_df = pd.DataFrame(
        {
            'bins': [10.0, 20.0, 10.0, 20.0],
            'stats': [0.5, 0.7, 0.2, 0.3],
            'cluster': [1, 1, 2, 2],
        }
    )
    sims_df = pd.DataFrame(
        {
            'bins': [10.0, 10.0, 20.0, 20.0],
            'stats': [0.1, 0.2, 0.3, 0.5],
        }
    )
    adata = _TempAnnData(
        {
            'ripley': {
                'L_stat': stat_df,
                'sims_stat': sims_df,
                'bins': np.array([10.0, 20.0], dtype=float),
                'pvalues': np.array([[0.01, 0.04], [0.20, 0.30]], dtype=float),
                'pvalues_fdr_bh': np.array([[0.04, 0.08], [0.30, 0.30]], dtype=float),
            }
        },
        categories,
    )
    dialog.sq_ripley_mode_combo.setCurrentText('L')

    export_df = dialog._build_sq_ripley_export_df(
        adata,
        roi_label='ROI_1',
        cluster_key='cluster',
    )

    assert {'simulation_mean', 'simulation_std', 'simulation_count', 'stat_value', 'p_value', 'p_value_adjusted'}.issubset(export_df.columns)
    assert export_df['simulation_count'].dropna().iloc[0] == 2
    assert export_df.loc[(export_df['cluster'] == 1) & (export_df['distance_um'] == 10.0), 'p_value'].iloc[0] == pytest.approx(0.01)
    assert export_df.loc[(export_df['cluster'] == 1) & (export_df['distance_um'] == 10.0), 'p_value_adjusted'].iloc[0] == pytest.approx(0.04)


def test_advanced_ripley_aggregation_combines_pvalues_across_rois(qtbot):
    dialog = _build_advanced_dialog(qtbot, n_clusters=4)
    stat_df_roi_1 = pd.DataFrame(
        {
            'bins': [10.0, 20.0, 10.0, 20.0],
            'stats': [0.5, 0.7, 0.2, 0.3],
            'cluster': [1, 1, 2, 2],
        }
    )
    stat_df_roi_2 = pd.DataFrame(
        {
            'bins': [10.0, 20.0, 10.0, 20.0],
            'stats': [0.4, 0.6, 0.25, 0.35],
            'cluster': [1, 1, 2, 2],
        }
    )
    sims_df = pd.DataFrame(
        {
            'bins': [10.0, 10.0, 20.0, 20.0],
            'stats': [0.1, 0.2, 0.3, 0.5],
        }
    )
    roi_1 = _TempAnnData(
        {
            'ripley': {
                'L_stat': stat_df_roi_1,
                'sims_stat': sims_df,
                'bins': np.array([10.0, 20.0], dtype=float),
                'pvalues': np.array([[0.01, 0.04], [0.20, 0.30]], dtype=float),
            }
        },
        [1, 2],
    )
    roi_2 = _TempAnnData(
        {
            'ripley': {
                'L_stat': stat_df_roi_2,
                'sims_stat': sims_df,
                'bins': np.array([10.0, 20.0], dtype=float),
                'pvalues': np.array([[0.02, 0.05], [0.10, 0.40]], dtype=float),
            }
        },
        [1, 2],
    )

    aggregated = dialog._aggregate_ripley_results({'ROI_1': roi_1, 'ROI_2': roi_2}, 'cluster', 'L', 'mean')

    combined = aggregated.uns['ripley']['pvalues']
    expected_cluster_1_bin_10 = combine_pvalues_fisher([0.01, 0.02])
    expected_cluster_2_bin_20 = combine_pvalues_fisher([0.30, 0.40])

    assert combined.shape == (2, 2)
    assert combined[0, 0] == pytest.approx(expected_cluster_1_bin_10)
    assert combined[1, 1] == pytest.approx(expected_cluster_2_bin_20)
    assert 'pvalues_fdr_bh' in aggregated.uns['ripley']


def test_explore_clusters_button_stays_disabled_without_masks(qtbot):
    parent = _QCTestParent(with_masks=False)
    dialog = _build_clustering_dialog(qtbot, n_clusters=4, n_features=6, parent=parent)

    dialog._update_cluster_action_buttons()

    assert not dialog.explore_btn.isEnabled()
    assert "Load segmentation masks" in dialog.explore_btn.toolTip()
