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
Unit tests for CLI functions.
"""
import builtins
import importlib
import pytest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile

from openimc.cli import load_data, parse_denoise_settings, qc_analysis_command
import openimc.cli as cli_module
from openimc.core import qc_analysis
from openimc.data.mcd_loader import AcquisitionInfo


class _DummyQCLoader:
    def __init__(self, stack, acquisition):
        self._stack = stack
        self._acquisition = acquisition

    def list_acquisitions(self):
        return [self._acquisition]

    def get_channels(self, _acq_id):
        return list(self._acquisition.channels)

    def get_all_channels(self, _acq_id):
        return self._stack

    def get_image(self, _acq_id, channel):
        index = self._acquisition.channels.index(channel)
        return self._stack[:, :, index]

    def close(self):
        return None


@pytest.mark.unit
class TestLoadData:
    """Tests for load_data CLI function."""
    
    def test_load_data_invalid_path(self):
        """Test load_data with invalid path raises error."""
        with pytest.raises(ValueError, match="Input path must be"):
            load_data("/nonexistent/path")
    
    def test_load_data_directory(self, mock_ometiff_directory):
        """Test load_data with OME-TIFF directory."""
        loader, loader_type = load_data(str(mock_ometiff_directory))
        
        assert loader_type == 'ometiff'
        assert loader is not None
    
    def test_load_data_channel_format(self, mock_ometiff_directory):
        """Test load_data with custom channel format."""
        loader, loader_type = load_data(str(mock_ometiff_directory), channel_format='HWC')
        
        assert loader_type == 'ometiff'
        assert loader.channel_format == 'HWC'


@pytest.mark.unit
class TestParseDenoiseSettings:
    """Tests for parse_denoise_settings CLI function."""
    
    def test_parse_denoise_settings_none(self):
        """Test parse_denoise_settings with None."""
        result = parse_denoise_settings(None)
        assert result == {}
    
    def test_parse_denoise_settings_empty_string(self):
        """Test parse_denoise_settings with empty string."""
        result = parse_denoise_settings("")
        assert result == {}
    
    def test_parse_denoise_settings_json_string(self):
        """Test parse_denoise_settings with JSON string."""
        json_str = '{"DAPI": {"hot": {"method": "median3"}}}'
        result = parse_denoise_settings(json_str)
        
        assert isinstance(result, dict)
        assert "DAPI" in result
    
    def test_parse_denoise_settings_json_file(self, temp_dir):
        """Test parse_denoise_settings with JSON file."""
        settings = {
            "DAPI": {
                "hot": {"method": "median3"},
                "speckle": {"method": "gaussian", "sigma": 0.8}
            }
        }
        
        json_file = temp_dir / "denoise_settings.json"
        with open(json_file, 'w') as f:
            json.dump(settings, f)
        
        result = parse_denoise_settings(str(json_file))
        
        assert isinstance(result, dict)
        assert "DAPI" in result
        assert "hot" in result["DAPI"]
    
    def test_parse_denoise_settings_invalid_json(self):
        """Test parse_denoise_settings with invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_denoise_settings("{invalid json}")


@pytest.mark.unit
class TestQCAnalysisCLI:
    """Tests for QC CLI signal-definition plumbing."""

    def test_qc_analysis_command_round_trips_cell_signal_options(self, temp_dir, monkeypatch):
        channels = ["MarkerA"]
        img = np.array(
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
            dtype=np.uint32,
        )
        stack = np.stack([img], axis=-1)
        acquisition = AcquisitionInfo(
            id="ROI_1",
            name="ROI_1",
            well="A1",
            size=img.shape,
            channels=channels,
            channel_metals=[""],
            channel_labels=[""],
            metadata={},
            source_file="dummy",
        )
        loader = _DummyQCLoader(stack, acquisition)
        mask_path = temp_dir / "mask.tif"
        output_path = temp_dir / "qc.csv"
        tifffile.imwrite(mask_path, mask)

        monkeypatch.setattr("openimc.cli.load_mcd", lambda *_args, **_kwargs: (loader, "ometiff"))

        args = SimpleNamespace(
            input="dummy_input",
            output=str(output_path),
            channel_format="CHW",
            acquisition=None,
            channels=None,
            mask=str(mask_path),
            mode="cell",
            cell_signal_method="upper_quantile",
            positive_threshold_sd=2.5,
            upper_quantile=0.75,
        )

        qc_analysis_command(args)

        expected = qc_analysis(
            loader=loader,
            acquisition=acquisition,
            channels=channels,
            mode="cell",
            mask=mask,
            cell_signal_method="upper_quantile",
            positive_threshold_sd=2.5,
            upper_quantile=0.75,
        ).reset_index(drop=True)
        actual = pd.read_csv(output_path).reset_index(drop=True)

        assert actual.loc[0, "cell_signal_method"] == "upper_quantile"
        assert actual.loc[0, "signal_quantile"] == pytest.approx(0.75)
        assert actual.loc[0, "snr"] == pytest.approx(expected.loc[0, "snr"])
        assert actual.loc[0, "cnr"] == pytest.approx(expected.loc[0, "cnr"])
        assert actual.loc[0, "signal_mean"] == pytest.approx(expected.loc[0, "signal_mean"])


@pytest.mark.unit
def test_cli_module_import_defers_optional_segmentation_backends(monkeypatch):
    attempted_imports = []
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split('.')[0]
        if top_level in {'cellpose', 'cellSAM'}:
            attempted_imports.append(top_level)
            raise AssertionError(f"unexpected optional import: {top_level}")
        return real_import(name, globals, locals, fromlist, level)

    def fake_find_spec(name, package=None):
        if name in {'cellpose', 'cellSAM'}:
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    importlib.reload(cli_module)

    assert cli_module._HAVE_CELLPOSE is False
    assert cli_module._HAVE_CELLSAM is False
    assert attempted_imports == []


def test_cluster_command_passes_pca_options(tmp_path, monkeypatch):
    features_path = tmp_path / "features.csv"
    output_path = tmp_path / "clustered.csv"
    pd.DataFrame(
        {
            "marker_1_mean": [0.1, 0.2, 1.0, 1.1],
            "marker_2_mean": [0.0, 0.1, 0.9, 1.0],
        }
    ).to_csv(features_path, index=False)

    captured = {}

    def fake_cluster(**kwargs):
        captured.update(kwargs)
        result = kwargs["features_df"].copy()
        result["cluster"] = [1, 1, 2, 2]
        return result

    monkeypatch.setattr(cli_module, "cluster", fake_cluster)
    args = SimpleNamespace(
        features=str(features_path),
        output=str(output_path),
        method="kmeans",
        columns="marker_1_mean,marker_2_mean",
        scaling="zscore",
        n_clusters=2,
        linkage="ward",
        resolution=1.0,
        seed=123,
        n_neighbors=15,
        metric="euclidean",
        use_jaccard=False,
        n_init=10,
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_method="eom",
        hdbscan_metric="euclidean",
        use_pca=True,
        pca_mode="components",
        pca_variance=0.95,
        pca_n_components=2,
    )

    cli_module.cluster_command(args)

    assert captured["use_pca"] is True
    assert captured["pca_mode"] == "components"
    assert captured["pca_variance"] == pytest.approx(0.95)
    assert captured["pca_n_components"] == 2
    assert captured["columns"] == ["marker_1_mean", "marker_2_mean"]


def test_workflow_clustering_passes_pca_config(tmp_path, monkeypatch):
    features_path = tmp_path / "features.csv"
    output_dir = tmp_path / "workflow_output"
    config_path = tmp_path / "workflow.yaml"
    pd.DataFrame({"marker_mean": [0.1, 0.2, 1.0]}).to_csv(features_path, index=False)
    config_path.write_text(
        f"output: {output_dir}\n"
        "clustering:\n"
        "  enabled: true\n"
        f"  input_features: {features_path}\n"
        "  method: kmeans\n"
        "  n_clusters: 2\n"
        "  use_pca: true\n"
        "  pca_mode: components\n"
        "  pca_n_components: 2\n"
    )

    captured = {}

    def fake_cluster_command(args):
        captured.update(vars(args))

    monkeypatch.setattr(cli_module, "cluster_command", fake_cluster_command)
    args = SimpleNamespace(config=str(config_path), output_dir=str(output_dir))

    cli_module.workflow_command(args)

    assert captured["use_pca"] is True
    assert captured["pca_mode"] == "components"
    assert captured["pca_variance"] == pytest.approx(0.95)
    assert captured["pca_n_components"] == 2
    assert captured["n_neighbors"] == 15
    assert captured["hdbscan_metric"] == "euclidean"


def test_workflow_harmony_uses_documented_default_iteration_count(tmp_path, monkeypatch):
    features_path = tmp_path / "features.csv"
    output_dir = tmp_path / "workflow_output"
    config_path = tmp_path / "workflow.yaml"
    features = pd.DataFrame(
        {
            "batch": ["A", "A", "B", "B"],
            "marker_mean": [0.1, 0.2, 1.0, 1.1],
        }
    )
    features.to_csv(features_path, index=False)
    config_path.write_text(
        f"output: {output_dir}\n"
        "batch_correction:\n"
        "  enabled: true\n"
        f"  input_features: {features_path}\n"
        "  method: harmony\n"
        "  batch_variable: batch\n"
        "  features: [marker_mean]\n"
    )
    captured = {}

    def fake_harmony(data, batch_var, feature_columns, **kwargs):
        captured.update(kwargs)
        return data.copy()

    monkeypatch.setattr(cli_module, "apply_harmony_correction", fake_harmony)

    cli_module.workflow_command(
        SimpleNamespace(config=str(config_path), output_dir=str(output_dir))
    )

    assert captured["max_iter"] == 20
