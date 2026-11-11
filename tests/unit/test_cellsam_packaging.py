"""Regression checks for native operators omitted from frozen applications."""

import importlib.util
from pathlib import Path
import runpy

import pytest

from openimc.release_validation import check_cellsam_cpu_runtime


@pytest.mark.parametrize(
    "native_files",
    [
        ["_C.cp312-win_amd64.pyd", "image.pyd"],
        ["_C_stable.pyd", "image_stable.pyd"],
        ["_C_stable.abi3.so", "image_stable.abi3.so"],
    ],
)
def test_torchvision_hook_collects_indirect_native_libraries(tmp_path, monkeypatch, native_files):
    hooks = pytest.importorskip("PyInstaller.utils.hooks")
    package = tmp_path / "torchvision"
    package.mkdir()
    (package / "__init__.py").touch()
    for name in native_files:
        (package / name).write_bytes(b"native-library-placeholder")
    (package / "extension.py").write_text("# Python source is not a native library")
    monkeypatch.setattr(hooks, "is_package", lambda name: name == "torchvision")
    monkeypatch.setattr(hooks, "get_all_package_paths", lambda name: [str(package)])

    hook = Path(__file__).resolve().parents[2] / "packaging/hooks/hook-torchvision.py"
    result = runpy.run_path(str(hook))
    assert {(Path(source).name, destination) for source, destination in result["binaries"]} == {
        (name, "torchvision") for name in native_files
    }


def test_real_cellsam_cpu_runtime_needs_no_credentials_or_download(monkeypatch):
    if importlib.util.find_spec("cellSAM") is None:
        pytest.skip("CellSAM is not installed")
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import torch
    import requests

    def unexpected_request(*args, **kwargs):
        raise AssertionError("CPU runtime validation must not download models")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = check_cellsam_cpu_runtime()
    assert result["device"] == "cpu"
    assert result["nms_indices"] == [0, 2]
    assert result["cellsam_imported"] is True


def test_cellsam_runtime_check_fails_if_nms_is_broken(monkeypatch):
    torchvision = pytest.importorskip("torchvision")

    def broken_nms(*args, **kwargs):
        raise RuntimeError("operator torchvision::nms does not exist")

    monkeypatch.setattr(torchvision.ops, "nms", broken_nms)
    with pytest.raises(RuntimeError, match="torchvision::nms"):
        check_cellsam_cpu_runtime()
