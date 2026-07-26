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

import builtins
import importlib
from types import SimpleNamespace

import pytest

import openimc.ui.dialogs.spatial_analysis as spatial_analysis_module


@pytest.mark.unit
def test_spatial_analysis_module_import_defers_squidpy_stack(monkeypatch):
    attempted_imports = []
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split('.')[0]
        if top_level in {'squidpy', 'scanpy', 'anndata'}:
            attempted_imports.append(top_level)
            raise AssertionError(f"unexpected optional import: {top_level}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.reload(spatial_analysis_module)

    assert spatial_analysis_module._HAVE_SQUIDPY is False
    assert attempted_imports == []


@pytest.mark.unit
def test_squidpy_available_handles_broken_optional_import(monkeypatch):
    importlib.reload(spatial_analysis_module)
    monkeypatch.setattr(spatial_analysis_module, "_should_probe_squidpy_import", lambda: False)

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "squidpy":
            raise OSError("DLL initialization failed")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert spatial_analysis_module.squidpy_available() is False
    assert spatial_analysis_module._HAVE_SQUIDPY is False
    assert "DLL initialization failed" in str(spatial_analysis_module._SQUIDPY_IMPORT_ERROR)

    spatial_analysis_module._SQUIDPY_IMPORT_ATTEMPTED = False
    spatial_analysis_module._SQUIDPY_IMPORT_ERROR = None
    spatial_analysis_module.sq = None
    spatial_analysis_module.sc = None
    spatial_analysis_module.ad = None


@pytest.mark.unit
def test_squidpy_available_handles_failed_windows_probe(monkeypatch):
    importlib.reload(spatial_analysis_module)

    monkeypatch.setattr(spatial_analysis_module, "_should_probe_squidpy_import", lambda: True)
    monkeypatch.setattr(
        spatial_analysis_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=3221225477, stderr="", stdout=""),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct import should not run")),
    )

    assert spatial_analysis_module.squidpy_available() is False
    assert spatial_analysis_module._HAVE_SQUIDPY is False
    assert "3221225477" in str(spatial_analysis_module._SQUIDPY_IMPORT_ERROR)


@pytest.mark.unit
def test_squidpy_available_keeps_windows_imports_out_of_process(monkeypatch):
    importlib.reload(spatial_analysis_module)

    monkeypatch.setattr(spatial_analysis_module, "_should_probe_squidpy_import", lambda: True)
    monkeypatch.setattr(
        spatial_analysis_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="", stdout=""),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direct import should not run")),
    )

    assert spatial_analysis_module.squidpy_available() is False
    assert spatial_analysis_module._HAVE_SQUIDPY is False
    assert "disabled in-process" in str(spatial_analysis_module._SQUIDPY_IMPORT_ERROR)


@pytest.mark.unit
def test_frozen_squidpy_probe_uses_bootstrap_command(monkeypatch):
    importlib.reload(spatial_analysis_module)
    commands = []

    monkeypatch.setattr(spatial_analysis_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(spatial_analysis_module.sys, "executable", "OpenIMC.exe")
    monkeypatch.setattr(
        spatial_analysis_module.subprocess,
        "run",
        lambda command, **kwargs: (
            commands.append(command)
            or SimpleNamespace(returncode=0, stderr="", stdout="")
        ),
    )

    assert spatial_analysis_module._probe_squidpy_import_in_subprocess() is True
    assert commands == [["OpenIMC.exe", "--openimc-squidpy-probe"]]
