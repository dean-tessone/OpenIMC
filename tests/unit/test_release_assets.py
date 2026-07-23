import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import build_desktop, prepare_release_assets


PACKAGE_NAMES = (
    "OpenIMC-0.1.0-linux-amd64.deb",
    "OpenIMC-0.1.0-windows-x86_64.zip",
    "OpenIMC-0.1.0-darwin-arm64.pkg",
    "OpenIMC-0.1.0-darwin-x86_64.pkg",
)
EXTRA_PACKAGE_NAMES = (
    "OpenIMC-0.1.0-linux-x86_64.tar.gz",
    "OpenIMC-0.1.0-darwin-arm64.dmg",
    "OpenIMC-0.1.0-darwin-x86_64.dmg",
)


def _write_package_and_checksum(root: Path, name: str) -> None:
    package = root / name
    package.write_bytes(f"payload for {name}".encode())
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    # Include CRLF for the Windows checksum to exercise cross-platform input.
    newline = "\r\n" if "windows" in name else "\n"
    (root / f"{name}.sha256").write_bytes(
        f"{digest}  {name}{newline}".encode()
    )


def test_write_sha256_uses_portable_lf(tmp_path):
    package = tmp_path / "OpenIMC.zip"
    package.write_bytes(b"release payload")

    checksum = build_desktop.write_sha256(package)

    assert checksum.read_bytes().endswith(b"\n")
    assert b"\r" not in checksum.read_bytes()


def test_prepare_release_assets_is_concise_and_verified(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in (*PACKAGE_NAMES, *EXTRA_PACKAGE_NAMES):
        _write_package_and_checksum(input_dir, name)
    for target in ("linux-x86_64", "windows-x86_64", "macos-arm64", "macos-x86_64"):
        (input_dir / f"openimc-{target}-sbom.cdx.json").write_text(
            '{"bomFormat": "CycloneDX"}',
            encoding="utf-8",
        )

    output_dir = tmp_path / "publish"
    assets = prepare_release_assets.prepare_release_assets(input_dir, output_dir)

    assert {asset.name for asset in assets} == {
        *PACKAGE_NAMES,
        "SHA256SUMS.txt",
        "openimc-sboms.zip",
    }
    assert not any(asset.suffix in {".dmg", ".gz"} for asset in assets)
    assert not any(asset.name.endswith(".sha256") for asset in assets)
    with zipfile.ZipFile(output_dir / "openimc-sboms.zip") as archive:
        assert len(archive.namelist()) == 4

    checksum_lines = (output_dir / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(checksum_lines) == 5
    assert all("  " in line for line in checksum_lines)


def test_prepare_release_assets_rejects_oversized_installer(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in (*PACKAGE_NAMES, *EXTRA_PACKAGE_NAMES):
        _write_package_and_checksum(input_dir, name)
    for target in ("linux-x86_64", "windows-x86_64", "macos-arm64", "macos-x86_64"):
        (input_dir / f"openimc-{target}-sbom.cdx.json").write_text("{}")

    monkeypatch.setattr(prepare_release_assets, "GITHUB_RELEASE_ASSET_LIMIT", 1)
    with pytest.raises(RuntimeError, match="below 2 GiB"):
        prepare_release_assets.prepare_release_assets(
            input_dir,
            tmp_path / "publish",
        )


def test_manual_desktop_build_publishes_a_test_prerelease():
    project_root = Path(__file__).resolve().parents[2]
    workflow = (project_root / ".github/workflows/desktop-builds.yml").read_text(
        encoding="utf-8"
    )

    assert "publish_prerelease:" in workflow
    assert "desktop-test-$short_sha" in workflow
    assert "--prerelease" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "inputs.publish_prerelease" in workflow


def test_windows_frozen_functional_check_exits_after_report_is_written(
    tmp_path, monkeypatch
):
    from openimc import gui_entry, release_validation

    calls = []
    monkeypatch.setattr(gui_entry.sys, "platform", "win32")
    monkeypatch.setattr(
        gui_entry.sys,
        "argv",
        [
            "OpenIMC.exe",
            "--openimc-functional-test",
            "input.ome.tiff",
            "mask.tiff",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        release_validation,
        "run_release_validation",
        lambda *arguments: calls.append(arguments),
    )

    def completed_exit(status):
        raise SystemExit(status)

    monkeypatch.setattr(gui_entry.os, "_exit", completed_exit)

    with pytest.raises(SystemExit) as exit_info:
        gui_entry._run_bootstrap_command()

    assert exit_info.value.code == 0
    assert calls == [("input.ome.tiff", "mask.tiff", str(tmp_path))]


def test_functional_test_requires_a_passed_report(tmp_path, monkeypatch):
    executable = tmp_path / "OpenIMC"
    executable.write_bytes(b"")
    monkeypatch.setattr(build_desktop, "built_executable", lambda app=None: executable)

    class CompletedValidation:
        def __init__(self, command, **kwargs):
            report_path = (
                Path(command[-1]) / "openimc-functional-validation.json"
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps({"status": "passed", "checks": {"example": {}}}),
                encoding="utf-8",
            )

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(build_desktop.subprocess, "Popen", CompletedValidation)

    build_desktop.functional_test()
