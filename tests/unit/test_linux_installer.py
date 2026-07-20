import os
from pathlib import Path

import pytest

from scripts import build_desktop


def test_debian_architecture_names():
    assert build_desktop._debian_architecture("x86_64") == "amd64"
    assert build_desktop._debian_architecture("AMD64") == "amd64"
    assert build_desktop._debian_architecture("aarch64") == "arm64"
    with pytest.raises(RuntimeError, match="Unsupported Debian package architecture"):
        build_desktop._debian_architecture("riscv64")


def test_debian_version_validation():
    assert build_desktop._validate_debian_version("0.1.0+abc1234") == (
        "0.1.0+abc1234"
    )
    with pytest.raises(ValueError, match="Invalid Debian package version"):
        build_desktop._validate_debian_version("0.1.0 invalid")


def test_make_linux_installer_stages_desktop_integration(tmp_path, monkeypatch):
    dist_dir = tmp_path / "dist"
    application = dist_dir / "OpenIMC"
    internal = application / "_internal"
    internal.mkdir(parents=True)
    executable = application / "OpenIMC"
    executable.write_bytes(b"frozen executable")
    executable.chmod(0o755)
    (internal / "scientific-library.so").write_bytes(b"native library")

    observed = {}

    def fake_run(command, **kwargs):
        assert command[:3] == ["dpkg-deb", "--root-owner-group", "--build"]
        package_root = Path(command[-2])
        package_path = Path(command[-1])
        observed["control"] = (package_root / "DEBIAN" / "control").read_text(
            encoding="utf-8"
        )
        observed["desktop"] = (
            package_root / "usr" / "share" / "applications" / "openimc.desktop"
        ).read_text(encoding="utf-8")
        observed["executable_mode"] = (
            package_root / "usr" / "lib" / "openimc" / "OpenIMC"
        ).stat().st_mode
        observed["launcher_target"] = os.readlink(
            package_root / "usr" / "bin" / "openimc"
        )
        observed["icon_exists"] = (
            package_root / "usr" / "share" / "pixmaps" / "openimc.png"
        ).is_file()
        observed["license_exists"] = (
            package_root / "usr" / "share" / "doc" / "openimc" / "copyright"
        ).is_file()
        package_path.write_bytes(b"!<arch>\nmock package")

    def fake_validate(package_path, *, version, architecture):
        observed["validated"] = (package_path, version, architecture)

    monkeypatch.setattr(build_desktop, "DIST_DIR", dist_dir)
    monkeypatch.setattr(build_desktop.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_desktop.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(build_desktop.shutil, "which", lambda name: "/usr/bin/dpkg-deb")
    monkeypatch.setattr(build_desktop, "run", fake_run)
    monkeypatch.setattr(
        build_desktop,
        "_validate_linux_installer",
        fake_validate,
    )

    package_path = build_desktop.make_linux_installer("0.1.0+abc1234")

    assert package_path == dist_dir / "OpenIMC-0.1.0+abc1234-linux-amd64.deb"
    assert package_path.is_file()
    assert "Package: openimc" in observed["control"]
    assert "Version: 0.1.0+abc1234" in observed["control"]
    assert "Architecture: amd64" in observed["control"]
    for dependency in build_desktop.LINUX_RUNTIME_DEPENDENCIES:
        assert dependency in observed["control"]
    assert "Exec=/usr/lib/openimc/OpenIMC" in observed["desktop"]
    assert observed["executable_mode"] & 0o111
    assert observed["launcher_target"] == "/usr/lib/openimc/OpenIMC"
    assert observed["icon_exists"] is True
    assert observed["license_exists"] is True
    assert observed["validated"] == (
        package_path,
        "0.1.0+abc1234",
        "amd64",
    )
