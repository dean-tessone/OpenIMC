#!/usr/bin/env python3
"""Build and optionally smoke-test an OpenIMC desktop distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build" / "pyinstaller"
SPEC_FILE = PROJECT_ROOT / "packaging" / "openimc.spec"
LINUX_DESKTOP_FILE = PROJECT_ROOT / "packaging" / "linux" / "openimc.desktop"
LINUX_RUNTIME_DEPENDENCIES = (
    "libegl1",
    "libgl1",
    "libxkbcommon-x11-0",
    "libxcb-cursor0",
    "libxcb-xinerama0",
)
SENSITIVE_ENVIRONMENT_VARIABLES = (
    "DEEPCELL_ACCESS_TOKEN",
    "OPENAI_API_KEY",
)
SECRET_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-)[A-Za-z0-9_-]{24,}"),
    re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9]{32,}(?![A-Za-z0-9_-])"),
)
TEXT_FILE_SUFFIXES = {
    ".cfg",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
FUNCTIONAL_TEST_TIMEOUT_SECONDS = 10 * 60


def sanitized_environment() -> dict[str, str]:
    """Copy the process environment without application credentials."""
    environment = os.environ.copy()
    for variable_name in SENSITIVE_ENVIRONMENT_VARIABLES:
        environment.pop(variable_name, None)
    return environment


def _file_contains_secret(
    path: Path,
    exact_values: tuple[bytes, ...],
    *,
    scan_key_patterns: bool,
) -> bool:
    """Scan a file without loading large native libraries fully into memory."""
    maximum_length = max([256, *(len(value) for value in exact_values)])
    previous = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            candidate = previous + chunk
            if any(value in candidate for value in exact_values):
                return True
            if scan_key_patterns and any(
                pattern.search(candidate) for pattern in SECRET_PATTERNS
            ):
                return True
            previous = candidate[-maximum_length:]
    return False


def audit_bundle_for_secrets(app_bundle: Path | None = None) -> None:
    """Fail a release if a credential or credential file entered the bundle."""
    if platform.system() == "Darwin":
        application_root = app_bundle or DIST_DIR / "OpenIMC.app"
    else:
        application_root = DIST_DIR / "OpenIMC"
    if not application_root.exists():
        raise FileNotFoundError(f"Packaged application not found: {application_root}")

    prohibited_names = {".env", "user_preferences.json", "credentials.json"}
    exact_values = tuple(
        value.encode("utf-8")
        for variable_name in SENSITIVE_ENVIRONMENT_VARIABLES
        if len(value := os.environ.get(variable_name, "")) >= 8
    )
    for path in application_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in prohibited_names:
            raise RuntimeError(f"Credential file was bundled: {path}")
        if _file_contains_secret(
            path,
            exact_values,
            scan_key_patterns=path.suffix.lower() in TEXT_FILE_SUFFIXES,
        ):
            raise RuntimeError(
                f"Possible API credential detected in packaged file: {path}"
            )
    print("Credential audit passed: no API keys or credential files were bundled.")


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: int | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        timeout=timeout,
    )


def built_executable(app_bundle: Path | None = None) -> Path:
    system = platform.system()
    if system == "Darwin":
        bundle = app_bundle or DIST_DIR / "OpenIMC.app"
        return bundle / "Contents" / "MacOS" / "OpenIMC"
    if system == "Windows":
        return DIST_DIR / "OpenIMC" / "OpenIMC.exe"
    return DIST_DIR / "OpenIMC" / "OpenIMC"


def smoke_test(app_bundle: Path | None = None) -> None:
    executable = built_executable(app_bundle)
    if not executable.exists():
        raise FileNotFoundError(f"Packaged executable not found: {executable}")

    application_root = (
        (app_bundle or DIST_DIR / "OpenIMC.app")
        if platform.system() == "Darwin"
        else DIST_DIR / "OpenIMC"
    )
    seed_files = list(
        application_root.rglob("openimc_bootstrap/matplotlib/fontlist-v*.json")
    )
    if not seed_files:
        raise FileNotFoundError("Packaged Matplotlib first-launch cache seed is missing")

    checkpoint_path = PROJECT_ROOT / "build" / "bundle-smoke-checkpoint.txt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="openimc-smoke-profile-") as profile:
        profile_path = Path(profile)
        environment = sanitized_environment()
        environment.pop("MPLCONFIGDIR", None)
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")
        environment["OPENIMC_SMOKE_CHECKPOINT"] = str(checkpoint_path)

        if platform.system() == "Windows":
            local_app_data = profile_path / "LocalAppData"
            environment["LOCALAPPDATA"] = str(local_app_data)
            expected_cache = local_app_data / "OpenIMC" / "Cache" / "matplotlib"
        elif platform.system() == "Darwin":
            environment["HOME"] = str(profile_path)
            expected_cache = profile_path / "Library" / "Caches" / "OpenIMC" / "matplotlib"
        else:
            xdg_cache = profile_path / "cache"
            environment["HOME"] = str(profile_path)
            environment["XDG_CACHE_HOME"] = str(xdg_cache)
            expected_cache = xdg_cache / "openimc" / "matplotlib"

        try:
            run(
                [str(executable), "--openimc-bundle-smoke-test"],
                environment=environment,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            if checkpoint_path.exists():
                print(
                    "Frozen smoke-test checkpoint:",
                    checkpoint_path.read_text(encoding="utf-8").strip(),
                    flush=True,
                )
            raise
        else:
            if checkpoint_path.exists():
                print(
                    "Frozen smoke-test checkpoint:",
                    checkpoint_path.read_text(encoding="utf-8").strip(),
                    flush=True,
                )
            for seed_file in seed_files:
                cached_file = expected_cache / seed_file.name
                if not cached_file.is_file():
                    raise FileNotFoundError(
                        f"Matplotlib first-launch cache was not seeded: {cached_file}"
                    )
            print("Matplotlib first-launch cache seed passed.")
        finally:
            checkpoint_path.unlink(missing_ok=True)


def functional_test(
    app_bundle: Path | None = None, *, allow_cellsam_download: bool = False
) -> None:
    """Exercise the packaged application's primary scientific workflows."""
    executable = built_executable(app_bundle)
    input_image = PROJECT_ROOT / "tests" / "data" / "Patient1_pos1_1.ome.tiff"
    mask_path = (
        PROJECT_ROOT
        / "tests"
        / "data"
        / "mask"
        / "Patient1_pos1_1.ome_Patient1_pos1_1_segmentation.tiff"
    )
    output_dir = PROJECT_ROOT / "build" / "functional-validation"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    environment = sanitized_environment()
    if allow_cellsam_download:
        token = os.environ.get("DEEPCELL_ACCESS_TOKEN")
        if not token:
            raise RuntimeError(
                "--allow-cellsam-download requires DEEPCELL_ACCESS_TOKEN"
            )
        environment["DEEPCELL_ACCESS_TOKEN"] = token
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    command = [
        str(executable),
        "--openimc-functional-test",
        str(input_image),
        str(mask_path),
        str(output_dir),
    ]
    print("+", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment)
    report_path = output_dir / "openimc-functional-validation.json"
    deadline = time.monotonic() + FUNCTIONAL_TEST_TIMEOUT_SECONDS
    last_report_text: str | None = None
    latest_report: dict[str, object] | None = None

    while True:
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return_code = None

        try:
            report_text = report_path.read_text(encoding="utf-8")
            if report_text != last_report_text:
                latest_report = json.loads(report_text)
                last_report_text = report_text
                current = latest_report.get("current_check", "finishing")
                completed = len(latest_report.get("checks", {}))
                print(
                    "Frozen functional validation progress: "
                    f"{completed} checks recorded; current={current}",
                    flush=True,
                )
        except (OSError, json.JSONDecodeError):
            pass

        if latest_report and latest_report.get("status") == "failed":
            try:
                process.kill()
            except OSError:
                pass
            process.wait()
            failed_checks = [
                (name, details)
                for name, details in latest_report.get("checks", {}).items()
                if details.get("status") == "failed"
            ]
            if failed_checks:
                failed_name, failed_details = failed_checks[0]
                traceback_text = failed_details.get("traceback", "")
                if traceback_text:
                    print(traceback_text, flush=True)
                raise RuntimeError(
                    f"Frozen functional validation failed during {failed_name}: "
                    f"{failed_details.get('error', 'unknown error')}"
                )
            raise RuntimeError("Frozen functional validation reported failure")

        if return_code is not None:
            break
        if time.monotonic() >= deadline:
            process.kill()
            process.wait()
            current = (
                latest_report.get("current_check", "before report creation")
                if latest_report
                else "before report creation"
            )
            raise TimeoutError(
                "Frozen functional validation exceeded "
                f"{FUNCTIONAL_TEST_TIMEOUT_SECONDS} seconds during {current}"
            )

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if latest_report is None or latest_report.get("status") != "passed":
        raise RuntimeError("Frozen functional validation exited without a passed report")


def _codesign_macos_bundle(app_bundle: Path, identity: str) -> None:
    run(["xattr", "-cr", str(app_bundle)])

    command = ["codesign", "--force", "--deep", "--sign", identity]
    if identity == "-":
        command.append("--timestamp=none")
    else:
        command.extend(["--options", "runtime", "--timestamp"])
    command.append(str(app_bundle))
    run(command)
    run(["codesign", "--verify", "--deep", "--strict", "--verbose", str(app_bundle)])


def sign_macos_bundle() -> Path | None:
    """Remove inherited metadata and apply a valid final app signature."""
    if platform.system() != "Darwin":
        return None

    app_bundle = DIST_DIR / "OpenIMC.app"
    identity = os.environ.get("OPENIMC_CODESIGN_IDENTITY", "-")

    try:
        _codesign_macos_bundle(app_bundle, identity)
        return app_bundle
    except subprocess.CalledProcessError:
        # iCloud/File Provider workspaces can immediately restore provenance
        # and FinderInfo xattrs after they are removed. Sign an attribute-free
        # copy outside the synced workspace and use it for testing/archiving.
        staging_dir = Path(tempfile.mkdtemp(prefix="openimc-signing-"))
        staged_bundle = staging_dir / "OpenIMC.app"
        run(
            [
                "ditto",
                "--noextattr",
                "--noqtn",
                str(app_bundle),
                str(staged_bundle),
            ]
        )
        _codesign_macos_bundle(staged_bundle, identity)
        print(f"Signed app staged at {staged_bundle}")
        return staged_bundle


def make_archive(version: str, app_bundle: Path | None = None) -> Path:
    system = platform.system()
    machine = platform.machine().lower().replace("amd64", "x86_64")
    base_name = f"OpenIMC-{version}-{system.lower()}-{machine}"

    if system == "Darwin":
        image_path = DIST_DIR / f"{base_name}.dmg"
        if image_path.exists():
            image_path.unlink()
        with tempfile.TemporaryDirectory(prefix="openimc-dmg-") as staging:
            staging_path = Path(staging)
            staged_app = staging_path / "OpenIMC.app"
            run(
                [
                    "ditto",
                    "--noextattr",
                    "--noqtn",
                    str(app_bundle or DIST_DIR / "OpenIMC.app"),
                    str(staged_app),
                ]
            )
            os.symlink("/Applications", staging_path / "Applications")
            (staging_path / "Install OpenIMC.txt").write_text(
                "Install OpenIMC\n"
                "===============\n\n"
                "Drag OpenIMC.app onto the Applications shortcut in this window.\n"
                "Then eject the OpenIMC disk image and launch OpenIMC from Applications.\n\n"
                "Alternatively, download the OpenIMC .pkg file to use the guided "
                "macOS Installer.\n",
                encoding="utf-8",
            )
            run(
                [
                    "hdiutil",
                    "create",
                    "-volname",
                    "OpenIMC",
                    "-srcfolder",
                    str(staging_path),
                    "-ov",
                    "-format",
                    "UDZO",
                    str(image_path),
                ]
            )
        return image_path

    archive = shutil.make_archive(
        str(DIST_DIR / base_name),
        "zip" if system == "Windows" else "gztar",
        root_dir=DIST_DIR,
        base_dir="OpenIMC",
    )
    return Path(archive)


def _debian_architecture(machine: str | None = None) -> str:
    """Translate a Python machine name into a Debian architecture."""
    normalized = (machine or platform.machine()).lower()
    architectures = {
        "amd64": "amd64",
        "x86_64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    try:
        return architectures[normalized]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported Debian package architecture: {normalized}"
        ) from exc


def _validate_debian_version(version: str) -> str:
    """Return a Debian-compatible version or fail before invoking dpkg-deb."""
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+:~\-]*", version):
        raise ValueError(f"Invalid Debian package version: {version!r}")
    return version


def _hardlink_or_copy(source: str, destination: str) -> str:
    """Hard-link large bundle files when possible to limit staging disk use."""
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _installed_size_kib(root: Path) -> int:
    """Calculate the Installed-Size value required by Debian package metadata."""
    total_bytes = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    return max(1, (total_bytes + 1023) // 1024)


def _validate_linux_installer(
    package_path: Path,
    *,
    version: str,
    architecture: str,
) -> None:
    """Validate the built Debian package's identity without installing it."""
    if not package_path.is_file() or package_path.stat().st_size == 0:
        raise FileNotFoundError(f"Debian installer was not created: {package_path}")

    expected_fields = {
        "Package": "openimc",
        "Version": version,
        "Architecture": architecture,
    }
    for field, expected in expected_fields.items():
        result = subprocess.run(
            ["dpkg-deb", "--field", str(package_path), field],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        actual = result.stdout.strip()
        if actual != expected:
            raise RuntimeError(
                f"Unexpected Debian {field}: expected {expected!r}, got {actual!r}"
            )
    print("Debian installer metadata validation passed.")


def make_linux_installer(version: str) -> Path:
    """Create a double-clickable Debian installer for the frozen application."""
    if platform.system() != "Linux":
        raise RuntimeError("Debian installers can only be built on Linux")
    if shutil.which("dpkg-deb") is None:
        raise RuntimeError("dpkg-deb is required to create the Ubuntu installer")

    version = _validate_debian_version(version)
    architecture = _debian_architecture()
    application_source = DIST_DIR / "OpenIMC"
    executable_source = application_source / "OpenIMC"
    if not executable_source.is_file():
        raise FileNotFoundError(
            f"Packaged OpenIMC executable not found: {executable_source}"
        )

    package_path = (
        DIST_DIR / f"OpenIMC-{version}-linux-{architecture}.deb"
    )
    package_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="openimc-deb-") as staging:
        package_root = Path(staging) / "openimc"
        control_dir = package_root / "DEBIAN"
        application_dir = package_root / "usr" / "lib" / "openimc"
        binary_dir = package_root / "usr" / "bin"
        desktop_dir = package_root / "usr" / "share" / "applications"
        icon_dir = package_root / "usr" / "share" / "pixmaps"
        documentation_dir = package_root / "usr" / "share" / "doc" / "openimc"
        for directory in (
            control_dir,
            binary_dir,
            desktop_dir,
            icon_dir,
            documentation_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        # The frozen application is several gigabytes on Linux. Hard links
        # avoid duplicating it in the staging tree before dpkg-deb compresses it.
        shutil.copytree(
            application_source,
            application_dir,
            symlinks=True,
            copy_function=_hardlink_or_copy,
        )
        os.symlink("/usr/lib/openimc/OpenIMC", binary_dir / "openimc")
        shutil.copy2(LINUX_DESKTOP_FILE, desktop_dir / "openimc.desktop")
        shutil.copy2(
            PROJECT_ROOT / "openimc" / "ui" / "resources" / "OpenIMC_Icon.png",
            icon_dir / "openimc.png",
        )
        shutil.copy2(PROJECT_ROOT / "LICENSE", documentation_dir / "copyright")

        installed_size = _installed_size_kib(package_root)
        control = (
            "Package: openimc\n"
            f"Version: {version}\n"
            "Section: science\n"
            "Priority: optional\n"
            f"Architecture: {architecture}\n"
            "Maintainer: OpenIMC Contributors <tessone@usc.edu>\n"
            f"Installed-Size: {installed_size}\n"
            f"Depends: {', '.join(LINUX_RUNTIME_DEPENDENCIES)}\n"
            "Homepage: https://github.com/dean-tessone/OpenIMC\n"
            "Description: Interactive analysis toolkit for IMC data\n"
            " OpenIMC provides visualization, segmentation, feature extraction,\n"
            " batch correction, clustering, and spatial analysis workflows for\n"
            " Imaging Mass Cytometry data.\n"
        )
        control_path = control_dir / "control"
        control_path.write_text(control, encoding="utf-8")
        control_path.chmod(0o644)
        (desktop_dir / "openimc.desktop").chmod(0o644)
        (icon_dir / "openimc.png").chmod(0o644)
        (documentation_dir / "copyright").chmod(0o644)

        run(
            [
                "dpkg-deb",
                "--root-owner-group",
                "--build",
                str(package_root),
                str(package_path),
            ]
        )

    _validate_linux_installer(
        package_path,
        version=version,
        architecture=architecture,
    )
    return package_path


def make_macos_installer(version: str, app_bundle: Path | None = None) -> Path:
    """Create a guided Installer package that places OpenIMC in Applications."""
    if platform.system() != "Darwin":
        raise RuntimeError("macOS Installer packages can only be built on macOS")

    machine = platform.machine().lower().replace("amd64", "x86_64")
    package_path = DIST_DIR / f"OpenIMC-{version}-darwin-{machine}.pkg"
    package_path.unlink(missing_ok=True)
    application = app_bundle or DIST_DIR / "OpenIMC.app"
    identity = os.environ.get("OPENIMC_INSTALLER_SIGNING_IDENTITY")

    command = ["productbuild"]
    if identity:
        command.extend(["--sign", identity])
    command.extend(
        [
            "--component",
            str(application),
            "/Applications",
            str(package_path),
        ]
    )
    run(command)

    payload = subprocess.run(
        ["pkgutil", "--payload-files", str(package_path)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if not any(
        entry.lstrip("./") == "OpenIMC.app" or entry.lstrip("./").startswith("OpenIMC.app/")
        for entry in payload
    ):
        raise RuntimeError("Installer package does not contain OpenIMC.app")

    if identity:
        run(["pkgutil", "--check-signature", str(package_path)])
    print("macOS Installer payload validation passed.")
    return package_path


def write_sha256(path: Path) -> Path:
    """Write a portable SHA-256 checksum file beside a release archive."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    checksum_path = path.with_name(f"{path.name}.sha256")
    # ``Path.write_text`` uses the platform newline on Windows. GNU and BSD
    # checksum tools can then interpret the trailing CR as part of the file
    # name. Force LF so one generated checksum works on every release target.
    with checksum_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{digest.hexdigest()}  {path.name}\n")
    return checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="1.1.0", help="Release version")
    parser.add_argument("--console", action="store_true", help="Keep a console for diagnostics")
    parser.add_argument("--no-clean", action="store_true", help="Reuse PyInstaller work files")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Validate/archive an existing dist/OpenIMC tree (for post-signing CI)",
    )
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument(
        "--functional-test",
        action="store_true",
        help="Run real segmentation, analysis, and persistence workflows in the bundle",
    )
    parser.add_argument(
        "--allow-cellsam-download",
        action="store_true",
        help=(
            "Optionally pass DEEPCELL_ACCESS_TOKEN only to the functional-test "
            "process for a live CellSAM model check"
        ),
    )
    parser.add_argument("--archive", action="store_true", help="Create a distributable archive")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_build:
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print(
                "PyInstaller is missing. Install the desktop build dependencies with "
                "'python -m pip install -r requirements-build.txt'.",
                file=sys.stderr,
            )
            return 2

    environment = sanitized_environment()
    environment["OPENIMC_VERSION"] = args.version
    environment["OPENIMC_CONSOLE"] = "1" if args.console else "0"

    if not args.skip_build:
        command = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            str(DIST_DIR),
            "--workpath",
            str(BUILD_DIR),
        ]
        if not args.no_clean:
            command.append("--clean")
        command.append(str(SPEC_FILE))
        run(command, environment=environment)

    signed_app_bundle = sign_macos_bundle()
    audit_bundle_for_secrets(signed_app_bundle)
    if not args.skip_smoke_test:
        smoke_test(signed_app_bundle)
    if args.functional_test:
        functional_test(
            signed_app_bundle,
            allow_cellsam_download=args.allow_cellsam_download,
        )
    if args.archive:
        artifacts = [make_archive(args.version, signed_app_bundle)]
        if platform.system() == "Darwin":
            artifacts.append(make_macos_installer(args.version, signed_app_bundle))
        elif platform.system() == "Linux":
            artifacts.append(make_linux_installer(args.version))
        for artifact in artifacts:
            checksum = write_sha256(artifact)
            print(f"Created {artifact}")
            print(f"Created {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
