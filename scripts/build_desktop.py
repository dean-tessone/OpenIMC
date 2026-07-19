#!/usr/bin/env python3
"""Build and optionally smoke-test an OpenIMC desktop distribution."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build" / "pyinstaller"
SPEC_FILE = PROJECT_ROOT / "packaging" / "openimc.spec"
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
    run(
        [
            str(executable),
            "--openimc-functional-test",
            str(input_image),
            str(mask_path),
            str(output_dir),
        ],
        environment=environment,
    )


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
        run(
            [
                "hdiutil",
                "create",
                "-volname",
                "OpenIMC",
                "-srcfolder",
                str(app_bundle or DIST_DIR / "OpenIMC.app"),
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


def write_sha256(path: Path) -> Path:
    """Write a portable SHA-256 checksum file beside a release archive."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    checksum_path = path.with_name(f"{path.name}.sha256")
    checksum_path.write_text(f"{digest.hexdigest()}  {path.name}\n", encoding="utf-8")
    return checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.0", help="Release version")
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
        archive = make_archive(args.version, signed_app_bundle)
        checksum = write_sha256(archive)
        print(f"Created {archive}")
        print(f"Created {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
