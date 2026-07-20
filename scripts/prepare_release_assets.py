#!/usr/bin/env python3
"""Prepare a concise, verified set of end-user GitHub Release assets."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import zipfile
from pathlib import Path


GITHUB_RELEASE_ASSET_LIMIT = 2 * 1024**3
PACKAGE_PATTERNS = (
    "OpenIMC-*-windows-x86_64.zip",
    "OpenIMC-*-linux-amd64.deb",
    "OpenIMC-*-darwin-arm64.pkg",
    "OpenIMC-*-darwin-x86_64.pkg",
)
CHECKSUM_PATTERN = re.compile(r"^([0-9a-fA-F]{64})  ([^/\\]+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _one_match(input_dir: Path, pattern: str) -> Path:
    matches = sorted(input_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one release package matching {pattern!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def verify_generated_checksums(input_dir: Path) -> dict[str, str]:
    """Verify every generated checksum, accepting Windows CRLF input."""
    checksum_files = sorted(input_dir.glob("*.sha256"))
    if len(checksum_files) != 7:
        raise RuntimeError(
            f"Expected seven generated checksum files, found {len(checksum_files)}"
        )

    verified: dict[str, str] = {}
    for checksum_file in checksum_files:
        line = checksum_file.read_text(encoding="utf-8").strip()
        match = CHECKSUM_PATTERN.fullmatch(line)
        if match is None:
            raise RuntimeError(f"Invalid checksum file: {checksum_file.name}")
        expected, file_name = match.groups()
        subject = input_dir / file_name
        if not subject.is_file():
            raise FileNotFoundError(
                f"Checksum {checksum_file.name} references missing {file_name}"
            )
        actual = sha256(subject)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"Checksum verification failed for {file_name}")
        verified[file_name] = actual
    return verified


def prepare_release_assets(input_dir: Path, output_dir: Path) -> list[Path]:
    """Select installers and consolidate checksums and SBOMs."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Release output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    verified = verify_generated_checksums(input_dir)
    packages = [_one_match(input_dir, pattern) for pattern in PACKAGE_PATTERNS]
    for package in packages:
        if package.stat().st_size >= GITHUB_RELEASE_ASSET_LIMIT:
            raise RuntimeError(
                f"{package.name} is {package.stat().st_size} bytes; GitHub Release "
                "assets must remain below 2 GiB"
            )
        if verified.get(package.name) != sha256(package):
            raise RuntimeError(f"No verified checksum found for {package.name}")
        _hardlink_or_copy(package, output_dir / package.name)

    sboms = sorted(input_dir.glob("openimc-*-sbom.cdx.json"))
    if len(sboms) != 4:
        raise RuntimeError(f"Expected four platform SBOMs, found {len(sboms)}")
    sbom_archive = output_dir / "openimc-sboms.zip"
    with zipfile.ZipFile(
        sbom_archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for sbom in sboms:
            archive.write(sbom, arcname=sbom.name)

    checksum_subjects = [*packages, sbom_archive]
    checksum_path = output_dir / "SHA256SUMS.txt"
    with checksum_path.open("w", encoding="utf-8", newline="\n") as stream:
        for subject in sorted(checksum_subjects, key=lambda item: item.name):
            stream.write(f"{sha256(subject)}  {subject.name}\n")

    assets = sorted(output_dir.iterdir())
    if len(assets) != 6:
        raise RuntimeError(f"Expected six concise release assets, found {len(assets)}")
    return assets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assets = prepare_release_assets(args.input, args.output)
    for asset in assets:
        print(asset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
