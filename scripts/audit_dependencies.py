#!/usr/bin/env python3
"""Run pip-audit with the documented target-specific release policy."""

from __future__ import annotations

import argparse
import subprocess
import sys


INTEL_MACOS_PYTORCH_EXCEPTIONS = (
    "PYSEC-2025-41",
    "PYSEC-2024-259",
    "PYSEC-2025-205",
    "PYSEC-2025-206",
    "PYSEC-2025-207",
    "PYSEC-2025-204",
    "PYSEC-2026-139",
    "PYSEC-2025-209",
    "PYSEC-2025-208",
    "PYSEC-2025-191",
    "PYSEC-2025-198",
    "PYSEC-2025-203",
    "PYSEC-2026-1970",
    "PYSEC-2026-2286",
    "GHSA-c678-jfcj-6jmf",
    "GHSA-x3gm-94wq-g975",
    "GHSA-f4hp-rmr7-r7v8",
    "GHSA-vgrw-7cvw-pwgx",
    "GHSA-rrmf-rvhw-rf47",
    "GHSA-qfhq-4f3w-5fph",
)


def build_command(target: str, output_format: str | None, output: str | None) -> list[str]:
    command = [sys.executable, "-m", "pip_audit"]
    if target == "macos-x86_64":
        for advisory in INTEL_MACOS_PYTORCH_EXCEPTIONS:
            command.extend(["--ignore-vuln", advisory])
    if output_format:
        command.extend(["--format", output_format])
    if output:
        command.extend(["--output", output])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--format", dest="output_format")
    parser.add_argument("--output")
    args = parser.parse_args()
    if bool(args.output_format) != bool(args.output):
        parser.error("--format and --output must be provided together")
    completed = subprocess.run(
        build_command(args.target, args.output_format, args.output),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
