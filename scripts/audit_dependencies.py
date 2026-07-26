#!/usr/bin/env python3
"""Run pip-audit with the documented target-specific release policy."""

from __future__ import annotations

import argparse
import subprocess
import sys


INTEL_MACOS_CONDA_PYTORCH_EXCEPTIONS = (
    "GHSA-rrmf-rvhw-rf47",
)


def build_command(target: str, output_format: str | None, output: str | None) -> list[str]:
    command = [sys.executable, "-m", "pip_audit"]
    if target == "macos-x86_64":
        for advisory in INTEL_MACOS_CONDA_PYTORCH_EXCEPTIONS:
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
