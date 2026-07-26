# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional per-user CUDA runtime support for frozen Linux/Windows releases."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


GPU_RUNTIME_SCHEMA = 2
GPU_TORCH_VERSION = "2.13.0"
GPU_TORCHVISION_VERSION = "0.28.0"
GPU_CUDA_VARIANT = "cu126"
GPU_INDEX_URL = f"https://download.pytorch.org/whl/{GPU_CUDA_VARIANT}"
MINIMUM_FREE_BYTES = 12 * 1024**3
LINUX_GPU_NATIVE_REQUIREMENTS = (
    "cuda-bindings==12.9.4",
    "cuda-pathfinder==1.2.2",
    "nvidia-cublas-cu12==12.6.4.1",
    "nvidia-cuda-cupti-cu12==12.6.80",
    "nvidia-cuda-nvrtc-cu12==12.6.85",
    "nvidia-cuda-runtime-cu12==12.6.77",
    "nvidia-cudnn-cu12==9.10.2.21",
    "nvidia-cufft-cu12==11.3.0.4",
    "nvidia-cufile-cu12==1.11.1.6",
    "nvidia-curand-cu12==10.3.7.77",
    "nvidia-cusolver-cu12==11.7.1.2",
    "nvidia-cusparse-cu12==12.5.4.2",
    "nvidia-cusparselt-cu12==0.7.1",
    "nvidia-nccl-cu12==2.29.3",
    "nvidia-nvjitlink-cu12==12.6.85",
    "nvidia-nvshmem-cu12==3.4.5",
    "nvidia-nvtx-cu12==12.6.77",
    "triton==3.7.1",
)


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_gpu_runtime_root() -> Path:
    """Return the writable directory used by the optional CUDA runtime."""
    override = os.environ.get("OPENIMC_GPU_RUNTIME_ROOT")
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = (
            Path(local_app_data)
            if local_app_data
            else Path.home() / "AppData" / "Local"
        ) / "OpenIMC"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "OpenIMC"
    else:
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        base = (
            Path(xdg_data_home).expanduser()
            if xdg_data_home
            else Path.home() / ".local" / "share"
        ) / "openimc"
    return base / "gpu-runtime"


def get_gpu_runtime_path() -> Path:
    return get_gpu_runtime_root() / "site-packages"


def get_gpu_setup_log_path() -> Path:
    return get_gpu_runtime_root() / "setup.log"


def _marker_path() -> Path:
    return get_gpu_runtime_root() / "installed.json"


def _read_marker() -> dict[str, Any] | None:
    try:
        marker = json.loads(_marker_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return marker if isinstance(marker, dict) else None


def gpu_runtime_is_installed() -> bool:
    """Return whether the exact release-tested CUDA runtime is complete."""
    marker = _read_marker()
    runtime_path = get_gpu_runtime_path()
    return bool(
        marker
        and marker.get("schema") == GPU_RUNTIME_SCHEMA
        and marker.get("platform") == sys.platform
        and marker.get("torch") == GPU_TORCH_VERSION
        and marker.get("torchvision") == GPU_TORCHVISION_VERSION
        and marker.get("cuda_variant") == GPU_CUDA_VARIANT
        and (runtime_path / "torch" / "__init__.py").is_file()
        and (runtime_path / "torchvision" / "__init__.py").is_file()
    )


def activate_gpu_runtime_path() -> Path | None:
    """Prepend a verified or explicitly probed CUDA runtime to ``sys.path``."""
    explicit_path = os.environ.get("OPENIMC_GPU_RUNTIME_PATH")
    if explicit_path:
        runtime_path = Path(explicit_path).expanduser()
    elif gpu_runtime_is_installed():
        runtime_path = get_gpu_runtime_path()
    else:
        return None

    if not runtime_path.is_dir():
        return None
    runtime_text = str(runtime_path)
    if runtime_text in sys.path:
        sys.path.remove(runtime_text)
    sys.path.insert(0, runtime_text)
    os.environ["OPENIMC_GPU_RUNTIME_ACTIVE"] = "1"
    return runtime_path


def detect_nvidia_cuda_device() -> str | None:
    """Return a short NVIDIA device description without importing PyTorch."""
    if _environment_flag("OPENIMC_DISABLE_GPU"):
        return None
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return None
    executable = shutil.which("nvidia-smi")
    if executable is None and sys.platform == "win32":
        candidates = []
        if system_root := os.environ.get("SystemRoot"):
            candidates.append(Path(system_root) / "System32" / "nvidia-smi.exe")
        if program_files := os.environ.get("ProgramFiles"):
            candidates.append(
                Path(program_files)
                / "NVIDIA Corporation"
                / "NVSMI"
                / "nvidia-smi.exe"
            )
        executable = next((str(path) for path in candidates if path.is_file()), None)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]
    return f"{devices[0]} and {len(devices) - 1} more NVIDIA GPU(s)"


def cuda_setup_prompt_device() -> str | None:
    """Return the detected device only when this bundle needs CUDA setup."""
    frozen = bool(getattr(sys, "frozen", False))
    supported_machine = platform.machine().lower() in {"amd64", "x86_64"}
    if (
        not frozen
        or sys.platform not in {"linux", "win32"}
        or not supported_machine
        or gpu_runtime_is_installed()
    ):
        return None
    return detect_nvidia_cuda_device()


def gpu_installer_command() -> tuple[str, list[str]]:
    """Build the command used by the GUI to launch the isolated installer."""
    if getattr(sys, "frozen", False):
        return sys.executable, ["--openimc-install-gpu-runtime"]
    return sys.executable, ["-m", "openimc.gui_entry", "--openimc-install-gpu-runtime"]


def _bootstrap_command(argument: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, argument]
    return [sys.executable, "-m", "openimc.gui_entry", argument]


def gpu_pip_arguments(
    target: Path, *, platform_name: str | None = None
) -> list[str]:
    """Return the wheel-only, dependency-isolated CUDA install arguments."""
    platform_name = platform_name or sys.platform
    requirements = [
        f"torch=={GPU_TORCH_VERSION}",
        f"torchvision=={GPU_TORCHVISION_VERSION}",
    ]
    if platform_name == "linux":
        # CUDA-enabled Linux Torch wheels use separate NVIDIA runtime wheels.
        requirements.extend(LINUX_GPU_NATIVE_REQUIREMENTS)
    elif platform_name != "win32":
        raise ValueError(f"Unsupported CUDA runtime platform: {platform_name!r}")

    return [
        "install",
        "--target",
        str(target),
        "--upgrade",
        "--only-binary=:all:",
        "--no-deps",
        "--no-cache-dir",
        "--index-url",
        GPU_INDEX_URL,
        *requirements,
    ]


def probe_gpu_runtime(result_path: Path) -> int:
    """Validate that the activated external runtime can execute on the GPU."""
    result: dict[str, Any]
    try:
        import torch
        import torchvision

        if torch.version.cuda is None:
            raise RuntimeError("the installed PyTorch build does not include CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA PyTorch was installed, but the NVIDIA driver could not use it"
            )
        device_count = torch.cuda.device_count()
        if device_count < 1:
            raise RuntimeError("CUDA reported no usable devices")
        result = {
            "ok": True,
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
            "cuda": str(torch.version.cuda),
            "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
        }
        return_code = 0
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return_code = 1
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return return_code


def _write_install_marker(probe: dict[str, Any]) -> None:
    marker = {
        "schema": GPU_RUNTIME_SCHEMA,
        "platform": sys.platform,
        "torch": GPU_TORCH_VERSION,
        "torchvision": GPU_TORCHVISION_VERSION,
        "cuda_variant": GPU_CUDA_VARIANT,
        "detected_cuda": probe.get("cuda"),
        "devices": probe.get("devices", []),
    }
    temporary_marker = _marker_path().with_suffix(".json.tmp")
    temporary_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    temporary_marker.replace(_marker_path())


def install_gpu_runtime() -> int:
    """Download, verify, and atomically activate the optional CUDA runtime."""
    root = get_gpu_runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    log_path = get_gpu_setup_log_path()
    active_path = get_gpu_runtime_path()
    staging_path: Path | None = None
    result_path: Path | None = None
    lock_stream = (root / "install.lock").open("a+b")

    try:
        if sys.platform not in {"linux", "win32"}:
            raise RuntimeError(
                "The optional CUDA runtime is supported on Linux and Windows only"
            )

        _acquire_install_lock(lock_stream)

        # A force-quit can leave a partial multi-gigabyte target behind. Once
        # this process owns the lock, those incomplete directories are safe to
        # remove before checking available space and starting again.
        for stale_path in root.glob(".install-*"):
            if stale_path.is_dir():
                shutil.rmtree(stale_path, ignore_errors=True)
        staging_path = Path(tempfile.mkdtemp(prefix=".install-", dir=root))
        result_path = staging_path / ".openimc-gpu-probe.json"

        free_bytes = shutil.disk_usage(root).free
        if free_bytes < MINIMUM_FREE_BYTES:
            raise RuntimeError(
                "At least 12 GiB of free disk space is required to download and "
                "stage the CUDA runtime."
            )

        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                "Installing the official OpenIMC CUDA runtime\n"
                f"Index: {GPU_INDEX_URL}\n"
                f"Torch: {GPU_TORCH_VERSION}\n"
                f"Torchvision: {GPU_TORCHVISION_VERSION}\n\n"
            )
            # Resolve no general dependencies into this high-priority path.
            # The one supported Torch wheel's native requirements are pinned
            # explicitly above; common scientific packages remain in OpenIMC's
            # audited bundle and cannot be shadowed by a resolver update.
            from pip._internal.cli.main import main as pip_main

            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = log
                sys.stderr = log
                return_code = pip_main(gpu_pip_arguments(staging_path))
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
            if return_code != 0:
                raise RuntimeError(f"pip exited with status {return_code}")

        environment = os.environ.copy()
        environment["OPENIMC_GPU_RUNTIME_PATH"] = str(staging_path)
        environment["OPENIMC_GPU_PROBE_RESULT"] = str(result_path)
        completed = subprocess.run(
            _bootstrap_command("--openimc-gpu-runtime-probe"),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if not result_path.is_file():
            raise RuntimeError(
                "The CUDA validation process did not produce a result. "
                f"Exit status: {completed.returncode}."
            )
        probe = json.loads(result_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not probe.get("ok"):
            raise RuntimeError(probe.get("error", "CUDA validation failed"))
        result_path.unlink()

        backup_path = root / ".previous-site-packages"
        if backup_path.exists():
            shutil.rmtree(backup_path)
        if active_path.exists():
            active_path.replace(backup_path)
        try:
            staging_path.replace(active_path)
            _write_install_marker(probe)
        except Exception:
            if active_path.exists():
                shutil.rmtree(active_path)
            if backup_path.exists():
                backup_path.replace(active_path)
            raise
        if backup_path.exists():
            shutil.rmtree(backup_path)
        return 0
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nCUDA setup failed: {type(exc).__name__}: {exc}\n")
        return 1
    finally:
        if result_path is not None:
            result_path.unlink(missing_ok=True)
        if staging_path is not None and staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)
        lock_stream.close()


def _acquire_install_lock(lock_stream) -> None:
    """Take a non-blocking, process-scoped installer lock on Linux or Windows."""
    if sys.platform == "win32":
        import msvcrt

        lock_stream.seek(0, os.SEEK_END)
        if lock_stream.tell() == 0:
            lock_stream.write(b"\0")
            lock_stream.flush()
        lock_stream.seek(0)
        try:
            msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another OpenIMC CUDA setup is already running") from exc
        return

    import fcntl

    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another OpenIMC CUDA setup is already running") from exc
