import json
from pathlib import Path
from types import SimpleNamespace

from openimc.utils import gpu_runtime


def _write_complete_runtime(root: Path) -> None:
    runtime = root / "site-packages"
    (runtime / "torch").mkdir(parents=True)
    (runtime / "torchvision").mkdir()
    (runtime / "torch" / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "torchvision" / "__init__.py").write_text("", encoding="utf-8")
    (root / "installed.json").write_text(
        json.dumps(
            {
                "schema": gpu_runtime.GPU_RUNTIME_SCHEMA,
                "platform": gpu_runtime.sys.platform,
                "torch": gpu_runtime.GPU_TORCH_VERSION,
                "torchvision": gpu_runtime.GPU_TORCHVISION_VERSION,
                "cuda_variant": gpu_runtime.GPU_CUDA_VARIANT,
            }
        ),
        encoding="utf-8",
    )


def test_gpu_runtime_uses_xdg_data_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENIMC_GPU_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(gpu_runtime.sys, "platform", "linux")

    assert gpu_runtime.get_gpu_runtime_root() == tmp_path / "openimc" / "gpu-runtime"


def test_gpu_runtime_requires_matching_marker_and_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIMC_GPU_RUNTIME_ROOT", str(tmp_path))
    assert gpu_runtime.gpu_runtime_is_installed() is False

    _write_complete_runtime(tmp_path)
    assert gpu_runtime.gpu_runtime_is_installed() is True

    marker_path = tmp_path / "installed.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["torch"] = "unexpected"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert gpu_runtime.gpu_runtime_is_installed() is False


def test_activate_gpu_runtime_prepends_explicit_probe_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIMC_GPU_RUNTIME_PATH", str(tmp_path))
    monkeypatch.delenv("OPENIMC_GPU_RUNTIME_ACTIVE", raising=False)
    original_path = list(gpu_runtime.sys.path)
    try:
        assert gpu_runtime.activate_gpu_runtime_path() == tmp_path
        assert gpu_runtime.sys.path[0] == str(tmp_path)
        assert gpu_runtime.os.environ["OPENIMC_GPU_RUNTIME_ACTIVE"] == "1"
    finally:
        gpu_runtime.sys.path[:] = original_path


def test_nvidia_detection_uses_driver_probe_without_importing_torch(monkeypatch):
    monkeypatch.delenv("OPENIMC_DISABLE_GPU", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu_runtime.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        gpu_runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="NVIDIA RTX 4090, 580.10\n",
        ),
    )

    assert gpu_runtime.detect_nvidia_cuda_device() == "NVIDIA RTX 4090, 580.10"


def test_startup_prompt_repeats_until_runtime_is_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIMC_GPU_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(gpu_runtime.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gpu_runtime.sys, "platform", "linux")
    monkeypatch.setattr(gpu_runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        gpu_runtime,
        "detect_nvidia_cuda_device",
        lambda: "NVIDIA RTX 4090, 580.10",
    )

    assert gpu_runtime.cuda_setup_prompt_device() == "NVIDIA RTX 4090, 580.10"
    assert gpu_runtime.cuda_setup_prompt_device() == "NVIDIA RTX 4090, 580.10"

    _write_complete_runtime(tmp_path)
    assert gpu_runtime.cuda_setup_prompt_device() is None


def test_windows_frozen_bundle_also_prompts_for_detected_nvidia_gpu(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENIMC_GPU_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(gpu_runtime.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gpu_runtime.sys, "platform", "win32")
    monkeypatch.setattr(gpu_runtime.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        gpu_runtime,
        "detect_nvidia_cuda_device",
        lambda: "NVIDIA RTX 4090, 580.10",
    )

    assert gpu_runtime.cuda_setup_prompt_device() == "NVIDIA RTX 4090, 580.10"


def test_macos_bundle_does_not_offer_cuda_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIMC_GPU_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(gpu_runtime.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gpu_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(gpu_runtime.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        gpu_runtime,
        "detect_nvidia_cuda_device",
        lambda: "unexpected NVIDIA device",
    )

    assert gpu_runtime.cuda_setup_prompt_device() is None


def test_windows_nvidia_detection_checks_the_standard_driver_location(
    tmp_path, monkeypatch
):
    executable = (
        tmp_path / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    monkeypatch.setattr(gpu_runtime.sys, "platform", "win32")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("SystemRoot", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OPENIMC_DISABLE_GPU", raising=False)
    monkeypatch.setattr(gpu_runtime.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        gpu_runtime.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=0, stdout="NVIDIA RTX 4090, 580.10\n"
        ),
    )

    assert gpu_runtime.detect_nvidia_cuda_device() == "NVIDIA RTX 4090, 580.10"


def test_installer_atomically_marks_only_a_verified_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIMC_GPU_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(gpu_runtime.sys, "platform", "linux")
    monkeypatch.setattr(gpu_runtime, "_acquire_install_lock", lambda stream: None)
    monkeypatch.setattr(
        gpu_runtime.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=gpu_runtime.MINIMUM_FREE_BYTES),
    )

    pip_calls = []

    def fake_pip(arguments):
        pip_calls.append(arguments)
        target = Path(arguments[arguments.index("--target") + 1])
        (target / "torch").mkdir(parents=True)
        (target / "torchvision").mkdir()
        (target / "torch" / "__init__.py").write_text("", encoding="utf-8")
        (target / "torchvision" / "__init__.py").write_text("", encoding="utf-8")
        return 0

    def fake_subprocess(command, *, env, **kwargs):
        result_path = Path(env["OPENIMC_GPU_PROBE_RESULT"])
        result_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "cuda": "12.6",
                    "devices": ["NVIDIA RTX 4090"],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("pip._internal.cli.main.main", fake_pip)
    monkeypatch.setattr(gpu_runtime.subprocess, "run", fake_subprocess)

    assert gpu_runtime.install_gpu_runtime() == 0
    assert gpu_runtime.gpu_runtime_is_installed() is True
    marker = json.loads((tmp_path / "installed.json").read_text(encoding="utf-8"))
    assert marker["devices"] == ["NVIDIA RTX 4090"]
    assert not list(tmp_path.glob(".install-*"))
    assert len(pip_calls) == 1
    arguments = pip_calls[0]
    assert "--no-deps" in arguments
    assert f"torch=={gpu_runtime.GPU_TORCH_VERSION}" in arguments
    assert set(gpu_runtime.LINUX_GPU_NATIVE_REQUIREMENTS).issubset(arguments)
    assert not any(requirement.startswith("numpy") for requirement in arguments)


def test_windows_cuda_wheel_is_self_contained_in_installer_arguments(tmp_path):
    arguments = gpu_runtime.gpu_pip_arguments(tmp_path, platform_name="win32")

    assert "--no-deps" in arguments
    assert f"torch=={gpu_runtime.GPU_TORCH_VERSION}" in arguments
    assert f"torchvision=={gpu_runtime.GPU_TORCHVISION_VERSION}" in arguments
    assert not set(gpu_runtime.LINUX_GPU_NATIVE_REQUIREMENTS).intersection(arguments)


def test_cuda_runtime_versions_match_the_cpu_bundle_pins():
    project_root = Path(__file__).resolve().parents[2]
    workflow = (project_root / ".github" / "workflows" / "desktop-builds.yml").read_text(
        encoding="utf-8"
    )

    assert f"torch=={gpu_runtime.GPU_TORCH_VERSION}" in workflow
    assert f"torchvision=={gpu_runtime.GPU_TORCHVISION_VERSION}" in workflow


def test_failed_cuda_probe_does_not_suppress_future_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIMC_GPU_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(gpu_runtime.sys, "platform", "linux")
    monkeypatch.setattr(gpu_runtime, "_acquire_install_lock", lambda stream: None)
    monkeypatch.setattr(
        gpu_runtime.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=gpu_runtime.MINIMUM_FREE_BYTES),
    )

    def fake_pip(arguments):
        target = Path(arguments[arguments.index("--target") + 1])
        (target / "torch").mkdir(parents=True, exist_ok=True)
        (target / "torchvision").mkdir(exist_ok=True)
        (target / "torch" / "__init__.py").write_text("", encoding="utf-8")
        (target / "torchvision" / "__init__.py").write_text("", encoding="utf-8")
        return 0

    def failed_probe(command, *, env, **kwargs):
        Path(env["OPENIMC_GPU_PROBE_RESULT"]).write_text(
            json.dumps({"ok": False, "error": "driver is too old"}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr("pip._internal.cli.main.main", fake_pip)
    monkeypatch.setattr(gpu_runtime.subprocess, "run", failed_probe)

    assert gpu_runtime.install_gpu_runtime() == 1
    assert gpu_runtime.gpu_runtime_is_installed() is False
    assert not (tmp_path / "installed.json").exists()
    assert "driver is too old" in (tmp_path / "setup.log").read_text(encoding="utf-8")
