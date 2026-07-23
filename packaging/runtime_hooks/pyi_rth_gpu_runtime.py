"""Activate OpenIMC's verified per-user CUDA runtime before package imports."""

from openimc.utils.gpu_runtime import activate_gpu_runtime_path


activate_gpu_runtime_path()
