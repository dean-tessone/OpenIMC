# SPDX-License-Identifier: GPL-3.0-or-later
"""Startup prompt for installing the optional Linux/Windows CUDA runtime."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from openimc.utils.gpu_runtime import (
    cuda_setup_prompt_device,
    get_gpu_setup_log_path,
    gpu_installer_command,
    gpu_runtime_is_installed,
)


def _run_gpu_installer(parent: QtWidgets.QWidget) -> bool:
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Setting up CUDA acceleration")
    dialog.setModal(True)
    dialog.setMinimumWidth(520)
    layout = QtWidgets.QVBoxLayout(dialog)

    message = QtWidgets.QLabel(
        "OpenIMC is downloading and verifying the official CUDA-enabled "
        "PyTorch packages. This is a several-gigabyte download and may take "
        "some time."
    )
    message.setWordWrap(True)
    layout.addWidget(message)
    progress = QtWidgets.QProgressBar()
    progress.setRange(0, 0)
    layout.addWidget(progress)
    cancel_button = QtWidgets.QPushButton("Cancel")
    button_row = QtWidgets.QHBoxLayout()
    button_row.addStretch()
    button_row.addWidget(cancel_button)
    layout.addLayout(button_row)

    process = QtCore.QProcess(dialog)
    process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
    state = {"completed": False, "success": False}

    def finish(exit_code: int, _exit_status: QtCore.QProcess.ExitStatus) -> None:
        state["completed"] = True
        state["success"] = exit_code == 0 and gpu_runtime_is_installed()
        dialog.accept()

    def cancel() -> None:
        if process.state() != QtCore.QProcess.NotRunning:
            process.kill()
            process.waitForFinished(3000)
        dialog.reject()

    process.finished.connect(finish)
    cancel_button.clicked.connect(cancel)
    program, arguments = gpu_installer_command()
    process.start(program, arguments)
    if not process.waitForStarted(5000):
        QtWidgets.QMessageBox.warning(
            parent,
            "CUDA Setup Could Not Start",
            "OpenIMC could not start the CUDA package installer.",
        )
        return False

    dialog.exec_()
    if process.state() != QtCore.QProcess.NotRunning:
        process.kill()
        process.waitForFinished(3000)
    return bool(state["completed"] and state["success"])


def maybe_offer_cuda_setup(parent: QtWidgets.QWidget) -> None:
    """Offer CUDA setup on every eligible startup until it succeeds."""
    device = cuda_setup_prompt_device()
    if device is None:
        return

    prompt = QtWidgets.QMessageBox(parent)
    prompt.setIcon(QtWidgets.QMessageBox.Information)
    prompt.setWindowTitle("Enable NVIDIA GPU acceleration?")
    prompt.setText("OpenIMC detected an NVIDIA CUDA-capable system.")
    prompt.setInformativeText(
        f"Detected: {device}\n\n"
        "The compact OpenIMC installer includes CPU-only PyTorch. Download the "
        "optional CUDA packages now to accelerate supported segmentation and "
        "analysis workflows? The download is several gigabytes and requires "
        "at least 12 GiB of free disk space."
    )
    install_button = prompt.addButton(
        "Download CUDA support", QtWidgets.QMessageBox.AcceptRole
    )
    prompt.addButton("Not now", QtWidgets.QMessageBox.RejectRole)
    prompt.setDefaultButton(install_button)
    prompt.exec_()
    if prompt.clickedButton() is not install_button:
        return

    if _run_gpu_installer(parent):
        QtWidgets.QMessageBox.information(
            parent,
            "CUDA Acceleration Ready",
            "CUDA support was downloaded and verified successfully. Restart "
            "OpenIMC to begin using the GPU. This startup prompt will no longer appear.",
        )
    else:
        QtWidgets.QMessageBox.warning(
            parent,
            "CUDA Setup Incomplete",
            "OpenIMC could not finish CUDA setup. The CPU workflows are still "
            "available, and OpenIMC will offer CUDA setup again next time it starts.\n\n"
            f"Details were written to:\n{get_gpu_setup_log_path()}",
        )
