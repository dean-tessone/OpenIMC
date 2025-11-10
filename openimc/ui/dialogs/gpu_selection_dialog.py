# SPDX-License-Identifier: GPL-3.0-or-later
#
# OpenIMC – Interactive analysis toolkit for IMC data
#
# Copyright (C) 2025 University of Southern California
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# Optional GPU runtime
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

class GPUSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Selection")
        self.setModal(True)
        self.setFixedSize(400, 300)
        self.selected_gpu = None
        self.available_gpus = self._detect_gpus()
        self._create_ui()

    def _detect_gpus(self):
        gpus = []
        if not _HAVE_TORCH:
            return gpus
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpus.append({'id': i, 'name': gpu_name, 'memory': gpu_memory, 'type': 'CUDA'})
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpus.append({'id': 'mps', 'name': 'Apple Metal Performance Shaders (MPS)', 'memory': None, 'type': 'MPS'})
        except Exception as e:
            print(f"Error detecting GPUs: {e}")
        return gpus

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        title_label = QtWidgets.QLabel("Select GPU for Segmentation")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 11pt; }")
        layout.addWidget(title_label)
        desc_label = QtWidgets.QLabel("GPU acceleration can significantly speed up segmentation. Select a device:")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        self.gpu_list = QtWidgets.QListWidget()
        self.gpu_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        cpu_item = QtWidgets.QListWidgetItem("CPU (No GPU acceleration)")
        cpu_item.setData(Qt.UserRole, None)
        self.gpu_list.addItem(cpu_item)
        for gpu in self.available_gpus:
            if gpu['memory'] is not None:
                text = f"{gpu['name']} ({gpu['memory']:.1f} GB VRAM)"
            else:
                text = gpu['name']
            item = QtWidgets.QListWidgetItem(text)
            item.setData(Qt.UserRole, gpu['id'])
            self.gpu_list.addItem(item)
        if self.gpu_list.count() > 0:
            self.gpu_list.setCurrentRow(0)
        layout.addWidget(self.gpu_list)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        button_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        self.gpu_list.itemSelectionChanged.connect(self._on_gpu_selection_changed)
        self._on_gpu_selection_changed()

    def _on_gpu_selection_changed(self):
        current_item = self.gpu_list.currentItem()
        if not current_item:
            return
        gpu_id = current_item.data(Qt.UserRole)
        if gpu_id is None:
            self.info_label.setText("Using CPU for segmentation. This will be slower but more compatible.")
        else:
            gpu = next((g for g in self.available_gpus if g['id'] == gpu_id), None)
            if gpu:
                if gpu['type'] == 'CUDA':
                    self.info_label.setText(f"Using CUDA GPU: {gpu['name']}\nMemory: {gpu['memory']:.1f} GB")
                elif gpu['type'] == 'MPS':
                    self.info_label.setText("Using Apple Metal Performance Shaders\nOptimized for Apple Silicon Macs")

    def get_selected_gpu(self):
        current_item = self.gpu_list.currentItem()
        if current_item:
            return current_item.data(Qt.UserRole)
        return None

