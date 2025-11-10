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

from typing import List

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# Data types
from openimc.data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
from openimc.ui.utils import combine_channels

# Optional GPU runtime
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False


class PreprocessingDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Segmentation Channels")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.channels = channels
        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        title_label = QtWidgets.QLabel("Select Segmentation Channels")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 11pt; }")
        layout.addWidget(title_label)
        desc_label = QtWidgets.QLabel("Configure channel combination for segmentation.")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        combo_group = QtWidgets.QGroupBox("Channel Combination")
        combo_layout = QtWidgets.QVBoxLayout(combo_group)
        
        # Nuclear channels section
        nuclear_layout = QtWidgets.QVBoxLayout()
        nuclear_layout.addWidget(QtWidgets.QLabel("Nuclear channels:"))
        
        # Nuclear channels search box
        self.nuclear_search = QtWidgets.QLineEdit()
        self.nuclear_search.setPlaceholderText("Search nuclear channels...")
        self.nuclear_search.textChanged.connect(self._filter_nuclear_channels)
        nuclear_layout.addWidget(self.nuclear_search)
        
        self.nuclear_list = QtWidgets.QListWidget()
        self.nuclear_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for channel in self.channels:
            self.nuclear_list.addItem(channel)
        nuclear_layout.addWidget(self.nuclear_list)
        combo_layout.addLayout(nuclear_layout)

        self.nuclear_auto_info = QtWidgets.QLabel("")
        self.nuclear_auto_info.setStyleSheet("QLabel { color: #0066cc; font-style: italic; font-size: 9pt; }")
        self.nuclear_auto_info.setWordWrap(True)
        combo_layout.addWidget(self.nuclear_auto_info)

        nuclear_combo_layout = QtWidgets.QHBoxLayout()
        nuclear_combo_layout.addWidget(QtWidgets.QLabel("Nuclear combination:"))
        self.nuclear_combo_method = QtWidgets.QComboBox()
        self.nuclear_combo_method.addItems(["single", "mean", "weighted", "max", "pca1"])
        self.nuclear_combo_method.currentTextChanged.connect(self._on_nuclear_combo_changed)
        nuclear_combo_layout.addWidget(self.nuclear_combo_method)
        nuclear_combo_layout.addStretch()
        combo_layout.addLayout(nuclear_combo_layout)
        self.nuclear_list.itemSelectionChanged.connect(self._on_nuclear_channels_changed)

        self.nuclear_weights_frame = QtWidgets.QFrame()
        nuclear_weights_layout = QtWidgets.QVBoxLayout(self.nuclear_weights_frame)
        nuclear_weights_layout.addWidget(QtWidgets.QLabel("Nuclear channel weights (leave empty for auto):"))
        self.nuclear_weights_edit = QtWidgets.QLineEdit()
        self.nuclear_weights_edit.setPlaceholderText("e.g., 0.5,0.3,0.2")
        nuclear_weights_layout.addWidget(self.nuclear_weights_edit)
        combo_layout.addWidget(self.nuclear_weights_frame)

        # Cytoplasm channels section wrapped in a frame for easy show/hide
        self.cyto_section_frame = QtWidgets.QFrame()
        cyto_section_v = QtWidgets.QVBoxLayout(self.cyto_section_frame)
        cyto_layout = QtWidgets.QVBoxLayout()
        cyto_layout.addWidget(QtWidgets.QLabel("Cytoplasm channels:"))
        
        # Cytoplasm channels search box
        self.cyto_search = QtWidgets.QLineEdit()
        self.cyto_search.setPlaceholderText("Search cytoplasm channels...")
        self.cyto_search.textChanged.connect(self._filter_cyto_channels)
        cyto_layout.addWidget(self.cyto_search)
        
        self.cyto_list = QtWidgets.QListWidget()
        self.cyto_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for channel in self.channels:
            self.cyto_list.addItem(channel)
        cyto_layout.addWidget(self.cyto_list)
        cyto_section_v.addLayout(cyto_layout)

        self.cyto_auto_info = QtWidgets.QLabel("")
        self.cyto_auto_info.setStyleSheet("QLabel { color: #0066cc; font-style: italic; font-size: 9pt; }")
        self.cyto_auto_info.setWordWrap(True)
        cyto_section_v.addWidget(self.cyto_auto_info)

        cyto_combo_layout = QtWidgets.QHBoxLayout()
        cyto_combo_layout.addWidget(QtWidgets.QLabel("Cytoplasm combination:"))
        self.cyto_combo_method = QtWidgets.QComboBox()
        self.cyto_combo_method.addItems(["single", "mean", "weighted", "max", "pca1"])
        self.cyto_combo_method.currentTextChanged.connect(self._on_cyto_combo_changed)
        cyto_combo_layout.addWidget(self.cyto_combo_method)
        cyto_combo_layout.addStretch()
        cyto_section_v.addLayout(cyto_combo_layout)
        self.cyto_list.itemSelectionChanged.connect(self._on_cyto_channels_changed)

        self.cyto_weights_frame = QtWidgets.QFrame()
        cyto_weights_layout = QtWidgets.QVBoxLayout(self.cyto_weights_frame)
        cyto_weights_layout.addWidget(QtWidgets.QLabel("Cytoplasm channel weights (leave empty for auto):"))
        self.cyto_weights_edit = QtWidgets.QLineEdit()
        self.cyto_weights_edit.setPlaceholderText("e.g., 0.5,0.3,0.2")
        cyto_weights_layout.addWidget(self.cyto_weights_edit)
        cyto_section_v.addWidget(self.cyto_weights_frame)

        combo_layout.addWidget(self.cyto_section_frame)

        layout.addWidget(combo_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self._on_nuclear_combo_changed()
        self._on_cyto_combo_changed()
        self._auto_parse_channels()

    # Programmatic setters and visibility controls
    def set_nuclear_channels(self, channel_names: List[str]):
        names = set(channel_names or [])
        for i in range(self.nuclear_list.count()):
            item = self.nuclear_list.item(i)
            item.setSelected(item.text() in names)
        self._on_nuclear_channels_changed()

    def set_cyto_channels(self, channel_names: List[str]):
        names = set(channel_names or [])
        for i in range(self.cyto_list.count()):
            item = self.cyto_list.item(i)
            item.setSelected(item.text() in names)
        self._on_cyto_channels_changed()

    def set_cytoplasm_visible(self, visible: bool):
        self.cyto_section_frame.setVisible(visible)

    def _on_nuclear_combo_changed(self):
        method = self.nuclear_combo_method.currentText()
        self.nuclear_weights_frame.setVisible(method == "weighted")

    def _on_cyto_combo_changed(self):
        method = self.cyto_combo_method.currentText()
        self.cyto_weights_frame.setVisible(method == "weighted")
        if self.cyto_auto_info.text().startswith("✓ Auto-selected"):
            self.cyto_auto_info.setText("")

    def _on_nuclear_channels_changed(self):
        selected_count = len(self.nuclear_list.selectedItems())
        self._update_combo_options(self.nuclear_combo_method, selected_count)
        if self.nuclear_auto_info.text().startswith("✓ Auto-selected"):
            self.nuclear_auto_info.setText("")

    def _on_cyto_channels_changed(self):
        selected_count = len(self.cyto_list.selectedItems())
        self._update_combo_options(self.cyto_combo_method, selected_count)
        if self.cyto_auto_info.text().startswith("✓ Auto-selected"):
            self.cyto_auto_info.setText("")

    def _auto_parse_channels(self):
        dna_channels = []
        dna_channel_names = []
        for i in range(self.nuclear_list.count()):
            item = self.nuclear_list.item(i)
            if 'DNA' in item.text().upper():
                dna_channels.append(item)
                dna_channel_names.append(item.text())
        for item in dna_channels:
            item.setSelected(True)
        if dna_channel_names:
            self.nuclear_auto_info.setText(f"✓ Auto-selected DNA channels: {', '.join(dna_channel_names)}")
            if dna_channels:
                self.nuclear_list.scrollToItem(dna_channels[0])
        else:
            self.nuclear_auto_info.setText("No DNA channels found for auto-selection")
        self.cyto_combo_method.setCurrentText("max")
        self.cyto_auto_info.setText("✓ Auto-selected 'max' as cytoplasm combination method")
        self._on_nuclear_channels_changed()
        self._on_cyto_channels_changed()

    def _update_combo_options(self, combo_box, selected_count):
        current_text = combo_box.currentText()
        combo_box.clear()
        if selected_count <= 1:
            combo_box.addItems(["single"])
            combo_box.setCurrentText("single")
        else:
            combo_box.addItems(["mean", "weighted", "max", "pca1"])
            if current_text in ["mean", "weighted", "max", "pca1"]:
                combo_box.setCurrentText(current_text)
            else:
                combo_box.setCurrentText("mean")

    # Accessors used by segmentation dialog
    def get_nuclear_channels(self):
        return [item.text() for item in self.nuclear_list.selectedItems()]

    def get_cyto_channels(self):
        return [item.text() for item in self.cyto_list.selectedItems()]

    def get_nuclear_combo_method(self) -> str:
        return self.nuclear_combo_method.currentText()

    def get_cyto_combo_method(self) -> str:
        return self.cyto_combo_method.currentText()

    def get_nuclear_weights(self):
        text = self.nuclear_weights_edit.text().strip()
        if not text:
            return None
        try:
            return [float(x.strip()) for x in text.split(',')]
        except ValueError:
            return None

    def get_cyto_weights(self):
        text = self.cyto_weights_edit.text().strip()
        if not text:
            return None
        try:
            return [float(x.strip()) for x in text.split(',')]
        except ValueError:
            return None

    def _filter_nuclear_channels(self):
        """Filter nuclear channels based on search text."""
        search_text = self.nuclear_search.text().lower()
        
        for i in range(self.nuclear_list.count()):
            item = self.nuclear_list.item(i)
            channel_name = item.text().lower()
            item.setHidden(search_text not in channel_name)

    def _filter_cyto_channels(self):
        """Filter cytoplasm channels based on search text."""
        search_text = self.cyto_search.text().lower()
        
        for i in range(self.cyto_list.count()):
            item = self.cyto_list.item(i)
            channel_name = item.text().lower()
            item.setHidden(search_text not in channel_name)

