from typing import List

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# Data types
from openmcd.data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
from openmcd.ui.utils import combine_channels

# Optional GPU runtime
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False


class PreprocessingDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preprocessing")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.channels = channels
        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        title_label = QtWidgets.QLabel("Image Preprocessing for Segmentation")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        layout.addWidget(title_label)
        desc_label = QtWidgets.QLabel("Configure normalization and channel combination for segmentation.")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        norm_group = QtWidgets.QGroupBox("Normalization")
        norm_layout = QtWidgets.QVBoxLayout(norm_group)
        norm_method_layout = QtWidgets.QHBoxLayout()
        norm_method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.norm_method_combo = QtWidgets.QComboBox()
        self.norm_method_combo.addItems(["None", "arcsinh", "percentile_clip"])
        self.norm_method_combo.currentTextChanged.connect(self._on_norm_method_changed)
        norm_method_layout.addWidget(self.norm_method_combo)
        norm_method_layout.addStretch()
        norm_layout.addLayout(norm_method_layout)

        self.arcsinh_frame = QtWidgets.QFrame()
        arcsinh_layout = QtWidgets.QHBoxLayout(self.arcsinh_frame)
        arcsinh_layout.addWidget(QtWidgets.QLabel("Cofactor:"))
        self.arcsinh_cofactor_spin = QtWidgets.QDoubleSpinBox()
        self.arcsinh_cofactor_spin.setRange(0.1, 100.0)
        self.arcsinh_cofactor_spin.setValue(5.0)
        self.arcsinh_cofactor_spin.setDecimals(1)
        arcsinh_layout.addWidget(self.arcsinh_cofactor_spin)
        arcsinh_layout.addStretch()
        norm_layout.addWidget(self.arcsinh_frame)

        self.percentile_frame = QtWidgets.QFrame()
        percentile_layout = QtWidgets.QHBoxLayout(self.percentile_frame)
        percentile_layout.addWidget(QtWidgets.QLabel("Low percentile:"))
        self.p_low_spin = QtWidgets.QDoubleSpinBox()
        self.p_low_spin.setRange(0.1, 50.0)
        self.p_low_spin.setValue(1.0)
        self.p_low_spin.setDecimals(1)
        percentile_layout.addWidget(self.p_low_spin)
        percentile_layout.addWidget(QtWidgets.QLabel("High percentile:"))
        self.p_high_spin = QtWidgets.QDoubleSpinBox()
        self.p_high_spin.setRange(50.0, 99.9)
        self.p_high_spin.setValue(99.0)
        self.p_high_spin.setDecimals(1)
        percentile_layout.addWidget(self.p_high_spin)
        percentile_layout.addStretch()
        norm_layout.addWidget(self.percentile_frame)
        layout.addWidget(norm_group)

        combo_group = QtWidgets.QGroupBox("Channel Combination")
        combo_layout = QtWidgets.QVBoxLayout(combo_group)
        nuclear_layout = QtWidgets.QHBoxLayout()
        nuclear_layout.addWidget(QtWidgets.QLabel("Nuclear channels:"))
        self.nuclear_list = QtWidgets.QListWidget()
        self.nuclear_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for channel in self.channels:
            self.nuclear_list.addItem(channel)
        nuclear_layout.addWidget(self.nuclear_list)
        combo_layout.addLayout(nuclear_layout)

        self.nuclear_auto_info = QtWidgets.QLabel("")
        self.nuclear_auto_info.setStyleSheet("QLabel { color: #0066cc; font-style: italic; font-size: 11px; }")
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

        cyto_layout = QtWidgets.QHBoxLayout()
        cyto_layout.addWidget(QtWidgets.QLabel("Cytoplasm channels:"))
        self.cyto_list = QtWidgets.QListWidget()
        self.cyto_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for channel in self.channels:
            self.cyto_list.addItem(channel)
        cyto_layout.addWidget(self.cyto_list)
        combo_layout.addLayout(cyto_layout)

        self.cyto_auto_info = QtWidgets.QLabel("")
        self.cyto_auto_info.setStyleSheet("QLabel { color: #0066cc; font-style: italic; font-size: 11px; }")
        self.cyto_auto_info.setWordWrap(True)
        combo_layout.addWidget(self.cyto_auto_info)

        cyto_combo_layout = QtWidgets.QHBoxLayout()
        cyto_combo_layout.addWidget(QtWidgets.QLabel("Cytoplasm combination:"))
        self.cyto_combo_method = QtWidgets.QComboBox()
        self.cyto_combo_method.addItems(["single", "mean", "weighted", "max", "pca1"])
        self.cyto_combo_method.currentTextChanged.connect(self._on_cyto_combo_changed)
        cyto_combo_layout.addWidget(self.cyto_combo_method)
        cyto_combo_layout.addStretch()
        combo_layout.addLayout(cyto_combo_layout)
        self.cyto_list.itemSelectionChanged.connect(self._on_cyto_channels_changed)

        self.cyto_weights_frame = QtWidgets.QFrame()
        cyto_weights_layout = QtWidgets.QVBoxLayout(self.cyto_weights_frame)
        cyto_weights_layout.addWidget(QtWidgets.QLabel("Cytoplasm channel weights (leave empty for auto):"))
        self.cyto_weights_edit = QtWidgets.QLineEdit()
        self.cyto_weights_edit.setPlaceholderText("e.g., 0.5,0.3,0.2")
        cyto_weights_layout.addWidget(self.cyto_weights_edit)
        combo_layout.addWidget(self.cyto_weights_frame)

        layout.addWidget(combo_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self._on_norm_method_changed()
        self._on_nuclear_combo_changed()
        self._on_cyto_combo_changed()
        self._auto_parse_channels()

    def _on_norm_method_changed(self):
        method = self.norm_method_combo.currentText()
        self.arcsinh_frame.setVisible(method == "arcsinh")
        self.percentile_frame.setVisible(method == "percentile_clip")

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
    def get_normalization_method(self) -> str:
        return self.norm_method_combo.currentText()

    def get_arcsinh_cofactor(self) -> float:
        return self.arcsinh_cofactor_spin.value()

    def get_percentile_params(self):
        return self.p_low_spin.value(), self.p_high_spin.value()

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

