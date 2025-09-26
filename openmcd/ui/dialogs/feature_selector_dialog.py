from typing import Dict, List

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class FeatureSelectorDialog(QtWidgets.QDialog):
    def __init__(self, available_feature_names: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Features for Clustering")
        self.setModal(True)
        self.resize(700, 520)

        # Split features into morphometric and intensity
        # Note: centroid_x and centroid_y are excluded as they are spatial coordinates, not biological features
        self._morpho_names = {
            'area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity',
            'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um',
            'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count'
        }
        self._intensity_suffixes = ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos']

        morpho = []
        intensity = []
        for name in available_feature_names:
            if name in self._morpho_names:
                morpho.append(name)
            elif any(name.endswith(suffix) for suffix in self._intensity_suffixes):
                intensity.append(name)

        self._morpho_all = sorted(set(morpho))
        self._intensity_all = sorted(set(intensity))

        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Intensity filter
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Intensity filter:"))
        self.cmb_filter = QtWidgets.QComboBox()
        self.cmb_filter.addItems(["All", "mean_", "median_", "std_", "mad_", "p10_", "p90_", "integrated_", "frac_pos_"])
        self.cmb_filter.currentTextChanged.connect(self._apply_intensity_filter)
        filter_row.addWidget(self.cmb_filter)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        # Lists side-by-side
        lists_row = QtWidgets.QHBoxLayout()

        morpho_group = QtWidgets.QGroupBox("Morphometric features")
        morpho_layout = QtWidgets.QVBoxLayout(morpho_group)
        self.lst_morpho = QtWidgets.QListWidget()
        self.lst_morpho.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        morpho_layout.addWidget(self.lst_morpho, 1)
        morpho_btns = QtWidgets.QHBoxLayout()
        self.btn_morpho_all = QtWidgets.QPushButton("Select All")
        self.btn_morpho_all.clicked.connect(lambda: self._set_all(self.lst_morpho, True))
        self.btn_morpho_none = QtWidgets.QPushButton("Select None")
        self.btn_morpho_none.clicked.connect(lambda: self._set_all(self.lst_morpho, False))
        morpho_btns.addWidget(self.btn_morpho_all)
        morpho_btns.addWidget(self.btn_morpho_none)
        morpho_layout.addLayout(morpho_btns)

        intensity_group = QtWidgets.QGroupBox("Intensity features")
        intensity_layout = QtWidgets.QVBoxLayout(intensity_group)
        self.lst_intensity = QtWidgets.QListWidget()
        self.lst_intensity.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        intensity_layout.addWidget(self.lst_intensity, 1)
        intensity_btns = QtWidgets.QHBoxLayout()
        self.btn_int_all = QtWidgets.QPushButton("Select All")
        self.btn_int_all.clicked.connect(lambda: self._set_all(self.lst_intensity, True))
        self.btn_int_none = QtWidgets.QPushButton("Select None")
        self.btn_int_none.clicked.connect(lambda: self._set_all(self.lst_intensity, False))
        intensity_btns.addWidget(self.btn_int_all)
        intensity_btns.addWidget(self.btn_int_none)
        intensity_layout.addLayout(intensity_btns)

        lists_row.addWidget(morpho_group, 1)
        lists_row.addWidget(intensity_group, 1)
        layout.addLayout(lists_row, 1)

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(self.btn_ok)
        btns.addWidget(self.btn_cancel)
        layout.addLayout(btns)

        # Populate lists (all selected by default)
        self._populate_checklist(self.lst_morpho, self._morpho_all)
        self._populate_checklist(self.lst_intensity, self._intensity_all)
        # Auto-select only _mean intensity features by default
        self._apply_intensity_default_selection()

    def _populate_checklist(self, widget: QtWidgets.QListWidget, names: List[str]):
        widget.clear()
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked)
            widget.addItem(item)

    def _apply_intensity_filter(self):
        sel = self.cmb_filter.currentText()
        if sel == "All":
            filtered = self._intensity_all
        else:
            # Convert prefix to suffix for filtering
            suffix = f"_{sel.lower()}"
            filtered = [n for n in self._intensity_all if n.endswith(suffix)]
        self._populate_checklist(self.lst_intensity, filtered)
        # When changing filter, keep default preference of _mean if All
        if sel == "All":
            self._apply_intensity_default_selection()

    def _set_all(self, widget: QtWidgets.QListWidget, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(widget.count()):
            it = widget.item(i)
            it.setCheckState(state)

    def _apply_intensity_default_selection(self):
        for i in range(self.lst_intensity.count()):
            it = self.lst_intensity.item(i)
            if it.text().endswith("_mean"):
                it.setCheckState(Qt.Checked)
            else:
                it.setCheckState(Qt.Unchecked)

    def get_selected_columns(self) -> List[str]:
        cols: List[str] = []
        for widget in (self.lst_morpho, self.lst_intensity):
            for i in range(widget.count()):
                it = widget.item(i)
                if it.checkState() == Qt.Checked:
                    cols.append(it.text())
        return sorted(cols)



