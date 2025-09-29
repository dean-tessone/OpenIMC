from typing import Dict, List

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class FeatureSelectorDialog(QtWidgets.QDialog):
    # Class-level cache to persist selections across dialog instances
    _last_selections: Dict[str, int] = {}
    
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

        # Track check states across filtering/searching
        self._checked_by_name: Dict[str, int] = {}
        
        # Restore previous selections from class cache
        for name in available_feature_names:
            if name in self._last_selections:
                self._checked_by_name[name] = self._last_selections[name]

        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # QC reminder note
        qc_note = QtWidgets.QLabel(
            "Note: Please perform QC on features prior to selecting them here."
        )
        qc_note.setWordWrap(True)
        qc_note.setStyleSheet("color: #a15d00;")  # subtle warning color
        layout.addWidget(qc_note)

        # Intensity filter
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Intensity filter:"))
        self.cmb_filter = QtWidgets.QComboBox()
        self.cmb_filter.addItems(["All", "_mean", "_median", "_std", "_mad", "_p10", "_p90", "_integrated", "_frac_pos"])
        self.cmb_filter.currentTextChanged.connect(self._apply_intensity_filter)
        filter_row.addWidget(self.cmb_filter)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        # Lists side-by-side
        lists_row = QtWidgets.QHBoxLayout()

        morpho_group = QtWidgets.QGroupBox("Morphometric features")
        morpho_layout = QtWidgets.QVBoxLayout(morpho_group)
        # Search bar for morphometric markers
        self.txt_search_morpho = QtWidgets.QLineEdit()
        self.txt_search_morpho.setPlaceholderText("Search morphometric markers…")
        self.txt_search_morpho.textChanged.connect(self._apply_morpho_search)
        morpho_layout.addWidget(self.txt_search_morpho)
        self.lst_morpho = QtWidgets.QListWidget()
        self.lst_morpho.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.lst_morpho.itemChanged.connect(self._on_item_check_changed)
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
        # Search bar for intensity markers
        self.txt_search_intensity = QtWidgets.QLineEdit()
        self.txt_search_intensity.setPlaceholderText("Search intensity markers…")
        self.txt_search_intensity.textChanged.connect(self._apply_intensity_filter)
        intensity_layout.addWidget(self.txt_search_intensity)
        self.lst_intensity = QtWidgets.QListWidget()
        self.lst_intensity.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.lst_intensity.itemChanged.connect(self._on_item_check_changed)
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

        # Initialize default checked state for all markers (only if not already set from cache)
        for name in self._morpho_all + self._intensity_all:
            if name not in self._checked_by_name:
                self._checked_by_name[name] = Qt.Checked

        # Populate lists
        self._populate_checklist(self.lst_morpho, self._morpho_all)
        # Default intensity filter to _mean and populate accordingly
        self.cmb_filter.setCurrentText("_mean")
        self._apply_intensity_filter()

    def _populate_checklist(self, widget: QtWidgets.QListWidget, names: List[str]):
        widget.blockSignals(True)
        widget.clear()
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            # Restore previous check state if available
            state = self._checked_by_name.get(name, Qt.Checked)
            item.setCheckState(state)
            widget.addItem(item)
        widget.blockSignals(False)

    def _on_item_check_changed(self, item: QtWidgets.QListWidgetItem):
        name = item.text()
        self._checked_by_name[name] = item.checkState()

    def _apply_intensity_filter(self):
        sel = self.cmb_filter.currentText()
        if sel == "All":
            filtered = self._intensity_all
        else:
            # Convert prefix to suffix for filtering
            suffix = sel.lower() if sel.startswith("_") else f"_{sel.lower()}"
            filtered = [n for n in self._intensity_all if n.endswith(suffix)]
        # Apply search filter as well (case-insensitive substring)
        query = (self.txt_search_intensity.text() if hasattr(self, 'txt_search_intensity') else "").strip().lower()
        if query:
            filtered = [n for n in filtered if query in n.lower()]
        self._populate_checklist(self.lst_intensity, filtered)
        # When changing filter, keep default preference of _mean if All
        if sel == "All":
            self._apply_intensity_default_selection()

    def _set_all(self, widget: QtWidgets.QListWidget, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        widget.blockSignals(True)
        for i in range(widget.count()):
            it = widget.item(i)
            it.setCheckState(state)
            self._checked_by_name[it.text()] = state
        widget.blockSignals(False)

    def _apply_intensity_default_selection(self):
        for i in range(self.lst_intensity.count()):
            it = self.lst_intensity.item(i)
            it.setCheckState(Qt.Checked if it.text().endswith("_mean") else Qt.Unchecked)

    def _apply_morpho_search(self):
        query = self.txt_search_morpho.text().strip().lower()
        if not query:
            names = self._morpho_all
        else:
            names = [n for n in self._morpho_all if query in n.lower()]
        self._populate_checklist(self.lst_morpho, names)

    def get_selected_columns(self) -> List[str]:
        cols: List[str] = []
        for widget in (self.lst_morpho, self.lst_intensity):
            for i in range(widget.count()):
                it = widget.item(i)
                if it.checkState() == Qt.Checked:
                    cols.append(it.text())
        
        # Save current selections to class cache for persistence
        self._last_selections.update(self._checked_by_name)
        
        return sorted(cols)



