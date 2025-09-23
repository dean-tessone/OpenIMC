from typing import Dict, List, Optional, Tuple
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pandas as pd
import multiprocessing as mp
from skimage.measure import regionprops, regionprops_table
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from openmcd.data.mcd_loader import MCDLoader, AcquisitionInfo
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from openmcd.ui.mpl_canvas import MplCanvas
from openmcd.ui.utils import (
    PreprocessingCache,
    robust_percentile_scale,
    arcsinh_normalize,
    percentile_clip_normalize,
    stack_to_rgb,
    combine_channels,
)
from openmcd.ui.dialogs.progress_dialog import ProgressDialog
from openmcd.ui.dialogs.gpu_selection_dialog import GPUSelectionDialog
from openmcd.ui.dialogs.preprocessing_dialog import PreprocessingDialog
from openmcd.ui.dialogs.segmentation_dialog import SegmentationDialog
from openmcd.ui.dialogs.export import ExportDialog
from openmcd.ui.dialogs.feature_extraction import FeatureExtractionDialog

# Optional runtime flags for GPU/TIFF
_HAVE_TORCH = False
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

_HAVE_TIFFFILE = False
try:
    import tifffile  # type: ignore  # noqa: F401
    _HAVE_TIFFFILE = True
except Exception:
    _HAVE_TIFFFILE = False
from openmcd.ui.dialogs.clustering import CellClusteringDialog, ClusterExplorerDialog
from openmcd.ui.dialogs.comparison_dialog import DynamicComparisonDialog

# Optional runtime flags for extra deps
_HAVE_CELLPOSE = False
try:
    from cellpose import models as _cp_models  # type: ignore  # noqa: F401
    import skimage  # type: ignore  # noqa: F401
    _HAVE_CELLPOSE = True
except Exception:
    _HAVE_CELLPOSE = False
else:
    # Import models under the expected name when available
    from cellpose import models  # type: ignore

# --------------------------
# Main Window
# --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMC .mcd File Viewer")
        
        # Set window size to full screen
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.resize(screen.width(), screen.height())

        # State
        self.loader: Optional[MCDLoader] = None
        self.current_path: Optional[str] = None
        self.acquisitions: List[AcquisitionInfo] = []
        self.current_acq_id: Optional[str] = None
        # Image cache and prefetching
        self.image_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._prefetch_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Per (acq_id, channel) annotations → label
        self.annotations: Dict[Tuple[str, str], str] = {}
        self.annotation_labels = ["Unlabeled", "High-quality", "Low-quality", "Artifact/Exclude"]
        
        # Store last selected channels for auto-selection
        self.last_selected_channels: List[str] = []

        # Widgets
        self.canvas = MplCanvas(width=6, height=6, dpi=100)
        self.open_btn = QtWidgets.QPushButton("Open .mcd")
        self.acq_combo = QtWidgets.QComboBox()
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.deselect_all_btn = QtWidgets.QPushButton("Deselect all")
        # Removed 'View selected' button; auto-refresh is enabled
        self.view_btn = QtWidgets.QPushButton("View selected")
        self.view_btn.setVisible(False)
        self.comparison_btn = QtWidgets.QPushButton("Comparison mode")
        self.segment_btn = QtWidgets.QPushButton("Cell Segmentation")
        self.extract_features_btn = QtWidgets.QPushButton("Extract Features")
        self.clustering_btn = QtWidgets.QPushButton("Cell Clustering")
        
        # Visualization options
        self.grayscale_chk = QtWidgets.QCheckBox("Grayscale mode")
        self.grid_view_chk = QtWidgets.QCheckBox("Grid view for multiple channels")
        # Auto-refresh on toggle
        self.grayscale_chk.toggled.connect(lambda _: self._view_selected())
        self.grid_view_chk.toggled.connect(self._on_grid_view_toggled)
        self.grid_view_chk.setChecked(True)
        self.segmentation_overlay_chk = QtWidgets.QCheckBox("Show segmentation overlay")
        self.segmentation_overlay_chk.toggled.connect(self._on_segmentation_overlay_toggled)
        
        # Custom scaling controls
        self.custom_scaling_chk = QtWidgets.QCheckBox("Custom scaling")
        self.custom_scaling_chk.toggled.connect(self._on_custom_scaling_toggled)
        
        self.scaling_frame = QtWidgets.QFrame()
        self.scaling_frame.setFrameStyle(QtWidgets.QFrame.Box)
        scaling_layout = QtWidgets.QVBoxLayout(self.scaling_frame)
        scaling_layout.addWidget(QtWidgets.QLabel("Custom Intensity Range:"))
        
        # Channel selection for per-channel scaling
        channel_row = QtWidgets.QHBoxLayout()
        channel_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.scaling_channel_combo = QtWidgets.QComboBox()
        self.scaling_channel_combo.currentTextChanged.connect(self._on_scaling_channel_changed)
        channel_row.addWidget(self.scaling_channel_combo)
        channel_row.addStretch()
        scaling_layout.addLayout(channel_row)
        
        # Number input controls
        input_layout = QtWidgets.QVBoxLayout()
        
        # Min input
        min_row = QtWidgets.QHBoxLayout()
        min_row.addWidget(QtWidgets.QLabel("Min:"))
        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setRange(0.0, 10000.0)
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setValue(0.0)
        self.min_spinbox.setSingleStep(0.1)
        self.min_spinbox.valueChanged.connect(self._on_scaling_changed)
        min_row.addWidget(self.min_spinbox)
        min_row.addStretch()
        input_layout.addLayout(min_row)
        
        # Max input
        max_row = QtWidgets.QHBoxLayout()
        max_row.addWidget(QtWidgets.QLabel("Max:"))
        self.max_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_spinbox.setRange(0.0, 10000.0)
        self.max_spinbox.setDecimals(3)
        self.max_spinbox.setValue(1000.0)
        self.max_spinbox.setSingleStep(0.1)
        self.max_spinbox.valueChanged.connect(self._on_scaling_changed)
        max_row.addWidget(self.max_spinbox)
        max_row.addStretch()
        input_layout.addLayout(max_row)
        
        scaling_layout.addLayout(input_layout)
        
        # Arcsinh normalization controls
        arcsinh_layout = QtWidgets.QHBoxLayout()
        arcsinh_layout.addWidget(QtWidgets.QLabel("Arcsinh Co-factor:"))
        self.cofactor_spinbox = QtWidgets.QDoubleSpinBox()
        self.cofactor_spinbox.setRange(0.1, 100.0)
        self.cofactor_spinbox.setDecimals(1)
        self.cofactor_spinbox.setValue(5.0)
        self.cofactor_spinbox.setSingleStep(0.5)
        arcsinh_layout.addWidget(self.cofactor_spinbox)
        arcsinh_layout.addStretch()
        scaling_layout.addLayout(arcsinh_layout)
        
        # Control buttons
        button_row = QtWidgets.QHBoxLayout()
        self.percentile_btn = QtWidgets.QPushButton("Percentile Scaling")
        self.percentile_btn.clicked.connect(self._percentile_scaling)
        button_row.addWidget(self.percentile_btn)
        
        self.arcsinh_btn = QtWidgets.QPushButton("Arcsinh Normalization")
        self.arcsinh_btn.clicked.connect(self._arcsinh_normalization)
        button_row.addWidget(self.arcsinh_btn)
        
        self.default_range_btn = QtWidgets.QPushButton("Default Range")
        self.default_range_btn.clicked.connect(self._default_range)
        button_row.addWidget(self.default_range_btn)
        
        # Remove Apply button; auto-apply scaling on change
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setVisible(False)
        
        scaling_layout.addLayout(button_row)
        self.scaling_frame.setVisible(False)
        
        # Store per-channel scaling values
        self.channel_scaling = {}  # {channel_name: {'min': value, 'max': value}}
        
        # Arcsinh normalization state
        self.arcsinh_enabled = False
        # Per-channel normalization config: {channel: {"method": str, "cofactor": float}}
        self.channel_normalization: Dict[str, Dict[str, float or str]] = {}
        
        # Per-channel scaling method state
        self.current_scaling_method = "default"  # kept for backward compatibility
        self.channel_scaling_method: Dict[str, str] = {}  # {channel: "default"|"percentile"|"arcsinh"}
        
        # Segmentation state
        self.segmentation_masks = {}  # {acq_id: mask_array}
        self.segmentation_colors = {}  # {acq_id: colors_array}
        self.segmentation_overlay = False
        self.preprocessing_cache = PreprocessingCache()
        
        # Feature extraction state
        self.feature_dataframe = None  # Store extracted features in memory
        
        # Color assignment for RGB composite
        self.color_assignment_frame = QtWidgets.QFrame()
        self.color_assignment_frame.setFrameStyle(QtWidgets.QFrame.Box)
        color_layout = QtWidgets.QVBoxLayout(self.color_assignment_frame)
        color_layout.addWidget(QtWidgets.QLabel("Color Assignment (for RGB composite):"))
        
        # Red channel selection
        red_layout = QtWidgets.QHBoxLayout()
        red_layout.addWidget(QtWidgets.QLabel("Red:"))
        self.red_list = QtWidgets.QListWidget()
        self.red_list.setMaximumHeight(80)
        self.red_list.setMaximumWidth(200)
        self.red_list.itemChanged.connect(lambda _i: self._on_rgb_list_changed())
        red_layout.addWidget(self.red_list)
        color_layout.addLayout(red_layout)
        
        # Green channel selection
        green_layout = QtWidgets.QHBoxLayout()
        green_layout.addWidget(QtWidgets.QLabel("Green:"))
        self.green_list = QtWidgets.QListWidget()
        self.green_list.setMaximumHeight(80)
        self.green_list.setMaximumWidth(200)
        self.green_list.itemChanged.connect(lambda _i: self._on_rgb_list_changed())
        green_layout.addWidget(self.green_list)
        color_layout.addLayout(green_layout)
        
        # Blue channel selection
        blue_layout = QtWidgets.QHBoxLayout()
        blue_layout.addWidget(QtWidgets.QLabel("Blue:"))
        self.blue_list = QtWidgets.QListWidget()
        self.blue_list.setMaximumHeight(80)
        self.blue_list.setMaximumWidth(200)
        self.blue_list.itemChanged.connect(lambda _i: self._on_rgb_list_changed())
        blue_layout.addWidget(self.blue_list)
        color_layout.addLayout(blue_layout)

        self.ann_combo = QtWidgets.QComboBox()
        self.ann_combo.addItems(self.annotation_labels)
        self.ann_apply_btn = QtWidgets.QPushButton("Apply label")

        # Metadata display
        self.metadata_text = QtWidgets.QTextEdit()
        self.metadata_text.setMaximumHeight(150)
        self.metadata_text.setReadOnly(True)

        # Left panel layout
        controls = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(controls)
        v.addWidget(self.open_btn)

        v.addWidget(QtWidgets.QLabel("Acquisition:"))
        v.addWidget(self.acq_combo)

        v.addWidget(QtWidgets.QLabel("Channels:"))
        
        # Channel search box
        self.channel_search = QtWidgets.QLineEdit()
        self.channel_search.setPlaceholderText("Search channels...")
        self.channel_search.textChanged.connect(self._filter_channels)
        v.addWidget(self.channel_search)
        
        v.addWidget(self.channel_list, 1)
        
        # Channel control buttons
        channel_btn_row = QtWidgets.QHBoxLayout()
        channel_btn_row.addWidget(self.deselect_all_btn)
        channel_btn_row.addStretch()
        v.addLayout(channel_btn_row)
        
        # Visualization options
        v.addWidget(self.grayscale_chk)
        v.addWidget(self.grid_view_chk)
        v.addWidget(self.segmentation_overlay_chk)
        v.addWidget(self.custom_scaling_chk)
        v.addWidget(self.scaling_frame)
        v.addWidget(self.color_assignment_frame)

        ann_row = QtWidgets.QHBoxLayout()
        ann_row.addWidget(QtWidgets.QLabel("Annotation:"))
        ann_row.addWidget(self.ann_combo, 1)
        v.addLayout(ann_row)
        v.addWidget(self.ann_apply_btn)

        v.addSpacing(8)
        v.addWidget(self.view_btn)
        v.addWidget(self.comparison_btn)
        v.addWidget(self.segment_btn)
        v.addWidget(self.extract_features_btn)
        v.addWidget(self.clustering_btn)
        v.addSpacing(8)
        
        v.addWidget(QtWidgets.QLabel("Metadata:"))
        v.addWidget(self.metadata_text)
        v.addStretch(1)

        # Splitter
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        leftw = QtWidgets.QWidget()
        leftw.setLayout(v)
        splitter.addWidget(leftw)
        # Right pane with toolbar + canvas
        rightw = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(rightw)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.nav_toolbar)
        right_layout.addWidget(self.canvas, 1)
        splitter.addWidget(rightw)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        act_open = file_menu.addAction("Open .mcd…")
        act_open.triggered.connect(self._open_dialog)
        file_menu.addSeparator()
        
        # Export submenu
        export_submenu = file_menu.addMenu("Export")
        act_export_tiff = export_submenu.addAction("Export to OME-TIFF…")
        act_export_tiff.triggered.connect(self._export_ome_tiff)
        act_save_annotations = export_submenu.addAction("Save Annotations CSV…")
        act_save_annotations.triggered.connect(self._save_annotations)
        
        # Masks submenu
        masks_submenu = file_menu.addMenu("Segmentation Masks")
        act_load_masks = masks_submenu.addAction("Load Masks…")
        act_load_masks.triggered.connect(self._load_segmentation_masks)
        act_save_masks = masks_submenu.addAction("Save Masks…")
        act_save_masks.triggered.connect(self._save_segmentation_masks)
        
        # Import submenu
        import_submenu = file_menu.addMenu("Import")
        act_load_annotations = import_submenu.addAction("Load Annotations CSV…")
        act_load_annotations.triggered.connect(self._load_annotations)
        
        file_menu.addSeparator()
        act_quit = file_menu.addAction("Quit")
        act_quit.triggered.connect(self.close)

        # Analysis menu
        analysis_menu = self.menuBar().addMenu("&Analysis")
        act_clustering = analysis_menu.addAction("Cell Clustering…")
        act_clustering.triggered.connect(self._open_clustering_dialog)

        # Signals
        self.open_btn.clicked.connect(self._open_dialog)
        self.acq_combo.currentIndexChanged.connect(self._on_acq_changed)
        self.deselect_all_btn.clicked.connect(self._deselect_all_channels)
        self.channel_list.itemChanged.connect(self._on_channel_selection_changed)
        self.channel_search.textChanged.connect(self._filter_channels)
        # Auto-refresh: no manual 'View selected' action
        try:
            self.view_btn.clicked.disconnect()
        except Exception:
            pass
        self.comparison_btn.clicked.connect(self._comparison)
        self.segment_btn.clicked.connect(self._run_segmentation)
        self.extract_features_btn.clicked.connect(self._extract_features)
        self.clustering_btn.clicked.connect(self._open_clustering_dialog)
        self.ann_apply_btn.clicked.connect(self._apply_annotation)

        # Loader
        try:
            self.loader = MCDLoader()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Missing dependency", str(e))

        # Ensure RGB controls are hidden when grid view is enabled on startup
        try:
            self._update_rgb_controls_visibility()
        except Exception:
            pass

    # ---------- File open ----------
    def _open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open IMC .mcd file", "", "IMC MCD files (*.mcd);;All files (*.*)"
        )
        if path:
            self._load_mcd(path)

    def _load_mcd(self, path: str):
        if self.loader is None:
            try:
                self.loader = MCDLoader()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Dependency error", str(e))
                return
        try:
            self.loader.open(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", f"Failed to open {path}\n\n{e}")
            return
        self.current_path = path
        try:
            stem = os.path.splitext(os.path.basename(path))[0]
            self.setWindowTitle(f"IMC .mcd File Viewer - {stem}")
        except Exception:
            # Fallback to default title if something goes wrong
            self.setWindowTitle("IMC .mcd File Viewer")
        self.acquisitions = self.loader.list_acquisitions()
        self.acq_combo.clear()
        for ai in self.acquisitions:
            label = ai.name + (f"({ai.well})" if ai.well else "")
            self.acq_combo.addItem(label, ai.id)
        if self.acquisitions:
            self._populate_channels(self.acquisitions[0].id)

    # ---------- Acquisition / channels ----------
    def _on_acq_changed(self, idx: int):
        acq_id = self.acq_combo.itemData(idx)
        if acq_id:
            # Store current scaling state before changing acquisition
            preserve_scaling = self.custom_scaling_chk.isChecked()
            current_scaling_method = self.current_scaling_method
            
            self._populate_channels(acq_id)
            # Start background prefetch of all channels for the new acquisition
            self._start_prefetch_all_channels(acq_id)
            
            # Update scaling channel combo when acquisition changes
            if preserve_scaling:
                self._update_scaling_channel_combo()
                # Restore scaling method state
                self.current_scaling_method = current_scaling_method
                self._update_minmax_controls_state()

    def _populate_channels(self, acq_id: str):
        self.current_acq_id = acq_id
        self.channel_list.clear()
        try:
            chans = self.loader.get_channels(acq_id)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Channels error", str(e))
            return
        
        # Pre-select channels that were selected in the previous acquisition
        selected_channels = []
        for ch in chans:
            item = QtWidgets.QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            
            # Check if this channel was selected in the previous acquisition
            if ch in self.last_selected_channels:
                item.setCheckState(Qt.Checked)
                selected_channels.append(ch)
            else:
                item.setCheckState(Qt.Unchecked)
            
            self.channel_list.addItem(item)
        
        # Kick off prefetch if not already running for this acq
        self._start_prefetch_all_channels(acq_id)

        # Update RGB color assignment lists with only currently selected channels
        self._populate_color_assignments(selected_channels)
        
        # Auto-load image if channels were pre-selected
        if selected_channels:
            self._auto_load_image(selected_channels)
        
        # Update metadata display
        ai = next(a for a in self.acquisitions if a.id == acq_id)
        metadata_text = f"Acquisition: {ai.name}\n"
        if ai.well:
            metadata_text += f"{ai.well}\n"
        if ai.size[0] and ai.size[1]:
            metadata_text += f"Size: {ai.size[1]} x {ai.size[0]} pixels\n"
        metadata_text += f"Channels: {len(ai.channels)}\n\n"
        
        # Add GPU info if available
        if _HAVE_TORCH:
            gpu_info = self._get_gpu_info()
            if gpu_info:
                metadata_text += f"GPU: {gpu_info}\n\n"
        
        if ai.metadata:
            metadata_text += "Metadata:\n"
            for key, value in ai.metadata.items():
                metadata_text += f"  {key}: {value}\n"
        
        self.metadata_text.setPlainText(metadata_text)

    def _on_channel_selection_changed(self):
        """Update color assignment dropdowns when channel selection changes."""
        selected_channels = self._selected_channels()
        self._populate_color_assignments(selected_channels)
        
        # Clear any color assignments that are no longer in the selected channels
        self._clear_invalid_color_assignments(selected_channels)
        
        # Update scaling channel combo to reflect current selection
        if self.custom_scaling_chk.isChecked():
            self._update_scaling_channel_combo()
        # Auto-refresh view when channels change
        self._view_selected()

        # Update RGB control visibility and selections on change
        self._update_rgb_controls_visibility()

    def _populate_color_assignments(self, channels: List[str]):
        """Populate the color assignment dropdowns with selected channels only."""
        # Clear existing items
        # Preserve current checks
        prev_red = {self.red_list.item(i).text(): self.red_list.item(i).checkState() == Qt.Checked for i in range(self.red_list.count())}
        prev_green = {self.green_list.item(i).text(): self.green_list.item(i).checkState() == Qt.Checked for i in range(self.green_list.count())}
        prev_blue = {self.blue_list.item(i).text(): self.blue_list.item(i).checkState() == Qt.Checked for i in range(self.blue_list.count())}

        self.red_list.clear()
        self.green_list.clear()
        self.blue_list.clear()
        
        # Only add selected channels; if none, keep lists empty
        for ch in channels:
            for lst, prev in [(self.red_list, prev_red), (self.green_list, prev_green), (self.blue_list, prev_blue)]:
                it = QtWidgets.QListWidgetItem(ch)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                # Restore previous check state if present
                checked = prev.get(ch, False)
                it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                lst.addItem(it)

        # Do not auto-select by default; leave empty if no channels are selected

    def _clear_invalid_color_assignments(self, selected_channels: List[str]):
        """Clear color assignments that are no longer in the selected channels."""
        # For list-based multi-select, deselect any items not in current selection list
        def _prune_list(lst: QtWidgets.QListWidget):
            for i in range(lst.count()):
                item = lst.item(i)
                if item.text() not in selected_channels:
                    item.setCheckState(Qt.Unchecked)
    def _on_rgb_list_changed(self):
        # Ensure lists only keep checks for currently selected channels
        selected_channels = self._selected_channels()
        def _prune(lst: QtWidgets.QListWidget):
            for i in range(lst.count()):
                item = lst.item(i)
                if item.text() not in selected_channels:
                    item.setCheckState(Qt.Unchecked)
        _prune(self.red_list)
        _prune(self.green_list)
        _prune(self.blue_list)
        
        # Update arcsinh button state based on new RGB assignments
        self._update_minmax_controls_state()
        
        # Refresh view
        self._view_selected()

    def _on_grid_view_toggled(self):
        self._update_rgb_controls_visibility()
        
        # Handle arcsinh/percentile state when switching between RGB and grid view
        if self.grid_view_chk.isChecked():
            # Switching to grid view - arcsinh/percentile becomes available for all channels
            self._enable_auto_scaling_for_grid_view()
        else:
            # Switching to RGB view - revert channels that had arcsinh/percentile applied in grid view
            self._revert_auto_scaling_for_rgb_view()
        
        self._view_selected()

    def _update_rgb_controls_visibility(self):
        """Show RGB assignment panel only when grid view is off."""
        # Guard if called before widgets are constructed
        if not hasattr(self, 'color_assignment_frame'):
            return
        show_rgb = not self.grid_view_chk.isChecked()
        self.color_assignment_frame.setVisible(show_rgb)

    def _deselect_all_channels(self):
        """Deselect all channels in the channel list."""
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setCheckState(Qt.Unchecked)
        self.channel_list.clearSelection()
        
        # Clear all color assignments when deselecting all channels
        self._populate_color_assignments([])

    def _selected_channels(self) -> List[str]:
        chans: List[str] = []
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            if it.checkState() == Qt.Checked or it.isSelected():
                chans.append(it.text())
        # unique, preserve order
        seen = set()
        uniq = []
        for c in chans:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def _get_acquisition_subtitle(self, acq_id: str) -> str:
        """Get acquisition subtitle showing well/description instead of acquisition number."""
        acq_info = next((ai for ai in self.acquisitions if ai.id == acq_id), None)
        if not acq_info:
            return "Unknown"
        
        # Use well if available, otherwise use name (which might be more descriptive)
        if acq_info.well:
            return f"{acq_info.well}"
        else:
            return acq_info.name

    def _on_custom_scaling_toggled(self):
        """Handle custom scaling checkbox toggle."""
        self.scaling_frame.setVisible(self.custom_scaling_chk.isChecked())
        if self.custom_scaling_chk.isChecked():
            self._update_scaling_channel_combo()
            self._load_channel_scaling()
            # Initialize controls state
            self._update_minmax_controls_state()
        # Auto-refresh when toggled
        self._view_selected()

    def _update_scaling_channel_combo(self):
        """Update the scaling channel combo box with selected channels only."""
        self.scaling_channel_combo.clear()
        if self.current_acq_id is None:
            return
        
        # Only show currently selected channels
        selected_channels = self._selected_channels()
        for channel in selected_channels:
            self.scaling_channel_combo.addItem(channel)
        
        # Select first channel if available
        if self.scaling_channel_combo.count() > 0:
            self.scaling_channel_combo.setCurrentIndex(0)
            self._load_channel_scaling()

    def _on_scaling_channel_changed(self):
        """Handle changes to the scaling channel selection."""
        if self.custom_scaling_chk.isChecked():
            self._load_channel_scaling()
            # Update controls state based on current scaling method
            self._update_minmax_controls_state()
        else:
            # Even if custom scaling is off, ensure controls reflect per-channel method
            self._update_minmax_controls_state()
        # Auto-refresh
        self._view_selected()

    def _on_scaling_changed(self):
        """Handle changes to the min/max spinboxes."""
        if self.custom_scaling_chk.isChecked():
            # Save current values
            self._save_channel_scaling()
            # Auto-refresh display
            self._view_selected()
    
    def _filter_channels(self):
        """Filter channels based on search text."""
        search_text = self.channel_search.text().lower()
        
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            channel_name = item.text().lower()
            item.setHidden(search_text not in channel_name)
    
    def _update_minmax_controls_state(self):
        """Enable/disable min/max controls based on scaling method."""
        # Determine current channel's method
        current_channel = self.scaling_channel_combo.currentText()
        method = self.channel_scaling_method.get(current_channel, "default")
        
        # Check if automatic scaling should be disabled for this channel
        auto_scaling_disabled = self._is_auto_scaling_disabled_for_channel(current_channel)
        
        if method in ["percentile", "arcsinh"]:
            # Disable min/max controls for automatic scaling methods
            self.min_spinbox.setEnabled(False)
            self.max_spinbox.setEnabled(False)
            self.min_spinbox.setStyleSheet("QDoubleSpinBox { background-color: #f0f0f0; color: #666; }")
            self.max_spinbox.setStyleSheet("QDoubleSpinBox { background-color: #f0f0f0; color: #666; }")
        else:
            # Enable min/max controls for manual/default scaling
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            self.min_spinbox.setStyleSheet("")
            self.max_spinbox.setStyleSheet("")
        
        # Update arcsinh and percentile button states
        if auto_scaling_disabled:
            self.arcsinh_btn.setEnabled(False)
            self.arcsinh_btn.setToolTip("Arcsinh disabled: Multiple markers assigned to same RGB color")
            self.cofactor_spinbox.setEnabled(False)
            self.percentile_btn.setEnabled(False)
            self.percentile_btn.setToolTip("Percentile scaling disabled: Multiple markers assigned to same RGB color")
        else:
            self.arcsinh_btn.setEnabled(True)
            self.arcsinh_btn.setToolTip("Apply arcsinh normalization to current channel")
            self.cofactor_spinbox.setEnabled(True)
            self.percentile_btn.setEnabled(True)
            self.percentile_btn.setToolTip("Apply percentile scaling to current channel")

    def _is_auto_scaling_disabled_for_channel(self, channel: str) -> bool:
        """Check if automatic scaling (arcsinh/percentile) should be disabled for a channel because multiple markers are assigned to the same RGB color."""
        if not channel:
            return False
        
        # In grid view, all channels are displayed individually, so arcsinh/percentile is always available
        if self.grid_view_chk.isChecked():
            return False
        
        # Get current RGB color assignments
        def _checked(lst: QtWidgets.QListWidget) -> List[str]:
            vals: List[str] = []
            for i in range(lst.count()):
                item = lst.item(i)
                if item.checkState() == Qt.Checked:
                    vals.append(item.text())
            return vals
        
        red_selection = _checked(self.red_list)
        green_selection = _checked(self.green_list)
        blue_selection = _checked(self.blue_list)
        
        # Check if this channel is assigned to any RGB color that has multiple channels
        if channel in red_selection and len(red_selection) > 1:
            return True
        if channel in green_selection and len(green_selection) > 1:
            return True
        if channel in blue_selection and len(blue_selection) > 1:
            return True
        
        return False

    def _enable_auto_scaling_for_grid_view(self):
        """Enable arcsinh/percentile for all channels when switching to grid view."""
        # In grid view, all channels are displayed individually, so arcsinh/percentile is always available
        # No special action needed - the _update_minmax_controls_state will handle enabling the buttons
        pass

    def _revert_auto_scaling_for_rgb_view(self):
        """Revert channels to default range when switching back to RGB view if they had arcsinh/percentile applied in grid view."""
        # Get current RGB color assignments
        def _checked(lst: QtWidgets.QListWidget) -> List[str]:
            vals: List[str] = []
            for i in range(lst.count()):
                item = lst.item(i)
                if item.checkState() == Qt.Checked:
                    vals.append(item.text())
            return vals
        
        red_selection = _checked(self.red_list)
        green_selection = _checked(self.green_list)
        blue_selection = _checked(self.blue_list)
        
        # Find channels that have multiple assignments and had arcsinh/percentile applied
        channels_to_revert = []
        
        # Check red channels
        if len(red_selection) > 1:
            for channel in red_selection:
                if (channel in self.channel_scaling_method and 
                    self.channel_scaling_method[channel] in ["arcsinh", "percentile"]):
                    channels_to_revert.append(channel)
        
        # Check green channels
        if len(green_selection) > 1:
            for channel in green_selection:
                if (channel in self.channel_scaling_method and 
                    self.channel_scaling_method[channel] in ["arcsinh", "percentile"]):
                    channels_to_revert.append(channel)
        
        # Check blue channels
        if len(blue_selection) > 1:
            for channel in blue_selection:
                if (channel in self.channel_scaling_method and 
                    self.channel_scaling_method[channel] in ["arcsinh", "percentile"]):
                    channels_to_revert.append(channel)
        
        # Revert each channel to default range
        for channel in channels_to_revert:
            try:
                if self.current_acq_id:
                    img = self.loader.get_image(self.current_acq_id, channel)
                    min_val = float(np.min(img))
                    max_val = float(np.max(img))
                    
                    # Update scaling method to default
                    self.channel_scaling_method[channel] = "default"
                    
                    # Clear normalization settings
                    if channel in self.channel_normalization:
                        self.channel_normalization.pop(channel, None)
                    
                    # Update scaling values
                    self.channel_scaling[channel] = {'min': min_val, 'max': max_val}
                    
            except Exception as e:
                print(f"Error reverting channel {channel} to default range: {e}")
        
        # Update UI controls to reflect the reverted state
        self._update_minmax_controls_state()

    def _load_channel_scaling(self):
        """Load scaling values for the currently selected channel."""
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        if current_channel in self.channel_scaling:
            # Load saved values
            min_val = self.channel_scaling[current_channel]['min']
            max_val = self.channel_scaling[current_channel]['max']
        else:
            # Use default range (full image range)
            if self.current_acq_id is None:
                return
            try:
                img = self.loader.get_image(self.current_acq_id, current_channel)
                min_val = float(np.min(img))
                max_val = float(np.max(img))
            except Exception as e:
                print(f"Error loading channel scaling: {e}")
                return
        
        # Update spinboxes based on actual values
        self._update_spinboxes_from_values(min_val, max_val)
        
        # Load per-channel arcsinh cofactor if available
        if current_channel in self.channel_normalization:
            norm_cfg = self.channel_normalization[current_channel]
            if norm_cfg.get("method") == "arcsinh":
                cofactor = norm_cfg.get("cofactor", 5.0)
                self.cofactor_spinbox.setValue(float(cofactor))

    def _save_channel_scaling(self):
        """Save current scaling values for the selected channel."""
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        # Get values directly from spinboxes
        min_val = self.min_spinbox.value()
        max_val = self.max_spinbox.value()
        
        self.channel_scaling[current_channel] = {'min': min_val, 'max': max_val}

    def _update_spinboxes_from_values(self, min_val, max_val):
        """Update spinboxes based on actual min/max values."""
        # Update spinboxes without triggering valueChanged
        self.min_spinbox.blockSignals(True)
        self.max_spinbox.blockSignals(True)
        self.min_spinbox.setValue(min_val)
        self.max_spinbox.setValue(max_val)
        self.min_spinbox.blockSignals(False)
        self.max_spinbox.blockSignals(False)

    def _percentile_scaling(self):
        """Set scaling using robust percentile scaling (1st-99th percentiles)."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        # Check if automatic scaling is disabled for this channel
        if self._is_auto_scaling_disabled_for_channel(current_channel):
            QtWidgets.QMessageBox.warning(self, "Percentile Scaling Disabled", 
                f"Percentile scaling is disabled for '{current_channel}' because multiple markers are assigned to the same RGB color.")
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            # Use robust percentile scaling function
            scaled_img = robust_percentile_scale(img, low=1.0, high=99.0)
            
            # Get the actual min/max values that were used for scaling
            min_val = float(np.percentile(img, 1))
            max_val = float(np.percentile(img, 99))
            
            self._update_spinboxes_from_values(min_val, max_val)
            
            # Update scaling method state
            self.current_scaling_method = "percentile"
            self.channel_scaling_method[current_channel] = "percentile"
            self.arcsinh_enabled = False
            self._update_minmax_controls_state()
            
            # Auto-apply the scaling
            self._save_channel_scaling()
            self._view_selected()
        except Exception as e:
            print(f"Error in percentile scaling: {e}")

    def _arcsinh_normalization(self):
        """Apply arcsinh normalization with configurable co-factor."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        # Check if automatic scaling is disabled for this channel
        if self._is_auto_scaling_disabled_for_channel(current_channel):
            QtWidgets.QMessageBox.warning(self, "Arcsinh Disabled", 
                f"Arcsinh normalization is disabled for '{current_channel}' because multiple markers are assigned to the same RGB color.")
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            cofactor = self.cofactor_spinbox.value()
            
            # Apply arcsinh normalization
            normalized_img = arcsinh_normalize(img, cofactor=cofactor)
            
            # Get the min/max values of the normalized image for scaling
            min_val = float(np.min(normalized_img))
            max_val = float(np.max(normalized_img))
            
            self._update_spinboxes_from_values(min_val, max_val)
            
            # Update scaling method state
            self.current_scaling_method = "arcsinh"
            self.channel_scaling_method[current_channel] = "arcsinh"
            # Only set normalization for the selected channel
            self.channel_normalization[current_channel] = {"method": "arcsinh", "cofactor": cofactor}
            self._update_minmax_controls_state()
            
            # Auto-apply the scaling
            self._save_channel_scaling()
            self._view_selected()
        except Exception as e:
            print(f"Error in arcsinh normalization: {e}")

    def _default_range(self):
        """Set scaling to the image's actual min/max range."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            min_val = float(np.min(img))
            max_val = float(np.max(img))
            
            self._update_spinboxes_from_values(min_val, max_val)
            
            # Update scaling method state
            self.current_scaling_method = "default"
            self.channel_scaling_method[current_channel] = "default"
            # Clear per-channel normalization for this channel
            if current_channel in self.channel_normalization:
                self.channel_normalization.pop(current_channel, None)
            self._update_minmax_controls_state()
            
            # Auto-apply the scaling and reload image in original range
            self._save_channel_scaling()
            self._view_selected()
        except Exception as e:
            print(f"Error in default range: {e}")

    def _load_image_with_normalization(self, acq_id: str, channel: str) -> np.ndarray:
        """Load image and apply arcsinh normalization if enabled."""
        # Try cache first
        cache_key = (acq_id, channel)
        with self._cache_lock:
            img = self.image_cache.get(cache_key)
        if img is None:
            img = self.loader.get_image(acq_id, channel)
            with self._cache_lock:
                self.image_cache[cache_key] = img
        
        # Apply per-channel normalization (if configured)
        norm_cfg = self.channel_normalization.get(channel)
        if norm_cfg and norm_cfg.get("method") == "arcsinh":
            cofactor = float(norm_cfg.get("cofactor", 5.0))
            img = arcsinh_normalize(img, cofactor=cofactor)
        
        return img

    def _start_prefetch_all_channels(self, acq_id: str):
        """Prefetch all channels for the given acquisition in the background (non-blocking)."""
        if self.loader is None or not acq_id:
            return
        # If a previous prefetch is running, let it finish; avoid stacking tasks
        if self._prefetch_future and not self._prefetch_future.done():
            return

        channels = []
        try:
            channels = self.loader.get_channels(acq_id)
        except Exception:
            return

        def _prefetch():
            try:
                # Load the full stack once, then split into channels for faster access
                stack = self.loader.get_all_channels(acq_id)
                # Store in cache
                with self._cache_lock:
                    for i, ch in enumerate(channels):
                        try:
                            self.image_cache[(acq_id, ch)] = stack[..., i]
                        except Exception:
                            continue
            except Exception:
                # Swallow errors silently to avoid UI disruption
                return

        self._prefetch_future = self._executor.submit(_prefetch)

    def _apply_scaling(self):
        """Apply the current scaling settings to the selected channel and refresh display."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        # Save current scaling values
        self._save_channel_scaling()
        
        # Refresh display
        self._view_selected()


    # ---------- View ----------
    def _view_selected(self):
        if self.current_acq_id is None:
            # Silent no-op during auto-refresh before an acquisition is selected
            return
        chans = self._selected_channels()
        if not chans:
            # Silent no-op during auto-refresh when no channels are selected
            return
        
        # Store selected channels for auto-selection in next acquisition
        self.last_selected_channels = chans.copy()
        
        grayscale = self.grayscale_chk.isChecked()
        grid_view = self.grid_view_chk.isChecked()
        
        # Get custom scaling values if enabled
        # For single channel view, use that channel's scaling
        # For RGB/grid view, we'll handle per-channel scaling in the display methods
        custom_min = None
        custom_max = None
        if self.custom_scaling_chk.isChecked() and len(chans) == 1:
            # For single channel, use the scaling for that specific channel
            channel = chans[0]
            if channel in self.channel_scaling:
                custom_min = self.channel_scaling[channel]['min']
                custom_max = self.channel_scaling[channel]['max']
        
        try:
            if not grid_view:
                # RGB composite view using user-selected color assignments (supports single or multiple channels per RGB)
                self._show_rgb_composite(chans, grayscale)
            else:
                # Grid view for multiple channels (when grid_view is True)
                images = [self._load_image_with_normalization(self.current_acq_id, c) for c in chans]
                
                # Apply segmentation overlay to all images if enabled
                if self.segmentation_overlay:
                    images = [self._get_segmentation_overlay(img) for img in images]
                
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                # Add acquisition subtitle to each channel title
                titles = [f"{ch}\n{acq_subtitle}" for ch in chans]
                if self.segmentation_overlay:
                    titles = [f"{ch}\n{acq_subtitle} (segmented)" for ch in chans]
                self.canvas.show_grid(images, titles, grayscale=grayscale, raw_images=images, 
                                    channel_names=chans, channel_scaling=self.channel_scaling, 
                                    custom_scaling_enabled=self.custom_scaling_chk.isChecked())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "View error", str(e))

    def _auto_load_image(self, selected_channels: List[str]):
        """Automatically load and display image for pre-selected channels."""
        try:
            grayscale = self.grayscale_chk.isChecked()
            grid_view = self.grid_view_chk.isChecked()
            
            # Get custom scaling values if enabled
            custom_min = None
            custom_max = None
            if self.custom_scaling_chk.isChecked() and len(selected_channels) == 1:
                # For single channel, use the scaling for that specific channel
                channel = selected_channels[0]
                if channel in self.channel_scaling:
                    custom_min = self.channel_scaling[channel]['min']
                    custom_max = self.channel_scaling[channel]['max']
            
            if not grid_view:
                # RGB composite view using user-selected color assignments (supports single or multiple channels per RGB)
                self._show_rgb_composite(selected_channels, grayscale)
            else:
                # Grid view for multiple channels (when grid_view is True)
                images = [self._load_image_with_normalization(self.current_acq_id, c) for c in selected_channels]
                
                # Apply segmentation overlay to all images if enabled
                if self.segmentation_overlay:
                    images = [self._get_segmentation_overlay(img) for img in images]
                
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                # Add acquisition subtitle to each channel title
                titles = [f"{ch}\n{acq_subtitle}" for ch in selected_channels]
                if self.segmentation_overlay:
                    titles = [f"{ch}\n{acq_subtitle} (segmented)" for ch in selected_channels]
                self.canvas.show_grid(images, titles, grayscale=grayscale, raw_images=images, 
                                    channel_names=selected_channels, channel_scaling=self.channel_scaling, 
                                    custom_scaling_enabled=self.custom_scaling_chk.isChecked())
        except Exception as e:
            print(f"Auto-load error: {e}")

    def _show_rgb_composite(self, selected_channels: List[str], grayscale: bool):
        """Show RGB composite using user-selected color assignments."""
        # Get user-selected color assignments
        # Read multi-selections for each color
        def _checked(lst: QtWidgets.QListWidget) -> List[str]:
            vals: List[str] = []
            for i in range(lst.count()):
                item = lst.item(i)
                if item.checkState() == Qt.Checked:
                    vals.append(item.text())
            return vals
        red_selection = _checked(self.red_list)
        green_selection = _checked(self.green_list)
        blue_selection = _checked(self.blue_list)
        
        # If only one channel is selected and no RGB assignments are made, assign it to red
        if (len(selected_channels) == 1 and 
            not red_selection and not green_selection and not blue_selection):
            red_selection = selected_channels.copy()
        
        # Create RGB stack based on user selections
        rgb_channels = []
        rgb_titles = []
        raw_channels = []  # Store raw images for colorbar
        
        # Get the first selected channel to determine image size
        first_img = None
        if selected_channels:
            first_img = self._load_image_with_normalization(self.current_acq_id, selected_channels[0])
        
        if first_img is None:
            QtWidgets.QMessageBox.information(self, "No RGB channels", "Please select at least one channel for RGB composite.")
            return
        
        # Always create 3 channels (R, G, B) even if some are empty
        def _sum_channels(names: List[str]) -> np.ndarray:
            if not names:
                return np.zeros_like(first_img)
            acc = np.zeros_like(first_img, dtype=np.float32)
            for ch_name in names:
                try:
                    img = self._load_image_with_normalization(self.current_acq_id, ch_name)
                except Exception:
                    img = np.zeros_like(first_img)
                acc += img.astype(np.float32)
            # Clip to max of original dtype range
            acc = np.clip(acc, 0, np.max(acc))
            return acc.astype(first_img.dtype)

        # Build R, G, B channels by summing selections per color
        r_img = _sum_channels(red_selection)
        g_img = _sum_channels(green_selection)
        b_img = _sum_channels(blue_selection)

        rgb_channels.append(r_img)
        raw_channels.append(r_img)
        rgb_titles.append(f"{'+'.join(red_selection) if red_selection else 'None'} (Red)")

        rgb_channels.append(g_img)
        raw_channels.append(g_img)
        rgb_titles.append(f"{'+'.join(green_selection) if green_selection else 'None'} (Green)")

        rgb_channels.append(b_img)
        raw_channels.append(b_img)
        rgb_titles.append(f"{'+'.join(blue_selection) if blue_selection else 'None'} (Blue)")
        
        # Ensure we have exactly 3 channels
        while len(rgb_channels) < 3:
            rgb_channels.append(np.zeros_like(first_img))
            raw_channels.append(np.zeros_like(first_img))
            rgb_titles.append(f"None ({['Red', 'Green', 'Blue'][len(rgb_channels)-1]})")
        
        # Apply per-channel custom scaling before stacking (for RGB display)
        if self.custom_scaling_chk.isChecked():
            scaled_channels = []
            color_selections = [red_selection, green_selection, blue_selection]
            
            for i, ch_img in enumerate(rgb_channels):
                # Skip empty channels (all zeros)
                if np.all(ch_img == 0):
                    scaled_channels.append(ch_img)
                    continue
                
                # Get the channels assigned to this RGB color
                assigned_channels = color_selections[i] if i < len(color_selections) else []
                
                # For custom scaling, we need to determine which channel's scaling to use
                # If multiple channels are assigned to this color, we'll use the first one that has scaling
                scaling_channel = None
                for ch_name in assigned_channels:
                    if ch_name in self.channel_scaling:
                        scaling_channel = ch_name
                        break
                
                if scaling_channel:
                    vmin = self.channel_scaling[scaling_channel]['min']
                    vmax = self.channel_scaling[scaling_channel]['max']
                    if vmax <= vmin:
                        vmax = vmin + 1e-6
                    
                    # For multiple channels with different arcsinh settings, we need to be more careful
                    # about the scaling range. The summed result might have a different range than
                    # any individual channel's scaling range.
                    if len(assigned_channels) > 1:
                        # When multiple channels are summed, use the actual range of the summed result
                        # but still apply the custom scaling logic
                        actual_min = float(np.min(ch_img))
                        actual_max = float(np.max(ch_img))
                        
                        # If the custom range is within the actual range, use it
                        if vmin >= actual_min and vmax <= actual_max:
                            ch_img = np.clip((ch_img.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
                        else:
                            # Otherwise, use the actual range but still normalize to 0-1
                            if actual_max > actual_min:
                                ch_img = (ch_img.astype(np.float32) - actual_min) / (actual_max - actual_min)
                            else:
                                ch_img = np.zeros_like(ch_img)
                    else:
                        # Single channel case - use the custom range directly
                        ch_img = np.clip((ch_img.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
                
                scaled_channels.append(ch_img)
            rgb_channels = scaled_channels

        # Stack channels
        stack = np.dstack(rgb_channels)
        raw_stack = np.dstack(raw_channels)
        
        # Note: Do NOT apply overlay per-channel (will be 3-channel) — apply to final RGB image below
        
        # Get acquisition subtitle
        acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
        title = " + ".join(rgb_titles) + f"\n{acq_subtitle}"
        if self.segmentation_overlay:
            title += " (segmented)"
        
        # Clear canvas and show RGB composite with individual colorbars
        self.canvas.fig.clear()
        
        if grayscale:
            # Grayscale background from assigned channels (mean of non-empty channels)
            ax = self.canvas.fig.add_subplot(111)

            nonzero_channels = [ch for ch in rgb_channels if not np.all(ch == 0)]
            if len(nonzero_channels) == 0:
                gray_base = stack[..., 0]
            elif len(nonzero_channels) == 1:
                gray_base = nonzero_channels[0]
            else:
                gray_base = np.mean(np.dstack(nonzero_channels), axis=2)

            if self.segmentation_overlay:
                # Apply colored overlay on top of grayscale background
                blended = self._get_segmentation_overlay(gray_base)
                ax.imshow(blended, interpolation="nearest")
                ax.set_title(title)
                ax.axis("off")
            else:
                # Show pure grayscale with colorbar
                vmin, vmax = np.min(gray_base), np.max(gray_base)
                im = ax.imshow(gray_base, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
                cbar = self.canvas.fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                cbar.set_ticks([vmin, vmax])
                cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
                ax.set_title(title)
                ax.axis("off")
        else:
            # RGB composite with slimmer individual channel colorbars
            # Create a grid with a much shorter bottom row
            gs = self.canvas.fig.add_gridspec(2, 3, height_ratios=[10, 1], hspace=0.12, wspace=0.2)
            
            # Main RGB composite image (spans top row)
            ax_main = self.canvas.fig.add_subplot(gs[0, :])
            rgb_img = stack_to_rgb(stack)
            if self.segmentation_overlay:
                rgb_img = self._get_segmentation_overlay(rgb_img)
            im = ax_main.imshow(rgb_img, interpolation="nearest")
            ax_main.set_title(title)
            ax_main.axis("off")
            
            # Individual channel colorbars (bottom row)
            for i, (channel_name, color_name) in enumerate(zip(rgb_titles, ['Red', 'Green', 'Blue'])):
                ax_cbar = self.canvas.fig.add_subplot(gs[1, i])
                
                # Create a colorbar for this channel
                if i < len(raw_channels) and raw_channels[i] is not None and not np.all(raw_channels[i] == 0):
                    # This channel has data
                    raw_min, raw_max = np.min(raw_channels[i]), np.max(raw_channels[i])
                    
                    # Check for per-channel scaling
                    if self.custom_scaling_chk.isChecked() and selected_channels and i < len(selected_channels):
                        channel = selected_channels[i]
                        if channel in self.channel_scaling:
                            raw_min = self.channel_scaling[channel]['min']
                            raw_max = self.channel_scaling[channel]['max']
                    
                    if raw_max > raw_min:  # Valid data range
                        # Create a gradient for the colorbar
                        gradient = np.linspace(0, 1, 256).reshape(1, -1)
                        ax_cbar.imshow(gradient, aspect='auto', cmap='gray' if grayscale else ['Reds', 'Greens', 'Blues'][i])
                        ax_cbar.set_xticks([0, 255])
                        ax_cbar.set_xticklabels([f'{raw_min:.1f}', f'{raw_max:.1f}'])
                        ax_cbar.tick_params(axis='x', labelsize=8, pad=1)
                        ax_cbar.set_yticks([])
                        ax_cbar.set_title(f"{channel_name}", fontsize=8, pad=2)
                    else:
                        # No data
                        ax_cbar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_cbar.transAxes, fontsize=8)
                        ax_cbar.set_title(f"{channel_name}", fontsize=8, pad=2)
                else:
                    # No data for this channel
                    ax_cbar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_cbar.transAxes, fontsize=8)
                    ax_cbar.set_title(f"{channel_name}", fontsize=8, pad=2)
                
                ax_cbar.set_xlim(0, 255)
                for spine in ax_cbar.spines.values():
                    spine.set_visible(False)
        
        self.canvas.draw()


    # ---------- Annotations ----------
    def _apply_annotation(self):
        if self.current_acq_id is None:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Select an acquisition first.")
            return
        chans = self._selected_channels()
        if not chans:
            QtWidgets.QMessageBox.information(self, "No channels", "Select channel(s) to annotate.")
            return
        label = self.ann_combo.currentText()
        for ch in chans:
            self.annotations[(self.current_acq_id, ch)] = label
        QtWidgets.QMessageBox.information(self, "Annotated", f"Labeled {len(chans)} channel(s) as '{label}'.")

    def _save_annotations(self):
        if not self.annotations:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "No annotations yet.")
            return
        rows = [
            {"acquisition_id": a, "channel": c, "label": lab}
            for (a, c), lab in self.annotations.items()
        ]
        df = pd.DataFrame(rows).sort_values(["acquisition_id", "channel"])
        base = "channel_annotations.csv"
        if self.current_path:
            stem = os.path.splitext(os.path.basename(self.current_path))[0]
            base = f"{stem}_annotations.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save annotations CSV", base, "CSV (*.csv)")
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Saved", f"Annotations saved to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _load_annotations(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load annotations CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if not {"acquisition_id", "channel", "label"}.issubset(df.columns):
                raise ValueError("CSV must have columns: acquisition_id, channel, label")
            for _, row in df.iterrows():
                self.annotations[(str(row["acquisition_id"]), str(row["channel"]))] = str(row["label"])
            QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded {len(df)} annotations.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    # ---------- Comparison ----------
    def _comparison(self):
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a .mcd first.")
            return
        
        # Open the dynamic comparison dialog
        dlg = DynamicComparisonDialog(self.acquisitions, self.loader, self)
        dlg.exec_()

    # ---------- Export ----------
    def _export_ome_tiff(self):
        """Export acquisitions to OME-TIFF format."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a .mcd first.")
            return
        
        if not _HAVE_TIFFFILE:
            QtWidgets.QMessageBox.critical(
                self, "Missing dependency", 
                "tifffile library is required for OME-TIFF export.\n"
                "Install it with: pip install tifffile"
            )
            return
        
        # Open export dialog
        dlg = ExportDialog(self.acquisitions, self.current_acq_id, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        export_type = dlg.get_export_type()
        output_dir = dlg.get_output_directory()
        include_metadata = dlg.get_include_metadata()
        apply_normalization = dlg.get_apply_normalization()
        
        # Create and show progress dialog
        progress_dlg = ProgressDialog("Export to OME-TIFF", self)
        progress_dlg.show()
        
        try:
            if export_type == "single":
                success = self._export_single_acquisition(
                    output_dir, include_metadata, apply_normalization, progress_dlg
                )
            else:
                success = self._export_whole_slide(
                    output_dir, include_metadata, apply_normalization, progress_dlg
                )
            
            progress_dlg.close()
            
            if success and not progress_dlg.is_cancelled():
                QtWidgets.QMessageBox.information(
                    self, "Export Complete", 
                    f"Successfully exported to:\n{output_dir}"
                )
            elif progress_dlg.is_cancelled():
                QtWidgets.QMessageBox.information(
                    self, "Export Cancelled", 
                    "Export was cancelled by user."
                )
        except Exception as e:
            progress_dlg.close()
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", 
                f"Export failed with error:\n{str(e)}"
            )
    
    def _export_single_acquisition(self, output_dir: str, include_metadata: bool, 
                                 apply_normalization: bool, progress_dlg: ProgressDialog) -> bool:
        """Export the currently selected acquisition."""
        if not self.current_acq_id:
            raise ValueError("No acquisition selected")
        
        acq_info = next(ai for ai in self.acquisitions if ai.id == self.current_acq_id)
        
        # Get all channels for this acquisition
        all_channels = self.loader.get_channels(self.current_acq_id)
        if not all_channels:
            raise ValueError("No channels found for this acquisition")
        
        progress_dlg.set_maximum(len(all_channels) + 2)  # +2 for stacking and writing
        progress_dlg.update_progress(0, f"Exporting {acq_info.name}", "Loading channels...")
        
        # Load all channel data
        channel_data = []
        channel_names = []
        
        for i, channel in enumerate(all_channels):
            if progress_dlg.is_cancelled():
                return False
                
            progress_dlg.update_progress(
                i + 1, 
                f"Exporting {acq_info.name}", 
                f"Loading channel {i+1}/{len(all_channels)}: {channel}"
            )
            
            if apply_normalization:
                img = self._load_image_with_normalization(self.current_acq_id, channel)
            else:
                img = self.loader.get_image(self.current_acq_id, channel)
            channel_data.append(img)
            channel_names.append(channel)
        
        if progress_dlg.is_cancelled():
            return False
        
        # Stack channels (C, H, W) for OME-TIFF
        progress_dlg.update_progress(
            len(all_channels) + 1, 
            f"Exporting {acq_info.name}", 
            "Stacking channels..."
        )
        stack = np.stack(channel_data, axis=0)
        
        # Create filename from acquisition name
        safe_name = self._sanitize_filename(acq_info.name)
        if acq_info.well:
            safe_well = self._sanitize_filename(acq_info.well)
            filename = f"{safe_name}_{safe_well}.ome.tiff"
        else:
            filename = f"{safe_name}.ome.tiff"
        
        output_path = os.path.join(output_dir, filename)
        
        # Prepare comprehensive metadata
        metadata = self._create_ome_metadata(
            acq_info, channel_names, include_metadata, stack.shape
        )
        
        # Write OME-TIFF
        progress_dlg.update_progress(
            len(all_channels) + 2, 
            f"Exporting {acq_info.name}", 
            f"Writing {filename}..."
        )
        
        if progress_dlg.is_cancelled():
            return False
            
        tifffile.imwrite(
            output_path,
            stack,
            imagej=True,
            metadata=metadata,
            ome=True
        )
        
        return True
    
    def _export_whole_slide(self, output_dir: str, include_metadata: bool, 
                          apply_normalization: bool, progress_dlg: ProgressDialog) -> bool:
        """Export all acquisitions from the slide."""
        total_acquisitions = len(self.acquisitions)
        progress_dlg.set_maximum(total_acquisitions)
        
        for acq_idx, acq_info in enumerate(self.acquisitions):
            if progress_dlg.is_cancelled():
                return False
            
            progress_dlg.update_progress(
                acq_idx, 
                f"Exporting acquisition {acq_idx + 1}/{total_acquisitions}", 
                f"Processing {acq_info.name}..."
            )
            
            # Get all channels for this acquisition
            all_channels = self.loader.get_channels(acq_info.id)
            if not all_channels:
                print(f"Warning: No channels found for acquisition {acq_info.name}")
                continue
            
            # Load all channel data
            channel_data = []
            channel_names = []
            
            for channel in all_channels:
                if progress_dlg.is_cancelled():
                    return False
                    
                if apply_normalization:
                    img = self._load_image_with_normalization(acq_info.id, channel)
                else:
                    img = self.loader.get_image(acq_info.id, channel)
                channel_data.append(img)
                channel_names.append(channel)
            
            if progress_dlg.is_cancelled():
                return False
            
            # Stack channels (C, H, W) for OME-TIFF
            stack = np.stack(channel_data, axis=0)
            
            # Create filename from acquisition name
            safe_name = self._sanitize_filename(acq_info.name)
            if acq_info.well:
                safe_well = self._sanitize_filename(acq_info.well)
                filename = f"{safe_name}_{safe_well}.ome.tiff"
            else:
                filename = f"{safe_name}.ome.tiff"
            
            output_path = os.path.join(output_dir, filename)
            
            # Prepare comprehensive metadata
            metadata = self._create_ome_metadata(
                acq_info, channel_names, include_metadata, stack.shape
            )
            
            # Write OME-TIFF
            progress_dlg.update_progress(
                acq_idx + 1, 
                f"Exporting acquisition {acq_idx + 1}/{total_acquisitions}", 
                f"Writing {filename}..."
            )
            
            if progress_dlg.is_cancelled():
                return False
                
            tifffile.imwrite(
                output_path,
                stack,
                imagej=True,
                metadata=metadata,
                ome=True
            )
        
        return True
    
    def _create_ome_metadata(self, acq_info: AcquisitionInfo, channel_names: List[str], 
                            include_metadata: bool, stack_shape: Tuple[int, ...]) -> Dict:
        """Create comprehensive OME-TIFF metadata."""
        metadata = {}
        
        # Basic acquisition information
        metadata['AcquisitionID'] = acq_info.id
        metadata['AcquisitionName'] = acq_info.name
        if acq_info.well:
            metadata['Well'] = acq_info.well
        
        # Image dimensions
        if len(stack_shape) >= 3:
            metadata['SizeC'] = stack_shape[0]  # Number of channels
            metadata['SizeT'] = 1  # Time points
            metadata['SizeZ'] = 1  # Z slices
            metadata['SizeY'] = stack_shape[1]  # Height
            metadata['SizeX'] = stack_shape[2]  # Width
        
        # Channel information
        metadata['ChannelNames'] = channel_names
        
        # Get detailed channel information from acquisition
        acq_id = acq_info.id
        if hasattr(self.loader, '_acq_channel_metals') and acq_id in self.loader._acq_channel_metals:
            channel_metals = self.loader._acq_channel_metals[acq_id]
            channel_labels = self.loader._acq_channel_labels[acq_id]
            
            # Create detailed channel metadata
            channel_metadata = []
            for i, (metal, label, name) in enumerate(zip(channel_metals, channel_labels, channel_names)):
                channel_info = {
                    'ID': f"Channel:{i}",
                    'Name': name,
                    'Metal': metal if metal else f"Channel_{i+1}",
                    'Label': label if label else f"Channel_{i+1}"
                }
                channel_metadata.append(channel_info)
            
            metadata['Channels'] = channel_metadata
        
        # Pixel size information (try to extract from metadata)
        pixel_size_x = None
        pixel_size_y = None
        pixel_size_unit = "µm"  # Default unit
        
        if include_metadata and acq_info.metadata:
            # Look for common pixel size keys in metadata
            for key, value in acq_info.metadata.items():
                key_lower = key.lower()
                if 'pixel' in key_lower and 'size' in key_lower:
                    if 'x' in key_lower or 'width' in key_lower:
                        try:
                            pixel_size_x = float(value)
                        except (ValueError, TypeError):
                            pass
                    elif 'y' in key_lower or 'height' in key_lower:
                        try:
                            pixel_size_y = float(value)
                        except (ValueError, TypeError):
                            pass
                elif 'resolution' in key_lower:
                    try:
                        # Sometimes resolution is given as a single value
                        pixel_size_x = pixel_size_y = float(value)
                    except (ValueError, TypeError):
                        pass
                elif 'unit' in key_lower and 'pixel' in key_lower:
                    pixel_size_unit = str(value)
                elif 'microns' in key_lower or 'micrometers' in key_lower:
                    # Sometimes pixel size is given as "microns per pixel"
                    try:
                        pixel_size_x = pixel_size_y = float(value)
                        pixel_size_unit = "µm"
                    except (ValueError, TypeError):
                        pass
            
            # If we found pixel size information, add it to metadata
            if pixel_size_x is not None:
                metadata['PhysicalSizeX'] = pixel_size_x
                metadata['PhysicalSizeXUnit'] = pixel_size_unit
            if pixel_size_y is not None:
                metadata['PhysicalSizeY'] = pixel_size_y
                metadata['PhysicalSizeYUnit'] = pixel_size_unit
            
            # Add all original metadata
            metadata.update(acq_info.metadata)
        
        # OME-TIFF specific metadata
        metadata['ImageJ'] = '1.53c'  # ImageJ version
        metadata['hyperstack'] = 'true'
        metadata['mode'] = 'grayscale'
        metadata['unit'] = pixel_size_unit
        
        # Add acquisition timestamp if available
        if include_metadata and acq_info.metadata:
            for key, value in acq_info.metadata.items():
                if 'time' in key.lower() or 'date' in key.lower():
                    metadata['AcquisitionTime'] = str(value)
                    break
        
        return metadata

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem usage."""
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        
        # Ensure filename is not empty
        if not filename:
            filename = "unnamed"
        
        return filename

    # ---------- Segmentation ----------
    def _run_segmentation(self):
        """Run cell segmentation using Cellpose."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a .mcd first.")
            return
        
        if not _HAVE_CELLPOSE:
            QtWidgets.QMessageBox.critical(
                self, "Missing dependency", 
                "Cellpose library is required for segmentation.\n"
                "Install it with: pip install cellpose"
            )
            return
        
        if not self.current_acq_id:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Select an acquisition first.")
            return
        
        # Get available channels
        channels = self.loader.get_channels(self.current_acq_id)
        if not channels:
            QtWidgets.QMessageBox.information(self, "No channels", "No channels available for segmentation.")
            return
        
        # Open segmentation dialog
        dlg = SegmentationDialog(channels, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        # Get segmentation parameters
        model = dlg.get_model()
        diameter = dlg.get_diameter()
        flow_threshold = dlg.get_flow_threshold()
        cellprob_threshold = dlg.get_cellprob_threshold()
        show_overlay = dlg.get_show_overlay()
        save_masks = dlg.get_save_masks()
        masks_directory = dlg.get_masks_directory()
        gpu_id = dlg.get_selected_gpu()
        preprocessing_config = dlg.get_preprocessing_config()
        segment_all = dlg.get_segment_all()
        
        # Validate preprocessing configuration
        if not preprocessing_config:
            QtWidgets.QMessageBox.warning(self, "No preprocessing configured", "Please configure preprocessing to select channels for segmentation.")
            return
        
        # Get channels from preprocessing config
        nuclear_channels = preprocessing_config.get('nuclear_channels', [])
        cyto_channels = preprocessing_config.get('cyto_channels', [])
        
        if not nuclear_channels:
            QtWidgets.QMessageBox.warning(self, "No nuclear channels", "Please select at least one nuclear channel in the preprocessing configuration.")
            return
        
        if model == "cyto" and not cyto_channels:
            QtWidgets.QMessageBox.warning(self, "No cytoplasm channels", "Please select at least one cytoplasm channel in the preprocessing configuration for whole-cell segmentation.")
            return
        
        try:
            if segment_all:
                # Run segmentation on all acquisitions
                self._perform_segmentation_all_acquisitions(
                    model, diameter, flow_threshold, cellprob_threshold, 
                    show_overlay, save_masks, masks_directory, gpu_id, preprocessing_config
                )
            else:
                # Run segmentation on current acquisition only
                self._perform_segmentation(
                    model, diameter, flow_threshold, cellprob_threshold, 
                    show_overlay, save_masks, masks_directory, gpu_id, preprocessing_config
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Segmentation Failed", 
                f"Segmentation failed with error:\n{str(e)}"
            )
    
    def _perform_segmentation(self, model: str, diameter: int = None, flow_threshold: float = 0.4, 
                            cellprob_threshold: float = 0.0, show_overlay: bool = True, 
                            save_masks: bool = False, masks_directory: str = None, gpu_id = None, preprocessing_config = None):
        """Perform the actual segmentation using Cellpose."""
        # Create progress dialog
        progress_dlg = ProgressDialog("Cell Segmentation", self)
        progress_dlg.show()
        
        try:
            progress_dlg.update_progress(0, "Initializing Cellpose model", "Loading model...")
            
            # Determine GPU usage
            use_gpu = False
            gpu_device = None
            
            if gpu_id == "auto":
                # Auto-detect best GPU
                if _HAVE_TORCH and torch.cuda.is_available():
                    use_gpu = True
                    gpu_device = 0  # Use first CUDA device
                elif _HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    use_gpu = True
                    gpu_device = 'mps'
            elif gpu_id is not None:
                # Use specific GPU
                use_gpu = True
                gpu_device = gpu_id
            
            # Initialize Cellpose model
            if model == "nuclei":
                model_obj = models.Cellpose(gpu=use_gpu, model_type='nuclei')
            else:  # cyto3
                model_obj = models.Cellpose(gpu=use_gpu, model_type='cyto3')
            
            # Set device if using GPU
            if use_gpu and gpu_device is not None:
                if gpu_device == 'mps':
                    progress_dlg.update_progress(5, "Initializing Cellpose model", "Using Apple Metal Performance Shaders...")
                else:
                    progress_dlg.update_progress(5, "Initializing Cellpose model", f"Using CUDA GPU {gpu_device}...")
            else:
                progress_dlg.update_progress(5, "Initializing Cellpose model", "Using CPU...")
            
            progress_dlg.update_progress(20, "Preprocessing images", "Loading and preprocessing channels...")
            
            # Preprocess and combine channels
            nuclear_img, cyto_img = self._preprocess_channels_for_segmentation(
                preprocessing_config, progress_dlg
            )
            
            # Prepare input images
            if model == "nuclei":
                # For nuclei model, use only nuclear channel
                images = [nuclear_img]
                channels = [0, 0]  # [cytoplasm, nucleus] - both are nuclear channel
            else:  # cyto3
                # For cyto3 model, use both channels
                if cyto_img is None:
                    cyto_img = nuclear_img  # Fallback to nuclear channel
                images = [cyto_img, nuclear_img]
                channels = [0, 1]  # [cytoplasm, nucleus]
            
            progress_dlg.update_progress(60, "Running segmentation", "Processing with Cellpose...")
            
            # Run segmentation
            masks, flows, styles, diams = model_obj.eval(
                images, 
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                channels=channels
            )
            
            progress_dlg.update_progress(80, "Processing results", "Creating segmentation masks...")
            
            # Store segmentation results
            self.segmentation_masks[self.current_acq_id] = masks[0]  # First (and only) mask
            # Clear colors for this acquisition so they get regenerated
            if self.current_acq_id in self.segmentation_colors:
                del self.segmentation_colors[self.current_acq_id]
            self.segmentation_overlay = show_overlay
            
            # Save masks if requested
            if save_masks:
                self._save_segmentation_masks(masks_directory)
            
            progress_dlg.update_progress(100, "Segmentation complete", f"Found {len(np.unique(masks[0])) - 1} cells")
            
            # Update display if overlay is enabled
            if show_overlay:
                self.segmentation_overlay_chk.setChecked(True)
                self._update_display_with_segmentation()
            
            progress_dlg.close()
            
            # Get channel information from preprocessing config
            nuclear_channels = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
            cyto_channels = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
            
            channel_info = ""
            if nuclear_channels:
                channel_info += f"Nuclear: {len(nuclear_channels)} channels"
            if cyto_channels:
                if channel_info:
                    channel_info += f" + Cytoplasm: {len(cyto_channels)} channels"
                else:
                    channel_info += f"Cytoplasm: {len(cyto_channels)} channels"
            
            QtWidgets.QMessageBox.information(
                self, "Segmentation Complete", 
                f"Successfully segmented {len(np.unique(masks[0])) - 1} cells.\n"
                f"Model: {model}\n"
                f"Channels: {channel_info if channel_info else 'Not specified'}"
            )
            
        except Exception as e:
            progress_dlg.close()
            raise e
    
    def _perform_segmentation_all_acquisitions(self, model: str, diameter: int = None, 
                                             flow_threshold: float = 0.4, cellprob_threshold: float = 0.0, 
                                             show_overlay: bool = True, save_masks: bool = False, 
                                             masks_directory: str = None, gpu_id = None, preprocessing_config = None):
        """Perform efficient batch segmentation on all acquisitions."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions", "No acquisitions available for segmentation.")
            return
        
        # Create progress dialog for batch processing
        total_acquisitions = len(self.acquisitions)
        progress_dlg = ProgressDialog("Batch Cell Segmentation", self)
        progress_dlg.set_maximum(total_acquisitions)
        progress_dlg.show()
        
        try:
            # Set fixed batch size
            batch_size = 16
            progress_dlg.update_progress(0, "Initializing batch processing", f"Batch size: {batch_size} (0/{total_acquisitions} completed)")
            
            # Initialize Cellpose model once
            progress_dlg.update_progress(0, "Initializing Cellpose model", f"Loading model... (0/{total_acquisitions} completed)")
            
            # Determine GPU usage
            use_gpu = False
            gpu_device = None
            
            if gpu_id == "auto":
                if _HAVE_TORCH and torch.cuda.is_available():
                    use_gpu = True
                    gpu_device = 0
                elif _HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    use_gpu = True
                    gpu_device = 'mps'
            elif gpu_id is not None:
                use_gpu = True
                gpu_device = gpu_id
            
            # Initialize model
            if model == "nuclei":
                model_obj = models.Cellpose(gpu=use_gpu, model_type='nuclei')
            else:  # cyto3
                model_obj = models.Cellpose(gpu=use_gpu, model_type='cyto3')
            
            # Process acquisitions in batches
            successful_segmentations = 0
            
            for batch_start in range(0, total_acquisitions, batch_size):
                if progress_dlg.is_cancelled():
                    break
                
                batch_end = min(batch_start + batch_size, total_acquisitions)
                batch_acquisitions = self.acquisitions[batch_start:batch_end]
                
                progress_dlg.update_progress(
                    successful_segmentations, 
                    f"Processing batch {batch_start//batch_size + 1}", 
                    f"Loading {len(batch_acquisitions)} acquisitions... ({successful_segmentations}/{total_acquisitions} completed)"
                )
                
                try:
                    # Load and preprocess all acquisitions in this batch
                    batch_data = self._load_batch_acquisitions(
                        batch_acquisitions, preprocessing_config, progress_dlg
                    )
                    
                    if not batch_data:
                        continue
                    
                    # Run segmentation on the entire batch
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Segmenting batch {batch_start//batch_size + 1}",
                        f"Processing {len(batch_data['images'])} images... ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                    masks, flows, styles, diams = model_obj.eval(
                        batch_data['images'],
                        diameter=diameter,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        channels=batch_data['channels']
                    )
                    
                    # Store results using acquisition mapping to ensure correct order
                    acquisition_mapping = batch_data['acquisition_mapping']
                    processed_acquisitions = set()
                    
                    for i, mask in enumerate(masks):
                        if i < len(acquisition_mapping):
                            acq_id = acquisition_mapping[i]
                            
                            # Only process each acquisition once (use the first mask for each acquisition)
                            if acq_id not in processed_acquisitions:
                                self.segmentation_masks[acq_id] = mask
                                # Clear colors for this acquisition so they get regenerated
                                if acq_id in self.segmentation_colors:
                                    del self.segmentation_colors[acq_id]
                                successful_segmentations += 1
                                processed_acquisitions.add(acq_id)
                                
                                # Save masks if requested
                                if save_masks:
                                    self._save_segmentation_masks_for_acquisition(mask, acq_id, masks_directory)
                                
                                # Update progress after each acquisition
                                acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
                                progress_dlg.update_progress(
                                    successful_segmentations,
                                    f"Completed batch {batch_start//batch_size + 1}",
                                    f"Segmented {acq_info.name} ({successful_segmentations}/{total_acquisitions} completed)"
                                )
                    
                    # Clear batch data to free memory
                    del batch_data
                    if _HAVE_TORCH and use_gpu:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing batch {batch_start//batch_size + 1}: {e}")
                    # Fall back to individual processing for this batch
                    for acq in batch_acquisitions:
                        try:
                            self._process_single_acquisition_fallback(
                                acq, model_obj, model, diameter, flow_threshold, 
                                cellprob_threshold, preprocessing_config, save_masks, masks_directory
                            )
                            successful_segmentations += 1
                            
                            # Update progress after each fallback acquisition
                            progress_dlg.update_progress(
                                successful_segmentations,
                                f"Fallback processing",
                                f"Segmented {acq.name} ({successful_segmentations}/{total_acquisitions} completed)"
                            )
                        except Exception as e2:
                            print(f"Error segmenting acquisition {acq.name}: {e2}")
                            continue
            
            progress_dlg.update_progress(total_acquisitions, "Batch segmentation complete", 
                                       f"Successfully segmented {successful_segmentations}/{total_acquisitions} acquisitions")
            
            # Show completion message
            QtWidgets.QMessageBox.information(
                self, "Batch Segmentation Complete",
                f"Successfully segmented {successful_segmentations} out of {total_acquisitions} acquisitions.\n"
                f"Segmentation masks are available for overlay display."
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Batch Segmentation Failed", 
                f"Batch segmentation failed with error:\n{str(e)}"
            )
        finally:
            progress_dlg.close()
    
    
    def _load_batch_acquisitions(self, acquisitions, preprocessing_config: dict, progress_dlg) -> dict:
        """Load and preprocess a batch of acquisitions efficiently."""
        batch_images = []
        batch_channels = []
        acquisition_mapping = []  # Track which images belong to which acquisition
        
        for acq in acquisitions:
            try:
                # Temporarily set current acquisition for preprocessing
                original_acq_id = self.current_acq_id
                self.current_acq_id = acq.id
                
                # Preprocess channels for this acquisition
                nuclear_img, cyto_img = self._preprocess_channels_for_segmentation(
                    preprocessing_config, progress_dlg
                )
                
                # Prepare input images based on model type
                if nuclear_img is not None:
                    if cyto_img is not None:
                        # Both nuclear and cytoplasm available
                        batch_images.extend([cyto_img, nuclear_img])
                        batch_channels.extend([0, 1])  # cyto, nuclear
                        acquisition_mapping.extend([acq.id, acq.id])  # Both images belong to same acquisition
                    else:
                        # Only nuclear available
                        batch_images.extend([nuclear_img, nuclear_img])
                        batch_channels.extend([0, 0])  # nuclear, nuclear
                        acquisition_mapping.extend([acq.id, acq.id])  # Both images belong to same acquisition
                else:
                    print(f"Warning: No valid images for acquisition {acq.name}")
                    continue
                
                # Restore original acquisition
                self.current_acq_id = original_acq_id
                
            except Exception as e:
                print(f"Error preprocessing acquisition {acq.name}: {e}")
                continue
        
        if not batch_images:
            return None
        
        return {
            'images': batch_images,
            'channels': batch_channels,
            'acquisition_mapping': acquisition_mapping,
            'acquisition_count': len(acquisitions)
        }
    
    def _process_single_acquisition_fallback(self, acq, model_obj, model: str, diameter: int,
                                           flow_threshold: float, cellprob_threshold: float,
                                           preprocessing_config: dict, save_masks: bool, masks_directory: str = None):
        """Fallback method to process a single acquisition individually."""
        # Temporarily set current acquisition
        original_acq_id = self.current_acq_id
        self.current_acq_id = acq.id
        
        try:
            # Preprocess channels
            nuclear_img, cyto_img = self._preprocess_channels_for_segmentation(
                preprocessing_config, None
            )
            
            # Prepare input images
            if model == "nuclei":
                images = [nuclear_img]
                channels = [0, 0]
            else:  # cyto3
                if cyto_img is None:
                    cyto_img = nuclear_img
                images = [cyto_img, nuclear_img]
                channels = [0, 1]
            
            # Run segmentation
            masks, flows, styles, diams = model_obj.eval(
                images,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                channels=channels
            )
            
            # Store results
            self.segmentation_masks[acq.id] = masks[0]
            # Clear colors for this acquisition so they get regenerated
            if acq.id in self.segmentation_colors:
                del self.segmentation_colors[acq.id]
            
            # Save masks if requested
            if save_masks:
                self._save_segmentation_masks_for_acquisition(masks[0], acq.id, masks_directory)
                
        finally:
            # Restore original acquisition
            self.current_acq_id = original_acq_id
    
    def _save_segmentation_masks_for_acquisition(self, masks: np.ndarray, acq_id: str, masks_directory: str = None):
        """Save segmentation masks for a specific acquisition."""
        acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
        filename = f"{acq_info.name}_segmentation_masks.tif"
        
        # Use provided directory or fallback to .mcd directory
        if masks_directory and os.path.exists(masks_directory):
            filepath = os.path.join(masks_directory, filename)
        else:
            filepath = os.path.join(os.path.dirname(self.mcd_path), filename)
        
        try:
            tifffile.imwrite(filepath, masks.astype(np.uint16))
            print(f"Segmentation masks saved: {filepath}")
        except Exception as e:
            print(f"Error saving segmentation masks: {e}")
    
    def _save_segmentation_masks(self, masks_directory: str = None):
        """Save all segmentation masks to files."""
        if not self.segmentation_masks:
            QtWidgets.QMessageBox.information(
                self, "No Masks", 
                "No segmentation masks available to save."
            )
            return
        
        # Use provided directory or ask user to select
        if masks_directory and os.path.exists(masks_directory):
            output_dir = masks_directory
        else:
            output_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Segmentation Masks", ""
            )
            if not output_dir:
                return
        
        # Save all masks
        saved_count = 0
        for acq_id, mask in self.segmentation_masks.items():
            try:
                acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
                safe_name = self._sanitize_filename(acq_info.name)
                if acq_info.well:
                    safe_well = self._sanitize_filename(acq_info.well)
                    filename = f"{safe_name}_{safe_well}_segmentation.tiff"
                else:
                    filename = f"{safe_name}_segmentation.tiff"
                
                output_path = os.path.join(output_dir, filename)
                
                # Save mask as TIFF
                if _HAVE_TIFFFILE:
                    tifffile.imwrite(output_path, mask.astype(np.uint16))
                else:
                    # Fallback to numpy save
                    np.save(output_path.replace('.tiff', '.npy'), mask)
                
                saved_count += 1
            except Exception as e:
                print(f"Error saving mask for {acq_id}: {e}")
                continue
        
        QtWidgets.QMessageBox.information(
            self, "Masks Saved", 
            f"Successfully saved {saved_count} segmentation mask(s) to:\n{output_dir}"
        )
    
    def _update_display_with_segmentation(self):
        """Update the current display to show segmentation overlay."""
        if not self.segmentation_overlay or self.current_acq_id not in self.segmentation_masks:
            return
        
        # Refresh the current view
        self._view_selected()
    
    def _get_segmentation_overlay(self, img: np.ndarray) -> np.ndarray:
        """Create segmentation overlay for display."""
        if not self.segmentation_overlay or self.current_acq_id not in self.segmentation_masks:
            return img
        
        mask = self.segmentation_masks[self.current_acq_id]
        
        # Create colored overlay
        overlay = np.zeros((*img.shape[:2], 3), dtype=np.float32)
        
        # Get or generate colors for this acquisition
        unique_labels = np.unique(mask)
        if self.current_acq_id not in self.segmentation_colors:
            # Generate and store colors for this acquisition
            self.segmentation_colors[self.current_acq_id] = np.random.rand(len(unique_labels), 3)
        
        colors = self.segmentation_colors[self.current_acq_id]
        
        for i, label in enumerate(unique_labels):
            if label == 0:  # Background
                continue
            cell_mask = (mask == label)
            overlay[cell_mask, :] = colors[i]
        
        # Blend with original image
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img
        
        # Normalize images to [0, 1]
        img_norm = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
        overlay_norm = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
        
        # Blend (50% original, 50% overlay)
        blended = 0.7 * img_norm + 0.3 * overlay_norm
        
        return blended

    def _get_gpu_info(self):
        """Get GPU information for display."""
        if not _HAVE_TORCH:
            return None
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                return f"CUDA ({gpu_count} GPU{'s' if gpu_count > 1 else ''}): {', '.join(gpu_names)}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "Apple Metal Performance Shaders (MPS)"
            else:
                return "CPU only"
        except Exception:
            return "GPU detection failed"
    
    def _preprocess_channels_for_segmentation(self, preprocessing_config: dict, progress_dlg) -> tuple:
        """Preprocess and combine channels for segmentation."""
        if not preprocessing_config:
            raise ValueError("Preprocessing configuration is required for segmentation")
        
        config = preprocessing_config
        
        # Get nuclear channels
        nuclear_channels = config.get('nuclear_channels', [])
        if not nuclear_channels:
            raise ValueError("No nuclear channels specified in preprocessing configuration")
        
        # Get cytoplasm channels
        cyto_channels = config.get('cyto_channels', [])
        
        # Load and normalize nuclear channels
        progress_dlg.update_progress(25, "Preprocessing images", "Loading nuclear channels...")
        nuclear_imgs = []
        for channel in nuclear_channels:
            img = self.loader.get_image(self.current_acq_id, channel)
            # Apply normalization if configured
            img = self._apply_normalization(img, config, self.current_acq_id, channel)
            nuclear_imgs.append(img)
        
        # Combine nuclear channels
        nuclear_combo_method = config.get('nuclear_combo_method', 'single')
        nuclear_weights = config.get('nuclear_weights')
        nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights)
        
        # Load and normalize cytoplasm channels
        cyto_img = None
        if cyto_channels:
            progress_dlg.update_progress(35, "Preprocessing images", "Loading cytoplasm channels...")
            cyto_imgs = []
            for channel in cyto_channels:
                img = self.loader.get_image(self.current_acq_id, channel)
                # Apply normalization if configured
                img = self._apply_normalization(img, config, self.current_acq_id, channel)
                cyto_imgs.append(img)
            
            # Combine cytoplasm channels
            cyto_combo_method = config.get('cyto_combo_method', 'single')
            cyto_weights = config.get('cyto_weights')
            cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights)
        
        return nuclear_img, cyto_img
    
    def _apply_normalization(self, img: np.ndarray, config: dict, acq_id: str, channel: str) -> np.ndarray:
        """Apply normalization to an image based on configuration."""
        norm_method = config.get('normalization_method', 'None')
        
        if norm_method == 'None':
            return img
        
        # Check cache first
        cache_key = f"{acq_id}_{channel}_{norm_method}"
        if norm_method == 'arcsinh':
            cofactor = config.get('arcsinh_cofactor', 5.0)
            cache_key += f"_{cofactor}"
        elif norm_method == 'percentile_clip':
            p_low, p_high = config.get('percentile_params', (1.0, 99.0))
            cache_key += f"_{p_low}_{p_high}"
        
        # Apply normalization
        if norm_method == 'arcsinh':
            cofactor = config.get('arcsinh_cofactor', 5.0)
            return arcsinh_normalize(img, cofactor)
        elif norm_method == 'percentile_clip':
            p_low, p_high = config.get('percentile_params', (1.0, 99.0))
            return percentile_clip_normalize(img, p_low, p_high)
        
        return img
    
    def _on_segmentation_overlay_toggled(self):
        """Handle segmentation overlay checkbox toggle."""
        self.segmentation_overlay = self.segmentation_overlay_chk.isChecked()
        
        # Update display if we have segmentation masks
        if self.current_acq_id in self.segmentation_masks:
            self._view_selected()
            
            # Update checkbox text to show cell count
            if self.segmentation_overlay:
                cell_count = len(np.unique(self.segmentation_masks[self.current_acq_id])) - 1
                self.segmentation_overlay_chk.setText(f"Show segmentation overlay ({cell_count} cells)")
            else:
                self.segmentation_overlay_chk.setText("Show segmentation overlay")
    
    def _extract_features(self):
        """Open feature extraction dialog and perform feature extraction."""
        if not self.segmentation_masks:
            QtWidgets.QMessageBox.warning(
                self, 
                "No segmentation masks", 
                "No segmentation masks found. Please run segmentation first."
            )
            return
        
        # Open feature extraction dialog
        dlg = FeatureExtractionDialog(self, self.acquisitions, self.segmentation_masks)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        # Get extraction parameters
        selected_acquisitions = dlg.get_selected_acquisitions()
        selected_features = dlg.get_selected_features()
        output_path = dlg.get_output_path()
        
        if not selected_acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions selected", "Please select at least one acquisition.")
            return
        
        if not any(selected_features.values()):
            QtWidgets.QMessageBox.warning(self, "No features selected", "Please select at least one feature to extract.")
            return
        
        # Perform feature extraction
        try:
            self._perform_feature_extraction(selected_acquisitions, selected_features, output_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Feature Extraction Failed", 
                f"Feature extraction failed with error:\n{str(e)}"
            )
    
    def _extract_features_worker(self, args):
        """Worker function for multiprocessing feature extraction."""
        acq_id, mask, selected_features, acq_info, loader, arcsinh_enabled, cofactor = args
        
        try:
            # Get pixel size from metadata
            pixel_size_um = self._get_pixel_size_um(acq_id, acq_info)
            
            # Get all available channels
            channels = acq_info.channels  # channels are already strings, not objects
            
            # Initialize feature dictionary
            features = {
                'acquisition_id': acq_id,
                'acquisition_name': acq_info.name,
                'well': acq_info.well if acq_info.well else ''
            }
            
            # Get unique cell IDs (excluding background = 0)
            unique_cells = np.unique(mask)
            unique_cells = unique_cells[unique_cells > 0]  # Remove background
            
            if len(unique_cells) == 0:
                return None
            
            # Extract morphology features
            if any(selected_features[key] for key in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 
                                                     'eccentricity', 'solidity', 'extent', 'circularity',
                                                     'major_axis_len_um', 'minor_axis_len_um', 'aspect_ratio',
                                                     'bbox_area_um2', 'touches_border', 'holes_count']):
                morph_features = self._extract_morphology_features(mask, unique_cells, pixel_size_um, selected_features)
                features.update(morph_features)
            
            # Extract intensity features for each channel
            if any(selected_features[key] for key in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated', 'frac_pos']):
                for channel in channels:
                    try:
                        # Load channel image
                        channel_img = loader.get_image(acq_id, channel)
                        
                        # Apply arcsinh normalization if enabled
                        if arcsinh_enabled:
                            channel_img = arcsinh_normalize(channel_img, cofactor=cofactor)
                        
                        intensity_features = self._extract_intensity_features(
                            channel_img, mask, unique_cells, channel, selected_features
                        )
                        features.update(intensity_features)
                    except Exception as e:
                        print(f"Warning: Could not load channel {channel}: {e}")
                        continue
            
            # Create DataFrame
            features_df = pd.DataFrame(features)
            
            return features_df
            
        except Exception as e:
            print(f"Error extracting features for acquisition {acq_id}: {e}")
            return None

    def _perform_feature_extraction(self, selected_acquisitions, selected_features, output_path):
        """Perform the actual feature extraction using multiprocessing."""
        # Create progress dialog
        progress_dlg = ProgressDialog("Feature Extraction", self)
        progress_dlg.set_maximum(len(selected_acquisitions))
        progress_dlg.show()
        
        try:
            # Prepare arguments for multiprocessing
            mp_args = []
            for acq_id in selected_acquisitions:
                try:
                    current_acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
                    mask = self.segmentation_masks[acq_id]
                    cofactor = self.cofactor_spinbox.value() if hasattr(self, 'cofactor_spinbox') else 5.0
                    
                    mp_args.append((
                        acq_id, 
                        mask, 
                        selected_features, 
                        current_acq_info, 
                        self.loader, 
                        self.arcsinh_enabled, 
                        cofactor
                    ))
                except StopIteration:
                    continue
            
            if not mp_args:
                QtWidgets.QMessageBox.warning(self, "No valid acquisitions", "No valid acquisitions found for feature extraction.")
                return
            
            # Use multiprocessing with maximum CPU count
            max_workers = mp.cpu_count()
            progress_dlg.update_progress(0, "Starting feature extraction", f"Using {max_workers} CPU cores")
            
            all_features = []
            try:
                from openmcd.processing.feature_worker import extract_features_for_acquisition
                # Prepare arguments without non-picklable objects
                safe_args = []
                for (acq_id, mask, selected_features, current_acq_info, _loader, arcsinh_enabled, cofactor) in mp_args:
                    # Build user-facing label matching UI (name + optional well)
                    acq_label = None
                    try:
                        acq_obj = next(a for a in self.acquisitions if a.id == acq_id)
                        acq_label = acq_obj.name + (f" ({acq_obj.well})" if acq_obj.well else "")
                    except StopIteration:
                        acq_label = acq_id
                    safe_args.append((
                        acq_id,
                        mask,
                        selected_features,
                        {"channels": current_acq_info.channels if hasattr(current_acq_info, 'channels') else current_acq_info.get('channels', [])},
                        acq_label,
                        self.current_path or "",
                        self.channel_normalization is not None and len(self.channel_normalization) > 0,  # arcsinh_enabled (approx)
                        self.cofactor_spinbox.value() if hasattr(self, 'cofactor_spinbox') else 5.0,
                    ))

                # Use threads instead of multiprocessing to avoid hangs in some environments
                progress_dlg.update_progress(0, "Starting feature extraction", f"Using up to {max_workers} threads")
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {executor.submit(extract_features_for_acquisition, *args): idx for idx, args in enumerate(safe_args)}
                    completed = 0
                    total = len(safe_args)
                    for future in as_completed(future_to_idx):
                        if progress_dlg.is_cancelled():
                            break
                        try:
                            result = future.result()
                        except Exception as e:
                            print(f"Feature extraction task failed: {e}")
                            result = None
                        if result is not None and not result.empty:
                            all_features.append(result)
                        completed += 1
                        progress_dlg.update_progress(
                            completed,
                            f"Processed acquisition {completed}/{total}",
                            f"Extracted features from {len(all_features)} acquisitions"
                        )
            except Exception as thread_error:
                print(f"Threaded execution failed, falling back to single-threaded processing: {thread_error}")
                progress_dlg.update_progress(0, "Threading failed, using single-threaded processing", "Processing acquisitions sequentially")
                # Fallback to single-threaded processing
                for i, args in enumerate(mp_args):
                    if progress_dlg.is_cancelled():
                        break
                    result = self._extract_features_worker(args)
                    if result is not None and not result.empty:
                        all_features.append(result)
                    progress_dlg.update_progress(
                        i + 1,
                        f"Processed acquisition {i+1}/{len(mp_args)}",
                        f"Extracted features from {len(all_features)} acquisitions"
                    )
            
            if not all_features:
                QtWidgets.QMessageBox.warning(self, "No features extracted", "No features could be extracted from the selected acquisitions.")
                return
            
            # Combine all features
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # Store in memory
            self.feature_dataframe = combined_features
            
            # Save to CSV
            if output_path:
                combined_features.to_csv(output_path, index=False)
                progress_dlg.update_progress(
                    len(selected_acquisitions), 
                    "Feature extraction complete", 
                    f"Features saved to: {output_path}\nTotal cells: {len(combined_features)}"
                )
            else:
                progress_dlg.update_progress(
                    len(selected_acquisitions), 
                    "Feature extraction complete", 
                    f"Features stored in memory\nTotal cells: {len(combined_features)}"
                )
            
            # Show completion message
            QtWidgets.QMessageBox.information(
                self, 
                "Feature Extraction Complete",
                f"Successfully extracted features from {len(selected_acquisitions)} acquisitions.\n"
                f"Total cells: {len(combined_features)}\n"
                f"Features saved to: {output_path if output_path else 'memory only'}"
            )
            
        except Exception as e:
            progress_dlg.close()
            raise e
        finally:
            progress_dlg.close()
    
    def _extract_features_for_acquisition(self, acq_id, mask, selected_features, acq_info):
        """Extract features for a single acquisition."""
        try:
            # Get pixel size from metadata
            pixel_size_um = self._get_pixel_size_um(acq_id, acq_info)
            
            # Get all available channels
            channels = acq_info.channels  # channels are already strings, not objects
            
            # Initialize feature dictionary
            features = {
                'acquisition_id': acq_id,
                'acquisition_name': acq_info.name,
                'well': acq_info.well if acq_info.well else ''
            }
            
            # Get unique cell IDs (excluding background = 0)
            unique_cells = np.unique(mask)
            unique_cells = unique_cells[unique_cells > 0]  # Remove background
            
            if len(unique_cells) == 0:
                return None
            
            # Extract morphology features
            if any(selected_features[key] for key in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 
                                                     'eccentricity', 'solidity', 'extent', 'circularity',
                                                     'major_axis_len_um', 'minor_axis_len_um', 'aspect_ratio',
                                                     'bbox_area_um2', 'touches_border', 'holes_count']):
                morph_features = self._extract_morphology_features(mask, unique_cells, pixel_size_um, selected_features)
                features.update(morph_features)
            
            # Extract intensity features for each channel
            if any(selected_features[key] for key in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated', 'frac_pos']):
                for channel in channels:
                    try:
                        # Load channel image
                        channel_img = self._load_image_with_normalization(acq_id, channel)
                        intensity_features = self._extract_intensity_features(
                            channel_img, mask, unique_cells, channel, selected_features
                        )
                        features.update(intensity_features)
                    except Exception as e:
                        print(f"Warning: Could not load channel {channel}: {e}")
                        continue
            
            # Create DataFrame
            features_df = pd.DataFrame(features)
            
            return features_df
            
        except Exception as e:
            print(f"Error extracting features for acquisition {acq_id}: {e}")
            return None
    
    def _get_pixel_size_um(self, acq_id, acq_info=None):
        """Get pixel size in micrometers from acquisition metadata."""
        try:
            # Use provided acq_info or look it up
            if acq_info is None:
                acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
            
            # Try to get pixel size from metadata
            if hasattr(acq_info, 'metadata') and acq_info.metadata:
                # Look for common pixel size keys
                for key in ['pixel_size_x', 'pixel_size', 'PhysicalSizeX']:
                    if key in acq_info.metadata:
                        return float(acq_info.metadata[key])
            
            # Default pixel size (1 μm) if not found
            return 1.0
        except Exception as e:
            return 1.0
    
    def _extract_morphology_features(self, mask, unique_cells, pixel_size_um, selected_features):
        """Extract morphology features from segmentation mask."""
        features = {}
        
        # Get region properties - mask is already labeled, no need for label() function
        props = regionprops(mask)
        
        # Initialize feature arrays including cell_id
        features['cell_id'] = []
        for key in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity', 
                   'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um', 
                   'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count']:
            if selected_features[key]:
                features[key] = []
        
        for prop in props:
            cell_id = prop.label
            
            # Add cell_id to the features dictionary
            features['cell_id'].append(cell_id)
            
            if selected_features['area_um2']:
                features['area_um2'].append(prop.area * (pixel_size_um ** 2))
            
            if selected_features['perimeter_um']:
                features['perimeter_um'].append(prop.perimeter * pixel_size_um)
            
            if selected_features['equivalent_diameter_um']:
                features['equivalent_diameter_um'].append(prop.equivalent_diameter * pixel_size_um)
            
            if selected_features['eccentricity']:
                features['eccentricity'].append(prop.eccentricity)
            
            if selected_features['solidity']:
                features['solidity'].append(prop.solidity)
            
            if selected_features['extent']:
                features['extent'].append(prop.extent)
            
            if selected_features['circularity']:
                perimeter = prop.perimeter
                area = prop.area
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                features['circularity'].append(circularity)
            
            if selected_features['major_axis_len_um']:
                features['major_axis_len_um'].append(prop.major_axis_length * pixel_size_um)
            
            if selected_features['minor_axis_len_um']:
                features['minor_axis_len_um'].append(prop.minor_axis_length * pixel_size_um)
            
            if selected_features['aspect_ratio']:
                aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 0
                features['aspect_ratio'].append(aspect_ratio)
            
            if selected_features['bbox_area_um2']:
                bbox_area = (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])
                features['bbox_area_um2'].append(bbox_area * (pixel_size_um ** 2))
            
            if selected_features['touches_border']:
                # Check if cell touches image border
                touches = (prop.bbox[0] == 0 or prop.bbox[1] == 0 or 
                          prop.bbox[2] == mask.shape[0] or prop.bbox[3] == mask.shape[1])
                features['touches_border'].append(touches)
            
            if selected_features['holes_count']:
                # Count holes in the cell (simplified - count of background pixels in convex hull)
                # This is a simplified implementation
                features['holes_count'].append(0)  # Placeholder - would need more complex analysis
        
        return features
    
    def _extract_intensity_features(self, channel_img, mask, unique_cells, channel_name, selected_features):
        """Extract intensity features for a specific channel."""
        features = {}
        
        # Initialize feature arrays
        for key in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated', 'frac_pos']:
            if selected_features[key]:
                features[f"{key}_{channel_name}"] = []
        
        for cell_id in unique_cells:
            # Get mask for this cell
            cell_mask = (mask == cell_id)
            cell_pixels = channel_img[cell_mask]
            
            if len(cell_pixels) == 0:
                # Fill with NaN if no pixels
                for key in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated', 'frac_pos']:
                    if selected_features[key]:
                        features[f"{key}_{channel_name}"].append(np.nan)
                continue
            
            if selected_features['mean']:
                features[f"mean_{channel_name}"].append(np.mean(cell_pixels))
            
            if selected_features['median']:
                features[f"median_{channel_name}"].append(np.median(cell_pixels))
            
            if selected_features['std']:
                features[f"std_{channel_name}"].append(np.std(cell_pixels))
            
            if selected_features['mad']:
                features[f"mad_{channel_name}"].append(stats.median_abs_deviation(cell_pixels))
            
            if selected_features['p10']:
                features[f"p10_{channel_name}"].append(np.percentile(cell_pixels, 10))
            
            if selected_features['p90']:
                features[f"p90_{channel_name}"].append(np.percentile(cell_pixels, 90))
            
            if selected_features['integrated']:
                mean_intensity = np.mean(cell_pixels)
                area = np.sum(cell_mask)
                features[f"integrated_{channel_name}"].append(mean_intensity * area)
            
            if selected_features['frac_pos']:
                # Use 95th percentile of ROI as threshold
                threshold = np.percentile(channel_img, 95)
                frac_pos = np.sum(cell_pixels > threshold) / len(cell_pixels)
                features[f"frac_pos_{channel_name}"].append(frac_pos)
        
        return features
    
    def _load_segmentation_masks(self):
        """Load previously saved segmentation masks from a directory."""
        if not self.current_acq_id:
            QtWidgets.QMessageBox.warning(self, "No acquisition selected", "Please select an acquisition first.")
            return
        
        # Ask user to select directory containing masks
        masks_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, 
            "Select Directory Containing Segmentation Masks",
            "",  # Start from current directory
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if not masks_dir:
            return
        
        # Look for mask files for the current acquisition
        acq_info = next(ai for ai in self.acquisitions if ai.id == self.current_acq_id)
        safe_name = self._sanitize_filename(acq_info.name)
        
        # Try different possible filenames
        possible_filenames = []
        if acq_info.well:
            safe_well = self._sanitize_filename(acq_info.well)
            possible_filenames.append(f"{safe_name}_{safe_well}_segmentation.tiff")
            possible_filenames.append(f"{safe_name}_{safe_well}_segmentation_masks.tif")
        possible_filenames.append(f"{safe_name}_segmentation.tiff")
        possible_filenames.append(f"{safe_name}_segmentation_masks.tif")
        
        # Find the first existing mask file
        mask_file = None
        for filename in possible_filenames:
            filepath = os.path.join(masks_dir, filename)
            if os.path.exists(filepath):
                mask_file = filepath
                break
        
        if not mask_file:
            QtWidgets.QMessageBox.warning(
                self, 
                "No mask file found", 
                f"No segmentation mask file found for acquisition '{acq_info.name}' in the selected directory.\n\n"
                f"Looking for files matching:\n" + "\n".join(f"• {f}" for f in possible_filenames)
            )
            return
        
        try:
            # Load the mask file
            if _HAVE_TIFFFILE:
                mask = tifffile.imread(mask_file)
            else:
                # Fallback to PIL if tifffile not available
                from PIL import Image
                mask = np.array(Image.open(mask_file))
            
            # Store the loaded mask
            self.segmentation_masks[self.current_acq_id] = mask
            # Clear colors for this acquisition so they get regenerated
            if self.current_acq_id in self.segmentation_colors:
                del self.segmentation_colors[self.current_acq_id]
            self.segmentation_overlay = True
            self.segmentation_overlay_chk.setChecked(True)
            
            # Update display
            self._view_selected()
            
            # Show success message
            cell_count = len(np.unique(mask)) - 1  # Subtract 1 for background
            QtWidgets.QMessageBox.information(
                self, 
                "Masks loaded successfully", 
                f"Loaded segmentation masks from:\n{mask_file}\n\n"
                f"Found {cell_count} cells. Overlay is now enabled."
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Error loading masks", 
                f"Failed to load segmentation masks:\n{str(e)}"
            )

    def closeEvent(self, event):
        """Clean up when closing the application."""
        if self.loader:
            self.loader.close()
        event.accept()

    def _open_clustering_dialog(self):
        """Open the cell clustering analysis dialog."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(
                self, 
                "No Feature Data", 
                "No feature data available. Please extract features first using the 'Extract Features' button."
            )
            return
        
        # Open clustering dialog
        dlg = CellClusteringDialog(self.feature_dataframe, self)
        dlg.exec_()





