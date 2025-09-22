from typing import List

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# Data types
from openmcd.data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
import os
from openmcd.ui.utils import combine_channels
from openmcd.ui.dialogs.preprocessing_dialog import PreprocessingDialog

# Optional GPU runtime
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False



class SegmentationDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Segmentation")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        self.channels = channels
        self.segmentation_result = None
        self.preprocessing_config = None
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Model selection
        model_group = QtWidgets.QGroupBox("Segmentation Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["cyto3", "nuclei"])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        
        # Model description
        self.model_desc = QtWidgets.QLabel("Cytoplasm: Segments whole cells using cytoplasm + nuclear channels")
        self.model_desc.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        model_layout.addWidget(self.model_desc)
        
        layout.addWidget(model_group)
        
        # Preprocessing button
        preprocess_group = QtWidgets.QGroupBox("Image Preprocessing")
        preprocess_layout = QtWidgets.QVBoxLayout(preprocess_group)
        
        preprocess_btn = QtWidgets.QPushButton("Configure Preprocessing...")
        preprocess_btn.clicked.connect(self._open_preprocessing_dialog)
        preprocess_layout.addWidget(preprocess_btn)
        
        self.preprocess_info_label = QtWidgets.QLabel("No preprocessing configured")
        self.preprocess_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.preprocess_info_label.setWordWrap(True)
        preprocess_layout.addWidget(self.preprocess_info_label)
        
        layout.addWidget(preprocess_group)
        
        # Parameters
        params_group = QtWidgets.QGroupBox("Segmentation Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        
        # Diameter
        diameter_layout = QtWidgets.QHBoxLayout()
        diameter_layout.addWidget(QtWidgets.QLabel("Diameter (pixels):"))
        self.diameter_spinbox = QtWidgets.QSpinBox()
        self.diameter_spinbox.setRange(1, 200)
        self.diameter_spinbox.setValue(10)
        self.diameter_spinbox.setSuffix(" px")
        diameter_layout.addWidget(self.diameter_spinbox)
        
        self.auto_diameter_chk = QtWidgets.QCheckBox("Auto-estimate")
        self.auto_diameter_chk.setChecked(True)
        self.auto_diameter_chk.toggled.connect(self._on_auto_diameter_toggled)
        diameter_layout.addWidget(self.auto_diameter_chk)
        diameter_layout.addStretch()
        
        params_layout.addLayout(diameter_layout)
        
        # Flow threshold
        flow_layout = QtWidgets.QHBoxLayout()
        flow_layout.addWidget(QtWidgets.QLabel("Flow threshold:"))
        self.flow_spinbox = QtWidgets.QDoubleSpinBox()
        self.flow_spinbox.setRange(0.0, 10.0)
        self.flow_spinbox.setDecimals(2)
        self.flow_spinbox.setValue(0.4)
        self.flow_spinbox.setSingleStep(0.1)
        flow_layout.addWidget(self.flow_spinbox)
        flow_layout.addStretch()
        
        params_layout.addLayout(flow_layout)
        
        # Cell probability threshold
        cellprob_layout = QtWidgets.QHBoxLayout()
        cellprob_layout.addWidget(QtWidgets.QLabel("Cell probability threshold:"))
        self.cellprob_spinbox = QtWidgets.QDoubleSpinBox()
        self.cellprob_spinbox.setRange(-6.0, 6.0)
        self.cellprob_spinbox.setDecimals(1)
        self.cellprob_spinbox.setValue(0.0)
        self.cellprob_spinbox.setSingleStep(0.5)
        cellprob_layout.addWidget(self.cellprob_spinbox)
        cellprob_layout.addStretch()
        
        params_layout.addLayout(cellprob_layout)
        
        layout.addWidget(params_group)
        
        # GPU selection
        gpu_group = QtWidgets.QGroupBox("GPU Acceleration")
        gpu_layout = QtWidgets.QVBoxLayout(gpu_group)
        
        gpu_row = QtWidgets.QHBoxLayout()
        gpu_row.addWidget(QtWidgets.QLabel("Device:"))
        self.gpu_combo = QtWidgets.QComboBox()
        self.gpu_combo.addItem("Auto-detect", "auto")
        self.gpu_combo.addItem("CPU only", None)
        gpu_row.addWidget(self.gpu_combo)
        gpu_row.addStretch()
        gpu_layout.addLayout(gpu_row)
        
        self.gpu_info_label = QtWidgets.QLabel("")
        self.gpu_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.gpu_info_label.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_info_label)
        
        layout.addWidget(gpu_group)
        
        # Options
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        self.show_overlay_chk = QtWidgets.QCheckBox("Show segmentation overlay")
        self.show_overlay_chk.setChecked(True)
        options_layout.addWidget(self.show_overlay_chk)
        
        self.save_masks_chk = QtWidgets.QCheckBox("Save segmentation masks")
        self.save_masks_chk.setChecked(False)
        self.save_masks_chk.toggled.connect(self._on_save_masks_toggled)
        options_layout.addWidget(self.save_masks_chk)
        
        # Directory selection for saving masks
        self.masks_dir_layout = QtWidgets.QHBoxLayout()
        self.masks_dir_label = QtWidgets.QLabel("Save directory:")
        self.masks_dir_edit = QtWidgets.QLineEdit()
        self.masks_dir_edit.setPlaceholderText("Select directory for saving masks...")
        self.masks_dir_edit.setReadOnly(True)
        self.masks_dir_btn = QtWidgets.QPushButton("Browse...")
        self.masks_dir_btn.clicked.connect(self._select_masks_directory)
        
        self.masks_dir_layout.addWidget(self.masks_dir_label)
        self.masks_dir_layout.addWidget(self.masks_dir_edit)
        self.masks_dir_layout.addWidget(self.masks_dir_btn)
        
        self.masks_dir_frame = QtWidgets.QFrame()
        self.masks_dir_frame.setLayout(self.masks_dir_layout)
        self.masks_dir_frame.setVisible(False)
        options_layout.addWidget(self.masks_dir_frame)
        
        self.segment_all_chk = QtWidgets.QCheckBox("Segment all acquisitions in .mcd file")
        self.segment_all_chk.setChecked(False)
        self.segment_all_chk.toggled.connect(self._on_segment_all_toggled)
        options_layout.addWidget(self.segment_all_chk)
        
        # Warning for batch segmentation
        self.batch_warning = QtWidgets.QLabel("⚠️ For batch segmentation, consider enabling 'Save segmentation masks' to preserve results")
        self.batch_warning.setStyleSheet("QLabel { color: #d97706; font-size: 11px; font-weight: bold; }")
        self.batch_warning.setWordWrap(True)
        self.batch_warning.setVisible(False)
        options_layout.addWidget(self.batch_warning)
        
        # Info label for segment all
        self.segment_all_info = QtWidgets.QLabel("")
        self.segment_all_info.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.segment_all_info.setWordWrap(True)
        options_layout.addWidget(self.segment_all_info)
        
        layout.addWidget(options_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.segment_btn = QtWidgets.QPushButton("Run Segmentation")
        self.segment_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.segment_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        # Initialize
        self._on_model_changed()
        self._on_auto_diameter_toggled()
        self._detect_and_populate_gpus()
        self.gpu_combo.currentTextChanged.connect(self._on_gpu_selection_changed)
        self._on_segment_all_toggled()
        
    def _on_model_changed(self):
        """Update UI when model selection changes."""
        model = self.model_combo.currentText()
        if model == "nuclei":
            self.model_desc.setText("Nuclei: Segments cell nuclei using nuclear channel")
        else:  # cyto3
            self.model_desc.setText("Cytoplasm: Segments whole cells using cytoplasm + nuclear channels")
    
    def _on_auto_diameter_toggled(self):
        """Enable/disable diameter spinbox based on auto-estimate checkbox."""
        self.diameter_spinbox.setEnabled(not self.auto_diameter_chk.isChecked())
    
    def get_model(self):
        """Get selected model."""
        return self.model_combo.currentText()
    
    
    def get_diameter(self):
        """Get diameter value (None if auto-estimate)."""
        if self.auto_diameter_chk.isChecked():
            return None
        return self.diameter_spinbox.value()
    
    def get_flow_threshold(self):
        """Get flow threshold."""
        return self.flow_spinbox.value()
    
    def get_cellprob_threshold(self):
        """Get cell probability threshold."""
        return self.cellprob_spinbox.value()
    
    def get_show_overlay(self):
        """Get whether to show overlay."""
        return self.show_overlay_chk.isChecked()
    
    def get_save_masks(self):
        """Get whether to save masks."""
        return self.save_masks_chk.isChecked()
    
    def _detect_and_populate_gpus(self):
        """Detect available GPUs and populate the combo box."""
        if not _HAVE_TORCH:
            self.gpu_info_label.setText("PyTorch not available. Using CPU only.")
            return
        
        try:
            available_gpus = []
            
            # Check CUDA
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    available_gpus.append({
                        'id': i,
                        'name': f"{gpu_name} ({gpu_memory:.1f} GB)",
                        'type': 'CUDA'
                    })
            
            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available_gpus.append({
                    'id': 'mps',
                    'name': 'Apple Metal Performance Shaders (MPS)',
                    'type': 'MPS'
                })
            
            # Add GPU options to combo
            for gpu in available_gpus:
                self.gpu_combo.addItem(gpu['name'], gpu['id'])
            
            # Update info
            if available_gpus:
                self.gpu_info_label.setText(f"Found {len(available_gpus)} GPU(s) available for acceleration.")
            else:
                self.gpu_info_label.setText("No GPUs detected. Using CPU only.")
                
        except Exception as e:
            self.gpu_info_label.setText(f"Error detecting GPUs: {str(e)}")
    
    def _on_gpu_selection_changed(self):
        """Update info when GPU selection changes."""
        gpu_id = self.gpu_combo.currentData()
        
        if gpu_id is None:
            self.gpu_info_label.setText("Using CPU for segmentation. This will be slower but more compatible.")
        elif gpu_id == "auto":
            self.gpu_info_label.setText("Will automatically select the best available GPU.")
        else:
            gpu_name = self.gpu_combo.currentText()
            self.gpu_info_label.setText(f"Selected: {gpu_name}")
    
    def get_selected_gpu(self):
        """Get the selected GPU ID."""
        return self.gpu_combo.currentData()
    
    def _open_preprocessing_dialog(self):
        """Open the preprocessing configuration dialog."""
        dlg = PreprocessingDialog(self.channels, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.preprocessing_config = {
                'normalization_method': dlg.get_normalization_method(),
                'arcsinh_cofactor': dlg.get_arcsinh_cofactor(),
                'percentile_params': dlg.get_percentile_params(),
                'nuclear_channels': dlg.get_nuclear_channels(),
                'cyto_channels': dlg.get_cyto_channels(),
                'nuclear_combo_method': dlg.get_nuclear_combo_method(),
                'cyto_combo_method': dlg.get_cyto_combo_method(),
                'nuclear_weights': dlg.get_nuclear_weights(),
                'cyto_weights': dlg.get_cyto_weights()
            }
            self._update_preprocess_info()
    
    def _update_preprocess_info(self):
        """Update the preprocessing info label."""
        if not self.preprocessing_config:
            self.preprocess_info_label.setText("No preprocessing configured")
            return
        
        config = self.preprocessing_config
        info_parts = []
        
        # Normalization info
        if config['normalization_method'] != 'None':
            if config['normalization_method'] == 'arcsinh':
                info_parts.append(f"Arcsinh (cofactor={config['arcsinh_cofactor']})")
            elif config['normalization_method'] == 'percentile_clip':
                p_low, p_high = config['percentile_params']
                info_parts.append(f"Percentile clip ({p_low}-{p_high}%)")
        
        # Channel combination info
        if config['nuclear_channels']:
            nuclear_info = f"Nuclear: {config['nuclear_combo_method']}({len(config['nuclear_channels'])} channels)"
            info_parts.append(nuclear_info)
        
        if config['cyto_channels']:
            cyto_info = f"Cytoplasm: {config['cyto_combo_method']}({len(config['cyto_channels'])} channels)"
            info_parts.append(cyto_info)
        
        if info_parts:
            self.preprocess_info_label.setText(" | ".join(info_parts))
        else:
            self.preprocess_info_label.setText("No preprocessing configured")
    
    def get_preprocessing_config(self):
        """Get the preprocessing configuration."""
        return self.preprocessing_config
    
    def _on_segment_all_toggled(self):
        """Update UI when segment all checkbox is toggled."""
        if self.segment_all_chk.isChecked():
            # Get acquisition count from parent (MainWindow)
            parent = self.parent()
            if hasattr(parent, 'acquisitions'):
                acq_count = len(parent.acquisitions)
                self.segment_all_info.setText(f"Will segment all {acq_count} acquisitions in the .mcd file. This may take a while.")
            else:
                self.segment_all_info.setText("Will segment all acquisitions in the .mcd file. This may take a while.")
            
            # Show warning about saving masks
            self.batch_warning.setVisible(True)
        else:
            self.segment_all_info.setText("")
            self.batch_warning.setVisible(False)
    
    def get_segment_all(self):
        """Get whether to segment all acquisitions."""
        return self.segment_all_chk.isChecked()
    
    def _on_save_masks_toggled(self):
        """Update UI when save masks checkbox is toggled."""
        self.masks_dir_frame.setVisible(self.save_masks_chk.isChecked())
    
    def _select_masks_directory(self):
        """Open directory selection dialog for saving masks."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, 
            "Select Directory for Saving Segmentation Masks",
            "",  # Start from current directory
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.masks_dir_edit.setText(directory)
    
    def get_masks_directory(self):
        """Get the selected directory for saving masks."""
        if self.save_masks_chk.isChecked():
            directory = self.masks_dir_edit.text().strip()
            if directory and os.path.exists(directory):
                return directory
            else:
                # Fallback to .mcd file directory
                parent = self.parent()
                if hasattr(parent, 'mcd_path') and parent.mcd_path:
                    return os.path.dirname(parent.mcd_path)
        return None