from typing import List, Optional

from PyQt5 import QtWidgets

from openmcd.data.mcd_loader import AcquisitionInfo

# --------------------------
# Export Dialog
# --------------------------
class ExportDialog(QtWidgets.QDialog):
    def __init__(self, acquisitions: List[AcquisitionInfo], current_acq_id: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export to OME-TIFF")
        self.setModal(True)
        self.acquisitions = acquisitions
        self.current_acq_id = current_acq_id
        self.output_directory = ""
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Export type selection
        type_group = QtWidgets.QGroupBox("Export Type")
        type_layout = QtWidgets.QVBoxLayout(type_group)
        
        self.single_roi_radio = QtWidgets.QRadioButton("Single ROI (Current Acquisition)")
        self.whole_slide_radio = QtWidgets.QRadioButton("Whole Slide (All Acquisitions)")
        self.single_roi_radio.setChecked(True)
        
        type_layout.addWidget(self.single_roi_radio)
        type_layout.addWidget(self.whole_slide_radio)
        layout.addWidget(type_group)
        
        # Current acquisition info
        self.acq_info_label = QtWidgets.QLabel("")
        layout.addWidget(self.acq_info_label)
        
        # Output directory selection
        dir_group = QtWidgets.QGroupBox("Output Directory")
        dir_layout = QtWidgets.QVBoxLayout(dir_group)
        
        dir_row = QtWidgets.QHBoxLayout()
        self.dir_label = QtWidgets.QLabel("No directory selected")
        self.dir_label.setStyleSheet("QLabel { color: #666; }")
        dir_row.addWidget(self.dir_label)
        dir_row.addStretch()
        
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_directory)
        dir_row.addWidget(self.browse_btn)
        
        dir_layout.addLayout(dir_row)
        layout.addWidget(dir_group)
        
        # Options
        options_group = QtWidgets.QGroupBox("Export Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        self.include_metadata_chk = QtWidgets.QCheckBox("Include metadata in OME-TIFF")
        self.include_metadata_chk.setChecked(True)
        options_layout.addWidget(self.include_metadata_chk)
        
        self.apply_normalization_chk = QtWidgets.QCheckBox("Apply current normalization settings")
        self.apply_normalization_chk.setChecked(False)
        options_layout.addWidget(self.apply_normalization_chk)
        
        layout.addWidget(options_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.setEnabled(False)  # Disabled until directory is selected
        self.export_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        # Connect signals
        self.single_roi_radio.toggled.connect(self._on_export_type_changed)
        self.whole_slide_radio.toggled.connect(self._on_export_type_changed)
        
        # Initialize the display
        self._on_export_type_changed()
        
    def _browse_directory(self):
        """Browse for output directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        if directory:
            self.output_directory = directory
            self.dir_label.setText(directory)
            self.dir_label.setStyleSheet("QLabel { color: black; }")
            self.export_btn.setEnabled(True)
    
    def _on_export_type_changed(self):
        """Update UI when export type changes."""
        if self.single_roi_radio.isChecked():
            if self.current_acq_id:
                # Find current acquisition info
                current_acq = next((acq for acq in self.acquisitions if acq.id == self.current_acq_id), None)
                if current_acq:
                    info_text = f"Will export: {current_acq.name}\n"
                    info_text += f"Channels: {len(current_acq.channels)}\n"
                    if current_acq.well:
                        info_text += f"Well: {current_acq.well}"
                    self.acq_info_label.setText(info_text)
                else:
                    self.acq_info_label.setText("Will export only the currently selected acquisition.")
            else:
                self.acq_info_label.setText("Will export only the currently selected acquisition.")
        else:
            # Show more detailed information about what will be exported
            total_channels = sum(len(acq.channels) for acq in self.acquisitions)
            info_text = f"Will export all {len(self.acquisitions)} acquisitions from the slide.\n"
            info_text += f"Total channels: {total_channels}\n"
            info_text += f"Acquisitions: {', '.join([acq.name for acq in self.acquisitions[:3]])}"
            if len(self.acquisitions) > 3:
                info_text += f" and {len(self.acquisitions) - 3} more..."
            self.acq_info_label.setText(info_text)
    
    def get_export_type(self):
        """Get the selected export type."""
        return "single" if self.single_roi_radio.isChecked() else "whole"
    
    def get_output_directory(self):
        """Get the selected output directory."""
        return self.output_directory
    
    def get_include_metadata(self):
        """Get whether to include metadata."""
        return self.include_metadata_chk.isChecked()
    
    def get_apply_normalization(self):
        """Get whether to apply normalization."""
        return self.apply_normalization_chk.isChecked()
