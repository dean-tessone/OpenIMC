"""
UI dialog for Ilastik segmentation.

This dialog allows users to select an Ilastik project file and configure
segmentation parameters.
"""

from typing import List, Dict, Optional
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from openimc.processing.ilastik_inference import IlastikInferencePipeline


class IlastikSegmentationDialog(QtWidgets.QDialog):
    """Dialog for configuring and running Ilastik segmentation."""
    
    def __init__(self, img_stack: np.ndarray, channel_names: List[str], parent=None, preprocessing_config=None):
        super().__init__(parent)
        self.setWindowTitle("Ilastik Segmentation")
        self.setModal(True)
        self.resize(600, 400)
        
        self.img_stack = img_stack
        self.channel_names = channel_names
        self.preprocessing_config = preprocessing_config
        
        # Results
        self.results = None
        self.ilastik_pipeline = None
        
        # Create UI
        self._create_ui()
    
    def _create_ui(self):
        """Create the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Ilastik project file selection
        project_group = QtWidgets.QGroupBox("Ilastik Project")
        project_layout = QtWidgets.QVBoxLayout(project_group)
        
        project_file_layout = QtWidgets.QHBoxLayout()
        project_file_layout.addWidget(QtWidgets.QLabel("Project file (.ilp):"))
        self.project_file_edit = QtWidgets.QLineEdit()
        self.project_file_edit.setPlaceholderText("Select Ilastik project file...")
        self.project_file_edit.setReadOnly(True)
        project_file_layout.addWidget(self.project_file_edit, 1)
        
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._select_project_file)
        project_file_layout.addWidget(self.browse_btn)
        
        project_layout.addLayout(project_file_layout)
        
        # Info label
        self.project_info_label = QtWidgets.QLabel("No project file selected")
        self.project_info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        self.project_info_label.setWordWrap(True)
        project_layout.addWidget(self.project_info_label)
        
        layout.addWidget(project_group)
        
        # Output options
        options_group = QtWidgets.QGroupBox("Output Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        self.return_probabilities_chk = QtWidgets.QCheckBox("Return probability maps")
        self.return_probabilities_chk.setChecked(False)
        self.return_probabilities_chk.setToolTip(
            "If checked, returns probability maps for each class. "
            "Otherwise, returns segmentation labels."
        )
        options_layout.addWidget(self.return_probabilities_chk)
        
        layout.addWidget(options_group)
        
        # Progress bar (hidden initially)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run Segmentation")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; "
            "padding: 10px 20px; border-radius: 5px; } "
            "QPushButton:hover { background-color: #218838; } "
            "QPushButton:disabled { background-color: #6c757d; }"
        )
        self.run_btn.clicked.connect(self._run_segmentation)
        self.run_btn.setEnabled(False)
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; font-weight: bold; "
            "padding: 10px 20px; border-radius: 5px; } "
            "QPushButton:hover { background-color: #c82333; }"
        )
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
    
    def _select_project_file(self):
        """Open file dialog to select Ilastik project file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ilastik Project File",
            "",
            "Ilastik Project Files (*.ilp);;All Files (*)"
        )
        
        if file_path:
            self.project_file_edit.setText(file_path)
            self._validate_project_file(file_path)
    
    def _validate_project_file(self, file_path: str):
        """Validate the selected project file and update UI."""
        if not file_path:
            self.project_info_label.setText("No project file selected")
            self.run_btn.setEnabled(False)
            return
        
        if not file_path.endswith('.ilp'):
            self.project_info_label.setText(
                "<span style='color: red;'>Error: File must have .ilp extension</span>"
            )
            self.run_btn.setEnabled(False)
            return
        
        import os
        if not os.path.exists(file_path):
            self.project_info_label.setText(
                "<span style='color: red;'>Error: File does not exist</span>"
            )
            self.run_btn.setEnabled(False)
            return
        
        # Try to initialize the pipeline to validate the file
        try:
            self.ilastik_pipeline = IlastikInferencePipeline(file_path)
            info = self.ilastik_pipeline.get_model_info()
            self.project_info_label.setText(
                f"<span style='color: green;'>✓ Valid Ilastik project file</span><br>"
                f"Path: {os.path.basename(file_path)}"
            )
            self.run_btn.setEnabled(True)
        except Exception as e:
            self.project_info_label.setText(
                f"<span style='color: red;'>Error: {str(e)}</span>"
            )
            self.run_btn.setEnabled(False)
            self.ilastik_pipeline = None
    
    def _run_segmentation(self):
        """Run Ilastik segmentation."""
        if not self.ilastik_pipeline:
            QMessageBox.warning(self, "No Project", "Please select a valid Ilastik project file.")
            return
        
        # Disable UI during processing
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Running Ilastik inference... This may take a while.")
        
        # Process in a separate thread to avoid blocking UI
        # For now, we'll run it synchronously but with progress updates
        QtWidgets.QApplication.processEvents()
        
        try:
            return_probabilities = self.return_probabilities_chk.isChecked()
            
            # Run inference
            results = self.ilastik_pipeline.run_inference(
                self.img_stack,
                self.channel_names,
                return_probabilities=return_probabilities
            )
            
            self.results = results
            self.status_label.setText("✓ Segmentation completed successfully!")
            
            # Show completion message
            if 'labels' in results:
                n_cells = len(np.unique(results['labels'])) - 1  # Exclude background
                QMessageBox.information(
                    self,
                    "Segmentation Complete",
                    f"Segmentation completed successfully!\n"
                    f"Found {n_cells} cells/regions."
                )
            
            self.accept()
            
        except Exception as e:
            self.status_label.setText(f"<span style='color: red;'>Error: {str(e)}</span>")
            QMessageBox.critical(
                self,
                "Segmentation Failed",
                f"Ilastik segmentation failed:\n{str(e)}"
            )
            self.results = None
        finally:
            # Re-enable UI
            self.progress_bar.setVisible(False)
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)
    
    def get_results(self) -> Optional[Dict[str, np.ndarray]]:
        """Get segmentation results."""
        return self.results
    
    def get_project_path(self) -> Optional[str]:
        """Get the selected project file path."""
        return self.project_file_edit.text() if self.project_file_edit.text() else None

