"""
Quality Control Analysis Dialog for OpenIMC

This module provides QC analysis capabilities including:
- Pixel-level QC: SNR calculation using Otsu threshold
- Cell-level QC: SNR calculation using segmentation masks
- Quality metrics: SNR vs intensity, % covered area, cell density, etc.
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from openimc.utils.logger import get_logger
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.progress_dialog import ProgressDialog
import multiprocessing as mp
import traceback

# Optional scikit-image for Otsu thresholding
_HAVE_SCIKIT_IMAGE = False
try:
    from skimage.filters import threshold_otsu
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False


# Module-level worker function for multiprocessing (must be picklable)
# This function is isolated to avoid conflicts with feature extraction workers
def _qc_process_acquisition_worker(task_data):
    """Process all channels for one acquisition for QC analysis. Returns list of metrics dicts.
    
    This is a separate worker function specifically for QC analysis to avoid conflicts
    with feature extraction workers.
    """
    acq_id, acq_name, channels, analysis_mode, mask_path, loader_path, acq_to_file_map = task_data
    results = []
    
    try:
        # Recreate loader (can't pickle loader objects)
        # Import inside function to ensure isolation
        from openimc.data.mcd_loader import MCDLoader
        from openimc.data.ometiff_loader import OMETIFFLoader
        import os
        
        loader = None
        if loader_path and os.path.exists(loader_path):
            if loader_path.endswith('.mcd'):
                loader = MCDLoader()
                loader.open(loader_path)
            else:
                loader = OMETIFFLoader()
                loader.open(loader_path)
        
        if not loader:
            return results
        
        # Load mask if needed (for cell-level analysis)
        mask = None
        if analysis_mode == "cell" and mask_path and os.path.exists(mask_path):
            try:
                import tifffile
                mask = tifffile.imread(mask_path)
            except Exception:
                if hasattr(loader, 'close'):
                    loader.close()
                return results
        
        # Process each channel
        for channel in channels:
            try:
                # Load image (raw pixel intensities)
                img = loader.get_image(acq_id, channel)
                if img is None:
                    continue
                
                # Calculate metrics
                if analysis_mode == "pixel":
                    metrics = _qc_calculate_pixel_metrics_worker(img, channel)
                else:
                    if mask is not None:
                        metrics = _qc_calculate_cell_metrics_worker(img, channel, mask)
                    else:
                        continue
                
                if metrics:
                    metrics['acquisition_id'] = acq_id
                    metrics['acquisition_name'] = acq_name
                    metrics['channel'] = channel
                    results.append(metrics)
                        
            except Exception:
                continue
        
        # Close loader
        if hasattr(loader, 'close'):
            loader.close()
        
        return results
        
    except Exception:
        return results


def _qc_calculate_pixel_metrics_worker(img: np.ndarray, channel: str) -> Optional[Dict[str, Any]]:
    """Calculate pixel-level QC metrics using Otsu threshold (module-level for multiprocessing).
    
    This is a separate worker function for QC analysis to avoid conflicts.
    """
    if not _HAVE_SCIKIT_IMAGE:
        return None
    
    try:
        # Convert to float if needed
        img_float = img.astype(np.float32)
        
        # Calculate Otsu threshold
        threshold = threshold_otsu(img_float)
        
        # Separate signal (foreground) and background
        foreground = img_float[img_float > threshold]
        background = img_float[img_float <= threshold]
        
        if len(foreground) == 0 or len(background) == 0:
            return None
        
        # Calculate metrics
        signal_mean = np.mean(foreground)
        signal_std = np.std(foreground)
        background_mean = np.mean(background)
        background_std = np.std(background)
        
        # SNR: (signal_mean - background_mean) / background_std
        snr = (signal_mean - background_mean) / (background_std + 1e-8)
        
        # Intensity metrics (using raw pixel intensities)
        mean_intensity = np.mean(img_float)
        median_intensity = np.median(img_float)
        max_intensity = np.max(img_float)
        min_intensity = np.min(img_float)
        
        # Coverage: percentage of pixels above threshold
        coverage_pct = (len(foreground) / img_float.size) * 100
        
        # Calculate percentiles
        p1 = np.percentile(img_float, 1)
        p25 = np.percentile(img_float, 25)
        p75 = np.percentile(img_float, 75)
        p99 = np.percentile(img_float, 99)
        
        return {
            'snr': snr,
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'background_mean': background_mean,
            'background_std': background_std,
            'threshold': threshold,
            'mean_intensity': mean_intensity,  # Raw pixel intensity
            'median_intensity': median_intensity,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'coverage_pct': coverage_pct,
            'p1': p1,
            'p25': p25,
            'p75': p75,
            'p99': p99,
            'total_pixels': img_float.size,
            'foreground_pixels': len(foreground),
            'background_pixels': len(background)
        }
    except Exception as e:
        print(f"Error calculating pixel metrics for {channel}: {e}")
        return None


def _qc_calculate_cell_metrics_worker(img: np.ndarray, channel: str, mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """Calculate cell-level QC metrics using segmentation masks (module-level for multiprocessing).
    
    This is a separate worker function for QC analysis to avoid conflicts.
    """
    try:
        # Convert to float if needed
        img_float = img.astype(np.float32)
        
        # Ensure mask and image have same shape
        if mask.shape != img_float.shape:
            print(f"Warning: Mask shape {mask.shape} doesn't match image shape {img_float.shape}")
            return None
        
        # Separate signal (cells) and background
        cell_mask = mask > 0
        background_mask = mask == 0
        
        if np.sum(cell_mask) == 0 or np.sum(background_mask) == 0:
            return None
        
        foreground = img_float[cell_mask]
        background = img_float[background_mask]
        
        # Calculate metrics
        signal_mean = np.mean(foreground)
        signal_std = np.std(foreground)
        background_mean = np.mean(background)
        background_std = np.std(background)
        
        # SNR: (signal_mean - background_mean) / background_std
        snr = (signal_mean - background_mean) / (background_std + 1e-8)
        
        # Intensity metrics (cell-level, using raw pixel intensities)
        mean_intensity = np.mean(foreground)  # Mean intensity in cells
        median_intensity = np.median(foreground)
        max_intensity = np.max(foreground)
        min_intensity = np.min(foreground)
        
        # Coverage: percentage of pixels covered by cells
        coverage_pct = (np.sum(cell_mask) / img_float.size) * 100
        
        # Cell density: number of cells per unit area
        unique_cells = np.unique(mask[mask > 0])
        num_cells = len(unique_cells)
        area_pixels = img_float.size
        cell_density = num_cells / area_pixels if area_pixels > 0 else 0
        
        # Calculate percentiles for cell intensities
        p1 = np.percentile(foreground, 1)
        p25 = np.percentile(foreground, 25)
        p75 = np.percentile(foreground, 75)
        p99 = np.percentile(foreground, 99)
        
        # Per-cell statistics
        cell_intensities = []
        for cell_id in unique_cells:
            cell_pixels = img_float[mask == cell_id]
            if len(cell_pixels) > 0:
                cell_intensities.append(np.mean(cell_pixels))
        
        mean_cell_intensity = np.mean(cell_intensities) if cell_intensities else 0
        median_cell_intensity = np.median(cell_intensities) if cell_intensities else 0
        
        return {
            'snr': snr,
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'background_mean': background_mean,
            'background_std': background_std,
            'mean_intensity': mean_intensity,  # Raw pixel intensity
            'median_intensity': median_intensity,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'coverage_pct': coverage_pct,
            'cell_density': cell_density,
            'num_cells': num_cells,
            'p1': p1,
            'p25': p25,
            'p75': p75,
            'p99': p99,
            'mean_cell_intensity': mean_cell_intensity,
            'median_cell_intensity': median_cell_intensity,
            'total_pixels': img_float.size,
            'foreground_pixels': np.sum(cell_mask),
            'background_pixels': np.sum(background_mask)
        }
    except Exception as e:
        print(f"Error calculating cell metrics for {channel}: {e}")
        import traceback
        traceback.print_exc()
        return None


class QCAnalysisDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quality Control Analysis")
        self.setMinimumSize(1000, 700)
        
        # Get parent window to access images and masks
        self.parent_window = parent
        
        # Store results
        self.qc_results: Optional[pd.DataFrame] = None  # Raw results per ROI per channel
        self.qc_results_aggregated: Optional[pd.DataFrame] = None  # Aggregated results per channel
        self.pixel_level_results: Optional[pd.DataFrame] = None
        self.cell_level_results: Optional[pd.DataFrame] = None
        
        # Analysis mode
        self.analysis_mode = "pixel"  # "pixel" or "cell"
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Quality Control Analysis")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12pt; }")
        layout.addWidget(title_label)
        
        # Options panel
        options_group = QtWidgets.QGroupBox("Analysis Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        # Analysis mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Analysis Level:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Pixel-level", "Cell-level"])
        
        # Check if masks exist
        has_masks = self._check_masks_exist()
        if not has_masks:
            self.mode_combo.setItemText(1, "Cell-level (No masks available)")
            self.mode_combo.model().item(1).setEnabled(False)
        
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        options_layout.addLayout(mode_layout)
        
        # Acquisition selection
        acq_layout = QtWidgets.QHBoxLayout()
        acq_layout.addWidget(QtWidgets.QLabel("Acquisition:"))
        self.acq_combo = QtWidgets.QComboBox()
        self._populate_acq_combo()
        # Add "All Acquisitions" as the first (default) option
        self.acq_combo.insertItem(0, "All Acquisitions", "all")
        self.acq_combo.setCurrentIndex(0)  # Set as default
        acq_layout.addWidget(self.acq_combo, 1)
        acq_layout.addStretch()
        options_layout.addLayout(acq_layout)
        
        # Number of workers for multiprocessing
        workers_layout = QtWidgets.QHBoxLayout()
        workers_layout.addWidget(QtWidgets.QLabel("Number of workers:"))
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setMinimum(1)
        self.workers_spin.setMaximum(mp.cpu_count())
        self.workers_spin.setValue(mp.cpu_count())  # Default to max CPUs
        workers_layout.addWidget(self.workers_spin)
        workers_layout.addStretch()
        options_layout.addLayout(workers_layout)
        
        # Run button
        run_layout = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Calculate QC Metrics")
        self.run_btn.clicked.connect(self._run_analysis)
        run_layout.addWidget(self.run_btn)
        run_layout.addStretch()
        options_layout.addLayout(run_layout)
        
        layout.addWidget(options_group)
        
        # Results tabs
        self.tabs = QtWidgets.QTabWidget()
        
        # Summary tab
        summary_tab = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_tab)
        
        # Summary table
        self.summary_table = QtWidgets.QTableWidget()
        self.summary_table.setColumnCount(0)
        self.summary_table.setAlternatingRowColors(True)
        summary_layout.addWidget(self.summary_table)
        
        # Export button
        summary_btn_layout = QtWidgets.QHBoxLayout()
        self.export_summary_btn = QtWidgets.QPushButton("Export Results...")
        self.export_summary_btn.clicked.connect(self._export_results)
        self.export_summary_btn.setEnabled(False)
        summary_btn_layout.addWidget(self.export_summary_btn)
        summary_btn_layout.addStretch()
        summary_layout.addLayout(summary_btn_layout)
        
        self.tabs.addTab(summary_tab, "Summary")
        
        # SNR vs Intensity plot
        snr_intensity_tab = QtWidgets.QWidget()
        snr_intensity_layout = QtWidgets.QVBoxLayout(snr_intensity_tab)
        
        self.snr_intensity_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        snr_intensity_layout.addWidget(self.snr_intensity_canvas)
        
        snr_intensity_btn_layout = QtWidgets.QHBoxLayout()
        self.snr_intensity_save_btn = QtWidgets.QPushButton("Save Plot...")
        self.snr_intensity_save_btn.clicked.connect(self._save_snr_intensity_plot)
        self.snr_intensity_save_btn.setEnabled(False)
        snr_intensity_btn_layout.addWidget(self.snr_intensity_save_btn)
        snr_intensity_btn_layout.addStretch()
        snr_intensity_layout.addLayout(snr_intensity_btn_layout)
        
        self.tabs.addTab(snr_intensity_tab, "SNR vs Intensity")
        
        # Coverage tab
        coverage_tab = QtWidgets.QWidget()
        coverage_layout = QtWidgets.QVBoxLayout(coverage_tab)
        
        self.coverage_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        coverage_layout.addWidget(self.coverage_canvas)
        
        coverage_btn_layout = QtWidgets.QHBoxLayout()
        self.coverage_save_btn = QtWidgets.QPushButton("Save Plot...")
        self.coverage_save_btn.clicked.connect(self._save_coverage_plot)
        self.coverage_save_btn.setEnabled(False)
        coverage_btn_layout.addWidget(self.coverage_save_btn)
        coverage_btn_layout.addStretch()
        coverage_layout.addLayout(coverage_btn_layout)
        
        self.tabs.addTab(coverage_tab, "Coverage & Density")
        
        # Distribution tab (boxplots)
        distribution_tab = QtWidgets.QWidget()
        distribution_layout = QtWidgets.QVBoxLayout(distribution_tab)
        
        self.distribution_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        distribution_layout.addWidget(self.distribution_canvas)
        
        distribution_btn_layout = QtWidgets.QHBoxLayout()
        self.distribution_save_btn = QtWidgets.QPushButton("Save Plot...")
        self.distribution_save_btn.clicked.connect(self._save_distribution_plot)
        self.distribution_save_btn.setEnabled(False)
        distribution_btn_layout.addWidget(self.distribution_save_btn)
        distribution_btn_layout.addStretch()
        distribution_layout.addLayout(distribution_btn_layout)
        
        self.tabs.addTab(distribution_tab, "Distributions")
        
        layout.addWidget(self.tabs, 1)
        
        # Initialize
        self._on_mode_changed()
        
    def _check_masks_exist(self) -> bool:
        """Check if any segmentation masks exist."""
        if not self.parent_window or not hasattr(self.parent_window, 'segmentation_masks'):
            return False
        return len(self.parent_window.segmentation_masks) > 0
    
    def _populate_acq_combo(self):
        """Populate acquisition combo box."""
        self.acq_combo.clear()
        if not self.parent_window or not hasattr(self.parent_window, 'acquisitions'):
            return
        
        for acq in self.parent_window.acquisitions:
            label = f"{acq.name}"
            if acq.well:
                label += f" ({acq.well})"
            self.acq_combo.addItem(label, acq.id)
        
    def _on_mode_changed(self):
        """Handle mode change."""
        mode_idx = self.mode_combo.currentIndex()
        if mode_idx == 0:
            self.analysis_mode = "pixel"
        else:
            self.analysis_mode = "cell"
            # Check if masks exist for selected acquisition
            if not self._check_masks_exist():
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Masks Available",
                    "Cell-level analysis requires segmentation masks. Please segment cells first."
                )
    
    
    def _run_analysis(self):
        """Run QC analysis with multiprocessing."""
        if not self.parent_window:
            QtWidgets.QMessageBox.warning(self, "Error", "Parent window not available.")
            return
        
        acq_id = self.acq_combo.currentData()
        if not acq_id:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select an acquisition.")
            return
        
        # Determine which acquisitions to process
        if acq_id == "all":
            acquisitions = self.parent_window.acquisitions
        else:
            acq_info = self.parent_window._get_acquisition_info(acq_id)
            if not acq_info:
                QtWidgets.QMessageBox.warning(self, "Error", f"Could not find acquisition {acq_id}.")
                return
            acquisitions = [acq_info]
        
        # Check if cell-level analysis is requested but no masks available
        if self.analysis_mode == "cell":
            missing_masks = []
            for acq in acquisitions:
                if acq.id not in self.parent_window.segmentation_masks:
                    missing_masks.append(acq.name)
            if missing_masks:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Masks Available",
                    f"Segmentation masks not found for: {', '.join(missing_masks)}. Please segment cells first."
                )
                return
        
        # Get number of workers
        num_workers = self.workers_spin.value()
        
        # Show progress dialog
        progress_dlg = ProgressDialog("Calculating QC Metrics...", self)
        progress_dlg.update_progress(0, f"Processing {num_workers} acquisitions in parallel...", "Preparing tasks...")
        progress_dlg.show()
        QtWidgets.QApplication.processEvents()
        
        try:
            # Prepare tasks for multiprocessing - one task per acquisition
            tasks = []
            acq_to_file_map = {}  # Map acquisition ID to source file path
            
            for acq_info in acquisitions:
                acq_id = acq_info.id
                
                # Get channels
                channels = acq_info.channels
                if not channels:
                    continue
                
                # Get source file path for loader
                source_file = None
                if hasattr(acq_info, 'source_file') and acq_info.source_file:
                    source_file = acq_info.source_file
                elif acq_id in self.parent_window.acq_to_file:
                    source_file = self.parent_window.acq_to_file[acq_id]
                else:
                    # Try to get from current_path
                    if hasattr(self.parent_window, 'current_path') and self.parent_window.current_path:
                        if self.parent_window.current_path.endswith('.mcd'):
                            source_file = self.parent_window.current_path
                
                if not source_file:
                    continue
                
                acq_to_file_map[acq_id] = source_file
                
                # Get mask path if cell-level analysis
                mask_path = None
                if self.analysis_mode == "cell":
                    mask = self.parent_window.segmentation_masks.get(acq_id)
                    if mask is None:
                        continue
                    # Save mask temporarily to temp location
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    mask_path = os.path.join(temp_dir, f"qc_mask_{acq_id}.tif")
                    try:
                        import tifffile
                        tifffile.imwrite(mask_path, mask.astype(np.uint32))
                    except Exception as e:
                        continue
                
                # Create one task per acquisition (all channels)
                task = (acq_id, acq_info.name, channels, self.analysis_mode, mask_path, source_file, acq_to_file_map)
                tasks.append(task)
            
            if not tasks:
                progress_dlg.close()
                QtWidgets.QMessageBox.warning(self, "Error", "No tasks to process.")
                return
            
            # Process with multiprocessing
            results = []
            total_tasks = len(tasks)
            total_channels = sum(len(acq_info.channels) for acq_info in acquisitions if acq_info.channels)
            
            # Use ProcessPoolExecutor for true parallelization
            # Use spawn method to ensure isolation from other multiprocessing operations
            if num_workers > 1 and total_tasks > 1:
                ctx = mp.get_context('spawn')  # Use spawn to avoid conflicts with feature extraction
                with ctx.Pool(processes=num_workers) as pool:
                    # Submit all tasks (one per acquisition)
                    futures = []
                    for task in tasks:
                        future = pool.apply_async(_qc_process_acquisition_worker, (task,))
                        futures.append(future)
                    
                    # Collect results as they complete (for better progress updates)
                    completed = 0
                    processed_futures = set()
                    channels_processed = 0
                    
                    while completed < total_tasks:
                        # Check each future to see if it's ready
                        for i, future in enumerate(futures):
                            if i in processed_futures:
                                continue
                            
                            # Check if ready (non-blocking)
                            if future.ready():
                                try:
                                    acquisition_results = future.get(timeout=0.1)
                                    if acquisition_results:
                                        results.extend(acquisition_results)
                                        channels_processed += len(acquisition_results)
                                    completed += 1
                                    processed_futures.add(i)
                                    
                                    # Update progress immediately
                                    progress = int((completed / total_tasks) * 100)
                                    progress_dlg.update_progress(
                                        progress,
                                        f"Processing {num_workers} acquisitions in parallel...",
                                        f"Processed {completed}/{total_tasks} acquisitions ({channels_processed} channels)"
                                    )
                                    QtWidgets.QApplication.processEvents()
                                except Exception as e:
                                    # Handle errors
                                    completed += 1
                                    processed_futures.add(i)
                                    progress = int((completed / total_tasks) * 100)
                                    progress_dlg.update_progress(
                                        progress,
                                        f"Processing {num_workers} acquisitions in parallel...",
                                        f"Processed {completed}/{total_tasks} acquisitions..."
                                    )
                                    QtWidgets.QApplication.processEvents()
                        
                        # Small delay to avoid busy-waiting and allow UI to update
                        if completed < total_tasks:
                            QtCore.QThread.msleep(50)  # 50ms delay for smoother UI updates
                            QtWidgets.QApplication.processEvents()
                    
                    # Final update to ensure all are processed
                    if completed < total_tasks:
                        # Collect any remaining results
                        for i, future in enumerate(futures):
                            if i not in processed_futures:
                                try:
                                    acquisition_results = future.get(timeout=5)
                                    if acquisition_results:
                                        results.extend(acquisition_results)
                                        channels_processed += len(acquisition_results)
                                    completed += 1
                                except Exception:
                                    completed += 1
                        
                        # Final progress update
                        progress_dlg.update_progress(
                            100,
                            f"Processing {num_workers} acquisitions in parallel...",
                            f"Processed {completed}/{total_tasks} acquisitions ({channels_processed} channels)"
                        )
                        QtWidgets.QApplication.processEvents()
                    else:
                        progress_dlg.update_progress(
                            100,
                            f"Processing {num_workers} acquisitions in parallel...",
                            f"Processed {completed}/{total_tasks} acquisitions ({channels_processed} channels)"
                        )
                        QtWidgets.QApplication.processEvents()
            else:
                # Single-threaded processing
                channels_processed = 0
                for i, task in enumerate(tasks):
                    acquisition_results = _qc_process_acquisition_worker(task)
                    if acquisition_results:
                        results.extend(acquisition_results)
                        channels_processed += len(acquisition_results)
                    
                    # Update progress
                    progress = int(((i + 1) / total_tasks) * 100)
                    progress_dlg.update_progress(
                        progress,
                        "Processing acquisitions...",
                        f"Processed {i+1}/{total_tasks} acquisitions ({channels_processed} channels)"
                    )
                    QtWidgets.QApplication.processEvents()
            
            # Create DataFrame
            if results:
                self.qc_results = pd.DataFrame(results)
                
                # Aggregate results per channel across all ROIs
                self._aggregate_results_by_channel()
                
                # Update UI
                self._update_summary_table()
                self._update_plots()
                self.export_summary_btn.setEnabled(True)
                self.snr_intensity_save_btn.setEnabled(True)
                self.coverage_save_btn.setEnabled(True)
                self.distribution_save_btn.setEnabled(True)
                
                progress_dlg.update_progress(100, "Analysis complete!", f"QC metrics calculated for {len(results)} channels across {len(acquisitions)} acquisitions.")
                QtWidgets.QApplication.processEvents()
                QtCore.QTimer.singleShot(500, progress_dlg.close)
            else:
                progress_dlg.close()
                QtWidgets.QMessageBox.warning(self, "Error", "No results generated. Check console for errors.")
                
        except Exception as e:
            progress_dlg.close()
            QtWidgets.QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _calculate_pixel_metrics(self, img: np.ndarray, channel: str) -> Optional[Dict[str, Any]]:
        """Calculate pixel-level QC metrics using Otsu threshold."""
        if not _HAVE_SCIKIT_IMAGE:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Dependency",
                "scikit-image is required for pixel-level analysis. Please install it: pip install scikit-image"
            )
            return None
        
        try:
            # Convert to float if needed
            img_float = img.astype(np.float32)
            
            # Calculate Otsu threshold
            threshold = threshold_otsu(img_float)
            
            # Separate signal (foreground) and background
            foreground = img_float[img_float > threshold]
            background = img_float[img_float <= threshold]
            
            if len(foreground) == 0 or len(background) == 0:
                return None
            
            # Calculate metrics
            signal_mean = np.mean(foreground)
            signal_std = np.std(foreground)
            background_mean = np.mean(background)
            background_std = np.std(background)
            
            # SNR: (signal_mean - background_mean) / background_std
            snr = (signal_mean - background_mean) / (background_std + 1e-8)
            
            # Intensity metrics
            mean_intensity = np.mean(img_float)
            median_intensity = np.median(img_float)
            max_intensity = np.max(img_float)
            min_intensity = np.min(img_float)
            
            # Coverage: percentage of pixels above threshold
            coverage_pct = (len(foreground) / img_float.size) * 100
            
            # Calculate percentiles
            p1 = np.percentile(img_float, 1)
            p25 = np.percentile(img_float, 25)
            p75 = np.percentile(img_float, 75)
            p99 = np.percentile(img_float, 99)
            
            return {
                'snr': snr,
                'signal_mean': signal_mean,
                'signal_std': signal_std,
                'background_mean': background_mean,
                'background_std': background_std,
                'threshold': threshold,
                'mean_intensity': mean_intensity,
                'median_intensity': median_intensity,
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'coverage_pct': coverage_pct,
                'p1': p1,
                'p25': p25,
                'p75': p75,
                'p99': p99,
                'total_pixels': img_float.size,
                'foreground_pixels': len(foreground),
                'background_pixels': len(background)
            }
        except Exception as e:
            print(f"Error calculating pixel metrics for {channel}: {e}")
            return None
    
    def _calculate_cell_metrics(self, img: np.ndarray, channel: str, mask: np.ndarray) -> Optional[Dict[str, Any]]:
        """Calculate cell-level QC metrics using segmentation masks."""
        try:
            # Convert to float if needed
            img_float = img.astype(np.float32)
            
            # Ensure mask and image have same shape
            if mask.shape != img_float.shape:
                print(f"Warning: Mask shape {mask.shape} doesn't match image shape {img_float.shape}")
                return None
            
            # Separate signal (cells) and background
            cell_mask = mask > 0
            background_mask = mask == 0
            
            if np.sum(cell_mask) == 0 or np.sum(background_mask) == 0:
                return None
            
            foreground = img_float[cell_mask]
            background = img_float[background_mask]
            
            # Calculate metrics
            signal_mean = np.mean(foreground)
            signal_std = np.std(foreground)
            background_mean = np.mean(background)
            background_std = np.std(background)
            
            # SNR: (signal_mean - background_mean) / background_std
            snr = (signal_mean - background_mean) / (background_std + 1e-8)
            
            # Intensity metrics (cell-level)
            mean_intensity = np.mean(foreground)  # Mean intensity in cells
            median_intensity = np.median(foreground)
            max_intensity = np.max(foreground)
            min_intensity = np.min(foreground)
            
            # Coverage: percentage of pixels covered by cells
            coverage_pct = (np.sum(cell_mask) / img_float.size) * 100
            
            # Cell density: number of cells per unit area
            unique_cells = np.unique(mask[mask > 0])
            num_cells = len(unique_cells)
            area_pixels = img_float.size
            # Assuming square pixels, convert to cells per mm² if we have pixel size info
            # For now, just report cells per pixel²
            cell_density = num_cells / area_pixels if area_pixels > 0 else 0
            
            # Calculate percentiles for cell intensities
            p1 = np.percentile(foreground, 1)
            p25 = np.percentile(foreground, 25)
            p75 = np.percentile(foreground, 75)
            p99 = np.percentile(foreground, 99)
            
            # Per-cell statistics
            cell_intensities = []
            for cell_id in unique_cells:
                cell_pixels = img_float[mask == cell_id]
                if len(cell_pixels) > 0:
                    cell_intensities.append(np.mean(cell_pixels))
            
            mean_cell_intensity = np.mean(cell_intensities) if cell_intensities else 0
            median_cell_intensity = np.median(cell_intensities) if cell_intensities else 0
            
            return {
                'snr': snr,
                'signal_mean': signal_mean,
                'signal_std': signal_std,
                'background_mean': background_mean,
                'background_std': background_std,
                'mean_intensity': mean_intensity,
                'median_intensity': median_intensity,
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'coverage_pct': coverage_pct,
                'cell_density': cell_density,
                'num_cells': num_cells,
                'p1': p1,
                'p25': p25,
                'p75': p75,
                'p99': p99,
                'mean_cell_intensity': mean_cell_intensity,
                'median_cell_intensity': median_cell_intensity,
                'total_pixels': img_float.size,
                'foreground_pixels': np.sum(cell_mask),
                'background_pixels': np.sum(background_mask)
            }
        except Exception as e:
            print(f"Error calculating cell metrics for {channel}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _aggregate_results_by_channel(self):
        """Aggregate results per channel across all ROIs."""
        if self.qc_results is None or self.qc_results.empty:
            return
        
        # Group by channel and compute mean across ROIs
        agg_dict = {
            'snr': 'mean',
            'mean_intensity': 'mean',
            'coverage_pct': 'mean',
            'signal_mean': 'mean',
            'background_mean': 'mean',
            'median_intensity': 'mean',
            'max_intensity': 'mean',
            'min_intensity': 'mean',
            'p1': 'mean',
            'p25': 'mean',
            'p75': 'mean',
            'p99': 'mean'
        }
        
        # Add cell-specific metrics if available
        if 'cell_density' in self.qc_results.columns:
            agg_dict['cell_density'] = 'mean'
        if 'num_cells' in self.qc_results.columns:
            agg_dict['num_cells'] = 'mean'
        if 'mean_cell_intensity' in self.qc_results.columns:
            agg_dict['mean_cell_intensity'] = 'mean'
        
        # Only aggregate columns that exist
        available_agg = {k: v for k, v in agg_dict.items() if k in self.qc_results.columns}
        
        # Aggregate by channel
        self.qc_results_aggregated = self.qc_results.groupby('channel').agg(available_agg).reset_index()
        
        # Add count of ROIs per channel
        roi_counts = self.qc_results.groupby('channel').size().reset_index(name='n_rois')
        self.qc_results_aggregated = self.qc_results_aggregated.merge(roi_counts, on='channel')
    
    def _update_summary_table(self):
        """Update the summary table with aggregated results."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        # Select relevant columns for display
        display_cols = ['channel', 'n_rois', 'snr', 'mean_intensity', 'coverage_pct']
        if 'cell_density' in self.qc_results_aggregated.columns:
            display_cols.append('cell_density')
        
        # Get available columns
        available_cols = [col for col in display_cols if col in self.qc_results_aggregated.columns]
        
        # Set up table
        self.summary_table.setRowCount(len(self.qc_results_aggregated))
        self.summary_table.setColumnCount(len(available_cols))
        self.summary_table.setHorizontalHeaderLabels(available_cols)
        
        # Populate table
        for i, row in self.qc_results_aggregated.iterrows():
            for j, col in enumerate(available_cols):
                value = row[col]
                if isinstance(value, (int, np.integer)):
                    item = QtWidgets.QTableWidgetItem(str(value))
                elif isinstance(value, (float, np.floating)):
                    item = QtWidgets.QTableWidgetItem(f"{value:.3f}")
                else:
                    item = QtWidgets.QTableWidgetItem(str(value))
                self.summary_table.setItem(i, j, item)
        
        # Resize columns to content
        self.summary_table.setColumnWidth(0, 200)  # Channel name column
        self.summary_table.resizeColumnsToContents()
    
    def _update_plots(self):
        """Update all plots."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        # Update SNR vs Intensity plot (using aggregated values)
        self._plot_snr_vs_intensity()
        
        # Update Coverage plot (using aggregated values)
        self._plot_coverage()
        
        # Update Distribution plots (using raw data to show distributions)
        self._plot_distributions()
    
    def _plot_snr_vs_intensity(self):
        """Plot SNR vs mean intensity with log scales on both axes (using aggregated values)."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        fig = self.snr_intensity_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Get aggregated intensities and SNR
        intensities = self.qc_results_aggregated['mean_intensity'].values
        snr_values = self.qc_results_aggregated['snr'].values
        channels = self.qc_results_aggregated['channel'].values
        
        # Filter out zero or negative values for log scale
        valid_mask = (intensities > 0) & (snr_values > 0)
        intensities = intensities[valid_mask]
        snr_values = snr_values[valid_mask]
        channels = channels[valid_mask]
        
        if len(intensities) == 0:
            ax.text(0.5, 0.5, 'No valid data points for log scale', 
                   ha='center', va='center', transform=ax.transAxes)
            fig.tight_layout()
            self.snr_intensity_canvas.draw()
            return
        
        # Scatter plot
        ax.scatter(
            intensities,
            snr_values,
            alpha=0.6,
            s=50
        )
        
        # Add channel labels
        for i, channel in enumerate(channels):
            ax.annotate(
                channel,
                (intensities[i], snr_values[i]),
                fontsize=7,
                alpha=0.7
            )
        
        # Set log scale on both axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel('Mean Intensity (log scale, averaged across ROIs)', fontsize=10)
        ax.set_ylabel('SNR (Signal-to-Noise Ratio, log scale, averaged across ROIs)', fontsize=10)
        ax.set_title('SNR vs Mean Intensity by Channel (Mean across all ROIs)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        fig.tight_layout()
        self.snr_intensity_canvas.draw()
    
    def _plot_coverage(self):
        """Plot coverage and cell density (using aggregated values)."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        fig = self.coverage_canvas.figure
        fig.clear()
        
        # Create subplots
        n_plots = 2 if 'cell_density' in self.qc_results_aggregated.columns else 1
        if n_plots == 2:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            ax1 = fig.add_subplot(111)
            ax2 = None
        
        # Get aggregated values
        channels = self.qc_results_aggregated['channel'].values
        coverage = self.qc_results_aggregated['coverage_pct'].values
        cell_density = self.qc_results_aggregated['cell_density'].values if 'cell_density' in self.qc_results_aggregated.columns else None
        
        # Coverage plot
        ax1.bar(range(len(channels)), coverage, alpha=0.7, color='steelblue')
        ax1.set_xticks(range(len(channels)))
        ax1.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('% Coverage (mean across ROIs)', fontsize=10)
        ax1.set_title('Coverage by Channel (Mean across all ROIs)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_coverage = np.mean(coverage)
        ax1.axhline(y=mean_coverage, color='red', linestyle='--', alpha=0.5, label=f'Overall Mean: {mean_coverage:.1f}%')
        ax1.legend(fontsize=8)
        
        # Cell density plot (if available)
        if ax2 is not None and cell_density is not None:
            ax2.bar(range(len(channels)), cell_density, alpha=0.7, color='orange')
            ax2.set_xticks(range(len(channels)))
            ax2.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Cell Density (cells/pixel², mean across ROIs)', fontsize=10)
            ax2.set_title('Cell Density by Channel (Mean across all ROIs)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add mean line
            mean_density = np.mean(cell_density)
            ax2.axhline(y=mean_density, color='red', linestyle='--', alpha=0.5, label=f'Overall Mean: {mean_density:.2e}')
            ax2.legend(fontsize=8)
        
        fig.tight_layout()
        self.coverage_canvas.draw()
    
    def _plot_distributions(self):
        """Plot boxplots showing distributions of SNR, intensity, and coverage across ROIs."""
        if self.qc_results is None or self.qc_results.empty:
            return
        
        fig = self.distribution_canvas.figure
        fig.clear()
        
        # Create 3 subplots
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        
        # Get unique channels (sorted)
        channels = sorted(self.qc_results['channel'].unique())
        
        # Prepare data for boxplots
        snr_data = []
        intensity_data = []
        coverage_data = []
        
        for channel in channels:
            channel_data = self.qc_results[self.qc_results['channel'] == channel]
            # Filter valid values
            valid_snr = channel_data['snr'][channel_data['snr'] > 0].values
            valid_intensity = channel_data['mean_intensity'][channel_data['mean_intensity'] > 0].values
            valid_coverage = channel_data['coverage_pct'].values
            
            snr_data.append(valid_snr if len(valid_snr) > 0 else [0])
            intensity_data.append(valid_intensity if len(valid_intensity) > 0 else [0])
            coverage_data.append(valid_coverage if len(valid_coverage) > 0 else [0])
        
        # SNR boxplot
        bp1 = ax1.boxplot(snr_data, labels=channels, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_yscale('log')
        ax1.set_ylabel('SNR (log scale)', fontsize=10)
        ax1.set_title('SNR Distribution across ROIs', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Intensity boxplot
        bp2 = ax2.boxplot(intensity_data, labels=channels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
        ax2.set_yscale('log')
        ax2.set_ylabel('Mean Intensity (log scale)', fontsize=10)
        ax2.set_title('Intensity Distribution across ROIs', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Coverage boxplot
        bp3 = ax3.boxplot(coverage_data, labels=channels, patch_artist=True)
        for patch in bp3['boxes']:
            patch.set_facecolor('lightcoral')
        ax3.set_ylabel('% Coverage', fontsize=10)
        ax3.set_title('Coverage Distribution across ROIs', fontsize=11, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        self.distribution_canvas.draw()
    
    def _export_results(self):
        """Export results to CSV."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export QC Results",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                # Export aggregated results
                self.qc_results_aggregated.to_csv(filename, index=False)
                QtWidgets.QMessageBox.information(self, "Success", f"Results exported to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
    
    def _save_distribution_plot(self):
        """Save distribution plot."""
        fig = self.distribution_canvas.figure
        save_figure_with_options(fig, self, "QC_Distributions")
    
    def _save_snr_intensity_plot(self):
        """Save SNR vs Intensity plot."""
        fig = self.snr_intensity_canvas.figure
        save_figure_with_options(fig, self, "SNR_vs_Intensity")
    
    def _save_coverage_plot(self):
        """Save coverage plot."""
        fig = self.coverage_canvas.figure
        save_figure_with_options(fig, self, "Coverage_Density")

