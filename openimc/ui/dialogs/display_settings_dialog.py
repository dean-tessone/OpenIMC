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

"""
Display settings dialog for customizing application appearance.
"""

import json
import platform
from pathlib import Path
from typing import Optional
from PyQt5 import QtWidgets, QtGui

from openimc.ui.theme import (
    DEFAULT_THEME,
    THEME_DARK,
    THEME_LIGHT,
    THEME_SYSTEM,
    apply_application_theme,
    normalize_theme,
)


def _get_user_config_path() -> Path:
    """Get path to user configuration file for storing preferences."""
    # Use user's home directory with .openimc config folder
    home = Path.home()
    config_dir = home / ".openimc"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "user_preferences.json"


def _load_user_preferences() -> dict:
    """Load user preferences from config file."""
    config_path = _get_user_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_user_preferences(prefs: dict):
    """Save user preferences to config file."""
    config_path = _get_user_config_path()
    try:
        # Load existing preferences and update
        existing_prefs = _load_user_preferences()
        existing_prefs.update(prefs)
        with open(config_path, 'w') as f:
            json.dump(existing_prefs, f, indent=2)
    except IOError:
        # Silently fail if we can't write to config file
        pass


def get_font_size_preference() -> Optional[int]:
    """Get the saved font size preference, or None if not set."""
    prefs = _load_user_preferences()
    return prefs.get('font_size')


def save_font_size_preference(font_size: int):
    """Save the font size preference."""
    _save_user_preferences({'font_size': font_size})


def get_theme_preference() -> str:
    """Get the saved interface theme, defaulting to light on every OS."""
    prefs = _load_user_preferences()
    return normalize_theme(prefs.get('theme', DEFAULT_THEME))


def save_theme_preference(theme: str):
    """Save a validated interface theme preference."""
    normalized = normalize_theme(theme)
    if normalized != theme:
        raise ValueError(f"Unsupported interface theme: {theme!r}")
    _save_user_preferences({'theme': normalized})


def get_masks_directory_preference() -> Optional[str]:
    """Get the saved masks directory preference, or None if not set."""
    prefs = _load_user_preferences()
    return prefs.get('masks_directory')


def save_masks_directory_preference(masks_directory: str):
    """Save the masks directory preference."""
    _save_user_preferences({'masks_directory': masks_directory})


def get_default_font_size() -> int:
    """Get the default font size based on platform."""
    if platform.system() == 'Windows':
        return 13  # Pixel size for Windows
    else:
        return 10  # Point size for Mac/Linux


class DisplaySettingsDialog(QtWidgets.QDialog):
    """Dialog for customizing display settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display Settings")
        self.setModal(True)
        self.setMinimumWidth(430)
        
        self._create_ui()
        self._load_current_settings()
    
    def _create_ui(self):
        """Create the UI elements."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Display Settings")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 12pt; }")
        layout.addWidget(title_label)

        # Interface theme section
        theme_group = QtWidgets.QGroupBox("Appearance")
        theme_layout = QtWidgets.QVBoxLayout(theme_group)

        theme_desc = QtWidgets.QLabel(
            "Choose the colors used by the OpenIMC interface. Theme changes "
            "are applied immediately and remembered for future launches."
        )
        theme_desc.setWordWrap(True)
        theme_desc.setForegroundRole(QtGui.QPalette.PlaceholderText)
        theme_layout.addWidget(theme_desc)

        theme_control_layout = QtWidgets.QHBoxLayout()
        theme_control_layout.addWidget(QtWidgets.QLabel("Theme:"))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItem("Light (default)", THEME_LIGHT)
        self.theme_combo.addItem("Dark", THEME_DARK)
        self.theme_combo.addItem("Follow operating system", THEME_SYSTEM)
        theme_control_layout.addWidget(self.theme_combo)
        theme_control_layout.addStretch()
        theme_layout.addLayout(theme_control_layout)
        layout.addWidget(theme_group)
        
        # Font size section
        font_group = QtWidgets.QGroupBox("Font Size")
        font_layout = QtWidgets.QVBoxLayout(font_group)
        
        # Description
        desc_label = QtWidgets.QLabel(
            "Adjust the font size for the application interface. "
            "Changes will take effect after restarting the application."
        )
        desc_label.setWordWrap(True)
        desc_label.setForegroundRole(QtGui.QPalette.PlaceholderText)
        font_layout.addWidget(desc_label)
        
        # Font size control
        font_size_layout = QtWidgets.QHBoxLayout()
        font_size_layout.addWidget(QtWidgets.QLabel("Font Size:"))
        
        self.font_size_spinbox = QtWidgets.QSpinBox()
        if platform.system() == 'Windows':
            self.font_size_spinbox.setMinimum(8)
            self.font_size_spinbox.setMaximum(24)
            self.font_size_spinbox.setSuffix(" px")
            self.font_size_spinbox.setToolTip("Font size in pixels (Windows)")
        else:
            self.font_size_spinbox.setMinimum(8)
            self.font_size_spinbox.setMaximum(20)
            self.font_size_spinbox.setSuffix(" pt")
            self.font_size_spinbox.setToolTip("Font size in points (Mac/Linux)")
        
        # Set default value
        default_size = get_default_font_size()
        saved_size = get_font_size_preference()
        self.font_size_spinbox.setValue(saved_size if saved_size is not None else default_size)
        
        font_size_layout.addWidget(self.font_size_spinbox)
        font_size_layout.addStretch()
        font_layout.addLayout(font_size_layout)
        
        # Reset button
        reset_btn = QtWidgets.QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_to_default)
        font_layout.addWidget(reset_btn)
        
        layout.addWidget(font_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self._ok_clicked)
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _load_current_settings(self):
        """Load current settings from preferences."""
        saved_size = get_font_size_preference()
        if saved_size is not None:
            self.font_size_spinbox.setValue(saved_size)
        saved_theme = get_theme_preference()
        theme_index = self.theme_combo.findData(saved_theme)
        self.theme_combo.setCurrentIndex(max(theme_index, 0))
    
    def _reset_to_default(self):
        """Reset font size to default."""
        default_size = get_default_font_size()
        self.font_size_spinbox.setValue(default_size)
    
    def _ok_clicked(self):
        """Handle OK button click - save settings and close."""
        font_size = self.font_size_spinbox.value()
        theme = self.get_theme()
        save_font_size_preference(font_size)
        save_theme_preference(theme)

        application = QtWidgets.QApplication.instance()
        if application is not None:
            apply_application_theme(application, theme)

        QtWidgets.QMessageBox.information(
            self,
            "Settings Saved",
            "The interface theme has been applied.\n\n"
            "Please restart OpenIMC if you changed the font size."
        )
        
        self.accept()
    
    def get_font_size(self) -> int:
        """Get the selected font size."""
        return self.font_size_spinbox.value()

    def get_theme(self) -> str:
        """Get the selected interface theme."""
        return self.theme_combo.currentData()
