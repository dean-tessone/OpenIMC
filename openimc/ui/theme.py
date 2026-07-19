# SPDX-License-Identifier: GPL-3.0-or-later
"""Application-wide light, dark, and system appearance support."""

from PyQt5 import QtGui, QtWidgets


THEME_LIGHT = "light"
THEME_DARK = "dark"
THEME_SYSTEM = "system"
DEFAULT_THEME = THEME_LIGHT
VALID_THEMES = (THEME_LIGHT, THEME_DARK, THEME_SYSTEM)


def normalize_theme(theme: object) -> str:
    """Return a supported theme name, falling back to the light default."""
    if isinstance(theme, str):
        normalized = theme.strip().lower()
        if normalized in VALID_THEMES:
            return normalized
    return DEFAULT_THEME


def _set_disabled_colors(palette: QtGui.QPalette, color: QtGui.QColor) -> None:
    """Apply readable disabled text colors to the common text roles."""
    for role in (
        QtGui.QPalette.WindowText,
        QtGui.QPalette.Text,
        QtGui.QPalette.ButtonText,
    ):
        palette.setColor(QtGui.QPalette.Disabled, role, color)


def build_application_palette(theme: str) -> QtGui.QPalette:
    """Create the explicit application palette for a light or dark theme."""
    theme = normalize_theme(theme)
    if theme == THEME_SYSTEM:
        raise ValueError("The system theme uses Qt's native palette")

    palette = QtGui.QPalette()
    if theme == THEME_DARK:
        window = QtGui.QColor("#2b2d30")
        base = QtGui.QColor("#202124")
        alternate_base = QtGui.QColor("#34363a")
        text = QtGui.QColor("#f1f3f4")
        button = QtGui.QColor("#36383c")
        highlight = QtGui.QColor("#3f8fc4")
        disabled = QtGui.QColor("#8d9399")
        tooltip = QtGui.QColor("#303236")
        link = QtGui.QColor("#78bdf2")
        light = QtGui.QColor("#55595f")
        midlight = QtGui.QColor("#45484d")
        mid = QtGui.QColor("#3c3f43")
        dark = QtGui.QColor("#191a1c")
        shadow = QtGui.QColor("#0d0e0f")
    else:
        window = QtGui.QColor("#f4f6f8")
        base = QtGui.QColor("#ffffff")
        alternate_base = QtGui.QColor("#edf1f4")
        text = QtGui.QColor("#202428")
        button = QtGui.QColor("#f2f4f6")
        highlight = QtGui.QColor("#2f80b8")
        disabled = QtGui.QColor("#858b91")
        tooltip = QtGui.QColor("#fffbe6")
        link = QtGui.QColor("#1769aa")
        light = QtGui.QColor("#ffffff")
        midlight = QtGui.QColor("#dce1e5")
        mid = QtGui.QColor("#aab1b7")
        dark = QtGui.QColor("#747b82")
        shadow = QtGui.QColor("#3d4247")

    palette.setColor(QtGui.QPalette.Window, window)
    palette.setColor(QtGui.QPalette.WindowText, text)
    palette.setColor(QtGui.QPalette.Base, base)
    palette.setColor(QtGui.QPalette.AlternateBase, alternate_base)
    palette.setColor(QtGui.QPalette.ToolTipBase, tooltip)
    palette.setColor(QtGui.QPalette.ToolTipText, text)
    palette.setColor(QtGui.QPalette.Text, text)
    palette.setColor(QtGui.QPalette.Button, button)
    palette.setColor(QtGui.QPalette.ButtonText, text)
    palette.setColor(QtGui.QPalette.Light, light)
    palette.setColor(QtGui.QPalette.Midlight, midlight)
    palette.setColor(QtGui.QPalette.Mid, mid)
    palette.setColor(QtGui.QPalette.Dark, dark)
    palette.setColor(QtGui.QPalette.Shadow, shadow)
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#ff5252"))
    palette.setColor(QtGui.QPalette.Link, link)
    palette.setColor(QtGui.QPalette.LinkVisited, link)
    palette.setColor(QtGui.QPalette.Highlight, highlight)
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    if hasattr(QtGui.QPalette, "PlaceholderText"):
        palette.setColor(QtGui.QPalette.PlaceholderText, disabled)
    _set_disabled_colors(palette, disabled)
    return palette


def _remember_system_appearance(application: QtWidgets.QApplication) -> None:
    """Capture Qt's native appearance once so System can restore it later."""
    if hasattr(application, "_openimc_system_palette"):
        return
    application._openimc_system_palette = QtGui.QPalette(application.palette())
    application._openimc_system_style = application.style().objectName()
    application._openimc_system_stylesheet = application.styleSheet()


def apply_application_theme(
    application: QtWidgets.QApplication,
    theme: str,
) -> str:
    """Apply an interface theme immediately and return its normalized name."""
    normalized = normalize_theme(theme)
    _remember_system_appearance(application)

    if normalized == THEME_SYSTEM:
        native_style = QtWidgets.QStyleFactory.create(
            application._openimc_system_style
        )
        if native_style is not None:
            application.setStyle(native_style)
        application.setPalette(
            QtGui.QPalette(application._openimc_system_palette)
        )
        application.setStyleSheet(application._openimc_system_stylesheet)
    else:
        # Fusion consistently honors application palettes on macOS, Windows,
        # and Linux; native macOS controls otherwise retain the OS dark colors.
        fusion_style = QtWidgets.QStyleFactory.create("Fusion")
        if fusion_style is not None:
            application.setStyle(fusion_style)
        application.setPalette(build_application_palette(normalized))
        application.setStyleSheet(application._openimc_system_stylesheet)

    application.setProperty("openimcTheme", normalized)
    return normalized


def palette_is_dark(palette: QtGui.QPalette) -> bool:
    """Return whether a palette's main window background is dark."""
    return palette.color(QtGui.QPalette.Window).lightness() < 128
