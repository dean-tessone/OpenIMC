import json

import pytest
from PyQt5 import QtGui, QtWidgets

from openimc.ui import theme
from openimc.ui.dialogs import display_settings_dialog


def test_theme_defaults_to_light_and_persists(tmp_path, monkeypatch):
    preferences_path = tmp_path / "user_preferences.json"
    monkeypatch.setattr(
        display_settings_dialog,
        "_get_user_config_path",
        lambda: preferences_path,
    )

    assert display_settings_dialog.get_theme_preference() == theme.THEME_LIGHT

    display_settings_dialog.save_theme_preference(theme.THEME_DARK)
    assert display_settings_dialog.get_theme_preference() == theme.THEME_DARK
    assert json.loads(preferences_path.read_text(encoding="utf-8")) == {
        "theme": "dark"
    }


def test_invalid_saved_theme_falls_back_to_light(tmp_path, monkeypatch):
    preferences_path = tmp_path / "user_preferences.json"
    preferences_path.write_text('{"theme": "sepia"}', encoding="utf-8")
    monkeypatch.setattr(
        display_settings_dialog,
        "_get_user_config_path",
        lambda: preferences_path,
    )

    assert display_settings_dialog.get_theme_preference() == theme.THEME_LIGHT
    with pytest.raises(ValueError, match="Unsupported interface theme"):
        display_settings_dialog.save_theme_preference("sepia")


def test_light_and_dark_palettes_have_expected_contrast():
    light = theme.build_application_palette(theme.THEME_LIGHT)
    dark = theme.build_application_palette(theme.THEME_DARK)

    assert light.color(QtGui.QPalette.Window).lightness() >= 128
    assert dark.color(QtGui.QPalette.Window).lightness() < 128
    assert (
        light.color(QtGui.QPalette.WindowText).lightness()
        < light.color(QtGui.QPalette.Window).lightness()
    )
    assert (
        dark.color(QtGui.QPalette.WindowText).lightness()
        > dark.color(QtGui.QPalette.Window).lightness()
    )
    assert (
        dark.color(QtGui.QPalette.PlaceholderText).lightness()
        > dark.color(QtGui.QPalette.Window).lightness()
    )


@pytest.mark.ui
def test_display_settings_applies_and_saves_theme(
    qtbot, tmp_path, monkeypatch
):
    preferences_path = tmp_path / "user_preferences.json"
    monkeypatch.setattr(
        display_settings_dialog,
        "_get_user_config_path",
        lambda: preferences_path,
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: None)

    dialog = display_settings_dialog.DisplaySettingsDialog()
    qtbot.addWidget(dialog)
    dialog.theme_combo.setCurrentIndex(
        dialog.theme_combo.findData(theme.THEME_DARK)
    )
    dialog._ok_clicked()

    assert dialog.result() == QtWidgets.QDialog.Accepted
    assert display_settings_dialog.get_theme_preference() == theme.THEME_DARK
    application = QtWidgets.QApplication.instance()
    assert application.property("openimcTheme") == theme.THEME_DARK
    assert theme.palette_is_dark(application.palette())

    # Restore the shared test application's original style and palette.
    theme.apply_application_theme(application, theme.THEME_SYSTEM)
