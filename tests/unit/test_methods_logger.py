import json
from pathlib import Path

from openimc.ui.dialogs import display_settings_dialog
from openimc.utils import logger


def test_default_methods_log_uses_macos_application_support(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENIMC_LOG_FILE", raising=False)
    monkeypatch.setattr(logger.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    expected = (
        tmp_path
        / "Library"
        / "Application Support"
        / "OpenIMC"
        / "logs"
        / "methods_log.jsonl"
    )
    assert logger.get_default_log_file_path() == expected

    methods_logger = logger.MethodsLogger()
    assert Path(methods_logger.get_log_file_path()) == expected
    metadata = json.loads(expected.read_text(encoding="utf-8").splitlines()[0])
    assert metadata["type"] == "log_metadata"


def test_default_methods_log_uses_windows_local_app_data(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENIMC_LOG_FILE", raising=False)
    monkeypatch.setattr(logger.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert logger.get_default_log_file_path() == (
        tmp_path / "OpenIMC" / "logs" / "methods_log.jsonl"
    )


def test_methods_log_environment_override(tmp_path, monkeypatch):
    override = tmp_path / "project" / "analysis.jsonl"
    monkeypatch.setenv("OPENIMC_LOG_FILE", str(override))

    assert logger.get_default_log_file_path() == override


def test_methods_log_preference_persists(tmp_path, monkeypatch):
    preferences_path = tmp_path / "user_preferences.json"
    monkeypatch.setattr(
        display_settings_dialog,
        "_get_user_config_path",
        lambda: preferences_path,
    )
    selected_log = tmp_path / "study" / "methods.jsonl"

    assert display_settings_dialog.get_methods_log_file_preference() is None
    display_settings_dialog.save_methods_log_file_preference(str(selected_log))
    assert display_settings_dialog.get_methods_log_file_preference() == str(selected_log)
