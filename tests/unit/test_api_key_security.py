import json

from openimc.release_validation import ValidationRun
from openimc.ui.dialogs import segmentation_dialog
from scripts import build_desktop


def test_legacy_deepcell_key_is_removed_without_losing_preferences(
    tmp_path, monkeypatch
):
    preferences_path = tmp_path / "user_preferences.json"
    preferences_path.write_text(
        json.dumps({"deepcell_api_key": "secret-value", "font_size": 12}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        segmentation_dialog,
        "_get_user_config_path",
        lambda: preferences_path,
    )

    assert segmentation_dialog._remove_legacy_saved_api_key() is True
    saved = json.loads(preferences_path.read_text(encoding="utf-8"))
    assert saved == {"font_size": 12}
    assert segmentation_dialog._remove_legacy_saved_api_key() is False


def test_bundle_scanner_detects_exact_credentials_in_binary_files(tmp_path):
    credential = b"private-deepcell-token-for-test"
    binary_path = tmp_path / "native-library.dylib"
    binary_path.write_bytes(b"binary-prefix\x00" + credential + b"\x00binary-suffix")

    assert build_desktop._file_contains_secret(
        binary_path,
        (credential,),
        scan_key_patterns=False,
    )


def test_bundle_scanner_limits_key_shape_heuristics_to_text_files(tmp_path):
    key_shaped_bytes = b"sk-proj-examplecredential1234567890"
    path = tmp_path / "content.bin"
    path.write_bytes(key_shaped_bytes)

    assert not build_desktop._file_contains_secret(
        path,
        (),
        scan_key_patterns=False,
    )
    assert build_desktop._file_contains_secret(
        path,
        (),
        scan_key_patterns=True,
    )


def test_credential_dependent_validation_can_be_skipped(tmp_path):
    validation = ValidationRun(tmp_path)
    validation.skip("cellsam_model_and_segmentation", "runtime token not supplied")
    report_path = validation.finish()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["checks"]["cellsam_model_and_segmentation"] == {
        "status": "skipped",
        "reason": "runtime token not supplied",
    }


def test_validation_report_records_the_current_check(tmp_path):
    validation = ValidationRun(tmp_path)

    def inspect_running_report():
        report = json.loads(validation.report_path.read_text(encoding="utf-8"))
        assert report["current_check"] == "real_scientific_check"
        return {"result": "ok"}

    validation.check("real_scientific_check", inspect_running_report)
    report_path = validation.finish()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "current_check" not in report
    assert report["checks"]["real_scientific_check"]["status"] == "passed"
