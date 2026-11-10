import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from openimc.processing import cellsam_models as models


@pytest.fixture
def clean_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("OPENIMC_CELLSAM_MODEL_PATH", raising=False)
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN", raising=False)
    return tmp_path


def test_existing_checkpoint_needs_no_authentication(clean_home, monkeypatch):
    checkpoint = clean_home / ".deepcell/models" / models.MODEL_MEMBER
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"existing weights")
    download = Mock(side_effect=AssertionError("must not contact Hub"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert models.resolve_cellsam_checkpoint() == checkpoint
    download.assert_not_called()


def test_explicit_checkpoint_overrides_default(clean_home, monkeypatch):
    checkpoint = clean_home / "custom.pt"
    checkpoint.touch()
    monkeypatch.setenv("OPENIMC_CELLSAM_MODEL_PATH", str(checkpoint))
    assert models.resolve_cellsam_checkpoint() == checkpoint
    with pytest.raises(FileNotFoundError):
        models.resolve_cellsam_checkpoint(clean_home / "missing.pt")


def make_archive(path, *, symlink=False):
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(models.MODEL_MEMBER)
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = "/outside-cache"
            archive.addfile(info)
        else:
            info.size = 7
            archive.addfile(info, io.BytesIO(b"weights"))
        # Other archive paths must never be extracted.
        info = tarfile.TarInfo("../outside-cache")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))


@pytest.mark.parametrize("symlink", [False, True])
def test_hub_download_checks_and_extracts_only_checkpoint(clean_home, monkeypatch, symlink):
    archive = clean_home / models.MODEL_ARCHIVE
    make_archive(archive, symlink=symlink)
    monkeypatch.setattr(models, "MODEL_ARCHIVE_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    download = Mock(return_value=str(archive))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    if symlink:
        with pytest.raises(ValueError, match="regular file"):
            models.resolve_cellsam_checkpoint()
    else:
        checkpoint = models.resolve_cellsam_checkpoint()
        assert checkpoint.read_bytes() == b"weights"
        assert models.resolve_cellsam_checkpoint() == checkpoint
    download.assert_called_with(
        repo_id=models.MODEL_REPO, filename=models.MODEL_ARCHIVE, revision=models.MODEL_REVISION,
        local_files_only=False,
    )
    assert not (clean_home.parent / "outside-cache").exists()


def test_corrupt_archive_is_rejected(clean_home, monkeypatch):
    archive = clean_home / models.MODEL_ARCHIVE
    archive.write_bytes(b"corrupt")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(archive))
    with pytest.raises(ValueError, match="checksum"):
        models.resolve_cellsam_checkpoint()


def test_gated_download_explains_hugging_face_setup(clean_home, monkeypatch):
    from huggingface_hub.errors import GatedRepoError
    monkeypatch.setattr("huggingface_hub.hf_hub_download", Mock(side_effect=GatedRepoError("private details", response=Mock(headers={}))))
    with pytest.raises(RuntimeError, match="hf auth login") as error:
        models.resolve_cellsam_checkpoint()
    assert "private details" not in str(error.value)
    assert models.MODEL_URL in str(error.value)


def test_model_cache_changes_when_checkpoint_changes(clean_home, monkeypatch):
    from openimc.processing import custom_cellsam as pipeline
    first = clean_home / "first.pt"
    second = clean_home / "second.pt"
    first.touch()
    second.touch()
    for name, value in [("_CACHED_MODEL", None), ("_CACHED_MODEL_DEVICE", None),
                        ("_CACHED_MODEL_PATH", None), ("_CACHED_MODEL_BBOX_THRESHOLD", None),
                        ("_HAVE_CELLSAM", True)]:
        monkeypatch.setattr(pipeline, name, value)
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    def new_model(path):
        model = Mock()
        model.to.return_value = model
        return model
    load = Mock(side_effect=new_model)
    monkeypatch.setattr(pipeline, "get_local_model", load)
    one = pipeline._get_cached_model(first)
    assert pipeline._get_cached_model(first, 0.2) is one
    assert one.bbox_threshold == 0.2
    assert pipeline._get_cached_model(second) is not one
    assert load.call_count == 2
