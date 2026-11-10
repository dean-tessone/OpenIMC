"""Resolve official CellSAM weights without the retired DeepCell API.

The inference code still uses DeepCell's downloader as of commit 362a8e5.
Keep acquisition here and pass a local checkpoint to that code instead.
"""

import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


MODEL_REPO = "vanvalenlab/cellsam"
MODEL_REVISION = "3a2e42cab75a0eb50b9d3c04889a8ea1cdd73904"
MODEL_ARCHIVE = "cellsam-models_v1.2.tar.gz"
# Published by CellSAM in cellSAM/_auth.py for the v1.2 archive.
MODEL_ARCHIVE_MD5 = "f41e6899c49fc8ce12f77a8d25594604"
MODEL_MEMBER = "cellsam_v1.2/cellsam_general.pt"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}"


def resolve_cellsam_checkpoint(model_path=None, *, local_files_only=False) -> Path:
    """Use an explicit checkpoint, an existing v1.2 cache, or Hugging Face.

    Hugging Face handles authentication (HF_TOKEN or `hf auth login`), offline
    mode, and its download cache. License acceptance must happen on the Hub.
    No credentials are written by OpenIMC and no DeepCell API is contacted.
    """
    explicit = model_path or os.environ.get("OPENIMC_CELLSAM_MODEL_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CellSAM checkpoint does not exist: {path}")
        return path

    legacy = Path.home() / ".deepcell/models" / MODEL_MEMBER
    if legacy.is_file():
        return legacy

    return download_cellsam_checkpoint(local_files_only=local_files_only)


def download_cellsam_checkpoint(*, local_files_only=False) -> Path:
    """Fetch the pinned Hub archive and prepare the generalist checkpoint.

    Unlike resolve_cellsam_checkpoint, this explicitly exercises the Hub route
    even when a legacy DeepCell checkpoint is available.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as exc:
        raise ImportError("Install CellSAM support with: pip install 'openimc[cellsam]'") from exc

    try:
        archive = Path(hf_hub_download(
            repo_id=MODEL_REPO, filename=MODEL_ARCHIVE, revision=MODEL_REVISION,
            local_files_only=local_files_only,
        ))
    except GatedRepoError:
        raise RuntimeError(
            f"Sign in to Hugging Face and accept the CellSAM terms and model license at {MODEL_URL}. "
            "Then authenticate OpenIMC with `hf auth login` or HF_TOKEN using the same account. "
            "In the desktop app, enter a Hugging Face read token with CellSAM access in CellSAM Parameters. "
            "A DeepCell API key is no longer used."
        ) from None
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Could not download CellSAM weights from {MODEL_URL}. "
            "Check your connection and Hugging Face login, or set "
            "OPENIMC_CELLSAM_MODEL_PATH to a local checkpoint."
        ) from exc

    checkpoint = archive.parent / "openimc-cellsam-v1.2" / "cellsam_general.pt"
    if checkpoint.is_file():
        return checkpoint

    digest = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MODEL_ARCHIVE_MD5:
        raise ValueError("CellSAM archive checksum does not match the published v1.2 weights")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Copy only the expected regular file; never extract paths or symlinks from
    # an archive. Atomic replacement prevents partially written cached weights.
    temporary = None
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember(MODEL_MEMBER)
            if not member.isfile():
                raise ValueError("CellSAM checkpoint archive member must be a regular file")
            with bundle.extractfile(member) as source, tempfile.NamedTemporaryFile(
                dir=checkpoint.parent, delete=False,
            ) as target:
                temporary = Path(target.name)
                shutil.copyfileobj(source, target)
        os.replace(temporary, checkpoint)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return checkpoint
