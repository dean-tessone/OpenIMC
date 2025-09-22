from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np

_HAVE_READIMC = False
try:
    from readimc import MCDFile as McdFile  # type: ignore
    _HAVE_READIMC = True
except Exception:
    _HAVE_READIMC = False


@dataclass
class AcquisitionInfo:
    id: str
    name: str
    well: Optional[str]
    size: Tuple[Optional[int], Optional[int]]  # (H, W)
    channels: List[str]
    channel_metals: List[str]
    channel_labels: List[str]
    metadata: Dict


class MCDLoader:
    """Loader for IMC .mcd files using the readimc library with f.read_acquisition() method."""

    def __init__(self):
        if not _HAVE_READIMC:
            raise RuntimeError("readimc is not installed. Run: pip install readimc")
        self.mcd: Optional[McdFile] = None
        self._acq_map: Dict[str, object] = {}
        self._acq_channels: Dict[str, List[str]] = {}
        self._acq_channel_metals: Dict[str, List[str]] = {}
        self._acq_channel_labels: Dict[str, List[str]] = {}
        self._acq_size: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        self._acq_name: Dict[str, str] = {}
        self._acq_well: Dict[str, Optional[str]] = {}
        self._acq_metadata: Dict[str, Dict] = {}

    def open(self, path: str):
        """Open an .mcd file."""
        self.mcd = McdFile(path)
        if hasattr(self.mcd, "open"):
            self.mcd.open()
        self._index()

    def _index(self):
        """Index all acquisitions in the .mcd file."""
        self._acq_map.clear()
        self._acq_channels.clear()
        self._acq_channel_metals.clear()
        self._acq_channel_labels.clear()
        self._acq_size.clear()
        self._acq_name.clear()
        self._acq_well.clear()
        self._acq_metadata.clear()

        acq_counter = 0
        slides = getattr(self.mcd, "slides", [])

        if slides:
            for slide_idx, slide in enumerate(slides):
                for acq_idx, acq in enumerate(getattr(slide, "acquisitions", [])):
                    acq_id = f"slide_{slide_idx}_acq_{acq_idx}"
                    acq_counter += 1

                    name = getattr(acq, "name", f"Slide {slide_idx + 1} Acquisition {acq_idx + 1}")

                    well = getattr(acq, "well", getattr(slide, "well", None))
                    if well is None and hasattr(acq, "metadata"):
                        metadata = acq.metadata
                        if isinstance(metadata, dict) and 'Description' in metadata:
                            well = metadata['Description']

                    channel_metals = getattr(acq, "channel_names", [])
                    channel_labels = getattr(acq, "channel_labels", [])

                    channels: List[str] = []
                    for i, (metal, label) in enumerate(zip(channel_metals, channel_labels)):
                        if label and metal:
                            channels.append(f"{label}_{metal}")
                        elif label:
                            channels.append(label)
                        elif metal:
                            channels.append(metal)
                        else:
                            channels.append(f"Channel_{i+1}")

                    try:
                        H = getattr(acq, "height", None) or getattr(acq, "rows", None)
                        W = getattr(acq, "width", None) or getattr(acq, "cols", None)
                        size = (int(H), int(W)) if H and W else (None, None)
                    except Exception:
                        size = (None, None)

                    metadata = getattr(acq, "metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}

                    self._acq_map[acq_id] = acq
                    self._acq_channels[acq_id] = channels
                    self._acq_channel_metals[acq_id] = channel_metals
                    self._acq_channel_labels[acq_id] = channel_labels
                    self._acq_size[acq_id] = size
                    self._acq_name[acq_id] = name
                    self._acq_well[acq_id] = well
                    self._acq_metadata[acq_id] = metadata

        if not self._acq_map:
            raise RuntimeError("No acquisitions found in this .mcd file.")

    def list_acquisitions(self) -> List[AcquisitionInfo]:
        """List all acquisitions in the .mcd file."""
        infos: List[AcquisitionInfo] = []
        for acq_id in self._acq_map:
            infos.append(
                AcquisitionInfo(
                    id=acq_id,
                    name=self._acq_name.get(acq_id, acq_id),
                    well=self._acq_well.get(acq_id),
                    size=self._acq_size.get(acq_id, (None, None)),
                    channels=self._acq_channels.get(acq_id, []),
                    channel_metals=self._acq_channel_metals.get(acq_id, []),
                    channel_labels=self._acq_channel_labels.get(acq_id, []),
                    metadata=self._acq_metadata.get(acq_id, {}),
                )
            )
        return infos

    def get_channels(self, acq_id: str) -> List[str]:
        """Get channel names for a specific acquisition."""
        return self._acq_channels[acq_id]

    def get_image(self, acq_id: str, channel: str) -> np.ndarray:
        """Get image data for a specific acquisition and channel."""
        acq = self._acq_map[acq_id]
        channels = self._acq_channels[acq_id]
        if channel not in channels:
            raise ValueError(f"Channel '{channel}' not found in acquisition {acq_id}.")
        ch_idx = channels.index(channel)
        assert self.mcd is not None
        with self.mcd as f:  # type: ignore
            img = f.read_acquisition(acq)
            img = np.transpose(img, (1, 2, 0))
            return img[..., ch_idx]

    def get_all_channels(self, acq_id: str) -> np.ndarray:
        """Get all channels for a specific acquisition as a 3D array (H, W, C)."""
        acq = self._acq_map[acq_id]
        assert self.mcd is not None
        with self.mcd as f:  # type: ignore
            img = f.read_acquisition(acq)
            img = np.transpose(img, (1, 2, 0))
            return img

    def close(self):
        """Close the .mcd file."""
        if self.mcd and hasattr(self.mcd, "close"):
            self.mcd.close()



