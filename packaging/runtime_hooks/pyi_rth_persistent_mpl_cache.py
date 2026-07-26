"""Use a persistent Matplotlib cache in OpenIMC's folder-based desktop app."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _openimc_cache_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "OpenIMC"
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "OpenIMC" / "Cache"
        return Path.home() / "AppData" / "Local" / "OpenIMC" / "Cache"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "openimc"
    return Path.home() / ".cache" / "openimc"


if "MPLCONFIGDIR" not in os.environ:
    # Matplotlib stores its bundled font paths relative to mpl-data and system
    # font paths as absolute paths. The cache therefore remains valid when an
    # OpenIMC application folder is moved or replaced; tying it to _MEIPASS
    # would force another expensive scan after every install-location change.
    matplotlib_cache = _openimc_cache_root() / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)

    # Every platform build includes a cache generated on the matching CI host.
    # Seed a user's cache before Matplotlib imports so even the first launch
    # avoids a long, silent font scan. Matplotlib will still repair the cache if
    # an unusual system font is unavailable.
    frozen_root = Path(getattr(sys, "_MEIPASS", sys.executable)).resolve()
    seed_directory = frozen_root / "openimc_bootstrap" / "matplotlib"
    if seed_directory.is_dir():
        for seed_file in seed_directory.glob("fontlist-v*.json"):
            target_file = matplotlib_cache / seed_file.name
            if not target_file.exists():
                try:
                    shutil.copyfile(seed_file, target_file)
                except OSError:
                    # A read-only or locked cache must not prevent OpenIMC from
                    # starting; Matplotlib will fall back to its normal path.
                    pass
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

del _openimc_cache_root
