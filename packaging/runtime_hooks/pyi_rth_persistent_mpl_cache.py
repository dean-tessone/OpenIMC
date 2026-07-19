"""Use a persistent Matplotlib cache in OpenIMC's folder-based desktop app."""

from __future__ import annotations

import hashlib
import os
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
    # Font entries for Matplotlib's bundled fonts contain absolute paths.
    # Namespace the cache by the installed application location so moving a
    # portable OpenIMC folder rebuilds once instead of reusing stale paths.
    frozen_root = Path(getattr(sys, "_MEIPASS", sys.executable)).resolve()
    location_key = hashlib.sha256(
        os.fsencode(str(frozen_root))
    ).hexdigest()[:12]
    matplotlib_cache = _openimc_cache_root() / "matplotlib" / location_key
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

del _openimc_cache_root
