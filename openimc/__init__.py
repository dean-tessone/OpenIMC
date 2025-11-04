"""
OpenIMC modular package.
"""

# Re-export common types for convenience
try:
    from .data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
    from .data.ometiff_loader import OMETIFFLoader  # noqa: F401
except Exception:
    # During partial refactors, these may not yet be available
    pass



