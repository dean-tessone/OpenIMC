"""
OpenMCD modular package.

This package contains modularized components extracted from the original
`open_mcd.py` monolithic script.
"""

# Re-export common types for convenience
try:
    from .data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
except Exception:
    # During partial refactors, these may not yet be available
    pass



