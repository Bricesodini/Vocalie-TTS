"""Compatibility shim — canonical location is backend.shared.output_paths."""
from backend.shared.output_paths import *  # noqa: F401,F403
from backend.shared.output_paths import __all__ as _all  # noqa: F401

__all__ = list(_all)