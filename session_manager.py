"""Compatibility shim — canonical location is backend.shared.session_manager."""
from backend.shared.session_manager import *  # noqa: F401,F403
from backend.shared.session_manager import __all__ as _all  # noqa: F401

__all__ = list(_all)