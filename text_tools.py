"""Compatibility shim — canonical location is backend.shared.text_tools."""
from backend.shared.text_tools import *  # noqa: F401,F403
from backend.shared.text_tools import __all__ as _all  # noqa: F401

__all__ = list(_all)