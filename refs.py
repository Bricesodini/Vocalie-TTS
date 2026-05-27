"""Compatibility shim — canonical location is backend.shared.refs."""
from backend.shared.refs import *  # noqa: F401,F403
from backend.shared.refs import __all__ as _all  # noqa: F401

__all__ = list(_all)