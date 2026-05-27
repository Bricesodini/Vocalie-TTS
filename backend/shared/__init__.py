"""Shared modules between canonical backend and compatibility surfaces.

Canonical backend (``backend/routes/*``, ``backend/services/*``) should import
from ``backend.shared`` rather than from root-level modules.  Root-level
shims re-export from here for backward compatibility.
"""