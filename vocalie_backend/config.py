"""Resolved configuration for the vocalie-backend CLI.

All defaults are kept here in one place so the Swift app, the shell
scripts, and the CLI all read the same values. Env vars override
defaults (matching backend/config.py's convention).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ── paths ───────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT_DIR / ".run"
LOG_DIR = RUN_DIR  # keep logs next to PID files for now
VENV_DIR = ROOT_DIR / ".venv"
ASSETS_DIR = ROOT_DIR / ".assets"


# ── network ─────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8018"))


# ── process ─────────────────────────────────────────────────────────────
BACKEND_PID_FILE = RUN_DIR / "backend.pid"
BACKEND_LOG_FILE = RUN_DIR / "backend.log"


# ── env defaults applied on start (only if not already set) ─────────────
START_ENV_DEFAULTS: dict[str, str] = {
    # Local dev: trust the loopback bridge. Native runs (Mac app) need
    # this because the Swift app forwards browser requests via the host
    # network stack and 127.0.0.1:8018 is not exposed externally.
    "VOCALIE_TRUST_LOCALHOST": "1",
    # CORS: the Swift app opens the UI in a system browser; the
    # browser origin is http://localhost:<port>.
    "VOCALIE_CORS_ORIGINS": f"http://localhost:3018,http://127.0.0.1:3018",
    # Hosts the TrustedHostMiddleware will accept (incl. 'testserver'
    # so the test suite can hit the API).
    "VOCALIE_ALLOWED_HOSTS": "127.0.0.1,localhost,::1,testserver",
}


def apply_start_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of os.environ with the canonical defaults applied.

    Defaults only fill in keys that are not already set — the operator's
    explicit overrides always win.
    """
    base = os.environ.copy() if env is None else env.copy()
    for k, v in START_ENV_DEFAULTS.items():
        base.setdefault(k, v)
    return base


# ── helpers ─────────────────────────────────────────────────────────────
def ensure_run_dir() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")
