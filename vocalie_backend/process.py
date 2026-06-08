"""Process management: start, stop, and query the backend.

The backend is a uvicorn process launched with the project's
``.venv`` interpreter. PID and log live in ``.run/`` so a stale
process can be cleaned up on next start and so the operator can
``tail -f .run/backend.log`` if something goes wrong.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from vocalie_backend.config import (
    API_HOST,
    API_PORT,
    BACKEND_LOG_FILE,
    BACKEND_PID_FILE,
    RUN_DIR,
    VENV_DIR,
    apply_start_env,
    ensure_run_dir,
)


@dataclass
class BackendState:
    """Snapshot of the backend process — JSON-serialisable for the Swift app."""

    running: bool
    pid: Optional[int]
    pid_alive: bool
    host: str
    port: int
    started_at: Optional[float]  # POSIX seconds, or None
    log_file: str
    pid_file: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=True)


def _read_pid() -> Optional[int]:
    if not BACKEND_PID_FILE.exists():
        return None
    try:
        raw = BACKEND_PID_FILE.read_text(encoding="utf-8").strip()
        return int(raw) if raw else None
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user — treat as alive.
        return True
    return True


def get_state() -> BackendState:
    ensure_run_dir()
    pid = _read_pid()
    alive = bool(pid and _pid_alive(pid))
    started: Optional[float] = None
    if alive and pid:
        try:
            stat = os.stat(f"/proc/{pid}" if Path("/proc").exists() else BACKEND_PID_FILE)
            started = stat.st_ctime
        except OSError:
            started = None
    return BackendState(
        running=alive,
        pid=pid,
        pid_alive=alive,
        host=API_HOST,
        port=API_PORT,
        started_at=started,
        log_file=str(BACKEND_LOG_FILE),
        pid_file=str(BACKEND_PID_FILE),
    )


def port_in_use(host: str = API_HOST, port: int = API_PORT) -> bool:
    """Best-effort: is anything listening on host:port?"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        try:
            s.connect((host, port))
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False
        return True


def start(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
) -> BackendState:
    """Spawn uvicorn in the background. Returns the new state.

    Raises RuntimeError on port conflict or missing venv.
    """
    ensure_run_dir()
    host = host or API_HOST
    port = port or API_PORT

    state = get_state()
    if state.running:
        raise RuntimeError(
            f"backend already running (pid={state.pid}); use 'stop' first"
        )
    if port_in_use(host, port):
        raise RuntimeError(
            f"port {port} already in use; another process is listening"
        )
    python = VENV_DIR / "bin" / "python"
    if not python.exists():
        raise RuntimeError(
            f"venv python not found at {python}; run 'vocalie-backend install' first"
        )

    env = apply_start_env()
    # Pass explicit host/port through to uvicorn.
    env["API_HOST"] = host
    env["API_PORT"] = str(port)

    cmd = [
        str(python),
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host", host,
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")
        cmd += [
            "--reload-exclude", str(RUN_DIR.parent / ".assets"),
        ]

    log_fp = open(BACKEND_LOG_FILE, "ab", buffering=0)
    log_fp.write(f"\n--- start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n".encode())
    try:
        proc = subprocess.Popen(  # noqa: S603 — argv is controlled
            cmd,
            cwd=str(RUN_DIR.parent),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from current session
        )
    finally:
        log_fp.close()

    BACKEND_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    return get_state()


def stop(*, timeout_s: float = 5.0, force: bool = False) -> BackendState:
    """Stop the backend if it's running. Idempotent."""
    state = get_state()
    if not state.running or not state.pid:
        # Clean up stale PID file if any.
        BACKEND_PID_FILE.unlink(missing_ok=True)
        return get_state()
    pid = state.pid
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        BACKEND_PID_FILE.unlink(missing_ok=True)
        return get_state()
    # Wait for graceful shutdown.
    deadline = time.time() + timeout_s
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    if _pid_alive(pid) and force:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
    BACKEND_PID_FILE.unlink(missing_ok=True)
    return get_state()


def wait_ready(timeout_s: float = 30.0) -> bool:
    """Block until /v1/health returns 200 or timeout."""
    import urllib.request
    import urllib.error
    url = f"http://{API_HOST}:{API_PORT}/v1/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False
