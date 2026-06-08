"""Tests for the vocalie_backend CLI.

We test the pure-Python helpers (config, process.get_state, health)
and the argparse plumbing. We do NOT spawn a real uvicorn here —
the integration test (start a backend, hit /v1/health, stop) is
covered manually and lives in docs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def test_version_flag(capsys):
    from vocalie_backend import __version__
    from vocalie_backend.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert __version__ in captured.out


def test_help_lists_subcommands(capsys):
    from vocalie_backend.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("start", "stop", "status", "health", "install", "doctor", "logs"):
        assert cmd in out


def test_status_when_stopped_emits_json(tmp_path, monkeypatch, capsys):
    # Repoint RUN_DIR into tmp so we don't pollute the real .run/
    from vocalie_backend import config, process

    monkeypatch.setattr(config, "RUN_DIR", tmp_path)
    monkeypatch.setattr(config, "BACKEND_PID_FILE", tmp_path / "backend.pid")
    monkeypatch.setattr(config, "BACKEND_LOG_FILE", tmp_path / "backend.log")
    monkeypatch.setattr(process, "BACKEND_PID_FILE", tmp_path / "backend.pid")
    monkeypatch.setattr(process, "BACKEND_LOG_FILE", tmp_path / "backend.log")
    monkeypatch.setattr(process, "RUN_DIR", tmp_path)

    from vocalie_backend.cli import main
    rc = main(["status", "--json"])
    out = capsys.readouterr().out.strip()
    assert rc == 2  # not running
    data = json.loads(out)
    assert data["running"] is False
    assert data["pid"] is None


def test_start_without_venv_returns_missing_dep(tmp_path, monkeypatch, capsys):
    from vocalie_backend import config, process

    monkeypatch.setattr(config, "RUN_DIR", tmp_path)
    monkeypatch.setattr(config, "BACKEND_PID_FILE", tmp_path / "backend.pid")
    monkeypatch.setattr(config, "BACKEND_LOG_FILE", tmp_path / "backend.log")
    monkeypatch.setattr(config, "VENV_DIR", tmp_path / "no-such-venv")
    monkeypatch.setattr(process, "BACKEND_PID_FILE", tmp_path / "backend.pid")
    monkeypatch.setattr(process, "BACKEND_LOG_FILE", tmp_path / "backend.log")
    monkeypatch.setattr(process, "RUN_DIR", tmp_path)
    monkeypatch.setattr(process, "VENV_DIR", tmp_path / "no-such-venv")

    from vocalie_backend.cli import main
    rc = main(["start", "--port", "9999"])
    captured = capsys.readouterr()
    assert rc == 4  # EXIT_MISSING_DEP
    assert "venv python not found" in captured.err


def test_health_when_nothing_listening(monkeypatch, capsys):
    from vocalie_backend import health

    # Point at a definitely-closed port.
    monkeypatch.setattr(health, "API_HOST", "127.0.0.1")
    monkeypatch.setattr(health, "API_PORT", 1)
    from vocalie_backend.cli import main
    rc = main(["health", "--json", "--timeout", "0.3"])
    out = capsys.readouterr().out.strip()
    assert rc == 1
    data = json.loads(out)
    assert data["ok"] is False
    assert data["http_status"] is None


def test_doctor_reports_local_env(capsys):
    from vocalie_backend.cli import main
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "summary:" in out


def test_stop_is_idempotent(tmp_path, monkeypatch, capsys):
    from vocalie_backend import config, process

    monkeypatch.setattr(config, "RUN_DIR", tmp_path)
    monkeypatch.setattr(config, "BACKEND_PID_FILE", tmp_path / "backend.pid")
    monkeypatch.setattr(config, "BACKEND_LOG_FILE", tmp_path / "backend.log")
    monkeypatch.setattr(process, "BACKEND_PID_FILE", tmp_path / "backend.pid")
    monkeypatch.setattr(process, "BACKEND_LOG_FILE", tmp_path / "backend.log")
    monkeypatch.setattr(process, "RUN_DIR", tmp_path)

    from vocalie_backend.cli import main
    # No PID file at all: stop is a no-op and must succeed.
    rc = main(["stop"])
    assert rc == 0
