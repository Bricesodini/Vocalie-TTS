"""Contract tests for backend.services.work_service."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.services.work_service import clean_work_dir


@pytest.fixture
def work_root(tmp_path):
    """Create a temporary work directory with some content."""
    root = tmp_path / "work"
    root.mkdir()

    sessions_dir = root / ".sessions"
    sessions_dir.mkdir()
    session = sessions_dir / "session_001"
    session.mkdir()
    (session / "audio.wav").write_bytes(b"fake-audio")

    tmp_dir = root / ".tmp"
    tmp_dir.mkdir()
    (tmp_dir / "temp_file.raw").write_bytes(b"temp-data")

    return root


def test_clean_work_dir_removes_sessions_and_tmp(work_root):
    result = clean_work_dir(work_root)
    assert result == 1  # one session removed
    assert not (work_root / ".sessions" / "session_001").exists()
    assert not (work_root / ".tmp" / "temp_file.raw").exists()
    # Root still exists
    assert work_root.exists()


def test_clean_work_dir_handles_empty_dir(tmp_path):
    root = tmp_path / "empty_work"
    root.mkdir()
    result = clean_work_dir(root)
    assert result == 0
    assert root.exists()


def test_clean_work_dir_skips_when_keep_env_set(work_root, monkeypatch):
    monkeypatch.setenv("VOCALIE_KEEP_WORK", "1")
    # Preserve existing sessions
    session_file = work_root / ".sessions" / "session_001" / "audio.wav"
    assert session_file.exists()

    result = clean_work_dir(work_root)
    assert result == 0

    # Sessions still exist
    assert session_file.exists()


def test_clean_work_dir_removes_files_in_sessions(work_root):
    # Add a loose file in sessions dir
    (work_root / ".sessions" / "loose_file.json").write_text("{}")
    result = clean_work_dir(work_root)
    assert result == 2  # 1 session dir + 1 loose file


def test_clean_work_dir_handles_alt_tmp(work_root):
    # Create alt tmp dir
    alt_tmp = work_root / "tmp"
    alt_tmp.mkdir()
    (alt_tmp / "alt_temp.raw").write_bytes(b"temp_data_2")

    clean_work_dir(work_root)

    assert not (alt_tmp / "alt_temp.raw").exists()


def test_clean_work_dir_creates_missing_root(tmp_path):
    root = tmp_path / "new_work_dir"
    assert not root.exists()

    result = clean_work_dir(root)
    assert result == 0
    assert root.exists()


def test_clean_work_dir_with_nested_session_dirs(work_root):
    # Create another session
    session2 = work_root / ".sessions" / "session_002"
    session2.mkdir()
    (session2 / "output.wav").write_bytes(b"fake-output")

    result = clean_work_dir(work_root)
    assert result == 2