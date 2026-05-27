"""Contract tests for backend.services.asset_service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.asset_service import (
    get_asset_meta,
    resolve_asset_path,
    write_asset_meta,
)


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    """Set up a temporary ASSETS_META_DIR and OUTPUT_DIR."""
    from backend import config

    meta_dir = tmp_path / "output" / ".assets"
    output_dir = tmp_path / "output"
    meta_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "ASSETS_META_DIR", meta_dir)
    monkeypatch.setattr(config, "OUTPUT_DIR", output_dir)

    # Also patch the module-level imports in asset_service
    import backend.services.asset_service as svc
    monkeypatch.setattr(svc, "ASSETS_META_DIR", meta_dir)
    monkeypatch.setattr(svc, "OUTPUT_DIR", output_dir)

    return meta_dir, output_dir


def test_write_asset_meta_creates_file(assets_dir):
    meta_dir, _ = assets_dir
    payload = {"file_name": "test.wav", "status": "done"}
    result = write_asset_meta("job_abc123", payload)

    assert result["asset_id"] == "job_abc123"
    assert "created_at" in result
    assert result["file_name"] == "test.wav"
    assert result["status"] == "done"

    meta_file = meta_dir / "job_abc123.json"
    assert meta_file.exists()
    data = json.loads(meta_file.read_text(encoding="utf-8"))
    assert data["asset_id"] == "job_abc123"


def test_write_asset_meta_preserves_created_at(assets_dir):
    write_asset_meta("job_preserve", {"status": "queued"})
    # Write again with existing created_at — should preserve original
    meta = get_asset_meta("job_preserve")
    assert meta is not None
    original_created = meta["created_at"]

    result = write_asset_meta("job_preserve", {"status": "running", "created_at": original_created})
    assert result["created_at"] == original_created


def test_write_asset_meta_sets_created_at_if_missing(assets_dir):
    payload = {"status": "done"}
    result = write_asset_meta("job_new", payload)
    assert "created_at" in result
    assert result["created_at"]  # non-empty string


def test_get_asset_meta_not_found(assets_dir):
    result = get_asset_meta("nonexistent")
    assert result is None


def test_get_asset_meta_found(assets_dir):
    write_asset_meta("job_found", {"file_name": "out.wav"})
    result = get_asset_meta("job_found")
    assert result is not None
    assert result["file_name"] == "out.wav"
    assert result["asset_id"] == "job_found"


def test_resolve_asset_path_by_relative_path(assets_dir):
    _, output_dir = assets_dir
    # Create an actual output file
    out_file = output_dir / "2025" / "test_output.wav"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(b"RIFF")

    meta = {"relative_path": "2025/test_output.wav"}
    path = resolve_asset_path(meta)
    assert path is not None
    assert path.name == "test_output.wav"


def test_resolve_asset_path_by_file_name(assets_dir):
    _, output_dir = assets_dir
    out_file = output_dir / "direct_file.wav"
    out_file.write_bytes(b"RIFF")

    meta = {"file_name": "direct_file.wav"}
    path = resolve_asset_path(meta)
    assert path is not None
    assert path.name == "direct_file.wav"


def test_resolve_asset_path_not_found(assets_dir):
    meta = {"relative_path": "does_not_exist.wav", "file_name": "also_missing.wav"}
    path = resolve_asset_path(meta)
    assert path is None


def test_resolve_asset_path_no_path_keys(assets_dir):
    meta = {"status": "done"}
    path = resolve_asset_path(meta)
    assert path is None