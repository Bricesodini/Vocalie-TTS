"""Tests for audio route path validation and upload security."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.security

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("backend.config.VOCALIE_TRUST_LOCALHOST", True):
        from backend.app import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestResolveSafePath:
    """Test _resolve_safe_path security via the audio edit endpoint."""

    def test_reject_path_traversal(self, client):
        """Ensure path traversal attempts are rejected."""
        # This test uses the audio edit endpoint with a malicious path
        response = client.post(
            "/v1/audio/edit",
            json={"input_wav_path": "../../etc/passwd"},
        )
        # Should be 400 (path_not_allowed) or 403 (if auth fails first)
        assert response.status_code in {400, 403}

    def test_reject_absolute_path(self, client):
        """Ensure absolute paths outside allowed dirs are rejected."""
        response = client.post(
            "/v1/audio/edit",
            json={"input_wav_path": "/etc/passwd"},
        )
        assert response.status_code in {400, 403}


class TestUploadValidation:
    """Test upload MIME type and extension validation."""

    def test_reject_py_file(self, client):
        """Reject .py file uploads."""
        response = client.post(
            "/v1/audio/enhance",
            files={"file": ("test.py", io.BytesIO(b"print('hello')"), "text/x-python")},
        )
        assert response.status_code == 415

    def test_reject_html_file(self, client):
        """Reject .html file uploads."""
        response = client.post(
            "/v1/audio/enhance",
            files={"file": ("test.html", io.BytesIO(b"<html></html>"), "text/html")},
        )
        assert response.status_code == 415


class TestSafeFilenameSecurity:
    """Test safe_filename from security module."""

    def test_reject_long_filename(self):
        from backend.security import safe_filename

        with pytest.raises(ValueError, match="invalid_name"):
            safe_filename("a" * 200)

    def test_reject_path_separator(self):
        from backend.security import safe_filename

        with pytest.raises(ValueError):
            safe_filename("sub/file.wav")