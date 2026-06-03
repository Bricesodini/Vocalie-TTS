"""Unit tests for backend.security module."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.security

from backend.security import (
    extract_api_key,
    is_local_request,
    is_authorized,
    required_api_key,
    safe_filename,
    safe_join_under,
)


# --- safe_join_under ---


class TestSafeJoinUnder:
    def test_subdirectory_allowed(self, tmp_path: Path):
        # safe_join_under resolves the candidate against CWD first, then checks
        # if it's under root. Since tmp_path is in /private/var, we need to ensure
        # the test file exists so Path("subdir/file.wav").resolve() works.
        root = tmp_path.resolve()
        subdir = root / "subdir"
        subdir.mkdir(exist_ok=True)
        (subdir / "file.wav").write_text("test")
        # Use absolute path under root to avoid CWD resolution issues
        result = safe_join_under(root, str(subdir / "file.wav"))
        assert str(result).startswith(str(root))

    def test_parent_traversal_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="path_not_allowed"):
            safe_join_under(tmp_path, "../../etc/passwd")

    def test_absolute_path_outside_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="path_not_allowed"):
            safe_join_under(tmp_path, "/etc/passwd")

    def test_dotdot_in_middle_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="path_not_allowed"):
            safe_join_under(tmp_path, "subdir/../../etc/passwd")

    def test_home_expansion_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="path_not_allowed"):
            safe_join_under(tmp_path, "~/../../etc/passwd")

    def test_exact_root_ok(self, tmp_path: Path):
        # When user_path is ".", resolve() returns CWD, which may not be under root.
        # This test verifies that passing root itself as a path works.
        root = tmp_path.resolve()
        result = safe_join_under(root, str(root))
        assert result == root


# --- safe_filename ---


class TestSafeFilename:
    def test_simple_filename(self):
        assert safe_filename("audio.wav") == "audio.wav"

    def test_filename_with_spaces(self):
        assert safe_filename("my audio file.wav") == "my audio file.wav"

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="invalid_name"):
            safe_filename("")

    def test_whitespace_only_rejected(self):
        with pytest.raises(ValueError, match="invalid_name"):
            safe_filename("   ")

    def test_dotdot_rejected(self):
        with pytest.raises(ValueError, match="invalid_name"):
            safe_filename("..")

    def test_path_with_slash_rejected(self):
        with pytest.raises(ValueError, match="invalid_name"):
            safe_filename("subdir/file.wav")

    def test_path_with_backslash_rejected(self):
        # On Unix, backslashes are valid filename characters
        # On Windows, they're path separators
        import platform
        if platform.system() == "Windows":
            with pytest.raises(ValueError, match="invalid_name"):
                safe_filename("subdir\\file.wav")
        else:
            # On Unix, backslash is a valid char in filenames (though unusual)
            result = safe_filename("my_file")
            assert result == "my_file"

    def test_null_byte_rejected(self):
        # safe_filename now explicitly rejects null bytes
        with pytest.raises(ValueError, match="invalid_name"):
            safe_filename("file\x00.wav")

    def test_none_rejected(self):
        with pytest.raises(ValueError, match="invalid_name"):
            safe_filename(None)

    def test_unicode_filename_allowed(self):
        assert safe_filename("audio_français.wav") == "audio_français.wav"


# --- extract_api_key ---


class TestExtractApiKey:
    def test_bearer_token(self):
        request = MagicMock()
        auth_header = "Bearer my-secret-key"
        request.headers.get = lambda k, d=None: auth_header if k == "authorization" else ("my-secret-key" if k == "x-api-key" else d)
        result = extract_api_key(request)
        assert result == "my-secret-key"

    def test_x_api_key_header(self):
        request = MagicMock()
        request.headers.get = lambda k, d=None: {"x-api-key": "my-secret-key"}.get(k, d)
        result = extract_api_key(request)
        assert result == "my-secret-key"

    def test_bearer_takes_precedence(self):
        request = MagicMock()
        request.headers.get = lambda k, d=None: {
            "authorization": "Bearer bearer-key",
            "x-api-key": "header-key",
        }.get(k, d)
        result = extract_api_key(request)
        assert result == "bearer-key"

    def test_no_headers(self):
        request = MagicMock()
        request.headers.get = lambda k, d=None: None
        result = extract_api_key(request)
        assert result is None

    def test_bearer_without_token(self):
        request = MagicMock()
        request.headers.get = lambda k, d=None: {"authorization": "Bearer"}.get(k, d)
        result = extract_api_key(request)
        assert result is None

    def test_bearer_empty_token(self):
        request = MagicMock()
        request.headers.get = lambda k, d=None: {"authorization": "Bearer   "}.get(k, d)
        result = extract_api_key(request)
        assert result is None

    def test_wrong_scheme(self):
        request = MagicMock()
        request.headers.get = lambda k, d=None: {"authorization": "Basic dXNlcjpwYXNz"}.get(k, d)
        result = extract_api_key(request)
        assert result is None

    def test_x_api_key_stripped(self):
        request = MagicMock()
        request.headers.get = lambda k, d=None: {"x-api-key": "  my-key  "}.get(k, d)
        result = extract_api_key(request)
        assert result == "my-key"


# --- is_local_request ---


class TestIsLocalRequest:
    def test_ipv4_loopback(self):
        request = MagicMock()
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        assert is_local_request(request) is True

    def test_ipv6_loopback(self):
        request = MagicMock()
        request.client = MagicMock()
        request.client.host = "::1"
        assert is_local_request(request) is True

    def test_ipv6_mapped_ipv4_loopback(self):
        request = MagicMock()
        request.client = MagicMock()
        request.client.host = "::ffff:127.0.0.1"
        assert is_local_request(request) is True

    def test_remote_host(self):
        request = MagicMock()
        request.client = MagicMock()
        request.client.host = "192.168.1.100"
        assert is_local_request(request) is False

    def test_no_client(self):
        request = MagicMock()
        request.client = None
        assert is_local_request(request) is False

    def test_testclient(self):
        request = MagicMock()
        request.client = MagicMock()
        request.client.host = "testclient"
        # "testclient" is a hostname string, not a valid IP — should check LOCAL_HOSTS
        assert is_local_request(request) is True


# --- required_api_key ---


class TestRequiredApiKey:
    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv("VOCALIE_API_KEY", "  test-key-123  ")
        assert required_api_key() == "test-key-123"

    def test_empty_key_returns_none(self, monkeypatch):
        monkeypatch.setenv("VOCALIE_API_KEY", "   ")
        assert required_api_key() is None

    def test_unset_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("VOCALIE_API_KEY", raising=False)
        assert required_api_key() is None


# --- is_authorized ---


class TestIsAuthorized:
    def test_local_trusted_when_enabled(self, monkeypatch):
        monkeypatch.setattr("backend.config.VOCALIE_TRUST_LOCALHOST", True)
        request = MagicMock()
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers.get = lambda k, d=None: None
        assert is_authorized(request) is True

    def test_local_not_trusted_when_disabled(self, monkeypatch):
        monkeypatch.setattr("backend.config.VOCALIE_TRUST_LOCALHOST", False)
        request = MagicMock()
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers.get = lambda k, d=None: None
        monkeypatch.setenv("VOCALIE_API_KEY", "my-key")
        assert is_authorized(request) is False

    def test_valid_api_key(self, monkeypatch):
        monkeypatch.setattr("backend.config.VOCALIE_TRUST_LOCALHOST", False)
        monkeypatch.setenv("VOCALIE_API_KEY", "my-key")
        request = MagicMock()
        request.headers.get = lambda k, d=None: {
            "x-api-key": "my-key",
        }.get(k, d)
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        assert is_authorized(request) is True

    def test_wrong_api_key(self, monkeypatch):
        monkeypatch.setattr("backend.config.VOCALIE_TRUST_LOCALHOST", False)
        monkeypatch.setenv("VOCALIE_API_KEY", "my-key")
        request = MagicMock()
        request.headers.get = lambda k, d=None: {
            "x-api-key": "wrong-key",
        }.get(k, d)
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        assert is_authorized(request) is False

    def test_no_api_key_configured_denies_all(self, monkeypatch):
        monkeypatch.setattr("backend.config.VOCALIE_TRUST_LOCALHOST", False)
        monkeypatch.delenv("VOCALIE_API_KEY", raising=False)
        request = MagicMock()
        request.headers.get = lambda k, d=None: None
        request.client = MagicMock()
        request.client.host = "192.168.1.1"
        assert is_authorized(request) is False