from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request
import backend.config as backend_config


LOCAL_HOSTS = {"127.0.0.1", "::1", "testclient"}


def is_local_request(request: Request) -> bool:
    host = getattr(getattr(request, "client", None), "host", None)
    if not host:
        return False
    return host in LOCAL_HOSTS


def _bearer_token(auth_header: str | None) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def extract_api_key(request: Request) -> Optional[str]:
    token = _bearer_token(request.headers.get("authorization"))
    if token:
        return token
    header_key = request.headers.get("x-api-key")
    if header_key and str(header_key).strip():
        return str(header_key).strip()
    return None


def required_api_key() -> Optional[str]:
    value = os.environ.get("VOCALIE_API_KEY")
    return value.strip() if value and value.strip() else None


def is_authorized(request: Request) -> bool:
    if backend_config.VOCALIE_TRUST_LOCALHOST and is_local_request(request):
        return True
    required = required_api_key()
    if not required:
        return False
    provided = extract_api_key(request)
    if not provided:
        return False
    return hmac.compare_digest(provided, required)


def require_authorized(request: Request) -> None:
    if not is_authorized(request):
        raise HTTPException(status_code=403, detail="forbidden")


def safe_join_under(root: Path, user_path: str) -> Path:
    candidate = Path(user_path).expanduser()
    resolved = candidate.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("path_not_allowed") from exc
    return resolved


def safe_filename(name: str) -> str:
    candidate = str(name or "").strip()
    if not candidate:
        raise ValueError("invalid_name")
    if candidate != Path(candidate).name:
        raise ValueError("invalid_name")
    if ".." in candidate:
        raise ValueError("invalid_name")
    return candidate
