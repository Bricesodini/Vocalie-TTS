from __future__ import annotations

import hashlib
import ipaddress
import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from backend.config import VOCALIE_RATE_LIMIT_BURST, VOCALIE_RATE_LIMIT_RPS, VOCALIE_TRUSTED_PROXIES


@dataclass
class Bucket:
    tokens: float
    updated_at: float


_LOCK = threading.Lock()
_BUCKETS: dict[str, Bucket] = {}


def _client_ip(request: Request) -> str:
    host = getattr(getattr(request, "client", None), "host", None)
    return str(host or "unknown")


def _trusted_proxy_set() -> set[str]:
    return {str(host).strip() for host in VOCALIE_TRUSTED_PROXIES if str(host).strip()}


def _is_valid_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _first_forwarded_ip(request: Request) -> str | None:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        for token in xff.split(","):
            candidate = token.strip()
            if candidate and _is_valid_ip(candidate):
                return candidate
    x_real_ip = (request.headers.get("x-real-ip") or "").strip()
    if x_real_ip and _is_valid_ip(x_real_ip):
        return x_real_ip
    return None


def _effective_client_ip(request: Request) -> str:
    peer = _client_ip(request)
    # Trust forwarding headers only when the direct peer is explicitly trusted.
    if peer in _trusted_proxy_set():
        forwarded = _first_forwarded_ip(request)
        if forwarded:
            return forwarded
    return peer


def _request_api_key(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    parts = auth.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    key = (request.headers.get("x-api-key") or "").strip()
    if key:
        return key
    return None


def _bucket_key(request: Request) -> str:
    client_ip = _effective_client_ip(request)
    api_key = _request_api_key(request)
    if not api_key:
        return f"ip:{client_ip}|anon"
    fp = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"ip:{client_ip}|key:{fp}"


def consume(request: Request, *, rps: float, burst: int) -> bool:
    if rps <= 0 or burst <= 0:
        return True
    now = time.monotonic()
    key = _bucket_key(request)
    with _LOCK:
        bucket = _BUCKETS.get(key)
        if bucket is None:
            bucket = Bucket(tokens=float(burst), updated_at=now)
            _BUCKETS[key] = bucket
        elapsed = max(0.0, now - bucket.updated_at)
        bucket.updated_at = now
        bucket.tokens = min(float(burst), bucket.tokens + elapsed * float(rps))
        if bucket.tokens < 1.0:
            return False
        bucket.tokens -= 1.0
        return True


def enforce_heavy(request: Request) -> None:
    ok = consume(
        request,
        rps=float(VOCALIE_RATE_LIMIT_RPS),
        burst=int(VOCALIE_RATE_LIMIT_BURST),
    )
    if not ok:
        raise HTTPException(status_code=429, detail={"error": "rate_limited"})
