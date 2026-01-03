from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from backend.config import VOCALIE_RATE_LIMIT_BURST, VOCALIE_RATE_LIMIT_RPS


@dataclass
class Bucket:
    tokens: float
    updated_at: float


_LOCK = threading.Lock()
_BUCKETS: dict[str, Bucket] = {}


def _client_ip(request: Request) -> str:
    host = getattr(getattr(request, "client", None), "host", None)
    return str(host or "unknown")


def consume(request: Request, *, rps: float, burst: int) -> bool:
    if rps <= 0 or burst <= 0:
        return True
    now = time.monotonic()
    key = _client_ip(request)
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

