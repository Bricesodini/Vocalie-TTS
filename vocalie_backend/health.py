"""HTTP health check helper used by the CLI and the Swift app."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from vocalie_backend.config import API_HOST, API_PORT


@dataclass
class HealthResult:
    ok: bool
    http_status: Optional[int]
    latency_ms: Optional[float]
    body: Optional[dict]
    error: Optional[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": self.ok,
                "http_status": self.http_status,
                "latency_ms": self.latency_ms,
                "body": self.body,
                "error": self.error,
            },
            indent=2,
            ensure_ascii=True,
        )


def check(timeout_s: float = 2.0) -> HealthResult:
    url = f"http://{API_HOST}:{API_PORT}/v1/health"
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            raw = r.read()
            latency = (time.perf_counter() - started) * 1000.0
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                body = None
            return HealthResult(
                ok=(r.status == 200),
                http_status=r.status,
                latency_ms=round(latency, 2),
                body=body,
                error=None,
            )
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - started) * 1000.0
        return HealthResult(
            ok=False,
            http_status=e.code,
            latency_ms=round(latency, 2),
            body=None,
            error=f"http_error: {e.reason}",
        )
    except (urllib.error.URLError, ConnectionRefusedError, OSError, TimeoutError) as e:
        latency = (time.perf_counter() - started) * 1000.0
        return HealthResult(
            ok=False,
            http_status=None,
            latency_ms=round(latency, 2),
            body=None,
            error=str(e) or type(e).__name__,
        )
