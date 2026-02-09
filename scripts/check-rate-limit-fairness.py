#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

from starlette.requests import Request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backend.rate_limit as rate_limit


def make_request(*, peer: str, xff: str | None = None, api_key: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if xff:
        headers.append((b"x-forwarded-for", xff.encode("ascii")))
    if api_key:
        headers.append((b"x-api-key", api_key.encode("ascii")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/tts/jobs",
        "raw_path": b"/v1/tts/jobs",
        "query_string": b"",
        "headers": headers,
        "client": (peer, 12345),
        "server": ("testserver", 80),
    }
    return Request(scope)


def run_batch(requests: list[Request], *, rps: float, burst: int) -> list[bool]:
    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        futures = [pool.submit(rate_limit.consume, req, rps=rps, burst=burst) for req in requests]
    return [f.result() for f in futures]


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    # Scenario A: trusted proxy + XFF -> each client IP gets its own bucket.
    rate_limit._BUCKETS.clear()
    rate_limit.VOCALIE_TRUSTED_PROXIES = ["testclient"]
    reqs_a = [
        make_request(peer="testclient", xff="203.0.113.10", api_key="shared-key"),
        make_request(peer="testclient", xff="203.0.113.11", api_key="shared-key"),
        make_request(peer="testclient", xff="203.0.113.12", api_key="shared-key"),
    ]
    wave1 = run_batch(reqs_a, rps=0.01, burst=1)
    wave2 = run_batch(reqs_a, rps=0.01, burst=1)
    assert_true(all(wave1), f"scenario A wave1 expected all pass, got {wave1}")
    assert_true(not any(wave2), f"scenario A wave2 expected all rate-limited, got {wave2}")

    # Scenario B: same IP + different API keys -> isolated buckets by key fingerprint.
    rate_limit._BUCKETS.clear()
    rate_limit.VOCALIE_TRUSTED_PROXIES = []
    reqs_b = [
        make_request(peer="198.18.0.20", api_key="alpha"),
        make_request(peer="198.18.0.20", api_key="beta"),
        make_request(peer="198.18.0.20", api_key="gamma"),
    ]
    wave1 = run_batch(reqs_b, rps=0.01, burst=1)
    wave2 = run_batch(reqs_b, rps=0.01, burst=1)
    assert_true(all(wave1), f"scenario B wave1 expected all pass, got {wave1}")
    assert_true(not any(wave2), f"scenario B wave2 expected all rate-limited, got {wave2}")

    print("OK: rate-limit fairness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
