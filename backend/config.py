from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
API_VERSION = "v1"

MAX_TEXT_CHARS = int(os.environ.get("VOCALIE_MAX_TEXT_CHARS") or "50000")
MAX_CONCURRENT_JOBS = int(os.environ.get("VOCALIE_MAX_CONCURRENT_JOBS") or "2")


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    value = raw.strip()
    if not value:
        return []
    items = [part.strip() for part in value.split(",")]
    return [item for item in items if item]


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:7860",
    "http://127.0.0.1:7860",
]

VOCALIE_CORS_ORIGINS = _parse_csv_env("VOCALIE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
VOCALIE_ALLOWED_HOSTS = _parse_csv_env(
    "VOCALIE_ALLOWED_HOSTS",
    ["127.0.0.1", "localhost", "::1", "testserver"],
)

VOCALIE_RATE_LIMIT_RPS = float(os.environ.get("VOCALIE_RATE_LIMIT_RPS") or "5")
VOCALIE_RATE_LIMIT_BURST = int(os.environ.get("VOCALIE_RATE_LIMIT_BURST") or "10")
VOCALIE_MAX_UPLOAD_BYTES = int(os.environ.get("VOCALIE_MAX_UPLOAD_BYTES") or str(25 * 1024 * 1024))

VOCALIE_BARK_TIMEOUT_S = float(os.environ.get("VOCALIE_BARK_TIMEOUT_S") or "600")
VOCALIE_BARK_SMALL_MODELS = os.environ.get("VOCALIE_BARK_SMALL_MODELS") in {"1", "true", "True", "yes", "YES"}
VOCALIE_ENABLE_AUDIOSR = os.environ.get("VOCALIE_ENABLE_AUDIOSR", "1") in {"1", "true", "True", "yes", "YES"}
VOCALIE_AUDIOSR_TIMEOUT_S = float(os.environ.get("VOCALIE_AUDIOSR_TIMEOUT_S") or "900")
VOCALIE_ENABLE_API_DOCS = _parse_bool_env("VOCALIE_ENABLE_API_DOCS", default=False)
VOCALIE_TRUST_LOCALHOST = _parse_bool_env("VOCALIE_TRUST_LOCALHOST", default=False)
VOCALIE_EXPOSE_SYSTEM_INFO = _parse_bool_env("VOCALIE_EXPOSE_SYSTEM_INFO", default=False)

work_env = os.environ.get("VOCALIE_WORK_DIR")
WORK_DIR = Path(work_env).expanduser() if work_env else BASE_DIR / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

output_env = os.environ.get("VOCALIE_OUTPUT_DIR") or os.environ.get("CHATTERBOX_OUT_DIR")
OUTPUT_DIR = Path(output_env).expanduser() if output_env else BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRESETS_DIR = BASE_DIR / "presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_META_DIR = OUTPUT_DIR / ".assets"
ASSETS_META_DIR.mkdir(parents=True, exist_ok=True)
