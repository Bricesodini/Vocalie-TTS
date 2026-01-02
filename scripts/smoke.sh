#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
HEALTH_URL="$API_BASE/v1/health"
ENGINES_URL="$API_BASE/v1/tts/engines"
VOICES_URL="$API_BASE/v1/tts/voices?engine=chatterbox_native"
STARTED=0
PID=""

cleanup() {
  if [[ "$STARTED" -eq 1 && -n "$PID" ]]; then
    kill "$PID" >/dev/null 2>&1 || true
    wait "$PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

curl_ok() {
  curl -fsS --max-time 2 "$1" >/dev/null 2>&1
}

if ! curl_ok "$HEALTH_URL"; then
  if [[ "$API_BASE" != "http://127.0.0.1:8000" ]]; then
    echo "Health check failed for $API_BASE. Set API_BASE to a running server or start one locally." >&2
    exit 1
  fi
  if [[ ! -d "$ROOT_DIR/.venv" ]]; then
    echo ".venv missing. Run ./scripts/bootstrap.sh min first." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
  LOG_PATH="$ROOT_DIR/work/.tmp/smoke_uvicorn.log"
  mkdir -p "$(dirname "$LOG_PATH")"
  uvicorn backend.app:app --host 127.0.0.1 --port 8000 >"$LOG_PATH" 2>&1 &
  PID="$!"
  STARTED=1
  for _ in {1..30}; do
    if curl_ok "$HEALTH_URL"; then
      break
    fi
    sleep 0.5
  done
  if ! curl_ok "$HEALTH_URL"; then
    echo "Health check failed after startup. See $LOG_PATH" >&2
    exit 1
  fi
fi

health_payload="$(curl -fsS "$HEALTH_URL")"
HEALTH_PAYLOAD="$health_payload" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["HEALTH_PAYLOAD"])
if payload.get("status") != "ok":
    sys.exit("health status not ok")
PY

engines_payload="$(curl -fsS "$ENGINES_URL")"
ENGINES_PAYLOAD="$engines_payload" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["ENGINES_PAYLOAD"])
engines = payload.get("engines")
if not isinstance(engines, list) or not engines:
    sys.exit("engines list missing/empty")
PY

voices_payload="$(curl -fsS "$VOICES_URL")"
VOICES_PAYLOAD="$voices_payload" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["VOICES_PAYLOAD"])
voices = payload.get("voices")
if not isinstance(voices, list) or not voices:
    sys.exit("voices list missing/empty")
PY

echo "Smoke tests OK"
