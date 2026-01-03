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
ids = {e.get("id") for e in engines if isinstance(e, dict)}
if "bark" not in ids:
    sys.exit("bark engine missing from catalog")
PY

voices_payload="$(curl -fsS "$VOICES_URL")"
VOICES_PAYLOAD="$voices_payload" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["VOICES_PAYLOAD"])
voices = payload.get("voices")
if not isinstance(voices, list) or not voices:
    sys.exit("voices list missing/empty")
PY

schema_payload="$(curl -fsS \"$API_BASE/v1/tts/engine_schema?engine=bark\")"
SCHEMA_PAYLOAD="$schema_payload" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["SCHEMA_PAYLOAD"])
fields = payload.get("fields") or []
keys = {f.get("key") for f in fields if isinstance(f, dict)}
required = {"voice_preset", "text_temp", "waveform_temp", "seed", "device"}
missing = required - keys
if missing:
    sys.exit(f"bark schema missing: {sorted(missing)}")
PY

if [[ "${SMOKE_BARK:-0}" == "1" ]]; then
  job_payload='{"engine":"bark","text":"Hello from Bark.","direction":{"enabled":false},"options":{"voice_preset":"v2/en_speaker_6","text_temp":0.7,"waveform_temp":0.7,"seed":0,"device":"cpu"}}'
  job_json="$(curl -fsS -X POST "$API_BASE/v1/tts/jobs" -H 'Content-Type: application/json' -d "$job_payload")"
  job_id="$(JOB_JSON="$job_json" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["JOB_JSON"])
job_id = payload.get("job_id")
if not job_id:
    sys.exit("missing job_id for bark")
print(job_id)
PY
)"
  for _ in {1..180}; do
    status_json="$(curl -fsS "$API_BASE/v1/jobs/$job_id")" || true
    status="$(STATUS_JSON="$status_json" python3 - <<'PY'
import json, os
payload = json.loads(os.environ.get("STATUS_JSON") or "{}")
print(payload.get("status") or "")
PY
)"
    if [[ "$status" == "done" ]]; then
      break
    fi
    if [[ "$status" == "error" || "$status" == "canceled" ]]; then
      echo "Bark job failed: $status_json" >&2
      exit 1
    fi
    sleep 0.5
  done
fi

echo "Smoke tests OK"
