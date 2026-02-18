#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_BASE="${API_BASE:-http://127.0.0.1:8018}"
API_KEY="${API_KEY:-}"
HEALTH_URL="$API_BASE/v1/health"
ENGINES_URL="$API_BASE/v1/tts/engines"
VOICES_URL="$API_BASE/v1/tts/voices?engine=chatterbox_native"
CAPABILITIES_URL="$API_BASE/v1/capabilities"
STARTED=0
PID=""

curl_with_auth() {
  if [[ -n "$API_KEY" ]]; then
    curl "$@" -H "X-API-Key: $API_KEY"
  else
    curl "$@"
  fi
}

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
  if [[ "$API_BASE" != "http://127.0.0.1:8018" ]]; then
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
  uvicorn backend.app:app --host 127.0.0.1 --port 8018 >"$LOG_PATH" 2>&1 &
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

engines_body="$(mktemp)"
engines_status="$(curl_with_auth -sS -o "$engines_body" -w "%{http_code}" "$ENGINES_URL" || true)"
if [[ "$engines_status" == "403" ]]; then
  echo "WARN: /v1/tts/engines returned 403 (API key required). Set API_KEY to validate engine catalog." >&2
elif [[ "$engines_status" =~ ^2 ]]; then
engines_payload="$(cat "$engines_body")"
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
else
  echo "Engine catalog check failed with HTTP $engines_status" >&2
  rm -f "$engines_body"
  exit 1
fi
rm -f "$engines_body"

voices_body="$(mktemp)"
voices_status="$(curl_with_auth -sS -o "$voices_body" -w "%{http_code}" "$VOICES_URL" || true)"
if [[ "$voices_status" == "403" ]]; then
  echo "WARN: /v1/tts/voices returned 403 (API key required). Set API_KEY to validate voices." >&2
elif [[ "$voices_status" =~ ^2 ]]; then
voices_payload="$(cat "$voices_body")"
VOICES_PAYLOAD="$voices_payload" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["VOICES_PAYLOAD"])
voices = payload.get("voices")
if not isinstance(voices, list) or not voices:
    sys.exit("voices list missing/empty")
PY
else
  echo "Voices check failed with HTTP $voices_status" >&2
  rm -f "$voices_body"
  exit 1
fi
rm -f "$voices_body"

cap_payload=""
cap_body="$(mktemp)"
cap_status="$(curl_with_auth -sS -o "$cap_body" -w "%{http_code}" "$CAPABILITIES_URL" || true)"
if [[ "$cap_status" == "403" ]]; then
  echo "WARN: /v1/capabilities returned 403 (API key required). Set API_KEY to validate AudioSR capability." >&2
elif [[ "$cap_status" =~ ^2 ]]; then
cap_payload="$(cat "$cap_body")"
CAP_PAYLOAD="$cap_payload" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["CAP_PAYLOAD"])
if "audiosr" not in payload:
    sys.exit("capabilities missing audiosr key")
PY
else
  echo "Capabilities check failed with HTTP $cap_status" >&2
  rm -f "$cap_body"
  exit 1
fi
rm -f "$cap_body"

audiosr_available="0"
if [[ -n "$cap_payload" ]]; then
audiosr_available="$(CAP_PAYLOAD="$cap_payload" python3 - <<'PY'
import json, os
payload = json.loads(os.environ["CAP_PAYLOAD"])
audiosr = payload.get("audiosr") or {}
print("1" if audiosr.get("available") else "0")
PY
)"
fi

if [[ "$audiosr_available" == "1" ]]; then
  sample_path="$ROOT_DIR/tests/assets/audiosr_sample.wav"
  if [[ ! -f "$sample_path" ]]; then
    echo "Missing AudioSR sample wav at $sample_path" >&2
    exit 1
  fi
  enhance_json="$(curl_with_auth -fsS -X POST "$API_BASE/v1/audio/enhance" -F "file=@$sample_path" -F "engine=audiosr")"
  ENHANCE_JSON="$enhance_json" python3 - <<'PY'
import json, os, sys
payload = json.loads(os.environ["ENHANCE_JSON"])
if int(payload.get("sample_rate") or 0) != 48000:
    sys.exit("audiosr sample_rate not 48000")
output_file = payload.get("output_file")
if not output_file:
    sys.exit("audiosr output_file missing")
if not os.path.exists(output_file):
    sys.exit(f"audiosr output_file missing on disk: {output_file}")
PY
fi

schema_body="$(mktemp)"
schema_status="$(curl_with_auth -sS -o "$schema_body" -w "%{http_code}" "$API_BASE/v1/tts/engine_schema?engine=bark" || true)"
if [[ "$schema_status" == "403" ]]; then
  echo "WARN: /v1/tts/engine_schema returned 403 (API key required). Set API_KEY to validate schema." >&2
elif [[ "$schema_status" =~ ^2 ]]; then
schema_payload="$(cat "$schema_body")"
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
else
  echo "Engine schema check failed with HTTP $schema_status" >&2
  rm -f "$schema_body"
  exit 1
fi
rm -f "$schema_body"

if [[ "${SMOKE_BARK:-0}" == "1" ]]; then
  job_payload='{"engine":"bark","text":"Hello from Bark.","direction":{"enabled":false},"options":{"voice_preset":"v2/en_speaker_6","text_temp":0.7,"waveform_temp":0.7,"seed":0,"device":"cpu"}}'
  job_json="$(curl_with_auth -fsS -X POST "$API_BASE/v1/tts/jobs" -H 'Content-Type: application/json' -d "$job_payload")"
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
    status_json="$(curl_with_auth -fsS "$API_BASE/v1/jobs/$job_id")" || true
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
