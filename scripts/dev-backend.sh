#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.11 and try again." >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -c "import uvicorn" >/dev/null 2>&1; then
  echo "Installing Python dependencies (requirements.txt)"
  pip install -U pip
  pip install -r requirements.txt
fi

if [[ "${WITH_CHATTERBOX:-0}" == "1" ]]; then
  echo "Installing Chatterbox isolated venv"
  "$ROOT_DIR/scripts/install-chatterbox-venv.sh"
fi

echo "Installing AudioSR isolated venv"
"$ROOT_DIR/scripts/install-audiosr-venv.sh"
AUDIOSR_PY="$ROOT_DIR/.venvs/audiosr/bin/python"
if [[ -x "$AUDIOSR_PY" ]]; then
  if ! "$AUDIOSR_PY" -c "import cog, pyloudnorm; print('ok')"; then
    echo "AudioSR import failed in isolated venv ($AUDIOSR_PY)" >&2
    exit 1
  fi
else
  echo "AudioSR python missing at $AUDIOSR_PY" >&2
  exit 1
fi

API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-127.0.0.1}"

if [[ -z "${VOCALIE_CORS_ORIGINS:-}" ]]; then
  cors_origins="http://localhost:3000,http://127.0.0.1:3000"
  lan_ip=""
  if command -v ipconfig >/dev/null 2>&1; then
    iface=""
    if command -v route >/dev/null 2>&1; then
      iface="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}' | head -n1)"
    fi
    if [[ -n "$iface" ]]; then
      lan_ip="$(ipconfig getifaddr "$iface" 2>/dev/null || true)"
    fi
    if [[ -z "$lan_ip" ]]; then
      lan_ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
    fi
    if [[ -z "$lan_ip" ]]; then
      lan_ip="$(ipconfig getifaddr en1 2>/dev/null || true)"
    fi
    if [[ -z "$lan_ip" ]]; then
      lan_ip="$(ipconfig getifaddr en2 2>/dev/null || true)"
    fi
    if [[ -z "$lan_ip" ]] && command -v ifconfig >/dev/null 2>&1; then
      lan_ip="$(ifconfig 2>/dev/null | awk '/inet /{print $2}' | grep -v '^127\.' | head -n1)"
    fi
  elif command -v hostname >/dev/null 2>&1; then
    lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  if [[ -n "$lan_ip" ]]; then
    cors_origins="${cors_origins},http://${lan_ip}:3000"
  fi
  export VOCALIE_CORS_ORIGINS="$cors_origins"
fi

ASSETS_DIR="$ROOT_DIR/.assets"
uvicorn backend.app:app --reload --reload-exclude "$ASSETS_DIR" --reload-exclude "**/.assets/**" --host "$API_HOST" --port "$API_PORT"
