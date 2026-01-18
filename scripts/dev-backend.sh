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

if [[ "${VOCALIE_ENABLE_AUDIOSR:-0}" == "1" ]]; then
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
fi

API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-127.0.0.1}"

uvicorn backend.app:app --reload --host "$API_HOST" --port "$API_PORT"
