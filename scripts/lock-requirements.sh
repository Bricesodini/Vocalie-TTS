#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_VENV="$ROOT_DIR/.venv"
CHATTERBOX_VENV="$ROOT_DIR/.venvs/chatterbox"
BARK_VENV="$ROOT_DIR/.venvs/bark"
AUDIOSR_VENV="$ROOT_DIR/.venvs/audiosr"

if [[ ! -d "$CORE_VENV" ]]; then
  echo "Missing core venv at $CORE_VENV. Run ./scripts/bootstrap.sh min first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$CORE_VENV/bin/activate"
python -m pip freeze > "$ROOT_DIR/requirements.lock.txt"
deactivate || true

if [[ -d "$CHATTERBOX_VENV" ]]; then
  # shellcheck disable=SC1091
  source "$CHATTERBOX_VENV/bin/activate"
  python -m pip freeze > "$ROOT_DIR/requirements-chatterbox.lock.txt"
  deactivate || true
fi

if [[ -d "$BARK_VENV" ]]; then
  # shellcheck disable=SC1091
  source "$BARK_VENV/bin/activate"
  python -m pip freeze > "$ROOT_DIR/requirements-bark.lock.txt"
  deactivate || true
fi

if [[ -d "$AUDIOSR_VENV" ]]; then
  # shellcheck disable=SC1091
  source "$AUDIOSR_VENV/bin/activate"
  python -m pip freeze > "$ROOT_DIR/requirements-audiosr.lock.txt"
  deactivate || true
fi

echo "Wrote requirements.lock.txt"
if [[ -f "$ROOT_DIR/requirements-chatterbox.lock.txt" ]]; then
  echo "Wrote requirements-chatterbox.lock.txt"
fi
if [[ -f "$ROOT_DIR/requirements-bark.lock.txt" ]]; then
  echo "Wrote requirements-bark.lock.txt"
fi
if [[ -f "$ROOT_DIR/requirements-audiosr.lock.txt" ]]; then
  echo "Wrote requirements-audiosr.lock.txt"
fi
