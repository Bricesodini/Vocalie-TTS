#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_VENV="$ROOT_DIR/.venv"
CHATTERBOX_VENV="$ROOT_DIR/.venvs/chatterbox"
AUDIOSR_VENV="$ROOT_DIR/.venvs/audiosr"
QWEN3_VENV="$ROOT_DIR/.venvs/qwen3"
COSYVOICE_VENV="$ROOT_DIR/.venvs/cosyvoice"

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

if [[ -d "$QWEN3_VENV" ]]; then
  # shellcheck disable=SC1091
  source "$QWEN3_VENV/bin/activate"
  python -m pip freeze > "$ROOT_DIR/requirements-qwen3.lock.txt"
  deactivate || true
fi

if [[ -d "$COSYVOICE_VENV" ]]; then
  # shellcheck disable=SC1091
  source "$COSYVOICE_VENV/bin/activate"
  python -m pip freeze > "$ROOT_DIR/requirements-cosyvoice.lock.txt"
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
if [[ -f "$ROOT_DIR/requirements-qwen3.lock.txt" ]]; then
  echo "Wrote requirements-qwen3.lock.txt"
fi
if [[ -f "$ROOT_DIR/requirements-cosyvoice.lock.txt" ]]; then
  echo "Wrote requirements-cosyvoice.lock.txt"
fi
if [[ -f "$ROOT_DIR/requirements-audiosr.lock.txt" ]]; then
  echo "Wrote requirements-audiosr.lock.txt"
fi
