#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venvs/chatterbox"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 not found. Install Python 3.11 and try again." >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR"
  python3.11 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install -U pip setuptools wheel
export PIP_NO_BUILD_ISOLATION=1
pip install "numpy<1.26,>=1.24"
pip install -r requirements-chatterbox.txt

echo "Chatterbox venv ready: $VENV_DIR"
