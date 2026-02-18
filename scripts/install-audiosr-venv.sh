#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venvs/audiosr"

PYTHON_BIN="python3.11"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.11+ and try again." >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install -U pip "setuptools<81" wheel
if [[ -f "$ROOT_DIR/requirements-audiosr.lock.txt" ]]; then
  pip install -r "$ROOT_DIR/requirements-audiosr.lock.txt"
else
  pip install -r "$ROOT_DIR/requirements-audiosr.in"
fi

echo "Checking AudioSR dependencies..."
"$VENV_DIR/bin/python" -c "import cog, pyloudnorm, matplotlib, torchcodec; print('deps ok')"
"$VENV_DIR/bin/python" - <<'PY'
try:
    import pkg_resources  # noqa: F401
except ImportError as exc:
    raise RuntimeError("setuptools version incompatible: pkg_resources missing") from exc
print("pkg_resources ok")
PY
"$VENV_DIR/bin/python" -c "import librosa, soundfile; print('audio io ok')"
"$VENV_DIR/bin/python" -c "import sys; print(sys.version)"

echo "AudioSR venv ready: $VENV_DIR"
