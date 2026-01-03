#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing core venv at $VENV_DIR. Run ./scripts/bootstrap.sh min first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -c "from backend_install.installer import run_install; ok, logs = run_install('bark'); print('\n'.join(logs)); raise SystemExit(0 if ok else 1)"
deactivate || true
