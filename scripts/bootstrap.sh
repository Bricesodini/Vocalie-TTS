#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_VENV="$ROOT_DIR/.venv"
CHATTERBOX_VENV="$ROOT_DIR/.venvs/chatterbox"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/bootstrap.sh min [--force]
  ./scripts/bootstrap.sh std [--force]
  ./scripts/bootstrap.sh clean
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
FORCE=0
if [[ "${2:-}" == "--force" ]]; then
  FORCE=1
fi

require_python() {
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11 not found. Install Python 3.11 and try again." >&2
    exit 1
  fi
}

install_core() {
  local created=0
  if [[ ! -d "$CORE_VENV" ]]; then
    echo "Creating core venv at $CORE_VENV"
    python3.11 -m venv "$CORE_VENV"
    created=1
  fi
  # shellcheck disable=SC1091
  source "$CORE_VENV/bin/activate"
  python -m pip install -U pip
  if [[ "$created" -eq 1 || "$FORCE" -eq 1 ]]; then
    echo "Installing core requirements"
    pip install -r "$ROOT_DIR/requirements.txt"
  else
    echo "Core venv exists; skipping requirements install (use --force to reinstall)."
  fi
  deactivate || true
}

install_chatterbox() {
  local created=0
  if [[ ! -d "$CHATTERBOX_VENV" ]]; then
    echo "Creating Chatterbox venv at $CHATTERBOX_VENV"
    python3.11 -m venv "$CHATTERBOX_VENV"
    created=1
  fi
  # shellcheck disable=SC1091
  source "$CHATTERBOX_VENV/bin/activate"
  if [[ "$created" -eq 1 || "$FORCE" -eq 1 ]]; then
    echo "Installing Chatterbox requirements"
    pip install -U pip setuptools wheel
    export PIP_NO_BUILD_ISOLATION=1
    pip install "numpy<1.26,>=1.24"
    pip install -r "$ROOT_DIR/requirements-chatterbox.txt"
  else
    echo "Chatterbox venv exists; skipping install (use --force to reinstall)."
  fi
  deactivate || true
}

install_std_engines() {
  # shellcheck disable=SC1091
  source "$CORE_VENV/bin/activate"
  python -c "from backend_install.installer import run_install; print(run_install('xtts'))"
  python -c "from backend_install.installer import run_install; print(run_install('piper'))"
  deactivate || true
}

clean_all() {
  echo "Removing $CORE_VENV and $ROOT_DIR/.venvs"
  rm -rf "$CORE_VENV" "$ROOT_DIR/.venvs"
}

cd "$ROOT_DIR"

case "$MODE" in
  min)
    require_python
    install_core
    install_chatterbox
    ;;
  std)
    require_python
    install_core
    install_chatterbox
    install_std_engines
    ;;
  clean)
    clean_all
    ;;
  *)
    usage
    exit 1
    ;;
esac
