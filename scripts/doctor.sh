#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ok=0
warn=0

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    echo "[ok] $name: $(command -v "$name")"
    ok=$((ok+1))
  else
    echo "[warn] $name missing"
    warn=$((warn+1))
  fi
}

check_version_ge() {
  local name="$1"
  local current="$2"
  local required="$3"
  if [[ "$current" == "$required" || "$current" > "$required" ]]; then
    echo "[ok] $name version: $current"
  else
    echo "[warn] $name version: $current (expected >= $required)"
    warn=$((warn+1))
  fi
}

check_cmd python3
check_cmd node
check_cmd npm
check_cmd ffmpeg

if command -v python3 >/dev/null 2>&1; then
  py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  check_version_ge "python3" "$py_ver" "3.11"
fi

if command -v node >/dev/null 2>&1; then
  node_ver=$(node -p "process.versions.node.split('.').slice(0,2).join('.')")
  check_version_ge "node" "$node_ver" "20"
fi

if [[ -d "$ROOT_DIR/.venv" ]]; then
  echo "[ok] .venv exists"
else
  echo "[warn] .venv missing"
  warn=$((warn+1))
fi

if [[ -d "$ROOT_DIR/.venvs/xtts" ]]; then
  echo "[ok] .venvs/xtts exists"
else
  echo "[warn] .venvs/xtts missing"
  warn=$((warn+1))
fi

if [[ -d "$ROOT_DIR/.venvs/piper" ]]; then
  echo "[ok] .venvs/piper exists"
else
  echo "[warn] .venvs/piper missing"
  warn=$((warn+1))
fi

if [[ -d "$ROOT_DIR/.venvs/chatterbox" ]]; then
  echo "[ok] .venvs/chatterbox exists"
else
  echo "[warn] .venvs/chatterbox missing"
  warn=$((warn+1))
fi

echo "Summary: ok=$ok warn=$warn"

if [[ "$warn" -gt 0 ]]; then
  exit 1
fi
