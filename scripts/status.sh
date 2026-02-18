#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "No .run directory found. Nothing tracked."
  exit 0
fi

check_pid() {
  local name="$1"
  local pid_file="$RUN_DIR/${name}.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "$name: not tracked"
    return
  fi
  local pid
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "$name: running (pid=$pid)"
  else
    echo "$name: stopped (stale pid=$pid)"
  fi
}

check_pid "backend"
check_pid "frontend"
check_pid "cockpit"

if command -v lsof >/dev/null 2>&1; then
  echo ""
  echo "Ports:"
  for port in 8018 3018 7860; do
    if lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "- $port: listening"
    else
      echo "- $port: free"
    fi
  done
fi
