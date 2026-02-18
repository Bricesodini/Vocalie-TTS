#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"

mkdir -p "$RUN_DIR"

start_process() {
  local name="$1"
  local cmd="$2"
  echo "Starting $name..."
  bash -lc "$cmd" &
  local pid=$!
  echo "$pid" > "$RUN_DIR/${name}.pid"
}

start_process "backend" "$ROOT_DIR/scripts/dev-backend.sh"
start_process "frontend" "$ROOT_DIR/scripts/dev-frontend.sh"

if [[ "${WITH_COCKPIT:-0}" == "1" ]]; then
  start_process "cockpit" "python ui_gradio/cockpit.py"
fi

echo "Backend: http://127.0.0.1:8018"
echo "Frontend: http://localhost:3018"
if [[ "${WITH_COCKPIT:-0}" == "1" ]]; then
  echo "Cockpit: http://127.0.0.1:7860"
fi

echo "PIDs stored in $RUN_DIR"

echo "Press Ctrl+C to stop all."

cleanup() {
  "$ROOT_DIR/scripts/stop.sh" || true
}
trap cleanup INT TERM

wait
