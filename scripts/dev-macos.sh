#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/.run"
FRONTEND_DIR="$ROOT_DIR/frontend"

if ! command -v node >/dev/null 2>&1; then
  echo "node not found. Install Node.js >= 20 and try again." >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

install_frontend_deps() {
  if [[ -d "$FRONTEND_DIR/node_modules" ]]; then
    return
  fi

  echo "Installing frontend dependencies (macOS workflow)"
  (
    cd "$FRONTEND_DIR"
    npm install --include=optional --no-audit --progress=false
  )
}

start_process() {
  local name="$1"
  local cmd="$2"
  echo "Starting $name..."
  bash -lc "$cmd" &
  local pid=$!
  echo "$pid" > "$RUN_DIR/${name}.pid"
}

install_frontend_deps

start_process "backend" "$ROOT_DIR/scripts/dev-backend.sh"
start_process "frontend" "cd $FRONTEND_DIR && NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE:-http://127.0.0.1:8000} npm run dev"

echo "Backend: http://127.0.0.1:8000"
echo "Frontend: http://localhost:3000"
echo "PIDs stored in $RUN_DIR"
echo "Press Ctrl+C to stop all."

cleanup() {
  "$ROOT_DIR/scripts/stop.sh" || true
}
trap cleanup INT TERM

wait
