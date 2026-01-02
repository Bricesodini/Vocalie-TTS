#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Starting backend (API) on http://127.0.0.1:8000"
"$ROOT_DIR/scripts/dev-backend.sh" &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting frontend on http://localhost:3000"
"$ROOT_DIR/scripts/dev-frontend.sh"
