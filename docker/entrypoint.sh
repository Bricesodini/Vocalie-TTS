#!/usr/bin/env bash
# Vocalie-TTS — Docker entrypoint
# Starts both backend (uvicorn) and frontend (next start) in the container.
set -euo pipefail

# --- Start backend ---
echo "Starting Vocalie-TTS backend on :8018..."
uvicorn backend.app:app \
  --host 127.0.0.1 \
  --port 8018 \
  --workers 1 \
  --no-access-log &

BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend..."
for i in $(seq 1 30); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8018/v1/health')" 2>/dev/null; then
    echo "Backend ready."
    break
  fi
  sleep 1
done

# --- Start frontend ---
echo "Starting Vocalie-TTS frontend on :3018..."
cd /app/frontend
PORT=3018 HOSTNAME=0.0.0.0 node server.js &

FRONTEND_PID=$!

# --- Forward signals ---
cleanup() {
  echo "Shutting down..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
cleanup